import numpy as np
import tensorflow as tf
import random
import string
import re
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention, Input, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.data as tfdata
import tensorflow.strings as tf_strings
import matplotlib.pyplot as plt
import keras


# Constants
VOCAB_SIZE = 10000
MAX_LEN = 200  # Maximum sequence length
EMBEDDING_DIM = 100  # Dimension of the word embeddings
EPOCHS = 10
BATCH_SIZE = 128 # maybe increase for runtime
def load_train_val_data():
    with open('europarl-v7-en-train.txt') as english_file, open('europarl-v7-it-train.txt') as italian_file:
        english_sentences = english_file.readlines()
        italian_sentences = italian_file.readlines()

    text_pairs = []

    for line in zip(english_sentences, italian_sentences):
        eng = line[0].strip()
        it = line[1].strip()
        eng = eng.translate(str.maketrans('', '', string.punctuation))
        it = it.translate(str.maketrans('', '', string.punctuation))
        it = '[start] ' + it + ' [end]'
        text_pairs.append((eng, it))

    del english_sentences
    del italian_sentences

    random.shuffle(text_pairs)

    num_val_samples = int(0.2 * len(text_pairs))
    num_train_samples = len(text_pairs) - num_val_samples

    return text_pairs[:num_train_samples], text_pairs[num_train_samples:]


train_pairs, val_pairs = load_train_val_data()

train_pairs = train_pairs[:len(train_pairs) // 2]
val_pairs = val_pairs[:len(val_pairs) // 2]

train_eng_texts = [pair[0] for pair in train_pairs]
train_it_texts = [pair[1] for pair in train_pairs]

# Preprocessing English sentences
tokenizer_en = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')  #convert sentences into sequences of integers.
tokenizer_en.fit_on_texts(train_eng_texts) # dictionary that maps words to numbers.

# Preprocessing Italian sentences
tokenizer_it = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
tokenizer_it.fit_on_texts(train_it_texts)

# with open('tokenizer_en_none.pickle', 'wb') as handle:
#     pickle.dump(tokenizer_en, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('tokenizer_it_none.pickle', 'wb') as handle:
#     pickle.dump(tokenizer_it, handle, protocol=pickle.HIGHEST_PROTOCOL)


def format_dataset(eng, it):
    return (
        {
            'encoder_inputs': eng,
            'decoder_inputs': it[:, :-1],
        },
        it[:, 1:],
    )


def make_dataset(pairs):
    eng_texts, it_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    it_texts = list(it_texts)
    eng = tokenizer_en.texts_to_sequences(eng_texts)
    eng = pad_sequences(eng, maxlen=MAX_LEN, padding='post')
    it = tokenizer_it.texts_to_sequences(it_texts)
    it = pad_sequences(it, maxlen=MAX_LEN+1, padding='post')
    dataset = tfdata.Dataset.from_tensor_slices((eng, it))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(1024).prefetch(buffer_size=tfdata.AUTOTUNE)


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

del train_pairs
del val_pairs

encoder_inputs = Input(shape=(None,), name='encoder_inputs')

encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)

hidden_states, state_h, state_c = LSTM(1024, return_sequences=True, return_state=True)(encoder_embedding)

decoder_inputs = Input(shape=(None,), name='decoder_inputs')

decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)

decoder_lstm = LSTM(1024, return_sequences=True)

decoder_hidden_states = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

attention = Attention()([decoder_hidden_states, hidden_states])

combined = Concatenate()([attention, decoder_hidden_states])

combined_dense = Dense(1024, activation='tanh')(combined)

decoder_dense = Dense(VOCAB_SIZE, activation='softmax')(combined_dense)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_dense)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# keras.utils.plot_model(model, to_file='final_model_none.png', show_layer_activations=True)

callbacks = [
    ModelCheckpoint('lstm_attention_model_best_none_test.weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(patience=2, monitor='val_loss')
    ]

history = model.fit(train_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val_ds, verbose=1, callbacks=callbacks)

with open('modelTrainDictNone_test.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.save_weights("lstm_attention_model_final_none_test.weights.h5")

