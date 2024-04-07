
import numpy as np
import tensorflow as tf
import random
import string
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention, Input, TimeDistributed, Concatenate, TextVectorization, Bidirectional
import tensorflow.data as tfdata
import tensorflow.strings as tf_strings
import matplotlib.pyplot as plt
import keras


# Constants
#MAX_WORDS = 10000  # Consider only the top 10,000 most common words
VOCAB_SIZE = 10000
MAX_LEN = 200  # Maximum sequence length
EMBEDDING_DIM = 100  # Dimension of the word embeddings
EPOCHS = 20
BATCH_SIZE = 128 # maybe increase for runtime
def load_train_val_data():
    with open('europarl-v7-en-train.txt') as english_file, open('europarl-v7-it-train.txt') as italian_file:
        english_sentences = english_file.readlines()
        italian_sentences = italian_file.readlines()

    text_pairs = []

    for line in zip(english_sentences, italian_sentences):
        eng = line[0].strip()
        it = line[1].strip()
        it = '[start] ' + it + ' [end]'
        text_pairs.append((eng, it))

    del english_sentences
    del italian_sentences

    random.shuffle(text_pairs)

    num_val_samples = int(0.2 * len(text_pairs))
    num_train_samples = len(text_pairs) - num_val_samples

    return text_pairs[:num_train_samples], text_pairs[num_train_samples:]


def preprocess_text(input_string):
    strip_chars = string.punctuation
    strip_chars = strip_chars.replace('[', '')
    strip_chars = strip_chars.replace(']', '')
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, '[%s]' % re.escape(strip_chars), '')

eng_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_LEN
)

it_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_LEN + 1,
    standardize=preprocess_text
)

train_pairs, val_pairs = load_train_val_data()

train_pairs = train_pairs[:len(train_pairs) // 2]
val_pairs = val_pairs[:len(val_pairs) // 2]

train_eng_texts = [pair[0] for pair in train_pairs]
train_it_texts = [pair[1] for pair in train_pairs]

eng_vectorize_layer.adapt(train_eng_texts)
it_vectorize_layer.adapt(train_it_texts)

pickle.dump({'config': eng_vectorize_layer.get_config(),
             'weights': eng_vectorize_layer.get_weights()}
            , open("eng_vectorize_layer.pickle", "wb"))

pickle.dump({'config': it_vectorize_layer.get_config(),
             'weights': it_vectorize_layer.get_weights()}
            , open("it_vectorize_layer.pickle", "wb"))


def format_dataset(eng, it):
    eng = eng_vectorize_layer(eng)
    it = it_vectorize_layer(it)
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
    dataset = tfdata.Dataset.from_tensor_slices((eng_texts, it_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(1024).prefetch(buffer_size=tfdata.AUTOTUNE)


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

del train_pairs
del val_pairs

encoder_inputs = Input(shape=(MAX_LEN,), name='encoder_inputs')

encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
hidden_states, state_h, state_c = LSTM(1024, return_sequences=True, return_state=True)(encoder_embedding)

decoder_inputs = Input(shape=(MAX_LEN,), name='decoder_inputs')

decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)

decoder_lstm = LSTM(1024, return_sequences=True, return_state=True)

decoder_hidden_states, final_hidden_state, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

attention = Attention()([decoder_hidden_states, hidden_states])

combined = Concatenate()([attention, decoder_hidden_states])

combined_dense = Dense(1024, activation='tanh')(combined)

decoder_dense = Dense(VOCAB_SIZE, activation='softmax')(combined_dense)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_dense)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

keras.utils.plot_model(model, to_file='final_model.png', show_layer_activations=True)

callbacks = [
    ModelCheckpoint('lstm_attention_model_best.weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(patience=2, monitor='val_loss')
    ]

history = model.fit(train_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val_ds, verbose=1, callbacks=callbacks)

with open('modelTrainDict.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.save_weights("lstm_attention_model_final.weights.h5")
