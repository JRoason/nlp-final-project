import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.data as tfdata
import keras
from baseline_model import build_model
from preprocess import load_train_val_data
from dataset import make_dataset

# This file is identical to model_training.py, except that it uses the baseline model instead of the final model.

if __name__ == '__main__':
    # Constants
    VOCAB_SIZE = 10000
    MAX_LEN = 200  # Maximum sequence length
    EMBEDDING_DIM = 100  # Dimension of the word embeddings
    EPOCHS = 10
    BATCH_SIZE = 128  # maybe increase for runtime

    train_pairs, val_pairs = load_train_val_data()

    train_pairs = train_pairs[:len(train_pairs) // 2]
    val_pairs = val_pairs[:len(val_pairs) // 2]

    train_eng_texts = [pair[0] for pair in train_pairs]
    train_it_texts = [pair[1] for pair in train_pairs]

    tokenizer_en = Tokenizer(num_words=VOCAB_SIZE,
                             filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')  # convert sentences into sequences of integers.
    tokenizer_en.fit_on_texts(train_eng_texts)  # dictionary that maps words to numbers.

    tokenizer_it = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
    tokenizer_it.fit_on_texts(train_it_texts)

    with open('data/tokenizer_en_baseline.pickle', 'wb') as handle:
        pickle.dump(tokenizer_en, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/tokenizer_it_baseline.pickle', 'wb') as handle:
        pickle.dump(tokenizer_it, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_ds = make_dataset(train_pairs, tokenizer_en, tokenizer_it, MAX_LEN=MAX_LEN, BATCH_SIZE=BATCH_SIZE)
    val_ds = make_dataset(val_pairs, tokenizer_en, tokenizer_it, MAX_LEN=MAX_LEN, BATCH_SIZE=BATCH_SIZE)

    del train_pairs
    del val_pairs
    del train_eng_texts
    del train_it_texts

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    keras.utils.plot_model(model, to_file='plots/baseline_model.png', show_layer_activations=True)

    callbacks = [
        ModelCheckpoint('data/baseline_model_best.weights.h5', save_best_only=True, save_weights_only=True,
                        monitor='val_loss', mode='min'),
        EarlyStopping(patience=2, monitor='val_loss')
    ]

    history = model.fit(train_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val_ds, verbose=1,
                        callbacks=callbacks)

    with open('data/history_baseline.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model.save_weights('data/baseline_model_final.weights.h5')

