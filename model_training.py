import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.data as tfdata
import keras
from model import build_model
from preprocess import load_train_val_data
from dataset import make_dataset

# This script is used to train the final model.
# The model is trained on the training data and validated on the validation data.

if __name__ == '__main__':
    # Constants
    VOCAB_SIZE = 10000
    MAX_LEN = 200  # Maximum sequence length
    EMBEDDING_DIM = 100  # Dimension of the word embeddings
    EPOCHS = 10  # Number of epochs
    BATCH_SIZE = 128  # maybe increase for runtime

    train_pairs, val_pairs = load_train_val_data()

    # Reduce the size of the training and validation data to speed up training
    train_pairs = train_pairs[:len(train_pairs) // 2]
    val_pairs = val_pairs[:len(val_pairs) // 2]

    train_eng_texts = [pair[0] for pair in train_pairs]
    train_it_texts = [pair[1] for pair in train_pairs]

    # Fit the tokenizer on the English training data
    tokenizer_en = Tokenizer(num_words=VOCAB_SIZE,
                             filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')  # convert sentences into sequences of integers.
    tokenizer_en.fit_on_texts(train_eng_texts)  # dictionary that maps words to numbers.

    # Fit the tokenizer on the Italian training data
    tokenizer_it = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
    tokenizer_it.fit_on_texts(train_it_texts)

    # Save the tokenizers to disk
    with open('data/tokenizer_en.pickle', 'wb') as handle:
        pickle.dump(tokenizer_en, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/tokenizer_it.pickle', 'wb') as handle:
        pickle.dump(tokenizer_it, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Create the training and validation datasets using the make_dataset function
    train_ds = make_dataset(train_pairs, tokenizer_en, tokenizer_it, MAX_LEN=MAX_LEN, BATCH_SIZE=BATCH_SIZE)
    val_ds = make_dataset(val_pairs, tokenizer_en, tokenizer_it, MAX_LEN=MAX_LEN, BATCH_SIZE=BATCH_SIZE)

    # Free up memory
    del train_pairs
    del val_pairs
    del train_eng_texts
    del train_it_texts

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Save the model architecture to disk
    keras.utils.plot_model(model, to_file='plots/final_model.png', show_layer_activations=True)

    # Define the callbacks, which will save the best model weights and stop training early if the validation loss
    # does not improve
    callbacks = [
        ModelCheckpoint('data/lstm_attention_model_best.weights.h5', save_best_only=True, save_weights_only=True,
                        monitor='val_loss', mode='min'),
        EarlyStopping(patience=2, monitor='val_loss')
    ]

    history = model.fit(train_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val_ds, verbose=1,
                        callbacks=callbacks)

    # Save the training history to disk for later plotting
    with open('data/history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Save the final model weights to disk
    model.save_weights('data/lstm_attention_model_final.weights.h5')

