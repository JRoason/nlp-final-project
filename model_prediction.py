import numpy as np
import tensorflow as tf
import string
import pickle
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention, Input, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import tqdm
from preprocess import load_test_data
from model import load_model

# This file is used to make predictions on the test data using the trained model.


# This function takes the model and the test data as input and returns the predicted sequences of tokens.
def make_predictions(model, test_data, MAX_LEN=200):
    # Create a numpy array to store the predicted sequences of tokens
    predictions = np.zeros((len(test_data), MAX_LEN + 1))
    for i in tqdm.tqdm(range(len(test_data))):
        eng, it = test_data[i]

        # Tokenize and pad the English sentence for the model
        eng = eng_tokenizer.texts_to_sequences([eng])
        eng = pad_sequences(eng, maxlen=MAX_LEN, padding='post')
        target = np.zeros((1, MAX_LEN + 1))

        # Start token
        target[0, 0] = it_tokenizer.word_index['[start]']

        for t in range(1, MAX_LEN + 1):

            # Predict the next token
            prediction = model.predict([eng, target[:, :MAX_LEN + 1]], verbose=0)

            # Use greedy decoding to select the token with the highest probability
            predicted_id = np.argmax(prediction[0, t - 1, :])

            # Store the predicted token
            target[0, t] = predicted_id

            # If the model predicts the end token, stop predicting
            if predicted_id == it_tokenizer.word_index['[end]']:
                break

        predictions[i] = target

    return predictions


# This function takes a sequence of tokens as input and returns the corresponding sentence.
def decode_sequence(input_seq):
    # Create a dictionary to map token indices to words
    it_index_word = {v: k for k, v in it_tokenizer.word_index.items()}
    decoded_sentence = ''
    for i in range(1, MAX_LEN + 1):

        # Get the token index
        sampled_token_index = input_seq[i]

        # Convert the token index to a word
        sampled_token = it_index_word[sampled_token_index]

        # If the token is the end token, stop decoding
        if sampled_token == '[end]':
            break

        decoded_sentence += sampled_token + ' '

    return decoded_sentence


if __name__ == '__main__':
    # Constants
    VOCAB_SIZE = 10000 # Size of the vocabulary
    MAX_LEN = 200  # Maximum sequence length
    EMBEDDING_DIM = 100  # Dimension of the word embeddings

    test_pairs = load_test_data()

    test_pairs = test_pairs[:5000]

    # Load the tokenizers that were used to preprocess the training data
    with open('data/tokenizer_en.pickle', 'rb') as handle:
        eng_tokenizer = pickle.load(handle)

    with open('data/tokenizer_it.pickle', 'rb') as handle:
        it_tokenizer = pickle.load(handle)

    # Load the trained model
    model = load_model(VOCAB_SIZE, EMBEDDING_DIM)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    predictions = make_predictions(model, test_pairs, MAX_LEN=MAX_LEN)

    # Save the predictions to a file
    for prediction in predictions:
        with open('data/predictions.txt', 'a') as f:
            f.write(decode_sequence(prediction) + '\n')
