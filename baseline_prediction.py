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
from baseline_model import load_model
from model_prediction import make_predictions, decode_sequence

# This script is used to generate predictions using the baseline model.
# The predictions are saved to a file called 'predictions_baseline.txt'.
# Implementation is based on the code from model_prediction.py.

if __name__ == '__main__':
    # Constants
    VOCAB_SIZE = 10000
    MAX_LEN = 200  # Maximum sequence length
    EMBEDDING_DIM = 100  # Dimension of the word embeddings

    test_pairs = load_test_data()

    test_pairs = test_pairs[:5000]

    # Load the trained tokenizers for the baseline training data
    with open('data/tokenizer_en_baseline.pickle', 'rb') as handle:
        eng_tokenizer = pickle.load(handle)

    with open('data/tokenizer_it_baseline.pickle', 'rb') as handle:
        it_tokenizer = pickle.load(handle)

    model = load_model(VOCAB_SIZE, EMBEDDING_DIM)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    predictions = make_predictions(model, test_pairs)

    for prediction in predictions:
        with open('data/predictions_baseline.txt', 'a') as f:
            f.write(decode_sequence(prediction) + '\n')
