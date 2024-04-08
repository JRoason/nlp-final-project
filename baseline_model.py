import tensorflow as tf
import random
import string
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.data as tfdata
import keras
from preprocess import load_train_val_data
from dataset import make_dataset

# This file contains the code to build the model for the baseline model. The model is a simple encoder-decoder model
# with an LSTM layer for the encoder and decoder, without any attention mechanism


# Load the saved model weights
def load_model(VOCAB_SIZE, EMBEDDING_DIM):
    model_skeleton = build_model(VOCAB_SIZE, EMBEDDING_DIM)

    model_skeleton.load_weights('data/baseline_model_best.weights.h5')

    return model_skeleton


# Build the model
def build_model(VOCAB_SIZE, EMBEDDING_DIM):

    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
    _, state_h, state_c = LSTM(1024, return_state=True)(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)
    decoder_lstm = LSTM(1024, return_sequences=True)
    decoder_hidden_states = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

    # Output layer
    decoder_dense = Dense(VOCAB_SIZE, activation='softmax')(decoder_hidden_states)

    return tf.keras.Model([encoder_inputs, decoder_inputs], decoder_dense)

