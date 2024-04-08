from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention, Input, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# This file contains the model architecture for the LSTM with Attention model.


# Load the model weights from the file
def load_model(VOCAB_SIZE, EMBEDDING_DIM):
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM)

    model.load_weights('data/lstm_attention_model_best.weights.h5')

    return model


# Build the model architecture
def build_model(VOCAB_SIZE, EMBEDDING_DIM):

    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
    hidden_states, state_h, state_c = LSTM(1024, return_sequences=True, return_state=True)(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)
    decoder_lstm = LSTM(1024, return_sequences=True)
    decoder_hidden_states = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

    # Attention
    attention = Attention()([decoder_hidden_states, hidden_states])
    combined = Concatenate()([attention, decoder_hidden_states])
    combined_dense = Dense(1024, activation='tanh')(combined)

    # Output
    decoder_dense = Dense(VOCAB_SIZE, activation='softmax')(combined_dense)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_dense)

    return model
