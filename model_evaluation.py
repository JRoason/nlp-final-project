import numpy as np
import tensorflow as tf
import string
import pickle
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention, Input, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import tqdm

# Constants
VOCAB_SIZE = 10000
MAX_LEN = 200  # Maximum sequence length
EMBEDDING_DIM = 100  # Dimension of the word embeddings

def load_test_data():
    with open('europarl-v7-en-test.txt') as english_file, open('europarl-v7-it-test.txt') as italian_file:
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

    return text_pairs


test_pairs = load_test_data()

test_pairs = test_pairs[:5000]

with open('tokenizer_en.pickle', 'rb') as handle:
    eng_tokenizer = pickle.load(handle)

with open('tokenizer_it.pickle', 'rb') as handle:
    it_tokenizer = pickle.load(handle)

def load_model():
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

    model.load_weights('lstm_attention_model_best.weights.h5')

    return model


model = load_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def make_predictions(model, test_data):
    predictions = np.zeros((len(test_data), MAX_LEN + 1))
    for i in tqdm.tqdm(range(len(test_data))):
        eng, it = test_data[i]
        if i % 1000 == 0:
            print(i)
        eng = eng_tokenizer.texts_to_sequences([eng])
        eng = pad_sequences(eng, maxlen=MAX_LEN, padding='post')
        target = np.zeros((1, MAX_LEN + 1))
        target[0, 0] = it_tokenizer.word_index['[start]']
        for t in range(1, MAX_LEN + 1):
            prediction = model.predict([eng, target[:, :MAX_LEN+1]], verbose=0)
            predicted_id = np.argmax(prediction[0, t - 1, :])
            target[0, t] = predicted_id
            if predicted_id == it_tokenizer.word_index['[end]']:
                break
        predictions[i] = target

    return predictions


def decode_sequence(input_seq):
    it_index_word = {v: k for k, v in it_tokenizer.word_index.items()}
    decoded_sentence = ''
    for i in range(1, MAX_LEN + 1):
        sampled_token_index = input_seq[i]
        sampled_token = it_index_word[sampled_token_index]
        if sampled_token == '[end]':
            break
        decoded_sentence += sampled_token + ' '
    return decoded_sentence

predictions = make_predictions(model, test_pairs)

for prediction in predictions:
    with open('predictions.txt', 'a') as f:
        f.write(decode_sequence(prediction) + '\n')

# for i in range(5):
#     print('English:', test_data[i][0])
#     print('Italian:', test_data[i][1])
#     print('Predicted:', decode_sequence(predictions[i]))
#     print()

def plot_loss():
    with open('modelTrainDict.pickle', 'rb') as handle:
        history = pickle.load(handle)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend()

    plt.savefig('model_loss.png')
    plt.show()

