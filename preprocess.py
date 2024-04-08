import string
import random

# This file contains functions to load the training, validation, and test data
# from the Europarl dataset. The data is preprocessed by removing punctuation
# and adding start and end tokens to the Italian sentences.
# The training and validation data is split into two lists of tuples, where
# each tuple contains an English sentence and its corresponding Italian sentence.
# The training data is also randomly shuffled.

# The test data is loaded into a list of tuples, where each tuple contains an
# English sentence and its corresponding Italian sentence.

def load_train_val_data():
    # Load the English and Italian sentences from the training files
    with open('data/europarl-v7-en-train.txt') as english_file, open('data/europarl-v7-it-train.txt') as italian_file:
        english_sentences = english_file.readlines()
        italian_sentences = italian_file.readlines()

    text_pairs = []

    # Preprocess the sentences and add them to the list of text pairs
    for line in zip(english_sentences, italian_sentences):
        eng = line[0].strip()
        it = line[1].strip()
        eng = eng.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation from the English sentences
        it = it.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation from the Italian sentences
        it = '[start] ' + it + ' [end]' # Add start and end tokens to the Italian sentences
        text_pairs.append((eng, it))

    del english_sentences
    del italian_sentences

    random.shuffle(text_pairs)

    # Split the data into training and validation sets (80% training, 20% validation)
    num_val_samples = int(0.2 * len(text_pairs))
    num_train_samples = len(text_pairs) - num_val_samples

    return text_pairs[:num_train_samples], text_pairs[num_train_samples:]


def load_test_data():
    with open('data/europarl-v7-en-test.txt') as english_file, open('data/europarl-v7-it-test.txt') as italian_file:
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

