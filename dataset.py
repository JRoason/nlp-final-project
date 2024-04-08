from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.data as tfdata
from sklearn.model_selection import train_test_split


# The following code is based on the following tutorial:
# https://keras.io/examples/nlp/neural_machine_translation_with_transformer/#vectorizing-the-text-data


# This function formats the dataset into the format that the model expects.
# The model expects a dictionary with two keys: 'encoder_inputs' and 'decoder_inputs'.
# The 'encoder_inputs' key should have the English sentences as the values.
# The 'decoder_inputs' key should have the Italian sentences as the values.
# The Italian sentences should be shifted by one position to the right. This is because the model is trained to predict
# the next word in the Italian sentence given the previous words in the Italian sentence.

def format_dataset(eng, it):
    return (
        {
            'encoder_inputs': eng,
            'decoder_inputs': it[:, :-1],
        },
        it[:, 1:],
    )


# This function creates a dataset from the pairs of English and Italian sentences.
# The dataset is created using the TensorFlow Dataset API.
# The English sentences are tokenized using the tokenizer_en tokenizer.
# The Italian sentences are tokenized using the tokenizer_it tokenizer.
# The English sentences are padded to a maximum length of MAX_LEN.
# The Italian sentences are padded to a maximum length of MAX_LEN + 1.
# The dataset is batched with a batch size of BATCH_SIZE.
# The dataset is then cached, shuffled, and prefetched.
# The goal of this is to speed up the training process by preloading the data into memory and shuffling the data.
def make_dataset(pairs, tokenizer_en, tokenizer_it, MAX_LEN=200, BATCH_SIZE=128):
    eng_texts, it_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    it_texts = list(it_texts)
    eng = tokenizer_en.texts_to_sequences(eng_texts)
    eng = pad_sequences(eng, maxlen=MAX_LEN, padding='post')
    it = tokenizer_it.texts_to_sequences(it_texts)
    it = pad_sequences(it, maxlen=MAX_LEN + 1, padding='post')
    dataset = tfdata.Dataset.from_tensor_slices((eng, it))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(1024).prefetch(buffer_size=tfdata.AUTOTUNE)


if __name__ == '__main__':

    # Load English-Italian Europarl dataset
    with open('europarl-v7-it.txt') as italian_file, open('europarl-v7-en.txt') as english_file:
        italian_sentences = italian_file.readlines()
        english_sentences = english_file.readlines()

    X_en_train, X_en_test, X_it_train, X_it_test = train_test_split(english_sentences, italian_sentences, test_size=0.2)

    with open('data/europarl-v7-it-train.txt', 'w') as italian_file:
        italian_file.writelines(X_it_train)

    with open('data/europarl-v7-en-train.txt', 'w') as english_file:
        english_file.writelines(X_en_train)

    with open('data/europarl-v7-it-test.txt', 'w') as italian_file:
        italian_file.writelines(X_it_test)

    with open('data/europarl-v7-en-test.txt', 'w') as english_file:
        english_file.writelines(X_en_test)

