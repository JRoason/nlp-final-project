import string
import nltk
from nltk.translate.bleu_score import corpus_bleu

def load_test_data():
    with open('europarl-v7-it-test.txt') as italian_file:
        italian_sentences = italian_file.readlines()

    text = []

    for line in italian_sentences:
        it = line.strip()
        it = it.translate(str.maketrans('', '', string.punctuation))
        it = it.lower()
        it = it.split()
        text.append(it)

    return text


test_data = load_test_data()

test_data = test_data[:5000]

with open('predictions.txt') as f:
    predictions = f.readlines()
    text = []
    for prediction in predictions:
        pred = prediction.strip()
        pred = pred.split()
        text.append(pred)

predictions = text

test_data = [[sentence] for sentence in test_data]

bleu_score = corpus_bleu(test_data, predictions, weights=(0.25, 0.25, 0.25, 0.25))

print(f'BLEU score: {bleu_score}')