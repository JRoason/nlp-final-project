import string
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer

nltk.download('wordnet')

# This file is used to evaluate the performance of the model. It calculates the BLEU score, METEOR score, and BERTScore


# Load the test data. This function differs from the one in preprocess.py because it returns the text as a list of strings
# I am aware that this function is duplicated in this file, but I decided to keep it here for clarity
def load_test_data():
    with open('data/europarl-v7-it-test.txt') as italian_file:
        italian_sentences = italian_file.readlines()

    text = []

    for line in italian_sentences:
        it = line.strip()
        it = it.translate(str.maketrans('', '', string.punctuation))
        it = it.lower()
        text.append(it)

    return text


if __name__ == '__main__':
    # Load the first 5000 test data samples and the predictions
    test_data = load_test_data()
    test_data = test_data[:5000]

    with open('data/predictions.txt') as f:
        predictions = f.readlines()
        text = []
        for prediction in predictions:
            pred = prediction.strip()
            text.append(pred)

    predictions = text

    # Load the baseline predictions
    with open('data/predictions_baseline.txt') as f:
        predictions_baseline = f.readlines()
        text = []
        for prediction in predictions_baseline:
            pred = prediction.strip()
            text.append(pred)

    predictions_baseline = text

    # Calculate the BERTScore
    scorer = BERTScorer(lang='it')

    P, R, F1 = scorer.score(test_data, predictions, verbose=True)

    P_baseline, R_baseline, F1_baseline = scorer.score(test_data, predictions_baseline, verbose=True)

    print()
    print(f'Precision Attention Model: {P.mean()}')
    print(f'Recall Attention Model: {R.mean()}')
    print(f'F1 Score Attention Model: {F1.mean()}')
    print()

    print()
    print(f'Precision Baseline Model: {P_baseline.mean()}')
    print(f'Recall Baseline Model: {R_baseline.mean()}')
    print(f'F1 Score Baseline Model: {F1_baseline.mean()}')
    print()

    # Convert the predictions and test data to lists of lists

    for i in range(len(predictions)):
        test_data[i] = test_data[i].split()
        predictions[i] = predictions[i].split()
        predictions_baseline[i] = predictions_baseline[i].split()
    test_data = [[sentence] for sentence in test_data]

    # Calculate the BLEU score
    bleu_score = corpus_bleu(test_data, predictions, weights=(0.25, 0.25, 0.25, 0.25))

    print()
    print(f'BLEU Score Attention Model: {bleu_score}')
    print()

    bleu_score_baseline = corpus_bleu(test_data, predictions_baseline, weights=(0.25, 0.25, 0.25, 0.25))

    print()
    print(f'BLEU Score Baseline Model: {bleu_score_baseline}')
    print()

    # Calculate the mean METEOR score
    meteor_scores = []
    for reference, hypothesis in zip(test_data, predictions):
        current_meteor_score = meteor_score(reference, hypothesis)
        meteor_scores.append(current_meteor_score)
    meteor_avg_score = sum(meteor_scores) / len(meteor_scores)

    meteor_scores_baseline = []

    for reference, hypothesis in zip(test_data, predictions_baseline):
        current_meteor_score = meteor_score(reference, hypothesis)
        meteor_scores_baseline.append(current_meteor_score)
    meteor_avg_score_baseline = sum(meteor_scores_baseline) / len(meteor_scores_baseline)

    print()
    print(f'METEOR Score Attention Model:  {meteor_avg_score}')
    print()

    print()
    print(f'METEOR Score Baseline Model:  {meteor_avg_score_baseline}')
    print()
