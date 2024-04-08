# English-Italian Neural Machine Translation with RNNs

This repository contains the code for training and testing a Neural Machine Translation (NMT) system for translating English sentences into Italian.
This model was developed as part of the final project for the course "Natural Language Processing" at the University of Groningen during the academic year 2023/2024.
The system is based on a Recurrent Neural Network (RNN) architecture with an encoder-decoder structure. The code is written in Python and uses the Keras library.

The repository contains a Pipfile with the required dependencies. The code was developed using Python 3.10.12 and Keras 3.1.1. To install the dependencies, run the following command:

```bash
pipenv install
```

To activate the virtual environment, run:

```bash
pipenv shell
```

The code should be run from the root directory of the repository. The main script for training the NMT system is `model_training.py`, and the script for generating output from the model is `model_prediction.py`. The training and testing data should be placed in the `data` directory, as described below.

There is also a baseline model that does not contain an attention mechanism. This model can be trained using the `baseline_training.py` script, prediction generated with `baseline_prediction.py`.

To evaluate both models using BLEU, METEOR and BERT scores, run the `evaluation.py` script.

## Data

The training data for this project is not present in this repository (size constraints). The data used for training the model is the EuroParl parallel corpus, which contains the proceedings of the European Parliament in 21 European languages. The English-Italian portion of the corpus was used for training the NMT system. The data can be downloaded [here](https://www.statmt.org/europarl/).

The files in the repository expect the training data to be in the `data` directory. The training data should be split into two files: `europarl-v7-en-train.txt` for the English sentences and `europarl-v7-it-train.txt` for the corresponding Italian translations.
The test data should be in the `data` directory as well, with the files `europarl-v7-en-test.txt` and `europarl-v7-it-test.txt`.

To split the data into training and testing data, assuming the data is in the root directory of the this repository, run the following commands:

```bash
python dataset.py
```

Note that this file assumes your files are named `europarl-v7-it.txt` and `europarl-v7-en.txt`. If they are named differently, change the file names in the code above.

The file `preprocess_data.py` contains two functions for loading and preprocessing the data and preparing it for training the NMT system. These functions strip whitespace, remove punctuation, and add start and end tokens to the Italian sentence, for use with the decoder.

Note that the half the training data is used due to the size of the corpus. This can be changed, and ought to be to produce a better model.

Additionally, only the first 5000 sentences of the test data are used for evaluation. This can also be changed, but evaluation takes a long time.

## Model

Due to size constraints, the weights of the trained models are not included in this repository. Therefore, the models need to be trained from scratch using the training script. The weights are saved in the `data` directory after training. The main model's architecture is defined in the `model.py` file, while the baseline model's architecture is defined in `baseline_model.py`.

## Training

During training, the `dataset.py` file used the Tensorflow `tf.data.Dataset` API to create a dataset pipeline for training the model. The dataset pipeline reads the training data from the files, preprocesses it, and batches it for training the model. The dataset pipeline is used in the `model_training.py` script to train the NMT system.

To train the LSTM with attention model, run the following command:

```bash
python model_training.py
```

To train the baseline model, run the following command:

```bash
python baseline_training.py
```

## Prediction

To generate translations from the trained model, run the following command:

```bash
python model_prediction.py
```

To generate translations from the baseline model, run the following command:

```bash
python baseline_prediction.py
```

The predictions are saved in the `data` directory as `predictions.txt` and `baseline_predictions.txt`.

## Evaluation

To evaluate the models using BLEU, METEOR and BERT scores, run the following command:

```bash
python evaluation.py
```

The output will be printed to the console.

Note that the files 'predictions.txt' and 'baseline_predictions.txt' are present in the repository, so the evaluation script can be run without training the models.