#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage example for LstmTextClassifier on the Newsgroups Dataset."""

import keras
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from mlblocks.components.pipelines.text.lstm_text import LstmTextClassifier

print("""
============================================
Testing Text LSTM
============================================
""")

newsgroups = fetch_20newsgroups()
X, X_test, y, y_test = train_test_split(
    newsgroups.data,
    newsgroups.target,
    train_size=9051,
    test_size=2263)

lstm_text = LstmTextClassifier(num_classes=20)

# Check that the hyperparameters are correct.
for hyperparam in lstm_text.get_fixed_hyperparams():
    print(
        str(hyperparam) + ":",
        lstm_text.get_fixed_hyperparams()[hyperparam])
for hyperparam in lstm_text.get_tunable_hyperparams():
    print(hyperparam)

# Check that the blocks are correct.
expected_blocks = {
    'tokenizer', 'sequence_padder', 'lstm_text', 'convert_class_probs'
}
blocks = set(lstm_text.blocks.keys())
assert expected_blocks == blocks

# Only use 1/30 of the data for quick testing.
x_sample = X[:len(X) // 30]
y_sample = y[:len(y) // 30]
x_test_sample = X_test[:len(X_test) // 30]
y_test_sample = y_test[:len(y_test) // 30]

y_cat = keras.utils.np_utils.to_categorical(y_sample)

# Check that we can score properly.
print("\nFitting pipeline...")
lstm_text.fit(x_sample, y_cat)
print("\nFit pipeline.")

print("\nScoring pipeline...")
predicted_y_labels = lstm_text.predict(x_test_sample)
score = f1_score(predicted_y_labels, y_test_sample, average='micro')
print("\nf1 micro score: %f" % score)
