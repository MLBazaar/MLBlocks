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
    train_size=90,
    test_size=22)

lstm_text = LstmTextClassifier(num_classes=20, pad_length=1500)

# Check that the hyperparameters are correct.
for hyperparam in lstm_text.get_fixed_hyperparams():
    print(
        str(hyperparam) + ":",
        lstm_text.get_fixed_hyperparams()[hyperparam])
for hyperparam in lstm_text.get_tunable_hyperparams():
    print(hyperparam)

# Check that the steps are correct.
expected_steps = {
    'tokenizer', 'sequence_padder', 'lstm_text', 'convert_class_probs'
}
steps = set(lstm_text.steps_dict.keys())
assert expected_steps == steps

y_cat = keras.utils.np_utils.to_categorical(y)

# Check that we can score properly.
print("\nFitting pipeline...")
fit_params = {('lstm_text', 'epochs'): 3}
lstm_text.fit(X, y_cat)
print("\nFit pipeline.")

print("\nScoring pipeline...")
predicted_y_labels = lstm_text.predict(X_test)
score = f1_score(predicted_y_labels, y_test, average='micro')
print("\nf1 micro score: %f" % score)
