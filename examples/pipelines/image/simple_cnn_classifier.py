#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage example for SimpleCnnClassifier on MNIST Dataset."""

import keras
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from mlblocks.components.pipelines.image.simple_cnn import SimpleCnnClassifier

print("""
============================================
Testing Simple CNN
============================================
""")

mnist = fetch_mldata('MNIST original')
X, X_test, y, y_test = train_test_split(
    mnist.data, mnist.target, train_size=1000, test_size=300)

# 10 classes for digits.
simple_cnn = SimpleCnnClassifier(num_classes=10)

# Check that the hyperparameters are correct.
for hyperparam in simple_cnn.get_fixed_hyperparams():
    print(
        str(hyperparam) + ":",
        simple_cnn.get_fixed_hyperparams()[hyperparam])
for hyperparam in simple_cnn.get_tunable_hyperparams():
    print(hyperparam)

# Check that the blocks are correct.
expected_blocks = {'simple_cnn', 'convert_class_probs'}
blocks = set(simple_cnn.blocks.keys())
assert expected_blocks == blocks

# Only use 1/10 of the data for quick testing.
x_sample = X[:len(X) // 30]
y_sample = y[:len(y) // 30]
x_test_sample = X_test[:len(X_test) // 30]
y_test_sample = y_test[:len(y_test) // 30]

# Properly format data.
prep_x = np.array([np.resize(im, (224, 224, 3))
                   for im in x_sample]) / 255.0
cat_y = keras.utils.to_categorical(y_sample)
prep_x_test = np.array(
    [np.resize(im, (224, 224, 3)) for im in x_test_sample]) / 255.0

# Check that we can score properly.
print("\nFitting pipeline...")
simple_cnn.fit(prep_x, cat_y)
print("\nFit pipeline.")

print("\nScoring pipeline...")
predicted_y_labels = simple_cnn.predict(prep_x_test)
score = f1_score(predicted_y_labels, y_test_sample, average='micro')
print("\nf1 micro score: %f" % score)
