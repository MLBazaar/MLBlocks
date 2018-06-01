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
    mnist.data, mnist.target, train_size=10, test_size=3)

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

# Properly format data.
prep_x = np.array([np.resize(im, (224, 224, 3))
                   for im in X]) / 255.0
cat_y = keras.utils.to_categorical(y)
prep_x_test = np.array(
    [np.resize(im, (224, 224, 3)) for im in X_test]) / 255.0

# Check that we can score properly.
print("\nFitting pipeline...")
fit_params = {('simple_cnn', 'epochs'): 12}
simple_cnn.fit(prep_x, cat_y)
print("\nFit pipeline.")

print("\nScoring pipeline...")
predicted_y_labels = simple_cnn.predict(prep_x_test)
score = f1_score(predicted_y_labels, y_test, average='micro')
print("\nf1 micro score: %f" % score)
