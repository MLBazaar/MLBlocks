#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage example for RandomForestClassifier on the Wine Dataset."""

from sklearn.datasets import load_wine
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from mlblocks.components.pipelines.tabular.random_forest import RandomForestClassifier

print("""
============================================
Testing Random Forest Classifier
============================================
""")

wine = load_wine()
X, X_test, y, y_test = train_test_split(
    wine.data, wine.target, train_size=142, test_size=36)

rf_classifier = RandomForestClassifier()

# Check that the hyperparameters are correct.
for hyperparam in rf_classifier.get_tunable_hyperparams():
    print(hyperparam)

# Check that the blocks are correct.
expected_blocks = {'rf_classifier'}
blocks = set(rf_classifier.blocks.keys())
assert expected_blocks == blocks

# Check that we can score properly.
print("\nFitting pipeline...")
rf_classifier.fit(X, y)
print("\nFit pipeline.")

print("\nScoring pipeline...")
predicted_y_val = rf_classifier.predict(X_test)
score = f1_score(predicted_y_val, y_test, average='micro')
print("\nf1 micro score: %f" % score)
