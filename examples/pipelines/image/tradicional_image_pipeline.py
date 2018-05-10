#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage example for TraditionalImagePipeline on MNIST Dataset."""

from sklearn.datasets import fetch_mldata
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from mlblocks.components.pipelines.image.traditional_image import TraditionalImagePipeline

print("""
============================================
Testing Traditional Image Pipeline
============================================
""")

mnist = fetch_mldata('MNIST original')
X, X_test, y, y_test = train_test_split(
    mnist.data, mnist.target, train_size=1000, test_size=300)

traditional_image = TraditionalImagePipeline()

# Check that the hyperparameters are correct.
for hyperparam in traditional_image.get_fixed_hyperparams():
    print(
        str(hyperparam) + ":",
        traditional_image.get_fixed_hyperparams()[hyperparam])
for hyperparam in traditional_image.get_tunable_hyperparams():
    print(hyperparam)

# Check that the steps are correct.
expected_steps = {'HOG', 'rf_classifier'}
steps = set(traditional_image.steps_dict.keys())
assert expected_steps == steps

# Check that we can update our pipeline's tunable hyperparameter
# values.
hp_dict = {('rf_classifier', 'max_depth'): 9}
traditional_image.set_from_hyperparam_dict(hp_dict)

# Check that we can score properly.
print("\nFitting pipeline...")
traditional_image.fit(X, y)
print("\nFit pipeline.")

print("\nScoring pipeline...")
predicted_y_val = traditional_image.predict(X_test)
score = f1_score(predicted_y_val, y_test, average='micro')
print("\nf1 micro score: %f" % score)
