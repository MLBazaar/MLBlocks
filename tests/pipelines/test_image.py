#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration tests for image pipelines."""

import unittest

from mlblocks.components.pipelines.image.traditional_image import TraditionalImagePipeline

from sklearn.datasets import fetch_mldata
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class TestImageClassifiers(unittest.TestCase):
    """Integration test for image classifiers."""

    def setUp(self):
        """Set up image classification data"""
        mnist = fetch_mldata('MNIST original')
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            mnist.data, mnist.target, train_size=1000, test_size=300)

    def test_traditional_image(self):
        print("\n============================================" +
              "\nTesting Traditional Image Pipeline" +
              "\n============================================")

        traditional_image = TraditionalImagePipeline()

        # Check that the hyperparameters are correct.
        for hyperparam in traditional_image.get_tunable_hyperparams():
            print(hyperparam)

        # Check that the steps are correct.
        expected_steps = {'HOG', 'rf_classifier'}
        steps = set(traditional_image.steps_dict.keys())
        self.assertSetEqual(expected_steps, steps)

        # Check that we can score properly.
        print("\nFitting pipeline...")
        traditional_image.fit(self.X, self.y)
        print("\nFit pipeline.")

        print("\nScoring pipeline...")
        predicted_y_val = traditional_image.predict(self.X_test)
        score = f1_score(predicted_y_val, self.y_test, average='micro')
        print("\nf1 micro score: %f" % score)
