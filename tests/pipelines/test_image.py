#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration tests for image pipelines."""

import unittest

from mlblocks.components.pipelines.image.traditional_image import TraditionalImagePipeline
from mlblocks.components.pipelines.image.cnn_image import SimpleCNN

import keras
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import numpy as np


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

    def test_simple_cnn(self):
        print("\n============================================" +
              "\nTesting Simple CNN" +
              "\n============================================")

        simple_cnn = SimpleCNN()

        # Check that the hyperparameters are correct.
        for hyperparam in simple_cnn.get_tunable_hyperparams():
            print(hyperparam)

        # Check that the steps are correct.
        expected_steps = {'simple_cnn'}
        steps = set(simple_cnn.steps_dict.keys())
        self.assertSetEqual(expected_steps, steps)

        # Properly format data
        prep_X = np.array([np.resize(im, (224, 224, 3)) for im in self.X]) / 255.0
        cat_y = keras.utils.to_categorical(self.y)
        prep_X_test = np.array([np.resize(im, (224, 224, 3)) for im in self.X_test]) / 255.0

        # Check that we can score properly.
        print("\nFitting pipeline...")
        simple_cnn.fit(prep_X, cat_y)
        print("\nFit pipeline.")

        print("\nScoring pipeline...")
        predicted_y_probs = simple_cnn.predict(prep_X_test)
        predicted_y_labels = np.argmax(predicted_y_probs, axis=1)
        score = f1_score(predicted_y_labels, self.y_test, average='micro')
        print("\nf1 micro score: %f" % score)

