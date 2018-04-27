#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integration tests for image pipelines."""

import unittest

import keras
import numpy as np

from mlblocks.components.pipelines.image.traditional_image import TraditionalImagePipeline
from mlblocks.components.pipelines.image.simple_cnn import SimpleCnnClassifier

from sklearn.datasets import fetch_mldata
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class TestImageClassifiers(unittest.TestCase):
    """Integration tests for image classifiers."""

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
        for hyperparam in traditional_image.get_fixed_hyperparams():
            print(
                str(hyperparam) + ":",
                traditional_image.get_fixed_hyperparams()[hyperparam])
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
        # 10 classes for digits.
        simple_cnn = SimpleCnnClassifier(num_classes=10)

        # Check that the hyperparameters are correct.
        for hyperparam in simple_cnn.get_fixed_hyperparams():
            print(
                str(hyperparam) + ":",
                simple_cnn.get_fixed_hyperparams()[hyperparam])
        for hyperparam in simple_cnn.get_tunable_hyperparams():
            print(hyperparam)

        # Check that the steps are correct.
        expected_steps = {'simple_cnn', 'convert_class_probs'}
        steps = set(simple_cnn.steps_dict.keys())
        self.assertSetEqual(expected_steps, steps)

        # Only use 1/10 of the data for quick testing.
        x_sample = self.X[:len(self.X) // 10]
        y_sample = self.y[:len(self.y) // 10]
        x_test_sample = self.X_test[:len(self.X_test) // 10]
        y_test_sample = self.y_test[:len(self.y_test) // 10]

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
