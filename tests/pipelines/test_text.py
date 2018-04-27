#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integration tests for text pipelines."""

import unittest

import keras

from mlblocks.components.pipelines.text.traditional_text import TraditionalTextPipeline
from mlblocks.components.pipelines.text.lstm_text import LstmTextClassifier

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class TestImageClassifiers(unittest.TestCase):
    """Integration tests for text classifiers."""

    def setUp(self):
        """Set up text classification data"""
        newsgroups = fetch_20newsgroups()
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            newsgroups.data, newsgroups.target, train_size=9051, test_size=2263)

    def test_traditional_text(self):
        print("\n============================================" +
              "\nTesting Traditional Text Pipeline" +
              "\n============================================")

        traditional_text = TraditionalTextPipeline()

        # Check that the hyperparameters are correct.
        for hyperparam in traditional_text.get_fixed_hyperparams():
            print(
                str(hyperparam) + ":",
                traditional_text.get_fixed_hyperparams()[hyperparam])
        for hyperparam in traditional_text.get_tunable_hyperparams():
            print(hyperparam)

        # Check that the steps are correct.
        expected_steps = {'count_vectorizer', 'to_array', 'tfidf_transformer', 'multinomial_nb'}
        steps = set(traditional_text.steps_dict.keys())
        self.assertSetEqual(expected_steps, steps)

        # Check that we can score properly.
        print("\nFitting pipeline...")
        traditional_text.fit(self.X, self.y)
        print("\nFit pipeline.")

        print("\nScoring pipeline...")
        predicted_y_val = traditional_text.predict(self.X_test)
        score = f1_score(predicted_y_val, self.y_test, average='micro')
        print("\nf1 micro score: %f" % score)

    def test_lstm_text(self):
        print("\n============================================" +
              "\nTesting Text LSTM" +
              "\n============================================")
        # 10 classes for digits.
        lstm_text = LstmTextClassifier(num_classes=20)
        print(lstm_text.steps_dict['lstm_text'].model.summary())

        # Check that the hyperparameters are correct.
        for hyperparam in lstm_text.get_fixed_hyperparams():
            print(
                str(hyperparam) + ":",
                lstm_text.get_fixed_hyperparams()[hyperparam])
        for hyperparam in lstm_text.get_tunable_hyperparams():
            print(hyperparam)

        # Check that the steps are correct.
        expected_steps = {'tokenizer', 'sequence_padder', 'lstm_text', 'convert_class_probs'}
        steps = set(lstm_text.steps_dict.keys())
        self.assertSetEqual(expected_steps, steps)

        # Only use 1/30 of the data for quick testing.
        x_sample = self.X[:len(self.X) // 30]
        y_sample = self.y[:len(self.y) // 30]
        x_test_sample = self.X_test[:len(self.X_test) // 30]
        y_test_sample = self.y_test[:len(self.y_test) // 30]

        y_cat = keras.utils.np_utils.to_categorical(y_sample)

        # Check that we can score properly.
        print("\nFitting pipeline...")
        lstm_text.fit(x_sample, y_cat)
        print("\nFit pipeline.")

        print("\nScoring pipeline...")
        predicted_y_labels = lstm_text.predict(x_test_sample)
        score = f1_score(predicted_y_labels, y_test_sample, average='micro')
        print("\nf1 micro score: %f" % score)
