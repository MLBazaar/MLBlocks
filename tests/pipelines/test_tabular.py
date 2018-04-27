#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integration tests for tabular pipelines."""

import unittest

from mlblocks.components.pipelines.tabular.random_forest import RandomForestClassifier
from mlblocks.components.pipelines.tabular.random_forest import RandomForestRegressor

from sklearn.datasets import load_wine, load_boston
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split


class TestTabularClassifiers(unittest.TestCase):
    """Integration test for tabular classifiers."""

    def setUp(self):
        """Set up tabular classification data"""
        wine = load_wine()
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            wine.data, wine.target, train_size=142, test_size=36)

    def test_rf_classifier(self):
        print("\n============================================" +
              "\nTesting Random Forest Classifier" +
              "\n============================================")

        rf_classifier = RandomForestClassifier()

        # Check that the hyperparameters are correct.
        for hyperparam in rf_classifier.get_tunable_hyperparams():
            print(hyperparam)

        # Check that the steps are correct.
        expected_steps = {'rf_classifier'}
        steps = set(rf_classifier.steps_dict.keys())
        self.assertSetEqual(expected_steps, steps)

        # Check that we can score properly.
        print("\nFitting pipeline...")
        rf_classifier.fit(self.X, self.y)
        print("\nFit pipeline.")

        print("\nScoring pipeline...")
        predicted_y_val = rf_classifier.predict(self.X_test)
        score = f1_score(predicted_y_val, self.y_test, average='micro')
        print("\nf1 micro score: %f" % score)


class TestTabularRegressors(unittest.TestCase):
    """Integration test for tabular regressors."""

    def setUp(self):
        """Set up tabular regression data"""
        housing = load_boston()
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            housing.data, housing.target, train_size=405, test_size=101)

    def test_rf_regressor(self):
        print("\n============================================" +
              "\nTesting Random Forest Regressor" +
              "\n============================================")

        rf_regressor = RandomForestRegressor()

        # Check that the hyperparameters are correct.
        for hyperparam in rf_regressor.get_tunable_hyperparams():
            print(hyperparam)

        # Check that the steps are correct.
        expected_steps = {'rf_regressor'}
        steps = set(rf_regressor.steps_dict.keys())
        self.assertSetEqual(expected_steps, steps)

        # Check that we can score properly.
        print("\nFitting pipeline...")
        rf_regressor.fit(self.X, self.y)
        print("\nFit pipeline.")

        print("\nScoring pipeline...")
        predicted_y_val = rf_regressor.predict(self.X_test)
        score = r2_score(self.y_test, predicted_y_val)
        print("\nr2 score: %f" % score)
