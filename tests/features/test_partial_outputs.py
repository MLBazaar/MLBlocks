from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from mlblocks.datasets import load_iris
from mlblocks.mlpipeline import MLPipeline


def almost_equal(obj1, obj2):
    if isinstance(obj1, dict):
        if not isinstance(obj2, dict):
            raise AssertionError("{} is not equal to {}".format(type(obj2), dict))

        for key, value in obj1.items():
            if key not in obj2:
                raise AssertionError("{} not in {}".format(key, obj2))
            almost_equal(value, obj2[key])

    else:
        np.testing.assert_almost_equal(obj1, obj2)


class TestPartialOutputs(TestCase):
    def setUp(self):
        dataset = load_iris()

        self.X_train, self.X_test, self.y_train, self.y_test = dataset.get_splits(1)

    def test_fit_output(self):

        # Setup variables
        primitives = [
            'sklearn.preprocessing.StandardScaler',
            'sklearn.linear_model.LogisticRegression'
        ]
        pipeline = MLPipeline(primitives)

        int_block = 0
        invalid_int = 10
        str_block = 'sklearn.preprocessing.StandardScaler#1'
        invalid_block = 'InvalidBlockName'
        str_block_variable = 'sklearn.preprocessing.StandardScaler#1.y'
        invalid_variable = 'sklearn.preprocessing.StandardScaler#1.invalid'

        # Run
        int_out = pipeline.fit(self.X_train[0:5], self.y_train[0:5], output_=int_block)
        str_out = pipeline.fit(self.X_train[0:5], self.y_train[0:5], output_=str_block)
        str_out_variable = pipeline.fit(self.X_train[0:5], self.y_train[0:5],
                                        output_=str_block_variable)
        no_output = pipeline.fit(self.X_train, self.y_train)

        # Assert successful calls
        X = np.array([
            [0.71269665, -1.45152899, 0.55344946, 0.31740553],
            [0.26726124, 1.23648766, -1.1557327, -1.0932857],
            [-1.95991577, 0.967686, -1.1557327, -1.0932857],
            [0.71269665, -0.645124, 0.39067021, 0.31740553],
            [0.26726124, -0.10752067, 1.36734573, 1.55176035]
        ])
        y = np.array([1, 0, 0, 1, 2])
        context = {
            'X': X,
            'y': y
        }
        almost_equal(context, int_out)
        almost_equal(context, str_out)

        almost_equal(y, str_out_variable)

        assert no_output is None

        # Run asserting exceptions
        with self.assertRaises(IndexError):
            pipeline.fit(self.X_train[0:5], self.y_train[0:5], output_=invalid_int)

        with self.assertRaises(ValueError):
            pipeline.fit(self.X_train[0:5], self.y_train[0:5], output_=invalid_block)

        with self.assertRaises(ValueError):
            pipeline.fit(self.X_train[0:5], self.y_train[0:5], output_=invalid_variable)

    def test_fit_start(self):
        # Setup variables
        primitives = [
            'sklearn.preprocessing.StandardScaler',
            'sklearn.linear_model.LogisticRegression'
        ]
        pipeline = MLPipeline(primitives)

        # Mock the first block
        block_mock = Mock()
        pipeline.blocks['sklearn.preprocessing.StandardScaler#1'] = block_mock

        # Run first block
        context = {
            'X': self.X_train,
            'y': self.y_train
        }
        int_start = 1
        str_start = 'sklearn.linear_model.LogisticRegression#1'

        pipeline.fit(start_=int_start, **context)
        pipeline.fit(start_=str_start, **context)

        # Assert that mock has not been called
        block_mock.fit.assert_not_called()

    def test_predict_start(self):
        # Setup variables
        primitives = [
            'sklearn.preprocessing.StandardScaler',
            'sklearn.linear_model.LogisticRegression'
        ]
        pipeline = MLPipeline(primitives)
        pipeline.fit(self.X_train, self.y_train)

        # Mock the first block
        block_mock = Mock()
        pipeline.blocks['sklearn.preprocessing.StandardScaler#1'] = block_mock

        # Run first block
        context = {
            'X': self.X_train,
        }
        int_start = 1
        str_start = 'sklearn.linear_model.LogisticRegression#1'

        pipeline.predict(start_=int_start, **context)
        pipeline.predict(start_=str_start, **context)

        # Assert that mock has not been called
        block_mock.predict.assert_not_called()
