from unittest import TestCase
from unittest.mock import Mock

import numpy as np

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
        self.X = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])
        self.y = np.array([0, 0, 0, 0, 1])

    def test_fit_output(self):

        # Setup variables
        primitives = [
            'sklearn.preprocessing.StandardScaler',
            'sklearn.linear_model.LogisticRegression'
        ]
        pipeline = MLPipeline(primitives)

        named = 'default'
        list_ = ['default', 0]
        int_block = 0
        invalid_int = 10
        str_block = 'sklearn.preprocessing.StandardScaler#1'
        invalid_block = 'InvalidBlockName'
        str_block_variable = 'sklearn.preprocessing.StandardScaler#1.X'
        invalid_variable = 'sklearn.preprocessing.StandardScaler#1.invalid'

        # Run
        named_out = pipeline.fit(self.X, self.y, output_=named)
        list_out = pipeline.fit(self.X, self.y, output_=list_)
        int_out = pipeline.fit(self.X, self.y, output_=int_block)
        str_out = pipeline.fit(self.X, self.y, output_=str_block)
        str_out_variable = pipeline.fit(self.X, self.y,
                                        output_=str_block_variable)
        no_output = pipeline.fit(self.X, self.y)

        # Assert successful calls
        X = np.array([
            [2., -0.5, -0.5, -0.5, -0.5],
            [-0.5, 2., -0.5, -0.5, -0.5],
            [-0.5, -0.5, 2., -0.5, -0.5],
            [-0.5, -0.5, -0.5, 2., -0.5],
            [-0.5, -0.5, -0.5, -0.5, 2.],
        ])
        y = np.array([
            0, 0, 0, 0, 1
        ])
        context = {'X': X, 'y': y}

        almost_equal(named_out, y)
        assert len(list_out) == 2
        almost_equal(list_out[0], y)
        almost_equal(list_out[1], context)
        almost_equal(context, int_out)
        almost_equal(context, str_out)
        almost_equal(X, str_out_variable)
        assert no_output is None

        # Run asserting exceptions
        with self.assertRaises(IndexError):
            pipeline.fit(self.X, self.y, output_=invalid_int)

        with self.assertRaises(ValueError):
            pipeline.fit(self.X, self.y, output_=invalid_block)

        with self.assertRaises(ValueError):
            pipeline.fit(self.X, self.y, output_=invalid_variable)

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
            'X': self.X,
            'y': self.y
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
        pipeline.fit(self.X, self.y)

        # Mock the first block
        block_mock = Mock()
        pipeline.blocks['sklearn.preprocessing.StandardScaler#1'] = block_mock

        # Run first block
        context = {
            'X': self.X,
        }
        int_start = 1
        str_start = 'sklearn.linear_model.LogisticRegression#1'

        pipeline.predict(start_=int_start, **context)
        pipeline.predict(start_=str_start, **context)

        # Assert that mock has not been called
        block_mock.predict.assert_not_called()
