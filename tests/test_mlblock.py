# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch

from mlblocks.mlblock import MLBlock, import_object

# import pytest


class DummyClass:
    pass


def test_import_object():
    dummy_class = import_object(__name__ + '.DummyClass')

    assert dummy_class is DummyClass


class TestMLBlock(TestCase):

    def test__extract_params(self):
        pass

    def test__get_tunable_no_conditionals(self):
        """If there are no conditionals, tunables are returned unmodified."""

        # setup
        init_params = {
            'an_init_param': 'a_value'
        }
        hyperparameters = {
            'tunable': {
                'this_is_not_conditional': {
                    'type': 'int',
                    'default': 1
                }
            }
        }

        # run
        tunable = MLBlock._get_tunable(hyperparameters, init_params)

        # assert
        expected = {
            'this_is_not_conditional': {
                'type': 'int',
                'default': 1
            }
        }
        assert tunable == expected

    def test__get_tunable_no_condition(self):
        """If there is a conditiona but no condition, conditional is returned unmodified."""

        # setup
        init_params = {
            'an_init_param': 'a_value'
        }
        hyperparameters = {
            'tunable': {
                'this_is_not_conditional': {
                    'type': 'int',
                    'default': 1
                },
                'this_is_conditional': {
                    'type': 'conditional',
                    'condition': 'a_condition',
                    'default': 1,
                    'values': {
                        1: {
                            'type': 'int',
                            'default': 0
                        },
                        '*': {
                            'type': 'str',
                            'default': 'whatever'
                        }
                    }
                }
            }
        }

        # run
        tunable = MLBlock._get_tunable(hyperparameters, init_params)

        # assert
        expected = {
            'this_is_not_conditional': {
                'type': 'int',
                'default': 1
            },
            'this_is_conditional': {
                'type': 'conditional',
                'condition': 'a_condition',
                'default': 1,
                'values': {
                    1: {
                        'type': 'int',
                        'default': 0
                    },
                    '*': {
                        'type': 'str',
                        'default': 'whatever'
                    }
                }
            }
        }
        assert tunable == expected

    def test__get_tunable_condition_match(self):
        """If there is a conditional and it matches, only that part is returned."""

        # setup
        init_params = {
            'a_condition': 'match'
        }
        hyperparameters = {
            'tunable': {
                'this_is_not_conditional': {
                    'type': 'int',
                    'default': 1
                },
                'this_is_conditional': {
                    'type': 'conditional',
                    'condition': 'a_condition',
                    'default': 1,
                    'values': {
                        'match': {
                            'type': 'int',
                            'default': 0
                        },
                        '*': {
                            'type': 'str',
                            'default': 'whatever'
                        }
                    }
                }
            }
        }

        # run
        tunable = MLBlock._get_tunable(hyperparameters, init_params)

        # assert
        expected = {
            'this_is_not_conditional': {
                'type': 'int',
                'default': 1
            },
            'this_is_conditional': {
                'type': 'int',
                'default': 0
            }
        }
        assert tunable == expected

    def test__get_tunable_condition_wildcard_match(self):
        """If there is a conditional and it matches the wildcard, only that part is returned."""

        # setup
        init_params = {
            'a_condition': 'no_match'
        }
        hyperparameters = {
            'tunable': {
                'this_is_not_conditional': {
                    'type': 'int',
                    'default': 1
                },
                'this_is_conditional': {
                    'type': 'conditional',
                    'condition': 'a_condition',
                    'default': 1,
                    'values': {
                        'match': {
                            'type': 'int',
                            'default': 0
                        },
                        '*': {
                            'type': 'str',
                            'default': 'whatever'
                        }
                    }
                }
            }
        }

        # run
        tunable = MLBlock._get_tunable(hyperparameters, init_params)

        # assert
        expected = {
            'this_is_not_conditional': {
                'type': 'int',
                'default': 1
            },
            'this_is_conditional': {
                'type': 'str',
                'default': 'whatever'
            }
        }
        assert tunable == expected

    def test__get_tunable_condition_no_match(self):
        """If there is a conditional without match or wildcard, it is not returned."""

        # setup
        init_params = {
            'a_condition': 'no_match'
        }
        hyperparameters = {
            'tunable': {
                'this_is_not_conditional': {
                    'type': 'int',
                    'default': 1
                },
                'this_is_conditional': {
                    'type': 'conditional',
                    'condition': 'a_condition',
                    'default': 1,
                    'values': {
                        'match': {
                            'type': 'int',
                            'default': 0
                        }
                    }
                }
            }
        }

        # run
        tunable = MLBlock._get_tunable(hyperparameters, init_params)

        # assert
        expected = {
            'this_is_not_conditional': {
                'type': 'int',
                'default': 1
            }
        }
        assert tunable == expected

    @patch('mlblocks.mlblock.MLBlock.set_hyperparameters')
    @patch('mlblocks.mlblock.import_object')
    @patch('mlblocks.mlblock.load_primitive')
    def test___init__(self, load_primitive_mock, import_object_mock, set_hps_mock):
        load_primitive_mock.return_value = {
            'primitive': 'a_primitive_name',
            'produce': {
                'args': [
                    {
                        'name': 'argument'
                    }
                ],
                'output': [
                ]
            }
        }

        mlblock = MLBlock('given_primitive_name', argument='value')

        assert mlblock.name == 'given_primitive_name'
        assert mlblock.primitive == import_object_mock.return_value
        assert mlblock._fit == dict()
        assert mlblock.fit_args == list()
        assert mlblock.fit_method is None

        produce = {
            'args': [
                {
                    'name': 'argument'
                }
            ],
            'output': [
            ]
        }
        assert mlblock._produce == produce
        assert mlblock.produce_args == produce['args']
        assert mlblock.produce_output == produce['output']
        assert mlblock.produce_method is None
        assert mlblock._class is False

        assert mlblock._hyperparameters == dict()
        assert mlblock._fit_params == dict()
        assert mlblock._produce_params == {'argument': 'value'}

        assert mlblock._tunable == dict()

        set_hps_mock.assert_called_once_with(dict())

    @patch('mlblocks.mlblock.import_object')
    @patch('mlblocks.mlblock.load_primitive')
    def test___str__(self, load_primitive_mock, import_object_mock):
        load_primitive_mock.return_value = {
            'primitive': 'a_primitive_name',
            'produce': {
                'args': [],
                'output': []
            }
        }

        mlblock = MLBlock('given_primitive_name')

        assert str(mlblock) == 'MLBlock - given_primitive_name'

    @patch('mlblocks.mlblock.import_object')
    @patch('mlblocks.mlblock.load_primitive')
    def test_get_tunable_hyperparameters(self, load_primitive_mock, import_object_mock):
        """get_tunable_hyperparameters has to return a copy of the _tunables attribute."""
        load_primitive_mock.return_value = {
            'primitive': 'a_primitive_name',
            'produce': {
                'args': [],
                'output': []
            }
        }

        mlblock = MLBlock('given_primitive_name')

        tunable = dict()
        mlblock._tunable = tunable

        returned = mlblock.get_tunable_hyperparameters()

        assert returned == tunable
        assert returned is not tunable

    @patch('mlblocks.mlblock.import_object')
    @patch('mlblocks.mlblock.load_primitive')
    def test_get_hyperparameters(self, load_primitive_mock, import_object_mock):
        """get_hyperparameters has to return a copy of the _hyperparameters attribute."""
        load_primitive_mock.return_value = {
            'primitive': 'a_primitive_name',
            'produce': {
                'args': [],
                'output': []
            }
        }

        mlblock = MLBlock('given_primitive_name')

        hyperparameters = dict()
        mlblock._hyperparameters = hyperparameters

        returned = mlblock.get_hyperparameters()

        assert returned == hyperparameters
        assert returned is not hyperparameters

    def test_set_hyperparameters_function(self):
        pass

    def test_set_hyperparameters_class(self):
        pass

    def test_fit_no_fit(self):
        pass

    def test_fit(self):
        pass

    def test_produce_function(self):
        pass

    def test_produce_class(self):
        pass
