# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import pytest

from mlblocks.mlblock import MLBlock, import_object


class DummyClass:
    def a_method(self):
        pass


def dummy_function():
    pass


class TestImportObject(TestCase):

    def test_class(self):
        imported = import_object(__name__ + '.DummyClass')

        assert imported is DummyClass

    def test_class_method(self):
        imported = import_object(__name__ + '.DummyClass.a_method')

        assert imported is DummyClass.a_method

    def test_function(self):
        imported = import_object(__name__ + '.dummy_function')

        assert imported is dummy_function

    def test_bad_object_name(self):
        with pytest.raises(AttributeError):
            import_object(__name__ + '.InvalidName')

    def test_bad_module(self):
        with pytest.raises(ImportError):
            import_object('an.invalid.module')


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
                    'default': 1,
                    'range': [1, 10]
                }
            }
        }

        # run
        tunable = MLBlock._get_tunable(hyperparameters, init_params)

        # assert
        expected = {
            'this_is_not_conditional': {
                'type': 'int',
                'default': 1,
                'range': [1, 10]
            }
        }
        assert tunable == expected

    def test__get_tunable_no_condition(self):
        """If there is a conditional but no condition, the default is used."""

        # setup
        init_params = {
            'an_init_param': 'a_value'
        }
        hyperparameters = {
            'tunable': {
                'this_is_not_conditional': {
                    'type': 'int',
                    'default': 1,
                    'range': [1, 10]
                },
                'this_is_conditional': {
                    'type': 'conditional',
                    'condition': 'a_condition',
                    'default': {
                        'type': 'float',
                        'default': 0.1,
                        'values': [0, 1]
                    },
                    'values': {
                        'not_a_match': {
                            'type': 'str',
                            'default': 'a',
                            'values': ['a', 'b']
                        },
                        'neither_a_match': {
                            'type': 'int',
                            'default': 0,
                            'range': [1, 10]
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
                'default': 1,
                'range': [1, 10]
            },
            'this_is_conditional': {
                'type': 'float',
                'default': 0.1,
                'values': [0, 1]
            }
        }
        assert tunable == expected

    def test__get_tunable_condition_match(self):
        """If there is a conditional and it matches, only that part is returned."""

        # setup
        init_params = {
            'a_condition': 'a_match'
        }
        hyperparameters = {
            'tunable': {
                'this_is_not_conditional': {
                    'type': 'int',
                    'default': 1,
                    'range': [1, 10]
                },
                'this_is_conditional': {
                    'type': 'conditional',
                    'condition': 'a_condition',
                    'default': {
                        'type': 'float',
                        'default': 0.1,
                        'values': [0, 1]
                    },
                    'values': {
                        'not_a_match': {
                            'type': 'str',
                            'default': 'a',
                            'values': ['a', 'b']
                        },
                        'a_match': {
                            'type': 'int',
                            'default': 0,
                            'range': [1, 10]
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
                'default': 1,
                'range': [1, 10]
            },
            'this_is_conditional': {
                'type': 'int',
                'default': 0,
                'range': [1, 10]
            }
        }
        assert tunable == expected

    def test__get_tunable_condition_no_match(self):
        """If there is a conditional and it does not match, the default is used."""

        # setup
        init_params = {
            'a_condition': 'not_a_match'
        }
        hyperparameters = {
            'tunable': {
                'this_is_not_conditional': {
                    'type': 'int',
                    'default': 1,
                    'range': [1, 10]
                },
                'this_is_conditional': {
                    'type': 'conditional',
                    'condition': 'a_condition',
                    'default': {
                        'type': 'float',
                        'default': 0.1,
                        'values': [0, 1]
                    },
                    'values': {
                        'also_not_a_match': {
                            'type': 'str',
                            'default': 'a',
                            'values': ['a', 'b']
                        },
                        'neither_a_match': {
                            'type': 'int',
                            'default': 0,
                            'range': [1, 10]
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
                'default': 1,
                'range': [1, 10]
            },
            'this_is_conditional': {
                'type': 'float',
                'default': 0.1,
                'values': [0, 1]
            }
        }
        assert tunable == expected

    def test__get_tunable_condition_default_null(self):
        """If there is no match and default is null (None), this param is not included."""

        # setup
        init_params = {
            'a_condition': 'not_a_match'
        }
        hyperparameters = {
            'tunable': {
                'this_is_not_conditional': {
                    'type': 'int',
                    'default': 1,
                    'range': [1, 10]
                },
                'this_is_conditional': {
                    'type': 'conditional',
                    'condition': 'a_condition',
                    'default': None,
                    'values': {
                        'also_not_a_match': {
                            'type': 'str',
                            'default': 'a',
                            'values': ['a', 'b']
                        },
                        'neither_a_match': {
                            'type': 'int',
                            'default': 0,
                            'range': [1, 10]
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
                'default': 1,
                'range': [1, 10]
            }
        }
        assert tunable == expected

    def test__get_tunable_condition_match_null(self):
        """If there is a match and it is null (None), this param is not included.

        This stands even if the default is not null.
        """

        # setup
        init_params = {
            'a_condition': 'a_match'
        }
        hyperparameters = {
            'tunable': {
                'this_is_not_conditional': {
                    'type': 'int',
                    'default': 1,
                    'range': [1, 10]
                },
                'this_is_conditional': {
                    'type': 'conditional',
                    'condition': 'a_condition',
                    'default': {
                        'type': 'float',
                        'default': 0.1,
                        'values': [0, 1]
                    },
                    'values': {
                        'not_a_match': {
                            'type': 'str',
                            'default': 'a',
                            'values': ['a', 'b']
                        },
                        'a_match': None
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
                'default': 1,
                'range': [1, 10]
            }
        }
        assert tunable == expected

    @patch('mlblocks.mlblock.MLBlock.set_hyperparameters')
    @patch('mlblocks.mlblock.import_object')
    @patch('mlblocks.mlblock.load_primitive')
    def test___init__(self, load_primitive_mock, import_object_mock, set_hps_mock):
        load_primitive_mock.return_value = {
            'name': 'a_primitive_name',
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

        mlblock = MLBlock('a_primitive_name', argument='value')

        assert mlblock.metadata == {
            'name': 'a_primitive_name',
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
        assert mlblock.name == 'a_primitive_name'
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
            'name': 'a_primitive_name',
            'primitive': 'a_primitive_name',
            'produce': {
                'args': [],
                'output': []
            }
        }

        mlblock = MLBlock('a_primitive_name')

        assert str(mlblock) == 'MLBlock - a_primitive_name'

    @patch('mlblocks.mlblock.import_object')
    @patch('mlblocks.mlblock.load_primitive')
    def test_get_tunable_hyperparameters(self, load_primitive_mock, import_object_mock):
        """get_tunable_hyperparameters has to return a copy of the _tunables attribute."""
        load_primitive_mock.return_value = {
            'name': 'a_primitive_name',
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

    @patch('mlblocks.mlblock.import_object', new=Mock())
    @patch('mlblocks.mlblock.load_primitive', new=MagicMock())
    def test_get_hyperparameters(self):
        """get_hyperparameters has to return a deepcopy of the _hyperparameters attribute."""
        mlblock = MLBlock('given_primitive_name')

        hyperparameters = {
            'a_list_param': ['a']
        }
        mlblock._hyperparameters = hyperparameters

        returned = mlblock.get_hyperparameters()

        assert returned == hyperparameters
        assert returned is not hyperparameters

        returned['a_list_param'].append('b')
        assert 'b' not in hyperparameters['a_list_param']

    @patch('mlblocks.mlblock.import_object')
    @patch('mlblocks.mlblock.load_primitive')
    def test_modify_hyperparameters(self, lp_mock, io_mock):
        """If a primitive method modifies the hyperparameters, changes should not persist."""

        def primitive(a_list_param):
            a_list_param.append('b')

        io_mock.return_value = primitive

        lp_mock.return_value = {
            'name': 'a_primitive',
            'primitive': 'a_primitive',
            'produce': {
                'args': [],
                'output': []
            }
        }

        mlblock = MLBlock('a_primitive')

        hyperparameters = {
            'a_list_param': ['a']
        }
        mlblock._hyperparameters = hyperparameters

        mlblock.produce()

        assert 'b' not in hyperparameters['a_list_param']

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
