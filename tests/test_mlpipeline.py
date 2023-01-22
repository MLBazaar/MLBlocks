# -*- coding: utf-8 -*-

from collections import OrderedDict
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import pytest

from mlblocks.mlblock import MLBlock
from mlblocks.mlpipeline import MLPipeline


def get_mlblock_mock(*args, **kwargs):
    return MagicMock(autospec=MLBlock)


class TestMLPipline(TestCase):

    @patch('mlblocks.mlpipeline.LOGGER')
    @patch('mlblocks.mlpipeline.MLBlock')
    def test___init__(self, mlblock_mock, logger_mock):
        blocks = [
            get_mlblock_mock(),
            get_mlblock_mock(),
            get_mlblock_mock(),
            get_mlblock_mock()
        ]
        last_block = blocks[-1]
        last_block.produce_output = [
            {
                'name': 'y',
                'type': 'array'
            }
        ]
        mlblock_mock.side_effect = blocks

        primitives = [
            'a.primitive.Name',
            'a.primitive.Name',
            'another.primitive.Name',
            'another.primitive.Name',
        ]
        expected_primitives = primitives.copy()

        init_params = {
            'a.primitive.Name': {
                'an_argument': 'value',
            },
            'another.primitive.Name#2': {
                'another': 'argument_value',
            }
        }
        expected_init_params = init_params.copy()
        input_names = {
            'another.primitive.Name#1': {
                'a_name': 'another_name',
            }
        }
        expected_input_names = input_names.copy()

        mlpipeline = MLPipeline(
            primitives=primitives,
            init_params=init_params,
            input_names=input_names
        )

        assert mlpipeline.primitives == expected_primitives
        assert mlpipeline.init_params == expected_init_params
        assert mlpipeline.blocks == OrderedDict((
            ('a.primitive.Name#1', blocks[0]),
            ('a.primitive.Name#2', blocks[1]),
            ('another.primitive.Name#1', blocks[2]),
            ('another.primitive.Name#2', blocks[3])
        ))
        assert mlpipeline.input_names == expected_input_names
        assert mlpipeline.output_names == dict()
        assert mlpipeline._tunable_hyperparameters == {
            'a.primitive.Name#1': blocks[0].get_tunable_hyperparameters.return_value,
            'a.primitive.Name#2': blocks[1].get_tunable_hyperparameters.return_value,
            'another.primitive.Name#1': blocks[2].get_tunable_hyperparameters.return_value,
            'another.primitive.Name#2': blocks[3].get_tunable_hyperparameters.return_value
        }
        assert mlpipeline.outputs == {
            'default': [
                {
                    'name': 'y',
                    'type': 'array',
                    'variable': 'another.primitive.Name#2.y'
                }
            ]
        }
        assert mlpipeline.verbose

        expected_calls = [
            call('a.primitive.Name', an_argument='value'),
            call('a.primitive.Name', an_argument='value'),
            call('another.primitive.Name'),
            call('another.primitive.Name', another='argument_value'),
        ]
        assert mlblock_mock.call_args_list == expected_calls

        logger_mock.warning.assert_called_once_with(
            'Non-numbered init_params are being used for more than one block %s.',
            'a.primitive.Name'
        )

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_tunable_hyperparameters(self):
        mlpipeline = MLPipeline(['a_primitive'])
        tunable = dict()
        mlpipeline._tunable_hyperparameters = tunable

        returned = mlpipeline.get_tunable_hyperparameters()

        assert returned == tunable
        assert returned is not tunable

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_tunable_hyperparameters_flat(self):
        mlpipeline = MLPipeline(['a_primitive'])
        mlpipeline._tunable_hyperparameters = {
            'block_1': {
                'hp_1': {
                    'type': 'int',
                    'range': [
                        1,
                        10
                    ],
                }
            },
            'block_2': {
                'hp_1': {
                    'type': 'str',
                    'default': 'a',
                    'values': [
                        'a',
                        'b',
                        'c'
                    ],
                },
                'hp_2': {
                    'type': 'bool',
                    'default': True,
                }
            }
        }

        returned = mlpipeline.get_tunable_hyperparameters(flat=True)

        expected = {
            ('block_1', 'hp_1'): {
                'type': 'int',
                'range': [
                    1,
                    10
                ],
            },
            ('block_2', 'hp_1'): {
                'type': 'str',
                'default': 'a',
                'values': [
                    'a',
                    'b',
                    'c'
                ],
            },
            ('block_2', 'hp_2'): {
                'type': 'bool',
                'default': True,
            }
        }
        assert returned == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_hyperparameters(self):
        block_1 = get_mlblock_mock()
        block_1.get_hyperparameters.return_value = {
            'a': 'a'
        }
        block_2 = get_mlblock_mock()
        block_2.get_hyperparameters.return_value = {
            'b': 'b',
            'c': 'c',
        }
        blocks = OrderedDict((
            ('a.primitive.Name#1', block_1),
            ('a.primitive.Name#2', block_2),
        ))
        mlpipeline = MLPipeline(['a_primitive'])
        mlpipeline.blocks = blocks

        hyperparameters = mlpipeline.get_hyperparameters()

        assert hyperparameters == {
            'a.primitive.Name#1': {
                'a': 'a',
            },
            'a.primitive.Name#2': {
                'b': 'b',
                'c': 'c',
            },
        }
        block_1.get_hyperparameters.assert_called_once_with()
        block_2.get_hyperparameters.assert_called_once_with()

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_hyperparameters_flat(self):
        block_1 = get_mlblock_mock()
        block_1.get_hyperparameters.return_value = {
            'a': 'a'
        }
        block_2 = get_mlblock_mock()
        block_2.get_hyperparameters.return_value = {
            'b': 'b',
            'c': 'c',
        }
        blocks = OrderedDict((
            ('a.primitive.Name#1', block_1),
            ('a.primitive.Name#2', block_2),
        ))
        mlpipeline = MLPipeline(['a_primitive'])
        mlpipeline.blocks = blocks

        hyperparameters = mlpipeline.get_hyperparameters(flat=True)

        assert hyperparameters == {
            ('a.primitive.Name#1', 'a'): 'a',
            ('a.primitive.Name#2', 'b'): 'b',
            ('a.primitive.Name#2', 'c'): 'c',
        }
        block_1.get_hyperparameters.assert_called_once_with()
        block_2.get_hyperparameters.assert_called_once_with()

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_set_hyperparameters(self):
        block_1 = get_mlblock_mock()
        block_2 = get_mlblock_mock()
        blocks = OrderedDict((
            ('a.primitive.Name#1', block_1),
            ('a.primitive.Name#2', block_2),
        ))
        mlpipeline = MLPipeline(['a_primitive'])
        mlpipeline.blocks = blocks

        hyperparameters = {
            'a.primitive.Name#2': {
                'some': 'arg'
            }
        }
        mlpipeline.set_hyperparameters(hyperparameters)

        block_1.set_hyperparameters.assert_not_called()
        block_2.set_hyperparameters.assert_called_once_with({'some': 'arg'})

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_set_hyperparameters_flat(self):
        block_1 = get_mlblock_mock()
        block_2 = get_mlblock_mock()
        blocks = OrderedDict((
            ('a.primitive.Name#1', block_1),
            ('a.primitive.Name#2', block_2),
        ))
        mlpipeline = MLPipeline(['a_primitive'])
        mlpipeline.blocks = blocks

        hyperparameters = {
            ('a.primitive.Name#2', 'some'): 'arg'
        }
        mlpipeline.set_hyperparameters(hyperparameters)

        block_1.set_hyperparameters.assert_not_called()
        block_2.set_hyperparameters.assert_called_once_with({'some': 'arg'})

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test__get_block_args(self):
        input_names = {
            'a_block': {
                'arg_3': 'arg_3_alt'
            }
        }
        pipeline = MLPipeline(['a_primitive'], input_names=input_names)

        block_args = [
            {
                'name': 'arg_1',
            },
            {
                'name': 'arg_2',
                'default': 'arg_2_value'
            },
            {
                'name': 'arg_3',
            },
            {
                'name': 'arg_4',
                'required': False
            },
        ]
        context = {
            'arg_1': 'arg_1_value',
            'arg_3_alt': 'arg_3_value'
        }

        args = pipeline._get_block_args('a_block', block_args, context)

        expected = {
            'arg_1': 'arg_1_value',
            'arg_3': 'arg_3_value',
        }
        assert args == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test__get_outputs_no_outputs(self):
        self_ = MagicMock(autospec=MLPipeline)

        self_._last_block_name = 'last_block'
        self_._get_block_outputs.return_value = ['some', 'outputs']

        pipeline = dict()
        outputs = None
        returned = MLPipeline._get_outputs(self_, pipeline, outputs)

        expected = {
            'default': ['some', 'outputs']
        }
        assert returned == expected

        self_._get_block_outputs.assert_called_once_with('last_block')

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test__get_outputs_defaults(self):
        self_ = MagicMock(autospec=MLPipeline)

        pipeline = dict()
        outputs = {
            'default': ['some', 'outputs']
        }
        returned = MLPipeline._get_outputs(self_, pipeline, outputs)

        expected = {
            'default': ['some', 'outputs']
        }
        assert returned == expected
        self_._get_block_outputs.assert_not_called()

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test__get_outputs_additional(self):
        self_ = MagicMock(autospec=MLPipeline)

        pipeline = {
            'outputs': {
                'default': ['some', 'outputs'],
                'additional': ['other', 'outputs']
            }
        }
        outputs = None
        returned = MLPipeline._get_outputs(self_, pipeline, outputs)

        expected = {
            'default': ['some', 'outputs'],
            'additional': ['other', 'outputs']
        }
        assert returned == expected
        self_._get_block_outputs.assert_not_called()

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_outputs_str_named(self):
        outputs = {
            'default': [
                {
                    'name': 'a_name',
                    'variable': 'a_variable',
                    'type': 'a_type',
                }
            ],
            'debug': [
                {
                    'name': 'another_name',
                    'variable': 'another_variable',
                }
            ]
        }
        pipeline = MLPipeline(['a_primitive', 'another_primitive'], outputs=outputs)

        returned = pipeline.get_outputs('debug')

        expected = [
            {
                'name': 'another_name',
                'variable': 'another_variable',
            }
        ]
        assert returned == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_outputs_str_variable(self):
        pipeline = MLPipeline(['a_primitive', 'another_primitive'])
        pipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'output',
                'type': 'whatever'
            }
        ]

        returned = pipeline.get_outputs('a_primitive#1.output')

        expected = [
            {
                'name': 'output',
                'type': 'whatever',
                'variable': 'a_primitive#1.output'
            }
        ]
        assert returned == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_outputs_str_block(self):
        pipeline = MLPipeline(['a_primitive', 'another_primitive'])

        returned = pipeline.get_outputs('a_primitive#1')

        expected = [
            {
                'name': 'a_primitive#1',
                'variable': 'a_primitive#1',
            }
        ]
        assert returned == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_outputs_int(self):
        pipeline = MLPipeline(['a_primitive', 'another_primitive'])

        returned = pipeline.get_outputs(-1)

        expected = [
            {
                'name': 'another_primitive#1',
                'variable': 'another_primitive#1',
            }
        ]
        assert returned == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_outputs_combination(self):
        outputs = {
            'default': [
                {
                    'name': 'a_name',
                    'variable': 'a_variable',
                    'type': 'a_type',
                }
            ],
            'debug': [
                {
                    'name': 'another_name',
                    'variable': 'another_variable',
                }
            ]
        }
        pipeline = MLPipeline(['a_primitive', 'another_primitive'], outputs=outputs)
        pipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'output',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['another_primitive#1'].produce_output = [
            {
                'name': 'something',
            }
        ]

        returned = pipeline.get_outputs(['default', 'debug', -1, 'a_primitive#1.output'])

        expected = [
            {
                'name': 'a_name',
                'variable': 'a_variable',
                'type': 'a_type'
            },
            {
                'name': 'another_name',
                'variable': 'another_variable',
            },
            {
                'name': 'another_primitive#1',
                'variable': 'another_primitive#1',
            },
            {
                'name': 'output',
                'type': 'whatever',
                'variable': 'a_primitive#1.output'
            }
        ]
        assert returned == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_outputs_invalid(self):
        pipeline = MLPipeline(['a_primitive'])

        pipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'output',
                'type': 'whatever'
            }
        ]

        with pytest.raises(ValueError):
            pipeline.get_outputs('a_primitive#1.invalid')

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_output_names(self):
        outputs = {
            'default': [
                {
                    'name': 'a_name',
                    'variable': 'a_variable',
                    'type': 'a_type',
                }
            ]
        }
        pipeline = MLPipeline(['a_primitive'], outputs=outputs)

        names = pipeline.get_output_names()

        assert names == ['a_name']

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_output_variables(self):
        outputs = {
            'default': [
                {
                    'name': 'a_name',
                    'variable': 'a_variable',
                    'type': 'a_type',
                }
            ]
        }
        pipeline = MLPipeline(['a_primitive'], outputs=outputs)

        names = pipeline.get_output_variables()

        assert names == ['a_variable']

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test__get_block_variables_is_dict(self):
        pipeline = MLPipeline(['a_primitive'])
        pipeline.blocks['a_primitive#1'].produce_outputs = [
            {
                'name': 'output',
                'type': 'whatever'
            }
        ]

        outputs = pipeline._get_block_variables(
            'a_primitive#1',
            'produce_outputs',
            {'output': 'name_output'}
        )

        expected = {
            'name_output': {
                'name': 'output',
                'type': 'whatever',
            }
        }
        assert outputs == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test__get_block_variables_is_str(self):
        pipeline = MLPipeline(['a_primitive'])
        pipeline.blocks['a_primitive#1'].produce_outputs = 'get_produce_outputs'
        pipeline.blocks['a_primitive#1'].instance.get_produce_outputs.return_value = [
            {
                'name': 'output_from_function',
                'type': 'test'
            }

        ]

        outputs = pipeline._get_block_variables(
            'a_primitive#1',
            'produce_outputs',
            {'output': 'name_output'}
        )

        expected = {
            'output_from_function': {
                'name': 'output_from_function',
                'type': 'test',
            }
        }
        assert outputs == expected
        pipeline.blocks['a_primitive#1'].instance.get_produce_outputs.assert_called_once_with()

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_inputs_fit(self):
        pipeline = MLPipeline(['a_primitive', 'another_primitive'])
        pipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].fit_args = [
            {
                'name': 'fit_input',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'output',
                'type': 'another_whatever'
            }
        ]
        pipeline.blocks['another_primitive#1'].produce_args = [
            {
                'name': 'output',
                'type': 'another_whatever'
            },
            {
                'name': 'another_input',
                'type': 'another_whatever'
            }
        ]

        inputs = pipeline.get_inputs()

        expected = {
            'input': {
                'name': 'input',
                'type': 'whatever',
            },
            'fit_input': {
                'name': 'fit_input',
                'type': 'whatever',
            },
            'another_input': {
                'name': 'another_input',
                'type': 'another_whatever',
            }
        }
        assert inputs == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_inputs_no_fit(self):
        pipeline = MLPipeline(['a_primitive', 'another_primitive'])
        pipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].fit_args = [
            {
                'name': 'fit_input',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'output',
                'type': 'another_whatever'
            }
        ]
        pipeline.blocks['another_primitive#1'].produce_args = [
            {
                'name': 'output',
                'type': 'another_whatever'
            },
            {
                'name': 'another_input',
                'type': 'another_whatever'
            }
        ]

        inputs = pipeline.get_inputs(fit=False)

        expected = {
            'input': {
                'name': 'input',
                'type': 'whatever',
            },
            'another_input': {
                'name': 'another_input',
                'type': 'another_whatever',
            }
        }
        assert inputs == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_fit_args(self):
        pipeline = MLPipeline(['a_primitive'])
        pipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].fit_args = [
            {
                'name': 'fit_input',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'output',
                'type': 'another_whatever'
            }
        ]

        outputs = pipeline.get_fit_args()

        expected = [
            {
                'name': 'input',
                'type': 'whatever'
            },
            {
                'name': 'fit_input',
                'type': 'whatever',
            }
        ]
        assert outputs == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_predict_args(self):
        pipeline = MLPipeline(['a_primitive'])
        pipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].fit_args = [
            {
                'name': 'fit_input',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'output',
                'type': 'another_whatever'
            }
        ]
        outputs = pipeline.get_predict_args()

        expected = [
            {
                'name': 'input',
                'type': 'whatever'
            }
        ]
        assert outputs == expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_fit_pending_all_primitives(self):
        block_1 = get_mlblock_mock()
        block_2 = get_mlblock_mock()
        blocks = OrderedDict((
            ('a.primitive.Name#1', block_1),
            ('a.primitive.Name#2', block_2),
        ))

        self_ = MagicMock(autospec=MLPipeline)
        self_.blocks = blocks
        self_._last_fit_block = 'a.primitive.Name#2'

        MLPipeline.fit(self_)

        expected = [
            call('a.primitive.Name#1'),
            call('a.primitive.Name#2')
        ]
        self_._fit_block.call_args_list = expected

        expected = [
            call('a.primitive.Name#1'),
        ]
        self_._produce_block.call_args_list = expected

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_fit_pending_one_primitive(self):
        block_1 = get_mlblock_mock()
        block_2 = get_mlblock_mock()
        blocks = OrderedDict((
            ('a.primitive.Name#1', block_1),
            ('a.primitive.Name#2', block_2),
        ))

        self_ = MagicMock(autospec=MLPipeline)
        self_.blocks = blocks
        self_._last_fit_block = 'a.primitive.Name#1'

        MLPipeline.fit(self_)

        expected = [
            call('a.primitive.Name#1'),
        ]
        self_._fit_block.call_args_list = expected

        assert not self_._produce_block.called

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_fit_no_debug(self):
        mlpipeline = MLPipeline(['a_primitive'])
        mlpipeline.blocks['a_primitive#1'].fit_args = [
            {
                'name': 'fit_input',
                'type': 'whatever'
            }
        ]

        returned = mlpipeline.fit(debug=False)

        assert returned is None

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_fit_debug_bool(self):
        mlpipeline = MLPipeline(['a_primitive'])
        mlpipeline.blocks['a_primitive#1'].fit_args = [
            {
                'name': 'fit_input',
                'type': 'whatever'
            }
        ]

        expected_return = dict()
        expected_return['debug'] = 'tmio'
        expected_return['fit'] = {
            'a_primitive#1': {
                'time': 0,
                'input': {
                    'whatever'
                },
                'memory': 0,
            }
        }

        returned = mlpipeline.fit(debug=True)

        assert isinstance(returned, dict)
        assert set(returned.keys()) == set(expected_return.keys())  # fit / produce
        assert set(returned['fit'].keys()) == set(expected_return['fit'].keys())  # block name

        for block_name, dictionary in expected_return['fit'].items():
            assert set(returned['fit'][block_name].keys()) == set(dictionary.keys())

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_fit_debug_str(self):
        mlpipeline = MLPipeline(['a_primitive'])
        mlpipeline.blocks['a_primitive#1'].fit_args = [
            {
                'name': 'fit_input',
                'type': 'whatever'
            }
        ]

        expected_return = dict()
        expected_return['debug'] = 'tm'
        expected_return['fit'] = {
            'a_primitive#1': {
                'time': 0,
                'memory': 0,
            }
        }

        returned = mlpipeline.fit(debug='tm')

        assert isinstance(returned, dict)
        assert set(returned.keys()) == set(expected_return.keys())  # fit / produce
        assert set(returned['fit'].keys()) == set(expected_return['fit'].keys())  # block name

        for block_name, dictionary in expected_return['fit'].items():
            assert set(returned['fit'][block_name].keys()) == set(dictionary.keys())

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_fit_produce_debug(self):
        outputs = {
            'default': [
                {
                    'name': 'a_name',
                    'variable': 'a_primitive#1.a_variable',
                    'type': 'a_type',
                }
            ]
        }
        mlpipeline = MLPipeline(['a_primitive'], outputs=outputs)
        mlpipeline.blocks['a_primitive#1'].fit_args = [
            {
                'name': 'fit_input',
                'type': 'whatever'
            }
        ]

        mlpipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input',
                'type': 'whatever'
            }
        ]

        mlpipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'a_name',
                'type': 'a_type'
            }
        ]

        expected_return = dict()
        expected_return['debug'] = 'tmio'
        expected_return['fit'] = {
            'a_primitive#1': {
                'time': 0,
                'input': {
                    'whatever'
                },
                'memory': 0,
            }
        }
        expected_return['produce'] = {
            'a_primitive#1': {
                'time': 0,
                'input': {
                    'whatever'
                },
                'output': {
                    'whatever'
                },
                'memory': 0,
            }
        }

        returned, debug_returned = mlpipeline.fit(output_='default', debug=True)

        assert len([returned]) == len(outputs['default'])
        assert isinstance(debug_returned, dict)
        assert set(debug_returned.keys()) == set(expected_return.keys())  # fit / produce
        assert set(debug_returned['fit'].keys()) == set(expected_return['fit'].keys())
        assert set(debug_returned['produce'].keys()) == set(expected_return['produce'].keys())

        for block_name, dictionary in expected_return['fit'].items():
            assert set(debug_returned['fit'][block_name].keys()) == set(dictionary.keys())

        for block_name, dictionary in expected_return['produce'].items():
            assert set(debug_returned['produce'][block_name].keys()) == set(dictionary.keys())

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_fit_produce_debug_str(self):
        outputs = {
            'default': [
                {
                    'name': 'a_name',
                    'variable': 'a_primitive#1.a_variable',
                    'type': 'a_type',
                }
            ]
        }
        mlpipeline = MLPipeline(['a_primitive'], outputs=outputs)
        mlpipeline.blocks['a_primitive#1'].fit_args = [
            {
                'name': 'fit_input',
                'type': 'whatever'
            }
        ]

        mlpipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input',
                'type': 'whatever'
            }
        ]

        mlpipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'a_name',
                'type': 'a_type'
            }
        ]

        expected_return = dict()
        expected_return['debug'] = 'tm'
        expected_return['fit'] = {
            'a_primitive#1': {
                'time': 0,
                'memory': 0,
            }
        }
        expected_return['produce'] = {
            'a_primitive#1': {
                'time': 0,
                'memory': 0,
            }
        }

        returned, debug_returned = mlpipeline.fit(output_='default', debug='tm')

        assert len([returned]) == len(outputs['default'])
        assert isinstance(debug_returned, dict)
        assert set(debug_returned.keys()) == set(expected_return.keys())  # fit / produce
        assert set(debug_returned['fit'].keys()) == set(expected_return['fit'].keys())
        assert set(debug_returned['produce'].keys()) == set(expected_return['produce'].keys())

        for block_name, dictionary in expected_return['fit'].items():
            assert set(debug_returned['fit'][block_name].keys()) == set(dictionary.keys())

        for block_name, dictionary in expected_return['produce'].items():
            assert set(debug_returned['produce'][block_name].keys()) == set(dictionary.keys())

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_predict_no_debug(self):
        outputs = {
            'default': [
                {
                    'name': 'a_name',
                    'variable': 'a_primitive#1.a_variable',
                    'type': 'a_type',
                },
                {
                    'name': 'b_name',
                    'variable': 'a_primitive#1.b_variable',
                    'type': 'b_type',
                },
            ]
        }
        mlpipeline = MLPipeline(['a_primitive'], outputs=outputs)
        mlpipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input',
                'type': 'whatever'
            }
        ]

        mlpipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'a_name',
                'type': 'a_type'
            },
            {
                'name': 'b_name',
                'type': 'b_type'
            }
        ]

        returned = mlpipeline.predict(debug=False)
        assert len(returned) == len(outputs['default'])
        for returned_output, expected_output in zip(returned, outputs['default']):
            assert returned_output == expected_output['variable']

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_predict_debug(self):
        outputs = {
            'default': [
                {
                    'name': 'a_name',
                    'variable': 'a_primitive#1.a_variable',
                    'type': 'a_type',
                }
            ]
        }
        mlpipeline = MLPipeline(['a_primitive'], outputs=outputs)
        mlpipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input',
                'type': 'whatever'
            }
        ]

        mlpipeline.blocks['a_primitive#1'].produce_output = [
            {
                'name': 'a_name',
                'type': 'a_type'
            }
        ]

        expected_return = dict()
        expected_return = {
            'a_primitive#1': {
                'time': 0,
                'input': {
                    'whatever'
                },
                'output': {
                    'whatever'
                },
                'memory': 0
            }
        }

        returned, debug_returned = mlpipeline.predict(debug=True)
        debug_returned = debug_returned['produce']

        assert len([returned]) == len(outputs['default'])
        assert isinstance(debug_returned, dict)
        assert set(debug_returned.keys()) == set(expected_return.keys())

        for block_name, dictionary in expected_return.items():
            assert set(debug_returned[block_name].keys()) == set(dictionary.keys())

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_diagram_simple(self):
        f = open('tests/data/diagrams/diagram_simple.txt', 'r')
        expected = f.read()[:-1]
        f.close()

        output = [
            {
                'name': 'output_variable',
                'type': 'another_whatever',
                'variable': 'a_primitive#1.output_variable'
            }
        ]

        pipeline = MLPipeline(['a_primitive'], outputs={'default': output})
        pipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input_variable',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].produce_output = output

        assert str(pipeline.get_diagram()).strip() == expected.strip()

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_diagram_fit(self):
        f = open('tests/data/diagrams/diagram_fit.txt', 'r')
        expected = f.read()[:-1]
        f.close()

        output = [
            {
                'name': 'output_variable',
                'type': 'another_whatever',
                'variable': 'a_primitive#1.output_variable'
            }
        ]

        pipeline = MLPipeline(['a_primitive'], outputs={'default': output})
        pipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input_variable',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].fit_args = [
            {
                'name': 'input_variable',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].produce_output = output

        assert str(pipeline.get_diagram()).strip() == expected.strip()

    @patch('mlblocks.mlpipeline.MLBlock', new=get_mlblock_mock)
    def test_get_diagram_multiple_blocks(self):
        f = open('tests/data/diagrams/diagram_multiple_blocks.txt', 'r')
        expected = f.read()[:-1]
        f.close()

        first_output = [
            {
                'name': 'output_variable_a',
                'type': 'another_whatever',
                'variable': 'a_primitive#1.output_variable_a'
            }
        ]
        second_output = [
            {
                'name': 'output_variable_b',
                'type': 'another_whatever',
                'variable': 'b_primitive#1.output_variable_b'
            }
        ]

        pipeline = MLPipeline(['a_primitive', 'b_primitive'], outputs={'default': second_output})
        pipeline.blocks['a_primitive#1'].produce_args = [
            {
                'name': 'input_variable',
                'type': 'whatever'
            }
        ]
        pipeline.blocks['a_primitive#1'].produce_output = first_output
        pipeline.blocks['b_primitive#1'].produce_args = first_output
        pipeline.blocks['b_primitive#1'].produce_output = second_output

        assert str(pipeline.get_diagram()).strip() == expected.strip()

    def test_fit(self):
        pass

    def test_predict(self):
        pass

    def test_to_dict(self):
        pass

    def test_save(self):
        pass

    def test_from_dict(self):
        pass

    def test_load(self):
        pass
