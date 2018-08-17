# -*- coding: utf-8 -*-

import logging
from collections import Counter, OrderedDict

from mlblocks.mlblock import MLBlock

LOGGER = logging.getLogger(__name__)


class MLPipeline(object):

    def __init__(self, blocks, init_params=None, input_names=None, output_names=None):
        """Initialize a MLPipeline with a list of corresponding MLBlocks.

        Args:
            blocks: List with the names of the primitives that will
                    compose this pipeline.
            init_params: dictionary containing initialization arguments to
                         be passed when creating the MLBlocks instances.
                         The dictionary keys must be the corresponding primitive
                         names and the values must be another dictionary that will
                         be passed as `**kargs` to the MLBlock instance.
            input_names: dictionary that maps input variable names with the actual
                         names expected by each primitive. This allows reusing
                         the same input argument for multiple primitives that name
                         it differently, as well as passing different values to
                         primitives that expect arguments named similary.
            output_names: dictionary that maps output variable names with the name
                          these variables will be given when stored in the variables
                          dictionary. This allows storing the output of different
                          primitives in different variables, even if the primitive
                          output name is the same one.
        """
        init_params = init_params or dict()

        self.blocks = OrderedDict()
        block_names_count = Counter()
        for block in blocks:
            try:
                block_names_count.update([block])
                block_count = block_names_count[block]
                block_name = '{}#{}'.format(block, block_count)
                block_params = init_params.get(block_name, dict())
                if not block_params:
                    block_params = init_params.get(block, dict())
                    if block_params and block_count > 1:
                        LOGGER.warn(("Non-numbered init_params are being used "
                                     "for more than one block %s."), block)

                mlblock = MLBlock(block, **block_params)
                self.blocks[block_name] = mlblock
            except Exception as e:
                LOGGER.exception("Exception caught building MLBlock {}".format(block))
                raise

        self.input_names = input_names or dict()
        self.output_names = output_names or dict()

    def get_tunable_hyperparameters(self):
        tunable = {}
        for block_name, block in self.blocks.items():
            tunable[block_name] = block.get_tunable_hyperparameters()

        return tunable

    def get_hyperparameters(self):
        hyperparameters = {}
        for block_name, block in self.blocks.items():
            hyperparameters[block_name] = block.get_hyperparameters()

        return hyperparameters

    def set_hyperparameters(self, hyperparameters):
        for block_name, block_hyperparams in hyperparameters.items():
            self.blocks[block_name].set_hyperparameters(block_hyperparams)

    def _get_block_args(self, block_name, block_args, variables):
        # TODO: type validation and/or transformation should be done here

        input_names = self.input_names.get(block_name, dict())

        kwargs = dict()
        for arg in block_args:
            name = arg['name']
            keyword = arg.get('keyword', name)
            variable = input_names.get(name, name)

            if variable in variables:
                value = variables[variable]

            elif 'default' in arg:
                value = arg['default']

            else:
                raise TypeError(
                    "Expected argument '{}.{}' not found in variables"
                    .format(block_name, variable)
                )

            kwargs[keyword] = value

        return kwargs

    def _get_outputs(self, block_name, outputs, block_outputs):
        # TODO: type validation and/or transformation should be done here

        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        elif len(outputs) != len(block_outputs):
            error = 'Invalid number of outputs. Expected {} but got {}'.format(
                len(block_outputs), len(outputs))

            raise ValueError(error)

        output_names = self.output_names.get(block_name, dict())

        output_dict = dict()
        for output, block_output in zip(outputs, block_outputs):
            name = block_output['name']
            output_name = output_names.get(name, name)
            output_dict[output_name] = output

        return output_dict

    def fit(self, X=None, y=None, **kwargs):
        variables = {
            'X': X,
            'y': y
        }
        variables.update(kwargs)

        last_block_name = list(self.blocks.keys())[-1]
        for block_name, block in self.blocks.items():
            fit_args = self._get_block_args(block_name, block.fit_args, variables)

            LOGGER.debug("Fitting block %s", block_name)
            block.fit(**fit_args)

            if block_name != last_block_name:
                produce_args = self._get_block_args(block_name, block.produce_args, variables)

                LOGGER.debug("Producing block %s", block_name)
                outputs = block.produce(**produce_args)

                output_dict = self._get_outputs(block_name, outputs, block.produce_output)
                variables.update(output_dict)

    def predict(self, X=None, **kwargs):
        variables = {
            'X': X
        }
        variables.update(kwargs)

        last_block_name = list(self.blocks.keys())[-1]
        for block_name, block in self.blocks.items():
            produce_args = self._get_block_args(block_name, block.produce_args, variables)

            LOGGER.debug("Producing block %s", block_name)
            outputs = block.produce(**produce_args)

            if block_name != last_block_name:
                output_dict = self._get_outputs(block_name, outputs, block.produce_output)
                variables.update(output_dict)

        return outputs
