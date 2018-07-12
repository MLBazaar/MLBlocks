import logging
from collections import Counter, OrderedDict

from mlblocks.mlblock import MLBlock

LOGGER = logging.getLogger(__name__)


class MLPipeline(object):

    def __init__(self, blocks, init_params=None):
        """Initialize a MLPipeline with a list of corresponding MLBlocks.

        Args:
            blocks: A list of MLBlocks composing this pipeline. MLBlocks
                    can be either MLBlock instances or primitive names to
                    load from the configuration JSON files.
        """
        init_params = init_params or dict()

        self.blocks = OrderedDict()
        block_names_count = Counter()
        for block in blocks:
            block_names_count.update(block)
            block_name = '{}_{}'.format(block, block_names_count[block])

            block_params = init_params.get(block_name, dict())
            mlblock = MLBlock(block, **block_params)

            self.blocks[block_name] = mlblock

    def get_tunable_hyperparameters(self):
        tunable = {}
        for block_name, block in self.blocks.items():
            tunable[block_name] = block.get_tunable_hyperparameters()

    def get_hyperparameters(self):
        hyperparameters = {}
        for block_name, block in self.blocks.items():
            hyperparameters[block_name] = block.get_hyperparameters()

    def set_hyperparameters(self, hyperparameters):
        for block_name, block_hyperparams in hyperparameters.items():
            self.blocks[block_name].set_hyperparameters(block_hyperparams)

    @staticmethod
    def _get_block_args(block_args, variables):
        # TODO: type validation and/or transformation should be done here

        args = dict()
        for name in block_args.keys():
            # name = block_arg['name']
            args[name] = variables[name]

        return args

    def _get_outputs(outputs, block_outputs):
        # TODO: type validation and/or transformation should be done here

        if len(outputs) != len(block_outputs):
            error = 'Invalid number of outputs. Expected {} but got {}'.format(
                len(block_outputs), len(outputs))
            raise ValueError(error)

        output_dict = dict()
        for output, block_output in zip(outputs, block_outputs):
            name = block_output['name']
            output_dict[name] = output

        return output_dict

    def fit(self, **inputs):
        variables = inputs.copy()

        last_block_name = list(self.blocks.keys())[-1]
        for block_name, block in self.blocks.items():
            fit_args = self._get_block_args(block.fit_args, variables)

            LOGGER.debug("Fitting block %s", block_name)
            block.fit(**fit_args)

            if block_name != last_block_name:
                produce_args = self._get_block_args(block.produce_args, variables)

                LOGGER.debug("Producing block %s", block_name)
                outputs = block.produce(**produce_args)

                output_dict = self._get_outputs(outputs, block.produce_output)
                variables.update(output_dict)

    def predict(self, **inputs):
        variables = inputs.copy()

        last_block_name = list(self.blocks.keys())[-1]
        for block_name, block in self.blocks.items():
            produce_args = self._get_block_args(block.produce_args, variables)

            LOGGER.debug("Producing block %s", block_name)
            outputs = block.produce(**produce_args)

            if block_name != last_block_name:
                output_dict = self._get_outputs(outputs, block.produce_output)
                variables.update(output_dict)

        return outputs
