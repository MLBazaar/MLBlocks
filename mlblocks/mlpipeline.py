# -*- coding: utf-8 -*-

"""Package where the MLPipeline class is defined."""

import logging
from collections import Counter, OrderedDict

from mlblocks.mlblock import MLBlock

LOGGER = logging.getLogger(__name__)


class MLPipeline():
    """MLPipeline Class.

    The **MLPipeline** class represents a Machine Learning Pipeline, which
    is an ordered collection of Machine Learning tools or Primitives,
    represented by **MLBlock instances**, that will be fitted and then used
    sequentially in order to produce results.

    The MLPipeline has two working modes or phases: **fitting** and
    **predicting**.

    During the **fitting** phase, each MLBlock instance, or **block** will be
    fitted and immediately after used to produce results on the same
    fitting data.
    This results will be then passed to the next block of the sequence
    as its fitting data, and this process will be repeated until the last
    block is fitted.

    During the **predicting** phase, each block will be used to produce results
    on the output of the previous one, until the last one has produce its
    results, which will be returned as the prediction of the pipeline.

    Attributes:
        blocks (list): OrderedDict of the block names and the corresponding
                       MLBlock instances.
        input_names (dict): input_names dictionary, as given when the instance
                            was created..
        output_names (dict): output_names dictionary, as given when the instance
                             was created.

    Args:
        blocks (list): List with the names of the primitives that will
                       compose this pipeline.
        init_params (dict): dictionary containing initialization arguments to
                            be passed when creating the MLBlocks instances.
                            The dictionary keys must be the corresponding
                            primitive names and the values must be another
                            dictionary that will be passed as `**kargs` to the
                            MLBlock instance.
        input_names (dict): dictionary that maps input variable names with the
                            actual names expected by each primitive. This
                            allows reusing the same input argument for multiple
                            primitives that name it differently, as well as
                            passing different values to primitives that expect
                            arguments named similary.
        output_names (dict): dictionary that maps output variable names with
                             the name these variables will be given when stored
                             in the context dictionary. This allows storing
                             the output of different primitives in different
                             variables, even if the primitive output name is
                             the same one.
    """

    def __init__(self, blocks, init_params=None, input_names=None, output_names=None):
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
                        LOGGER.warning(("Non-numbered init_params are being used "
                                        "for more than one block %s."), block)

                mlblock = MLBlock(block, **block_params)
                self.blocks[block_name] = mlblock

            except Exception:
                LOGGER.exception("Exception caught building MLBlock %s", block)
                raise

        self.input_names = input_names or dict()
        self.output_names = output_names or dict()

    def get_tunable_hyperparameters(self):
        """Get the tunable hyperparamters of each block.

        Returns:
            dict: A dictionary containing the block names as keys and
                  the block tunable hyperparameters dictionary as values.
        """
        tunable = {}
        for block_name, block in self.blocks.items():
            tunable[block_name] = block.get_tunable_hyperparameters()

        return tunable

    def get_hyperparameters(self):
        """Get the current hyperparamters of each block.

        Returns:
            dict: A dictionary containing the block names as keys and
                  the current block hyperparameters dictionary as values.
        """
        hyperparameters = {}
        for block_name, block in self.blocks.items():
            hyperparameters[block_name] = block.get_hyperparameters()

        return hyperparameters

    def set_hyperparameters(self, hyperparameters):
        """Set new hyperparameter values for some blocks.

        Args:
            hyperparameters (dict): A dictionary containing the block names as
                                    keys and the new hyperparameters dictionary
                                    as values.
        """
        for block_name, block_hyperparams in hyperparameters.items():
            self.blocks[block_name].set_hyperparameters(block_hyperparams)

    def _get_block_args(self, block_name, block_args, context):
        # TODO: type validation and/or transformation should be done here

        input_names = self.input_names.get(block_name, dict())

        kwargs = dict()
        for arg in block_args:
            name = arg['name']
            keyword = arg.get('keyword', name)
            variable = input_names.get(name, name)

            if variable in context:
                value = context[variable]

            elif 'default' in arg:
                value = arg['default']

            else:
                raise TypeError(
                    "Expected argument '{}.{}' not found in context"
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
        """Fit the blocks of this pipeline.

        Sequentially call the `fit` and the `produce` methods of each block,
        capturing the outputs each `produce` method before calling the `fit`
        method of the next one.

        During the whole process a context dictionary is built, where both the
        passed arguments and the captured outputs of the `produce` methods
        are stored, and from which the arguments for the next `fit` and
        `produce` calls will be taken.

        Args:
            X: Fit Data, which the pipeline will learn from.
            y: Fit Data labels, which the pipeline will use to learn how to
               behave.
            **kwargs: Any additional keyword arguments will be directly added
                      to the context dictionary and available for the blocks.
        """
        context = {
            'X': X,
            'y': y
        }
        context.update(kwargs)

        last_block_name = list(self.blocks.keys())[-1]
        for block_name, block in self.blocks.items():
            fit_args = self._get_block_args(block_name, block.fit_args, context)

            LOGGER.debug("Fitting block %s", block_name)
            block.fit(**fit_args)

            if block_name != last_block_name:
                produce_args = self._get_block_args(block_name, block.produce_args, context)

                LOGGER.debug("Producing block %s", block_name)
                outputs = block.produce(**produce_args)

                output_dict = self._get_outputs(block_name, outputs, block.produce_output)
                context.update(output_dict)

    def predict(self, X=None, **kwargs):
        """Produce predictions using the blocks of this pipeline.

        Sequentially call the `produce` method of each block, capturing the
        outputs before calling the next one.

        During the whole process a context dictionary is built, where both the
        passed arguments and the captured outputs of the `produce` methods
        are stored, and from which the arguments for the next `produce` calls
        will be taken.

        Args:
            X: Data which the pipeline will use to make predictions.
            **kwargs: Any additional keyword arguments will be directly added
                      to the context dictionary and available for the blocks.
        """
        context = {
            'X': X
        }
        context.update(kwargs)

        last_block_name = list(self.blocks.keys())[-1]
        for block_name, block in self.blocks.items():
            produce_args = self._get_block_args(block_name, block.produce_args, context)

            LOGGER.debug("Producing block %s", block_name)
            outputs = block.produce(**produce_args)

            if block_name != last_block_name:
                output_dict = self._get_outputs(block_name, outputs, block.produce_output)
                context.update(output_dict)

        return outputs
