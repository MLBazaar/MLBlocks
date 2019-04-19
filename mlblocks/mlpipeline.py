# -*- coding: utf-8 -*-

"""Package where the MLPipeline class is defined."""

import json
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
        primitives (list): List of the names of the primitives that compose
                           this pipeline.
        blocks (list): OrderedDict of the block names and the corresponding
                       MLBlock instances.
        init_params (dict): init_params dictionary, as given when the instance
                            was created.
        input_names (dict): input_names dictionary, as given when the instance
                            was created.
        output_names (dict): output_names dictionary, as given when the instance
                             was created.

    Args:
        primitives (list): List with the names of the primitives that will
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

    def _get_tunable_hyperparameters(self):
        tunable = {}
        for block_name, block in self.blocks.items():
            tunable[block_name] = block.get_tunable_hyperparameters()

        return tunable

    def __init__(self, primitives, init_params=None, input_names=None, output_names=None):
        self.primitives = primitives
        self.init_params = init_params or dict()
        self.blocks = OrderedDict()

        block_names_count = Counter()
        for primitive in primitives:
            try:
                block_names_count.update([primitive])
                block_count = block_names_count[primitive]
                block_name = '{}#{}'.format(primitive, block_count)
                block_params = self.init_params.get(block_name, dict())
                if not block_params:
                    block_params = self.init_params.get(primitive, dict())
                    if block_params and block_count > 1:
                        LOGGER.warning(("Non-numbered init_params are being used "
                                        "for more than one block %s."), primitive)

                block = MLBlock(primitive, **block_params)
                self.blocks[block_name] = block

            except Exception:
                LOGGER.exception("Exception caught building MLBlock %s", primitive)
                raise

        self.input_names = input_names or dict()
        self.output_names = output_names or dict()
        self._tunable_hyperparameters = self._get_tunable_hyperparameters()

    def get_tunable_hyperparameters(self):
        """Get the tunable hyperparamters of each block.

        Returns:
            dict:
                A dictionary containing the block names as keys and
                the block tunable hyperparameters dictionary as values.
        """
        return self._tunable_hyperparameters.copy()

    def get_hyperparameters(self):
        """Get the current hyperparamters of each block.

        Returns:
            dict:
                A dictionary containing the block names as keys and
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

    def _extract_outputs(self, block_name, outputs, block_outputs):
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

    def _get_block_name(self, index):
        return list(self.blocks.keys())[index]

    def _get_output_spec(self, output):
        if output is None:
            return None, None

        if isinstance(output, int):
            output = self._get_block_name(output)

        if output in self.blocks:
            return output, None

        if '.' in output:
            output_block, output_variable = output.rsplit('.', 1)
            if output_block not in self.blocks:
                raise ValueError('Unknown block name: {}'.format(output_block))

            return output_block, output_variable

        last_block_name = self._get_block_name(-1)
        return last_block_name, output

    def _get_output(self, output_variable, context):
        if output_variable:
            if output_variable not in context:
                raise ValueError('Output variable {} not found in context'
                                 .format(output_variable))

            return context[output_variable]
        else:
            return context

    def fit(self, X=None, y=None, output=None, skip_to=None, **kwargs):
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

        output_block, output_variable = self._get_output_spec(output)
        last_block_name = self._get_block_name(-1)

        if isinstance(skip_to, int):
            skip_to = self._get_block_name(skip_to)

        for block_name, block in self.blocks.items():
            if block_name == skip_to:
                skip_to = False
            elif skip_to:
                LOGGER.debug("Skipping block %s fit", block_name)
                continue

            LOGGER.debug("Fitting block %s", block_name)
            try:
                fit_args = self._get_block_args(block_name, block.fit_args, context)
                block.fit(**fit_args)
            except Exception:
                LOGGER.exception("Exception caught fitting MLBlock %s", block_name)
                raise

            if (block_name != last_block_name) or (block_name == output_block):
                LOGGER.debug("Producing block %s", block_name)
                try:
                    produce_args = self._get_block_args(block_name, block.produce_args, context)
                    outputs = block.produce(**produce_args)

                    output_dict = self._extract_outputs(block_name, outputs, block.produce_output)
                    context.update(output_dict)
                except Exception:
                    LOGGER.exception("Exception caught producing MLBlock %s", block_name)
                    raise

            if block_name == output_block:
                return self._get_output(output_variable, context)

    def predict(self, X=None, output='y', skip_to=None, **kwargs):
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

        output_block, output_variable = self._get_output_spec(output)

        if isinstance(skip_to, int):
            skip_to = self._get_block_name(skip_to)

        for block_name, block in self.blocks.items():
            if block_name == skip_to:
                skip_to = False
            elif skip_to:
                LOGGER.debug("Skipping block %s produce", block_name)
                continue

            LOGGER.debug("Producing block %s", block_name)
            try:
                produce_args = self._get_block_args(block_name, block.produce_args, context)
                outputs = block.produce(**produce_args)
                output_dict = self._extract_outputs(block_name, outputs, block.produce_output)
                context.update(output_dict)

                if block_name == output_block:
                    return self._get_output(output_variable, context)

            except Exception:
                LOGGER.exception("Exception caught producing MLBlock %s", block_name)
                raise

        if skip_to:
            # We skipped all the blocks up to the end
            raise ValueError('Unknown block name: {}'.format(skip_to))

    def to_dict(self):
        """Return all the details of this MLPipeline in a dict.

        The dict structure contains all the `__init__` arguments of the
        MLPipeline, as well as the current hyperparameter values and the
        specification of the tunable_hyperparameters::

            {
                "primitives": [
                    "a_primitive",
                    "another_primitive"
                ],
                "init_params": {
                    "a_primitive": {
                        "an_argument": "a_value"
                    }
                },
                "hyperparameters": {
                    "a_primitive#1": {
                        "an_argument": "a_value",
                        "another_argument": "another_value",
                    },
                    "another_primitive#1": {
                        "yet_another_argument": "yet_another_value"
                     }
                },
                "tunable_hyperparameters": {
                    "another_primitive#1": {
                        "yet_another_argument": {
                            "type": "str",
                            "default": "a_default_value",
                            "values": [
                                "a_default_value",
                                "yet_another_value"
                            ]
                        }
                    }
                }
            }
        """
        return {
            'primitives': self.primitives,
            'init_params': self.init_params,
            'input_names': self.input_names,
            'output_names': self.output_names,
            'hyperparameters': self.get_hyperparameters(),
            'tunable_hyperparameters': self._tunable_hyperparameters
        }

    def save(self, path):
        """Save the specification of this MLPipeline in a JSON file.

        The content of the JSON file is the dict returned by the `to_dict` method.

        Args:
            path (str): Path to the JSON file to write.
        """
        with open(path, 'w') as out_file:
            json.dump(self.to_dict(), out_file, indent=4)

    @classmethod
    def from_dict(cls, metadata):
        """Create a new MLPipeline from a dict specification.

        The dict structure is the same as the one created by the `to_dict` method.

        Args:
            metadata (dict): Dictionary containing the pipeline specification.

        Returns:
            MLPipeline:
                A new MLPipeline instance with the details found in the
                given specification dictionary.
        """
        hyperparameters = metadata.get('hyperparameters')
        tunable = metadata.get('tunable_hyperparameters')

        pipeline = cls(
            metadata['primitives'],
            metadata.get('init_params'),
            metadata.get('input_names'),
            metadata.get('output_names'),
        )

        if hyperparameters:
            pipeline.set_hyperparameters(hyperparameters)

        if tunable is not None:
            pipeline._tunable_hyperparameters = tunable

        return pipeline

    @classmethod
    def load(cls, path):
        """Create a new MLPipeline from a JSON specification.

        The JSON file format is the same as the one created by the `to_dict` method.

        Args:
            path (str): Path of the JSON file to load.

        Returns:
            MLPipeline:
                A new MLPipeline instance with the specification found
                in the JSON file.
        """
        with open(path, 'r') as in_file:
            metadata = json.load(in_file)

        return cls.from_dict(metadata)
