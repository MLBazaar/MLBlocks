# -*- coding: utf-8 -*-

"""Package where the MLPipeline class is defined."""

import json
import logging
from collections import Counter, OrderedDict, defaultdict

import numpy as np

from mlblocks.discovery import load_pipeline
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
        primitives (list):
            List of the names of the primitives that compose this pipeline.
        blocks (list):
            OrderedDict of the block names and the corresponding MLBlock instances.
        init_params (dict):
            init_params dictionary, as given when the instance was created.
        input_names (dict):
            input_names dictionary, as given when the instance was created.
        output_names (dict):
            output_names dictionary, as given when the instance was created.

    Args:
        pipeline (str, list, dict or MLPipeline):
            The pipeline argument accepts four different types with different interpretations:
                * `str`: the name of the pipeline to search and load.
                * `list`: the primitives list.
                * `dict`: a complete pipeline specification.
                * `MLPipeline`: another pipeline to be cloned.
        primitives (list):
            List with the names of the primitives that will compose this pipeline.
        init_params (dict):
            dictionary containing initialization arguments to be passed when creating the
            MLBlocks instances. The dictionary keys must be the corresponding primitive names
            and the values must be another dictionary that will be passed as ``**kargs`` to the
            MLBlock instance.
        input_names (dict):
            dictionary that maps input variable names with the actual names expected by each
            primitive. This allows reusing the same input argument for multiple primitives that
            name it differently, as well as passing different values to primitives that expect
            arguments named similary.
        output_names (dict):
            dictionary that maps output variable names with the name these variables will be
            given when stored in the context dictionary. This allows storing the output of
            different primitives in different variables, even if the primitive output name is
            the same one.
    """

    def _get_tunable_hyperparameters(self):
        """Get the tunable hyperperparameters from all the blocks in this pipeline."""
        tunable = {}
        for block_name, block in self.blocks.items():
            tunable[block_name] = block.get_tunable_hyperparameters()

        return tunable

    def _build_blocks(self):
        blocks = OrderedDict()

        block_names_count = Counter()
        for primitive in self.primitives:
            if isinstance(primitive, str):
                primitive_name = primitive
            else:
                primitive_name = primitive['name']

            try:
                block_names_count.update([primitive_name])
                block_count = block_names_count[primitive_name]
                block_name = '{}#{}'.format(primitive_name, block_count)
                block_params = self.init_params.get(block_name, dict())
                if not block_params:
                    block_params = self.init_params.get(primitive_name, dict())
                    if block_params and block_count > 1:
                        LOGGER.warning(("Non-numbered init_params are being used "
                                        "for more than one block %s."), primitive_name)

                block = MLBlock(primitive, **block_params)
                blocks[block_name] = block

            except Exception:
                LOGGER.exception("Exception caught building MLBlock %s", primitive)
                raise

        return blocks

    @staticmethod
    def _get_pipeline_dict(pipeline, primitives):

        if isinstance(pipeline, dict):
            return pipeline

        elif isinstance(pipeline, str):
            return load_pipeline(pipeline)

        elif isinstance(pipeline, MLPipeline):
            return pipeline.to_dict()

        elif isinstance(pipeline, list):
            if primitives is not None:
                raise ValueError('if `pipeline` is a `list`, `primitives` must be `None`')

            return {'primitives': pipeline}

        elif pipeline is None:
            if primitives is None:
                raise ValueError('Either `pipeline` or `primitives` must be not `None`.')

            return dict()

    def __init__(self, pipeline=None, primitives=None, init_params=None,
                 input_names=None, output_names=None):

        pipeline = self._get_pipeline_dict(pipeline, primitives)

        self.primitives = primitives or pipeline['primitives']
        self.init_params = init_params or pipeline.get('init_params', dict())
        self.blocks = self._build_blocks()

        self.input_names = input_names or pipeline.get('input_names', dict())
        self.output_names = output_names or pipeline.get('output_names', dict())

        tunable = pipeline.get('tunable_hyperparameters')
        if tunable is not None:
            self._tunable_hyperparameters = tunable
        else:
            self._tunable_hyperparameters = self._get_tunable_hyperparameters()

        hyperparameters = pipeline.get('hyperparameters')
        if hyperparameters:
            self.set_hyperparameters(hyperparameters)

    @staticmethod
    def _flatten_dict(hyperparameters):
        return {
            (block, name): value
            for block, block_hyperparameters in hyperparameters.items()
            for name, value in block_hyperparameters.items()
        }

    def get_tunable_hyperparameters(self, flat=False):
        """Get the tunable hyperparamters of each block.

        Args:
            flat (bool): If True, return a flattened dictionary where each key
                is a two elements tuple containing the name of the block as the first
                element and the name of the hyperparameter as the second one.
                If False (default), return a dictionary where each key is the name of
                a block and each value is a dictionary containing the complete
                hyperparameter specification of that block.

        Returns:
            dict:
                A dictionary containing the block names as keys and
                the block tunable hyperparameters dictionary as values.
        """
        tunables = self._tunable_hyperparameters.copy()
        if flat:
            tunables = self._flatten_dict(tunables)

        return tunables

    @classmethod
    def _sanitize_value(cls, value):
        """Convert numpy values to their python primitive type equivalent.

        If a value is a dict, recursively sanitize its values.

        Args:
            value:
                value to sanitize.

        Returns:
            sanitized value.
        """
        if isinstance(value, dict):
            return {
                key: cls._sanitize_value(value)
                for key, value in value.items()
            }
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.bool_):
            return bool(value)
        elif value == 'None':
            return None

        return value

    @classmethod
    def _sanitize(cls, hyperparameters):
        """Convert tuple hyperparameter keys to nested dicts.

        Also convert numpy types to primary python types.

        The input hyperparameters dict can specify them in two formats:

        One is the native MLBlocks format, where each key is the name of a block and each value
        is a dict containing a complete hyperparameter specification for that block::

            {
                "block_name": {
                    "hyperparameter_name": "hyperparameter_value",
                    ...
                },
                ...
            }

        The other one is an alternative format where each key is a two element tuple containing
        the name of the block as the first element and the name of the hyperparameter as the
        second one::

            {
                ("block_name", "hyperparameter_name"): "hyperparameter_value",
                ...
            }


        Args:
            hyperparaeters (dict):
                hyperparameters dict to sanitize.

        Returns:
            dict:
                Sanitized dict.
        """
        params_tree = defaultdict(dict)
        for key, value in hyperparameters.items():
            value = cls._sanitize_value(value)
            if isinstance(key, tuple):
                block, hyperparameter = key
                params_tree[block][hyperparameter] = value
            else:
                params_tree[key] = value

        return params_tree

    def get_hyperparameters(self, flat=False):
        """Get the current hyperparamters of each block.

        Args:
            flat (bool): If True, return a flattened dictionary where each key
                is a two elements tuple containing the name of the block as the first
                element and the name of the hyperparameter as the second one.
                If False (default), return a dictionary where each key is the name of
                a block and each value is a dictionary containing the complete
                hyperparameter specification of that block.

        Returns:
            dict:
                A dictionary containing the block names as keys and
                the current block hyperparameters dictionary as values.
        """
        hyperparameters = dict()
        for block_name, block in self.blocks.items():
            hyperparameters[block_name] = block.get_hyperparameters()

        if flat:
            hyperparameters = self._flatten_dict(hyperparameters)

        return hyperparameters

    def set_hyperparameters(self, hyperparameters):
        """Set new hyperparameter values for some blocks.

        Args:
            hyperparameters (dict):
                A dictionary containing the block names as keys and the new hyperparameters
                dictionary as values.
        """
        hyperparameters = self._sanitize(hyperparameters)
        for block_name, block_hyperparams in hyperparameters.items():
            self.blocks[block_name].set_hyperparameters(block_hyperparams)

    def _get_block_args(self, block_name, block_args, context):
        """Get the arguments expected by the block method from the context.

        The arguments will be taken from the context using both the method
        arguments specification and the ``input_names`` given when the pipeline
        was created.

        Args:
            block_name (str):
                Name of this block. Used to find the corresponding input_names.
            block_args (list):
                list of method argument specifications from the primitive.
            context (dict):
                current context dictionary.

        Returns:
            dict:
                A dictionary containing the argument names and values to pass
                to the method.
        """
        # TODO: type validation and/or transformation should be done here

        input_names = self.input_names.get(block_name, dict())

        kwargs = dict()
        for arg in block_args:
            name = arg['name']
            variable = input_names.get(name, name)

            if variable in context:
                kwargs[name] = context[variable]

        return kwargs

    def _extract_outputs(self, block_name, outputs, block_outputs):
        """Extract the outputs of the method as a dict to be set into the context."""
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
        """Get the name of the block in the ``index`` position."""
        return list(self.blocks.keys())[index]

    def _get_output_spec(self, output):
        """Parse the output specification and get a block name and a variable name.

        The output specification can be of two types: int and str.

        If it is an integer, it is interpreted as a block index, and the variable name
        is considered to be ``None``, which means that the whole context will be returned.

        If it is a string, it can be interpreted in three ways:

            * **block name**: If the string matches a block name exactly, including
            its hash and counter number ``#n`` at the end, the whole context will be
            returned after that block is produced.
            * **variable_name**: If the string does not match any block name and does
            not contain any dot characted, ``'.'``, it will be considered a variable
            name. In this case, the indicated variable will be extracted from the
            context and returned after the last block has been produced.
            * **block_name + variable_name**: If the complete string does not match a
            block name but it contains at least one dot, ``'.'``, it will be split
            in two parts on the last dot. If the first part of the string matches a
            block name exactly, the second part of the string will be considered a
            variable name, assuming the format ``{block_name}.{variable_name}``, and
            the indicated variable will be extracted from the context and returned
            after the block has been produced. Otherwise, if the extracted
            ``block_name`` does not match a block name exactly, a ``ValueError``
            will be raised.

        Args:
            output (str or int):
                Output specification as either a string or an integer.

        Raises:
            ValueError:
                If the output string contains dots but it does not match a block
                name exactly

        Returns:
            tuple:
                The output is a tuple containing:
                    * block_name (str): name of the block from which the output will be
                        returned, including its counter number.
                    * variable_name (str): Name of the variable to extract from the context.
                        It can be ``None``, which means that the whole context is to be
                        returned.
        """
        # If None is given, both block and varialbe are None
        if output is None:
            return None, None

        # If an int is given, it is a block index and there is no variable
        if isinstance(output, int):
            output = self._get_block_name(output)
            return output, None

        # If the string matches a block name, there is no variable
        if output in self.blocks:
            return output, None

        # If there is at least one dot in the output, but it did not match
        # a block name, it is considered to be {block_name}.{variable_name}
        if '.' in output:
            output_block, output_variable = output.rsplit('.', 1)
            if output_block not in self.blocks:
                raise ValueError('Unknown block name: {}'.format(output_block))

            return output_block, output_variable

        # If the given string is not a block name and it has no dots,
        # it is considered to be a variable name to be extracted
        # from the context after the last block has been produced
        last_block_name = self._get_block_name(-1)
        return last_block_name, output

    def _get_output(self, output_variable, context):
        """Get the specified output variable from the context.

        If the variable name is ``None``, return the entire context.
        """
        if output_variable:
            if output_variable not in context:
                raise ValueError('Output variable {} not found in context'
                                 .format(output_variable))

            return context[output_variable]
        else:
            return context

    def fit(self, X=None, y=None, output_=None, start_=None, **kwargs):
        """Fit the blocks of this pipeline.

        Sequentially call the ``fit`` and the ``produce`` methods of each block,
        capturing the outputs each ``produce`` method before calling the ``fit``
        method of the next one.

        During the whole process a context dictionary is built, where both the
        passed arguments and the captured outputs of the ``produce`` methods
        are stored, and from which the arguments for the next ``fit`` and
        ``produce`` calls will be taken.

        Args:
            X:
                Fit Data, which the pipeline will learn from.

            y:
                Fit Data labels, which the pipeline will use to learn how to
                behave.

            output_ (str or int or None):
                Output specification, which can be a string or an integer or None.

                    * If it is None (default), nothing will be returned
                    * If an integer is given, it is interpreted as the block number, and the whole
                      context after running the specified block will be returned.
                    * If it is a string, it can be interpreted in three ways:

                        * **block name**: If the string matches a block name exactly, including
                          its hash and counter number ``#n`` at the end, the whole context will be
                          returned after that block is produced.
                        * **variable_name**: If the string does not match any block name and does
                          not contain any dot characted, ``'.'``, it will be considered a variable
                          name. In this case, the indicated variable will be extracted from the
                          context and returned after the last block has been produced.
                        * **block_name + variable_name**: If the complete string does not match a
                          block name but it contains at least one dot, ``'.'``, it will be split
                          in two parts on the last dot. If the first part of the string matches a
                          block name exactly, the second part of the string will be considered a
                          variable name, assuming the format ``{block_name}.{variable_name}``, and
                          the indicated variable will be extracted from the context and returned
                          after the block has been produced. Otherwise, if the extracted
                          ``block_name`` does not match a block name exactly, a ``ValueError``
                          will be raised.

            start_ (str or int or None):
                Block index or block name to start processing from. The
                value can either be an integer, which will be interpreted as a block index,
                or the name of a block, including the conter number at the end.
                If given, the execution of the pipeline will start on the specified block,
                and all the blocks before that one will be skipped.

            **kwargs:
                Any additional keyword arguments will be directly added
                to the context dictionary and available for the blocks.

        Returns:
            None or dict or object:
                * If no output is specified, nothing will be returned.
                * If an output block has been specified without and output variable, the
                  context dictionary will be returned after the produce method of that block
                  has been called.
                * If both an output block and an output variable have been specified,
                  the value of that variable from the context will extracted and returned
                  after the produce method of that block has been called.
        """
        context = kwargs.copy()
        if X is not None:
            context['X'] = X

        if y is not None:
            context['y'] = y

        output_block, output_variable = self._get_output_spec(output_)
        last_block_name = self._get_block_name(-1)

        if isinstance(start_, int):
            start_ = self._get_block_name(start_)

        for block_name, block in self.blocks.items():
            if start_:
                if block_name == start_:
                    start_ = False
                else:
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

        if start_:
            # We skipped all the blocks up to the end
            raise ValueError('Unknown block name: {}'.format(start_))

    def predict(self, X=None, output_=None, start_=None, **kwargs):
        """Produce predictions using the blocks of this pipeline.

        Sequentially call the ``produce`` method of each block, capturing the
        outputs before calling the next one.

        During the whole process a context dictionary is built, where both the
        passed arguments and the captured outputs of the ``produce`` methods
        are stored, and from which the arguments for the next ``produce`` calls
        will be taken.

        Args:
            X:
                Data which the pipeline will use to make predictions.

            output_ (str or int or None):
                Output specification, which can be a string or an integer or None.
                    * If it is None (default), the output of the last block will be returned.
                    * If an integer is given, it is interpreted as the block number, and the whole
                      context after running the specified block will be returned.
                    * If it is a string, it can be interpreted in three ways:

                        * **block name**: If the string matches a block name exactly, including
                          its hash and counter number ``#n`` at the end, the whole context will be
                          returned after that block is produced.
                        * **variable_name**: If the string does not match any block name and does
                          not contain any dot characted, ``'.'``, it will be considered a variable
                          name. In this case, the indicated variable will be extracted from the
                          context and returned after the last block has been produced.
                        * **block_name + variable_name**: If the complete string does not match a
                          block name but it contains at least one dot, ``'.'``, it will be split
                          in two parts on the last dot. If the first part of the string matches a
                          block name exactly, the second part of the string will be considered a
                          variable name, assuming the format ``{block_name}.{variable_name}``, and
                          the indicated variable will be extracted from the context and returned
                          after the block has been produced. Otherwise, if the extracted
                          ``block_name`` does not match a block name exactly, a ``ValueError``
                          will be raised.

            start_ (str or int or None):
                Block index or block name to start processing from. The
                value can either be an integer, which will be interpreted as a block index,
                or the name of a block, including the conter number at the end.
                If given, the execution of the pipeline will start on the specified block,
                and all the blocks before that one will be skipped.

            **kwargs:
                Any additional keyword arguments will be directly added
                to the context dictionary and available for the blocks.

        Returns:
            None or dict or object:
                * If no output is specified, the output of the last block will be returned.
                * If an output block has been specified without and output variable, the
                  context dictionary will be returned after the produce method of that block
                  has been called.
                * If both an output block and an output variable have been specified,
                  the value of that variable from the context will extracted and returned
                  after the produce method of that block has been called.
        """
        context = kwargs.copy()
        if X is not None:
            context['X'] = X

        output_block, output_variable = self._get_output_spec(output_)

        if isinstance(start_, int):
            start_ = self._get_block_name(start_)

        for block_name, block in self.blocks.items():
            if start_:
                if block_name == start_:
                    start_ = False
                else:
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

        if start_:
            # We skipped all the blocks up to the end
            raise ValueError('Unknown block name: {}'.format(start_))

        if output_ is None:
            return outputs

    def to_dict(self):
        """Return all the details of this MLPipeline in a dict.

        The dict structure contains all the ``__init__`` arguments of the
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

        The content of the JSON file is the dict returned by the ``to_dict`` method.

        Args:
            path (str):
                Path to the JSON file to write.
        """
        with open(path, 'w') as out_file:
            json.dump(self.to_dict(), out_file, indent=4)

    @classmethod
    def from_dict(cls, metadata):
        """Create a new MLPipeline from a dict specification.

        The dict structure is the same as the one created by the ``to_dict`` method.

        Args:
            metadata (dict):
                Dictionary containing the pipeline specification.

        Returns:
            MLPipeline:
                A new MLPipeline instance with the details found in the
                given specification dictionary.
        """
        return cls(metadata)

    @classmethod
    def load(cls, path):
        """Create a new MLPipeline from a JSON specification.

        The JSON file format is the same as the one created by the ``to_dict`` method.

        Args:
            path (str):
                Path of the JSON file to load.

        Returns:
            MLPipeline:
                A new MLPipeline instance with the specification found
                in the JSON file.
        """
        with open(path, 'r') as in_file:
            metadata = json.load(in_file)

        return cls.from_dict(metadata)
