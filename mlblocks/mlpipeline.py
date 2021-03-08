# -*- coding: utf-8 -*-

"""Package where the MLPipeline class is defined."""

import json
import logging
import os
import re
import warnings
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from datetime import datetime

import numpy as np
import psutil
from graphviz import Digraph

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
        outputs (dict):
            dictionary containing lists of output variables associated to a name.
        verbose (bool):
            whether to log the exceptions that occur when running the pipeline before
            raising them or not.
    """

    def _get_tunable_hyperparameters(self):
        """Get the tunable hyperperparameters from all the blocks in this pipeline."""
        tunable = {}
        for block_name, block in self.blocks.items():
            tunable[block_name] = block.get_tunable_hyperparameters()

        return tunable

    def _build_blocks(self):
        blocks = OrderedDict()
        last_fit_block = None

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
                        LOGGER.warning(('Non-numbered init_params are being used '
                                        'for more than one block %s.'), primitive_name)

                block = MLBlock(primitive, **block_params)
                blocks[block_name] = block

                if bool(block._fit):
                    last_fit_block = block_name

            except Exception:
                LOGGER.exception('Exception caught building MLBlock %s', primitive)
                raise

        return blocks, last_fit_block

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

    def _get_block_outputs(self, block_name):
        """Get the list of output variables for the given block."""
        outputs = self._get_block_variables(
            block_name,
            'produce_output',
            self.output_names.get(block_name, dict())
        )
        for context_name, output in outputs.items():
            output['variable'] = '{}.{}'.format(block_name, context_name)

        return list(outputs.values())

    def _get_block_variables(self, block_name, variables_attr, names):
        """Get dictionary of variable names to the variable for a given block

        Args:
            block_name (str):
                Name of the block for which to get the specification
            variables_attr (str):
                Name of the attribute that has the variables list. It can be
                `fit_args`, `produce_args` or `produce_output`.
            names (dict):
                Dictionary used to translate the variable names.
        """
        block = self.blocks[block_name]
        variables = deepcopy(getattr(block, variables_attr))
        if isinstance(variables, str):
            variables = getattr(block.instance, variables)()

        variable_dict = {}
        for variable in variables:
            name = variable['name']
            context_name = names.get(name, name)
            variable_dict[context_name] = variable

        return variable_dict

    def _get_outputs(self, pipeline, outputs):
        """Get the output definitions from the pipeline dictionary.

        If the ``"default"`` entry does not exist, it is built using the
        outputs from the last block in the pipeline.
        """
        outputs = outputs or pipeline.get('outputs')
        if outputs is None:
            outputs = dict()

        if 'default' not in outputs:
            outputs['default'] = self._get_block_outputs(self._last_block_name)

        return outputs

    def _get_block_name(self, index):
        """Get the name of the block in the ``index`` position."""
        return list(self.blocks.keys())[index]

    def __init__(self, pipeline=None, primitives=None, init_params=None,
                 input_names=None, output_names=None, outputs=None, verbose=True):

        pipeline = self._get_pipeline_dict(pipeline, primitives)

        self.primitives = primitives or pipeline['primitives']
        self.init_params = init_params or pipeline.get('init_params', dict())
        self.blocks, self._last_fit_block = self._build_blocks()
        self._last_block_name = self._get_block_name(-1)

        self.input_names = input_names or pipeline.get('input_names', dict())
        self.output_names = output_names or pipeline.get('output_names', dict())

        self.outputs = self._get_outputs(pipeline, outputs)
        self.verbose = verbose

        tunable = pipeline.get('tunable_hyperparameters')
        if tunable is not None:
            self._tunable_hyperparameters = tunable
        else:
            self._tunable_hyperparameters = self._get_tunable_hyperparameters()

        hyperparameters = pipeline.get('hyperparameters')
        if hyperparameters:
            self.set_hyperparameters(hyperparameters)

        self._re_block_name = re.compile(r'(^[^#]+#\d+)(\..*)?')

    def _get_str_output(self, output):
        """Get the outputs that correspond to the str specification."""
        if output in self.outputs:
            return self.outputs[output]
        elif output in self.blocks:
            return [{'name': output, 'variable': output}]
            # return self._get_block_outputs(output)
        elif '.' in output:
            block_name, variable_name = output.rsplit('.', 1)
            block = self.blocks.get(block_name)
            if not block:
                raise ValueError('Invalid block name: {}'.format(block_name))

            for variable in block.produce_output:
                if variable['name'] == variable_name:
                    output_variable = deepcopy(variable)
                    output_variable['variable'] = output
                    return [output_variable]

            raise ValueError('Block {} has no output {}'.format(block_name, variable_name))

        raise ValueError('Invalid Output Specification: {}'.format(output))

    def get_inputs(self, fit=True):
        """Get a relation of all the input variables required by this pipeline.

        The result is a list contains all of the input variables.
        Optionally include the fit arguments.

        Args:
            fit (bool):
                Optional argument to include fit arguments or not. Defaults to ``True``.

        Returns:
            list:
                Dictionary specifying all the input variables.
                Each dictionary contains the entry ``name``, as
                well as any other metadata that may have been included in the
                pipeline inputs specification.
        """
        inputs = dict()
        for block_name in reversed(self.blocks.keys()):  # iterates through pipeline backwards
            produce_outputs = self._get_block_variables(
                block_name,
                'produce_output',
                self.output_names.get(block_name, dict())
            )

            for produce_output_name in produce_outputs.keys():
                inputs.pop(produce_output_name, None)

            produce_inputs = self._get_block_variables(
                block_name,
                'produce_args',
                self.input_names.get(block_name, dict())
            )
            inputs.update(produce_inputs)

            if fit:
                fit_inputs = self._get_block_variables(
                    block_name,
                    'fit_args',
                    self.input_names.get(block_name, dict())
                )
                inputs.update(fit_inputs)

        return inputs

    def get_fit_args(self):
        return list(self.get_inputs(fit=True).values())

    def get_predict_args(self):
        return list(self.get_inputs(fit=False).values())

    def get_outputs(self, outputs='default'):
        """Get the list of output variables that correspond to the specified outputs.

        Outputs specification can either be a single string, a single integer, or a
        list of strings and integers.

        If strings are given, they can either be one of the named outputs that have
        been specified on the pipeline definition or the name of a block, including the
        counter number at the end, or a full variable specification following the format
        ``{block-name}.{variable-name}``.

        Alternatively, integers can be passed as indexes of the blocks from which to get
        the outputs.

        If output specifications that resolve to multiple output variables are given,
        such as the named outputs or block names, all the variables are concatenated
        together, in order, in a single variable list.

        Args:
            outputs (str, int or list[str or int]):
                Single or list of output specifications.

        Returns:
            list:
                List of dictionaries specifying all the output variables. Each
                dictionary contains the entries ``name`` and ``variable``, as
                well as any other metadata that may have been included in the
                pipeline outputs or block produce outputs specification.

        Raises:
            ValueError:
                If an output specification is not valid.
            TypeError:
                If the type of a specification is not an str or an int.
        """
        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs, )

        computed = list()
        for output in outputs:
            if isinstance(output, int):
                output = self._get_block_name(output)

            if isinstance(output, str):
                computed.extend(self._get_str_output(output))
            else:
                raise TypeError('Output Specification can only be str or int')

        return computed

    def get_output_names(self, outputs='default'):
        """Get the names of the outputs that correspond to the given specification.

        The indicated outputs will be resolved and the names of the output variables
        will be returned as a single list.

        Args:
            outputs (str, int or list[str or int]):
                Single or list of output specifications.

        Returns:
            list:
                List of variable names

        Raises:
            ValueError:
                If an output specification is not valid.
            TypeError:
                If the type of a specification is not an str or an int.
        """
        outputs = self.get_outputs(outputs)
        return [output['name'] for output in outputs]

    def get_output_variables(self, outputs='default'):
        """Get the list of variable specifications of the given outputs.

        The indicated outputs will be resolved and their variables specifications
        will be returned as a single list.

        Args:
            outputs (str, int or list[str or int]):
                Single or list of output specifications.

        Returns:
            list:
                List of variable specifications.

        Raises:
            ValueError:
                If an output specification is not valid.
            TypeError:
                If the type of a specification is not an str or an int.
        """
        outputs = self.get_outputs(outputs)
        return [output['variable'] for output in outputs]

    def _extract_block_name(self, variable_name):
        return self._re_block_name.search(variable_name).group(1)

    def _prepare_outputs(self, outputs):
        output_variables = self.get_output_variables(outputs)
        outputs = output_variables.copy()
        output_blocks = {
            self._extract_block_name(variable)
            for variable in output_variables
        }
        return output_variables, outputs, output_blocks

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
                'block_name': {
                    'hyperparameter_name': 'hyperparameter_value',
                    ...
                },
                ...
            }

        The other one is an alternative format where each key is a two element tuple containing
        the name of the block as the first element and the name of the hyperparameter as the
        second one::

            {
                ('block_name', 'hyperparameter_name'): 'hyperparameter_value',
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

        if isinstance(block_args, str):
            block = self.blocks[block_name]
            block_args = getattr(block.instance, block_args)()

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
        if isinstance(block_outputs, str):
            block = self.blocks[block_name]
            block_outputs = getattr(block.instance, block_outputs)()

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

    def _update_outputs(self, variable_name, output_variables, outputs, value):
        """Set the requested block outputs into the outputs list in the right place."""
        if variable_name in output_variables:
            index = output_variables.index(variable_name)
            outputs[index] = deepcopy(value)

    def _fit_block(self, block, block_name, context, debug_info=None):
        """Get the block args from the context and fit the block."""
        LOGGER.debug('Fitting block %s', block_name)
        try:
            fit_args = self._get_block_args(block_name, block.fit_args, context)
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            start = datetime.utcnow()
            block.fit(**fit_args)
            elapsed = datetime.utcnow() - start
            memory_after = process.memory_info().rss

            if debug_info is not None:
                debug = debug_info['debug']
                record = {}
                if 't' in debug:
                    record['time'] = elapsed.total_seconds()
                if 'm' in debug:
                    record['memory'] = memory_after - memory_before
                if 'i' in debug:
                    record['input'] = deepcopy(fit_args)

                debug_info['fit'][block_name] = record

        except Exception:
            if self.verbose:
                LOGGER.exception('Exception caught fitting MLBlock %s', block_name)

            raise

    def _produce_block(self, block, block_name, context, output_variables,
                       outputs, debug_info=None):
        """Get the block args from the context and produce the block.

        Afterwards, set the block outputs back into the context and update
        the outputs list if necessary.
        """
        LOGGER.debug('Producing block %s', block_name)
        try:
            produce_args = self._get_block_args(block_name, block.produce_args, context)
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            start = datetime.utcnow()
            block_outputs = block.produce(**produce_args)
            elapsed = datetime.utcnow() - start
            memory_after = process.memory_info().rss

            outputs_dict = self._extract_outputs(block_name, block_outputs, block.produce_output)
            context.update(outputs_dict)

            if output_variables:
                if block_name in output_variables:
                    self._update_outputs(block_name, output_variables, outputs, context)
                else:
                    for key, value in outputs_dict.items():
                        variable_name = '{}.{}'.format(block_name, key)
                        self._update_outputs(variable_name, output_variables, outputs, value)

            if debug_info is not None:
                debug = debug_info['debug']
                record = {}
                if 't' in debug:
                    record['time'] = elapsed.total_seconds()
                if 'm' in debug:
                    record['memory'] = memory_after - memory_before
                if 'i' in debug:
                    record['input'] = deepcopy(produce_args)
                if 'o' in debug:
                    record['output'] = deepcopy(outputs_dict)

                debug_info['produce'][block_name] = record

        except Exception:
            if self.verbose:
                LOGGER.exception('Exception caught producing MLBlock %s', block_name)

            raise

    def fit(self, X=None, y=None, output_=None, start_=None, debug=False, **kwargs):
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
            output_ (str or int or list or None):
                Output specification, as required by ``get_outputs``. If ``None`` is given,
                nothing will be returned.
            start_ (str or int or None):
                Block index or block name to start processing from. The
                value can either be an integer, which will be interpreted as a block index,
                or the name of a block, including the conter number at the end.
                If given, the execution of the pipeline will start on the specified block,
                and all the blocks before that one will be skipped.
            debug (bool or str):
                Debug a pipeline with the following options:

                    * ``t``:
                        Elapsed time for the primitive and the given stage (fit or predict).
                    * ``m``:
                        Amount of memory incrase (or decrease) for the primitive. This amount
                        is represented in bytes.
                    * ``i``:
                        The input values that the primitive takes for that step.
                    * ``o``:
                        The output values that the primitive generates.

                If provided, return a dictionary with the ``fit`` and ``predict`` performance.
                This argument can be a string containing a combination of the letters listed above,
                or ``True`` which will return a complete debug.

            **kwargs:
                Any additional keyword arguments will be directly added
                to the context dictionary and available for the blocks.

        Returns:
            None or dict or object:
                * If no ``output`` is specified, nothing will be returned.
                * If ``output_`` has been specified, either a single value or a
                  tuple of values will be returned.
        """
        context = kwargs.copy()
        if X is not None:
            context['X'] = X

        if y is not None:
            context['y'] = y

        if output_ is None:
            output_variables = None
            outputs = None
            output_blocks = set()
        else:
            output_variables, outputs, output_blocks = self._prepare_outputs(output_)

        if isinstance(start_, int):
            start_ = self._get_block_name(start_)

        debug_info = None
        if debug:
            debug_info = defaultdict(dict)
            debug_info['debug'] = debug.lower() if isinstance(debug, str) else 'tmio'

        fit_pending = True
        for block_name, block in self.blocks.items():
            if block_name == self._last_fit_block:
                fit_pending = False

            if start_:
                if block_name == start_:
                    start_ = False
                else:
                    LOGGER.debug('Skipping block %s fit', block_name)
                    continue

            self._fit_block(block, block_name, context, debug_info)

            if fit_pending or output_blocks:
                self._produce_block(
                    block, block_name, context, output_variables, outputs, debug_info)

                # We already captured the output from this block
                if block_name in output_blocks:
                    output_blocks.remove(block_name)

            # If there was an output_ but there are no pending
            # outputs we are done.
            if output_variables:
                if not output_blocks:
                    if len(outputs) > 1:
                        result = tuple(outputs)
                    else:
                        result = outputs[0]

                    if debug:
                        return result, debug_info

                    return result

            elif not fit_pending:
                if debug:
                    return debug_info

                return

        if start_:
            # We skipped all the blocks up to the end
            raise ValueError('Unknown block name: {}'.format(start_))

        if debug:
            return debug_info

    def predict(self, X=None, output_='default', start_=None, debug=False, **kwargs):
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
            output_ (str or int or list or None):
                Output specification, as required by ``get_outputs``. If not specified
                the ``default`` output will be returned.
            start_ (str or int or None):
                Block index or block name to start processing from. The
                value can either be an integer, which will be interpreted as a block index,
                or the name of a block, including the conter number at the end.
                If given, the execution of the pipeline will start on the specified block,
                and all the blocks before that one will be skipped.
            debug (bool or str):
                Debug a pipeline with the following options:

                    * ``t``:
                        Elapsed time for the primitive and the given stage (fit or predict).
                    * ``m``:
                        Amount of memory incrase (or decrease) for the primitive. This amount
                        is represented in bytes.
                    * ``i``:
                        The input values that the primitive takes for that step.
                    * ``o``:
                        The output values that the primitive generates.

                If ``True`` then a dictionary will be returned containing all the elements listed
                previously. If a ``string`` value with the combination of letters is given for
                each option, it will return a dictionary with the selected elements.

            **kwargs:
                Any additional keyword arguments will be directly added
                to the context dictionary and available for the blocks.

        Returns:
            object or tuple:
                * If a single output is requested, it is returned alone.
                * If multiple outputs have been requested, a tuple is returned.
                * If ``debug`` is given, a tupple will be returned where the first element
                  returned are the predictions and the second a dictionary containing the debug
                  information.
        """
        context = kwargs.copy()
        if X is not None:
            context['X'] = X

        output_variables, outputs, output_blocks = self._prepare_outputs(output_)

        if isinstance(start_, int):
            start_ = self._get_block_name(start_)

        debug_info = None
        if debug:
            debug_info = defaultdict(dict)
            debug_info['debug'] = debug.lower() if isinstance(debug, str) else 'tmio'

        for block_name, block in self.blocks.items():
            if start_:
                if block_name == start_:
                    start_ = False
                else:
                    LOGGER.debug('Skipping block %s produce', block_name)
                    continue

            self._produce_block(block, block_name, context, output_variables, outputs, debug_info)

            # We already captured the output from this block
            if block_name in output_blocks:
                output_blocks.remove(block_name)

            # If there was an output_ but there are no pending
            # outputs we are done.
            if not output_blocks:
                if len(outputs) > 1:
                    result = tuple(outputs)
                else:
                    result = outputs[0]

                if debug:
                    return result, debug_info

                return result

        if start_:
            # We skipped all the blocks up to the end
            raise ValueError('Unknown block name: {}'.format(start_))

    def to_dict(self):
        """Return all the details of this MLPipeline in a dict.

        The dict structure contains all the ``__init__`` arguments of the
        MLPipeline, as well as the current hyperparameter values and the
        specification of the tunable_hyperparameters::

            {
                'primitives': [
                    'a_primitive',
                    'another_primitive'
                ],
                'init_params': {
                    'a_primitive': {
                        'an_argument': 'a_value'
                    }
                },
                'hyperparameters': {
                    'a_primitive#1': {
                        'an_argument': 'a_value',
                        'another_argument': 'another_value',
                    },
                    'another_primitive#1': {
                        'yet_another_argument': 'yet_another_value'
                     }
                },
                'tunable_hyperparameters': {
                    'another_primitive#1': {
                        'yet_another_argument': {
                            'type': 'str',
                            'default': 'a_default_value',
                            'values': [
                                'a_default_value',
                                'yet_another_value'
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
            'tunable_hyperparameters': self._tunable_hyperparameters,
            'outputs': self.outputs,
        }

    def _get_simple_block_name(self, block_name):
        """
        Gets the most readable, simplest version of the block name,
        without the number of the block or excess modifiers.

        Args:
            block_name (str):
                Name of the block whose simple name is being extracted.

        Returns:
            str:
                block name stripped of number and other modifiers.
        """
        full_name = block_name.split('#')[0]
        simple_name = full_name.split('.')[-1]
        return simple_name

    def _get_context_name_from_variable(self, variable_name):
        """
        Gets the name of the context from the given variable.

        Args:
            variable_name (str):
                Name of the variable.

        Returns:
            str:
                Name of the context of the variable.
        """
        block_name = variable_name.split('#')[0]
        rest = variable_name[len(block_name) + 1:]
        block_index = rest.split('.')[0]
        context_name = rest[len(block_index) + 1:]
        if len(context_name) == 0:
            raise ValueError('Invalid variable name')
        return context_name

    def _get_relevant_output_variables(self, block_name, block, current_output_variables):
        """
        Gets the output variables of the given block that are in a given set of output variables

        Args:
            block_name (str):
                The name of the block from which the variables are outputted

            block (MLBlock):
                The block from which the variables are outputted

            current_output_variables (list):
                A list of possible output variables to return

        Returns:
            set:
                A set of strings containing the output variable name if and only if it is an
                output variable of the given block and its name is in the list of possible
                output variables
        """
        output_alt_names = self.output_names.get(block_name, dict())
        relevant_output = set()
        for block_output in block.produce_output:
            output_variable_name = block_output['name']
            if output_variable_name in output_alt_names.keys():
                output_variable_name = output_alt_names[output_variable_name]

            if output_variable_name in current_output_variables:
                relevant_output.add(block_output['name'])

        return relevant_output

    def _make_diagram_block(self, diagram, block_name):
        """
        Modifies the diagram to add the corresponding block of the pipeline as a visible node in
        the diagram.

        Args:
            diagram (Digraph):
                Diagram to be modified.

            block_name (str):
                Name of block to be added to the diagram
        """
        simple_name = self._get_simple_block_name(block_name)
        diagram.node(block_name, simple_name, penwidth='1')

    def _make_block_inputs(self, diagram, fit, block_name, block, cluster_edges, variable_blocks):
        """
        Modifies the diagram to add the corresponding input variables to the corresponding block
        and their edges as outputs to other blocks by modifying `variable_blocks`. Additionally
        modifies a set of edges to add any edges between an alternative input name and this block.

        Args:
            diagram (Digraph):
                Diagram to be modified.

            fit (bool):
                `True` if including fitted arguments, `False` otherwise.

            block_name (str):
                Name of block whose input variables are to be added to the diagram

            block (MLBlock):
                Block whose input variables are to be added to the diagram

            cluster_edges (set):
                Set of tuples representing edges between alternative variable names and their
                corresponding block and the type of arrowhead

            variable_blocks (dict):
                Dictionary of variable names and the set of tuples of blocks into which the
                variable connects and the type of arrowhead to use
        """
        input_alt_names = self.input_names.get(block_name, dict())
        input_variables = set(variable['name'] for variable in block.produce_args)

        if fit:
            for input_variable in block.fit_args:
                if input_variable['name'] not in input_variables:
                    input_variables.add(input_variable['name'])

        for input_name in input_variables:
            input_block = block_name
            arrowhead = 'normal'
            if input_name in input_alt_names:
                input_variable_label = block_name + ' ' + input_name + ' (input)'
                diagram.node(input_variable_label,
                             '(' + input_name + ')', fontcolor='blue')
                cluster_edges.add((input_variable_label, block_name, 'normal'))
                input_name = input_alt_names[input_name]
                input_block = input_variable_label
                arrowhead = 'none'

            if input_name in variable_blocks.keys():
                variable_blocks[input_name].add((input_block, arrowhead))
            else:
                variable_blocks[input_name] = {(input_block, arrowhead)}

    def _make_block_outputs(self, diagram, block_name, output_names, cluster_edges,
                            variable_blocks):
        """
        Modifies the diagram to add the corresponding output variables to the corresponding block
        and their edges as inputs to other blocks, as well as updating `variable_blocks`.
        Additionally modifies a set of edges to add any edges between an alternative output name
        and this block.

        Args:
            diagram (Digraph):
                Diagram to be modified.

            block_name (str):
                Name of block whose output variables are to be added to the diagram

            output_names (set):
                Set of output variable names to be added to the diagram

            cluster_edges (set):
                Set of tuples representing edges between alternative variable names and their
                corresponding block and the type of arrowhead

            variable_blocks (dict):
                Dictionary of variable names and the set of tuples of blocks into which the
                variable connects and the type of arrowhead to use
        """
        output_alt_names = self.output_names.get(block_name, dict())
        for output_name in output_names:
            output_block = block_name
            if output_name in output_alt_names.keys():
                alt_variable_label = block_name + ' ' + output_name + ' (output)'
                diagram.node(alt_variable_label,
                             '(' + output_name + ')', fontcolor='red')
                cluster_edges.add((block_name, alt_variable_label, 'none'))
                output_name = output_alt_names[output_name]
                output_block = alt_variable_label

            output_variable_label = block_name + ' ' + output_name
            diagram.node(output_variable_label, output_name)
            diagram.edge(output_block, output_variable_label, arrowhead='none')

            for block, arrow in variable_blocks[output_name]:
                diagram.edge(output_variable_label, block, arrowhead=arrow)

            del variable_blocks[output_name]

    def _make_diagram_inputs(self, diagram, input_variables_blocks):
        """
        Modifies the diagram to add the inputs of the pipeline

        Args:
            diagram (Digraph):
                Diagram to be modified.

            input_variables_blocks (dict):
                Dictionary of input variables of the pipeline and the set of tuples of blocks into
                which the variable connects and the type of arrowhead to use
        """
        with diagram.subgraph(name='cluster_inputs') as cluster:
            cluster.attr(tooltip='Input variables')
            cluster.attr('graph', rank='source', bgcolor='azure3', penwidth='0')
            cluster.attr('node', penwidth='0', fontsize='20')
            cluster.attr('edge', penwidth='0', arrowhead='none')
            cluster.node('Input', 'Input', fontsize='14', tooltip='Input variables')
            input_variables = []
            for input_name, blocks in input_variables_blocks.items():
                input_name_label = input_name + '_input'
                cluster.node(input_name_label, input_name)
                cluster.edge('Input', input_name_label)
                input_variables.append(input_name_label)

                for block, arrow in blocks:
                    diagram.edge(input_name_label, block, pendwith='1', arrowhead=arrow)

            with cluster.subgraph() as input_variables_subgraph:
                input_variables_subgraph.attr(None, rank='same')
                for index in range(1, len(input_variables)):
                    input_variables_subgraph.edge(input_variables[index - 1],
                                                  input_variables[index])
                    input_variables_subgraph.attr(None, rankdir='LR')

    def _make_diagram_outputs(self, diagram, outputs):
        """
        Modifies the diagram to add outputs of the pipeline in order from left to right.

        Args:
            diagram (Digraph):
                Diagram to be modified.

            outputs (str, int, or list[str or int]):
                Single or list of output specifications.

        Returns:
            list[str]:
                List of the human-readable names of the output variables in order
        """
        output_variables = []
        outputs_vars = self.get_outputs(outputs)

        with diagram.subgraph(name='cluster_outputs') as cluster:
            cluster.attr(tooltip='Output variables')
            cluster.attr('graph', rank='source', bgcolor='azure3', penwidth='0')
            cluster.attr('node', penwidth='0', fontsize='20')
            cluster.attr('edge', penwidth='0', arrowhead='none')
            cluster.node('Output', 'Output', fontsize='14', tooltip='Output variables')
            for output in outputs_vars:
                try:
                    variable_name = self._get_context_name_from_variable(output['variable'])
                except ValueError:
                    raise NotImplementedError(
                        'Can not deal with this type of output specification')
                cluster.node(variable_name + '_output', variable_name)
                output_variables.append(variable_name)
                cluster.edge(output_variables[-1] + '_output', 'Output')
            with cluster.subgraph() as output_variables_subgraph:
                output_variables_subgraph.attr(None, rank='same')
                for index in range(1, len(output_variables)):
                    output_variables_subgraph.edge(output_variables[index - 1] + '_output',
                                                   output_variables[index] + '_output')
                output_variables_subgraph.attr(None, rankdir='LR')

        return output_variables

    def _make_diagram_alignment(self, diagram, cluster_edges):
        """
        Modifies the diagram to add alignment edges and connect alternative names to the blocks.

        Args:
            diagram (Digraph):
                Diagram to be modified

            cluster_edges (set):
                Set of tuples that contain alternative variable names and its
                corresponding block in order
        """
        with diagram.subgraph() as alignment:
            alignment.attr('graph', penwidth='0')
            alignment.attr('node', penwidth='0')
            alignment.attr('edge', len='1', minlen='1', penwidth='1')

            for first_block, second_block, arrow in cluster_edges:
                with alignment.subgraph(name='cluster_' + first_block + second_block) as cluster:
                    cluster.edge(first_block, second_block, arrowhead=arrow)

    def get_diagram(self, fit=True, outputs='default', image_path=None):
        """
        Creates a png diagram for the pipeline, showing Pipeline Steps,
        Pipeline Inputs and Outputs, and block inputs and outputs.

        If strings are given, they can either be one of the named outputs that have
        been specified on the pipeline definition or a full variable specification
        following the format ``{block-name}.{variable-name}``.

        Args:
            fit (bool):
                Optional argument to include fit arguments or not. Defaults to `True`.

            outputs (str, int, or list[str or int]):
                Single or list of output specifications.

            image_path (str):
                Optional argument for the location at which to save the file.
                Defaults to `None`, which returns a `graphviz.Digraph` object instead of
                saving the file.

        Returns:
            None or `graphviz.Digraph` object:
                * `graphviz.Digraph` contains the information about the Pipeline Diagram
        """

        diagram = Digraph(format='png')
        diagram.attr('graph', splines='ortho')
        diagram.attr(tooltip=' ')  # hack to remove extraneous tooltips on edges
        diagram.attr('node', shape='box', penwidth='0')

        output_variables = self._make_diagram_outputs(diagram, outputs)

        cluster_edges = set()
        variable_blocks = dict((name, {(name + '_output', 'normal')}) for name in output_variables)
        for block_name, block in reversed(self.blocks.items()):
            relevant_output_names = self._get_relevant_output_variables(block_name, block,
                                                                        variable_blocks.keys())
            if len(relevant_output_names) == 0:
                continue  # skip this block

            self._make_diagram_block(diagram, block_name)
            self._make_block_outputs(diagram, block_name, relevant_output_names, cluster_edges,
                                     variable_blocks)
            self._make_block_inputs(diagram, fit, block_name, block, cluster_edges,
                                    variable_blocks)

        self._make_diagram_inputs(diagram, variable_blocks)
        self._make_diagram_alignment(diagram, cluster_edges)

        if image_path:
            diagram.render(filename='Diagram', directory=image_path, cleanup=True, format='png')
        else:
            return diagram

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
        warnings.warn(
            'MLPipeline.form_dict(pipeline_dict) is deprecated and will be removed in a '
            'later release. Please use MLPipeline(dict) instead,',
            DeprecationWarning
        )
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
        warnings.warn(
            'MLPipeline.load(path) is deprecated and will be removed in a later release. '
            'Please use MLPipeline(path) instead,',
            DeprecationWarning
        )
        with open(path, 'r') as in_file:
            metadata = json.load(in_file)

        return cls.from_dict(metadata)
