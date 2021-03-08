# -*- coding: utf-8 -*-

"""Package where the MLBlock class is defined."""

import importlib
import logging
from copy import deepcopy

from mlblocks.discovery import load_primitive

LOGGER = logging.getLogger(__name__)


def import_object(object_name):
    """Import an object from its Fully Qualified Name."""

    if isinstance(object_name, str):
        parent_name, attribute = object_name.rsplit('.', 1)
        try:
            parent = importlib.import_module(parent_name)
        except ImportError:
            grand_parent_name, parent_name = parent_name.rsplit('.', 1)
            grand_parent = importlib.import_module(grand_parent_name)
            parent = getattr(grand_parent, parent_name)

        return getattr(parent, attribute)

    return object_name


class MLBlock():
    """MLBlock Class.

    The MLBlock class represents a single step within an MLPipeline.

    It is responsible for loading and interpreting JSON primitives, as well
    as wrapping them and providing a common interface to run them.

    Attributes:
        name (str):
            Primitive name.
        metadata (dict):
            Additional information about this primitive
        primitive (object):
            the actual function or instance which this MLBlock wraps.
        fit_args (dict):
            specification of the arguments expected by the ``fit`` method.
        fit_method (str):
            name of the primitive method to call on ``fit``. ``None`` if the
            primitive is a function.
        produce_args (dict):
            specification of the arguments expected by the ``predict`` method.
        produce_output (dict):
            specification of the outputs of the ``produce`` method.
        produce_method (str):
            name of the primitive method to call on ``produce``. ``None`` if the primitive is a
            function.

    Args:
        primitive (str or dict):
            primitive name or primitive dictionary.
        **kwargs:
            Any additional arguments that will be used as hyperparameters or passed to the
            ``fit`` or ``produce`` methods.

    Raises:
        TypeError:
            A ``TypeError`` is raised if a required argument is not found within the ``kwargs``
            or if an unexpected argument has been given.
    """  # pylint: disable=too-many-instance-attributes

    def _extract_params(self, kwargs, hyperparameters):
        """Extract init, fit and produce params from kwargs.

        The ``init_params``, ``fit_params`` and ``produce_params`` are extracted
        from the passed ``kwargs`` taking the metadata hyperparameters as a
        reference.

        During this extraction, make sure that all the required hyperparameters
        have been given and that nothing unexpected exists in the input.

        Args:
            kwargs (dict):
                dict containing the Keyword arguments that have been passed to the ``__init__``
                method upon initialization.
            hyperparameters (dict):
                hyperparameters dictionary, as found in the JSON annotation.

        Raises:
            TypeError:
                A ``TypeError`` is raised if a required argument is not found in the
                ``kwargs`` dict, or if an unexpected argument has been given.
        """
        init_params = dict()
        fit_params = dict()
        produce_params = dict()

        for name, param in hyperparameters.get('fixed', dict()).items():
            if name in kwargs:
                value = kwargs.pop(name)

            elif 'default' in param:
                value = param['default']

            else:
                raise TypeError("{} required argument '{}' not found".format(self.name, name))

            init_params[name] = value

        for name, param in hyperparameters.get('tunable', dict()).items():
            if name in kwargs:
                init_params[name] = kwargs.pop(name)

        if not isinstance(self.fit_args, str):
            fit_args = [arg['name'] for arg in self.fit_args]
        else:
            fit_args = []

        if not isinstance(self.produce_args, str):
            produce_args = [arg['name'] for arg in self.produce_args]
        else:
            produce_args = []

        for name in list(kwargs.keys()):
            if name in fit_args:
                fit_params[name] = kwargs.pop(name)

            elif name in produce_args:
                produce_params[name] = kwargs.pop(name)

        if kwargs:
            error = "Unexpected hyperparameters '{}'".format(', '.join(kwargs.keys()))
            raise TypeError(error)

        return init_params, fit_params, produce_params

    @staticmethod
    def _filter_conditional(conditional, init_params):
        condition = conditional['condition']
        default = conditional.get('default')

        if condition not in init_params:
            return default

        condition_value = init_params[condition]
        values = conditional['values']
        return values.get(condition_value, default)

    @classmethod
    def _get_tunable(cls, hyperparameters, init_params):
        tunable = dict()
        for name, param in hyperparameters.get('tunable', dict()).items():
            if name not in init_params:
                if param['type'] == 'conditional':
                    param = cls._filter_conditional(param, init_params)
                    if param is not None:
                        tunable[name] = param

                else:
                    tunable[name] = param

        return tunable

    def __init__(self, primitive, **kwargs):
        if isinstance(primitive, str):
            primitive = load_primitive(primitive)

        self.metadata = primitive
        self.name = primitive['name']

        self.primitive = import_object(self.metadata['primitive'])

        self._fit = self.metadata.get('fit', dict())
        self.fit_args = self._fit.get('args', [])
        self.fit_method = self._fit.get('method')

        self._produce = self.metadata['produce']
        self.produce_args = self._produce['args']
        self.produce_output = self._produce['output']
        self.produce_method = self._produce.get('method')

        self._class = bool(self.produce_method)

        hyperparameters = self.metadata.get('hyperparameters', dict())
        init_params, fit_params, produce_params = self._extract_params(kwargs, hyperparameters)

        self._hyperparameters = init_params
        self._fit_params = fit_params
        self._produce_params = produce_params

        self._tunable = self._get_tunable(hyperparameters, init_params)

        default = {
            name: param['default']
            for name, param in self._tunable.items()
            # TODO: support undefined defaults
        }

        self.set_hyperparameters(default)

    def __str__(self):
        """Return a string that represents this block."""
        return 'MLBlock - {}'.format(self.name)

    def get_tunable_hyperparameters(self):
        """Get the hyperparameters that can be tuned for this MLBlock.

        The list of hyperparameters is taken from the JSON annotation,
        filtering out any hyperparameter for which a value has been given
        during the initalization.

        Returns:
            dict:
                the dictionary containing the hyperparameters that can be
                tuned, their types and, if applicable, the accepted
                ranges or values.
        """
        return deepcopy(self._tunable)

    def get_hyperparameters(self):
        """Get hyperparameters values that the current MLBlock is using.

        Returns:
            dict:
                the dictionary containing the hyperparameter values that the
                MLBlock is currently using.
        """
        return deepcopy(self._hyperparameters)

    def set_hyperparameters(self, hyperparameters):
        """Set new hyperparameters.

        Only the specified hyperparameters are modified, so any other
        hyperparameter keeps the value that had been previously given.

        If necessary, a new instance of the primitive is created.

        Args:
            hyperparameters (dict):
                Dictionary containing as keys the name of the hyperparameters and as
                values the values to be used.
        """
        self._hyperparameters.update(hyperparameters)

        if self._class:
            LOGGER.debug('Creating a new primitive instance for %s', self.name)
            self.instance = self.primitive(**self.get_hyperparameters())

    def _get_method_kwargs(self, kwargs, method_args):
        """Prepare the kwargs for the method.

        The kwargs dict will be altered according to the method_kwargs
        specification to make them ready for the primitive method to
        accept them.

        Args:
            kwargs (dict):
                keyword arguments that have been passed to the block method.
            method_args (list):
                method arguments as specified in the JSON annotation.

        Returns:
            dict:
                A dictionary containing the argument names and values to pass
                to the primitive method.
        """
        if isinstance(method_args, str):
            method_args = getattr(self.instance, method_args)()

        method_kwargs = dict()
        for arg in method_args:
            name = arg['name']
            keyword = arg.get('keyword', name)

            if name in kwargs:
                value = kwargs[name]
            elif 'default' in arg:
                value = arg['default']
            elif arg.get('required', True):
                raise TypeError("missing expected argument '{}'".format(name))

            method_kwargs[keyword] = value

        return method_kwargs

    def fit(self, **kwargs):
        """Call the fit method of the primitive.

        The given keyword arguments will be passed directly to the ``fit``
        method of the primitive instance specified in the JSON annotation.

        If any of the arguments expected by the produce method had been
        given during the MLBlock initialization, they will be passed as well.

        If the fit method was not specified in the JSON annotation, or if
        the primitive is a simple function, this will be a noop.

        Args:
            **kwargs:
                Any given keyword argument will be directly passed to the primitive fit method.

        Raises:
            TypeError:
                A ``TypeError`` might be raised if any argument not expected by the primitive fit
                method is given.
        """
        if self.fit_method is not None:
            fit_kwargs = self._fit_params.copy()
            fit_kwargs.update(kwargs)
            fit_kwargs = self._get_method_kwargs(fit_kwargs, self.fit_args)
            getattr(self.instance, self.fit_method)(**fit_kwargs)

    def produce(self, **kwargs):
        """Call the primitive function, or the predict method of the primitive.

        The given keyword arguments will be passed directly to the primitive,
        if it is a simple function, or to the ``produce`` method of the
        primitive instance specified in the JSON annotation, if it is a class.

        If any of the arguments expected by the fit method had been given
        during the MLBlock initialization, they will be passed as well.

        Returns:
            The output of the call to the primitive function or primitive
            produce method.
        """
        produce_kwargs = self._produce_params.copy()
        produce_kwargs.update(kwargs)
        produce_kwargs = self._get_method_kwargs(produce_kwargs, self.produce_args)
        if self._class:
            return getattr(self.instance, self.produce_method)(**produce_kwargs)

        produce_kwargs.update(self.get_hyperparameters())
        return self.primitive(**produce_kwargs)
