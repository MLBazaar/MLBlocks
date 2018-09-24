# -*- coding: utf-8 -*-

"""Package where the MLBlock class is defined."""

import importlib

import mlblocks


def import_object(object_name):
    """Import an object from its Fully Qualified Name."""
    package, name = object_name.rsplit('.', 1)
    return getattr(importlib.import_module(package), name)


class MLBlock():
    """MLBlock Class.

    The MLBlock class represents a single step within an MLPipeline.

    It is responsible for loading and interpreting JSON primitives, as well
    as wrapping them and providing a common interface to run them.

    Attributes:
        name (str): Name given to this MLBlock.
        primitive (object): the actual function or instance which this MLBlock
                            wraps.
        fit_args (dict): specification of the arguments expected by the `fit`
                         method.
        fit_method (str): name of the primitive method to call on `fit`.
                          `None` if the primitive is a function.
        produce_args (dict): specification of the arguments expected by the
                             `predict` method.
        produce_output (dict): specification of the outputs of the `produce`
                               method.
        produce_method (str): name of the primitive method to call on
                              `produce`. `None` if the primitive is a function.

    Args:
        name (str): Name given to this MLBlock.
        **kwargs: Any additional arguments that will be used as
                  hyperparameters or passed to the `fit` or `produce`
                  methods.

    Raises:
        TypeError: A `TypeError` is raised if a required argument is not
                   found within the `kwargs` or if an unexpected
                   argument has been given.
    """
    # pylint: disable=too-many-instance-attributes

    def _extract_params(self, kwargs, hyperparameters):
        """Extract init, fit and produce params from kwargs.

        The `init_params`, `fit_params` and `produce_params` are extracted
        from the passed `kwargs` taking the metadata hyperparameters as a
        reference.

        During this extraction, make sure that all the required hyperparameters
        have been given and that nothing unexpected exists in the input.

        Args:
            kwargs (dict): dict containing the Keyword arguments that have
                           been passed to the `__init__` method upon
                           initialization.
            hyperparameters (dict): hyperparameters dictionary, as found in
                                    the JSON annotation.

        Raises:
            TypeError: A `TypeError` is raised if a required argument is not
                       found in the `kwargs` dict, or if an unexpected
                       argument has been given.
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
                raise TypeError("Required argument '{}' not found".format(name))

            init_params[name] = value

        for name, param in hyperparameters.get('tunable', dict()).items():
            if name in kwargs:
                init_params[name] = kwargs.pop(name)

        fit_args = [arg['name'] for arg in self.fit_args]
        produce_args = [arg['name'] for arg in self.produce_args]

        for name in kwargs.keys():
            if name in fit_args:
                fit_params[name] = kwargs.pop(name)

            elif name in produce_args:
                produce_params[name] = kwargs.pop(name)

        if kwargs:
            error = "Unexpected hyperparameters '{}'".format(', '.join(kwargs.keys()))
            raise TypeError(error)

        return init_params, fit_params, produce_params

    def __init__(self, name, **kwargs):

        self.name = name

        metadata = mlblocks.load_primitive(name)

        self.primitive = import_object(metadata['primitive'])

        self._fit = metadata.get('fit', dict())
        self.fit_args = self._fit.get('args', [])
        self.fit_method = self._fit.get('method')

        self._produce = metadata['produce']
        self.produce_args = self._produce['args']
        self.produce_output = self._produce['output']
        self.produce_method = self._produce.get('method')

        self._class = bool(self.produce_method)

        hyperparameters = metadata.get('hyperparameters', dict())
        init_params, fit_params, produce_params = self._extract_params(
            kwargs, hyperparameters)
        self._hyperparamters = init_params
        self._fit_params = fit_params
        self._produce_params = produce_params

        tunable = hyperparameters.get('tunable', dict())
        self._tunable = {
            name: param
            for name, param in tunable.items()
            if name not in init_params
            # TODO: filter conditionals
        }

        default = {
            name: param['default']
            for name, param in self._tunable.items()
            # TODO: support undefined defaults
        }

        self.set_hyperparameters(default)

    def __str__(self):
        return 'MLBlock - {}'.format(self.name)

    def get_tunable_hyperparameters(self):
        """Get the hyperparameters that can be tuned for this MLBlock.

        The list of hyperparameters is taken from the JSON annotation,
        filtering out any hyperparameter for which a value has been given
        during the initalization.

        Returns:
            dict: the dictionary containing the hyperparameters that can be
                  tuned, their types and, if applicable, the accepted
                  ranges or values.
        """
        return self._tunable

    def get_hyperparameters(self):
        """Get hyperparameters values that the current MLBlock is using.

        Returns:
            dict: the dictionary containing the hyperparameter values that the
                  MLBlock is currently using.
        """
        return self._hyperparamters

    def set_hyperparameters(self, hyperparameters):
        """Set new hyperparameters.

        Only the specified hyperparameters are modified, so any other
        hyperparameter keeps the value that had been previously given.

        If necessary, a new instance of the primitive is created.

        Args:
            hyperparameters (dict): Dictionary containing as keys the name
                                    of the hyperparameters and as values
                                    the values to be used.
        """
        self._hyperparamters.update(hyperparameters)

        if self._class:
            self.instance = self.primitive(**self._hyperparamters)

    def fit(self, **kwargs):
        """Call the fit method of the primitive.

        The given keyword arguments will be passed directly to the `fit`
        method of the primitive instance specified in the JSON annotation.

        If any of the arguments expected by the produce method had been
        given during the MLBlock initialization, they will be passed as well.

        If the fit method was not specified in the JSON annotation, or if
        the primitive is a simple function, this will be a noop.

        Args:
            **kwargs: Any given keyword argument will be directly passed
                      to the primitive fit method.

        Raises:
            TypeError: A `TypeError` might be raised if any argument not
                       expected by the primitive fit method is given.
        """
        if self.fit_method is not None:
            fit_args = self._fit_params.copy()
            fit_args.update(kwargs)
            getattr(self.instance, self.fit_method)(**fit_args)

    def produce(self, **kwargs):
        """Call the primitive function, or the predict method of the primitive.

        The given keyword arguments will be passed directly to the primitive,
        if it is a simple function, or to the `produce` method of the
        primitive instance specified in the JSON annotation, if it is a class.

        If any of the arguments expected by the fit method had been given
        during the MLBlock initialization, they will be passed as well.

        Returns:
            The output of the call to the primitive function or primitive
            produce method.
        """
        produce_args = self._produce_params.copy()
        produce_args.update(kwargs)
        if self._class:
            return getattr(self.instance, self.produce_method)(**produce_args)

        produce_args.update(self._hyperparamters)
        return self.primitive(**produce_args)
