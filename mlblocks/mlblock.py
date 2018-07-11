# -*- coding: utf-8 -*-

import importlib
import inspect
import json

from mlblocks import get_primitive_path


def import_object(object_name):
    """Import an object from its Fully Qualified Name."""
    package, name = object_name.rsplit('.', 1)
    return getattr(importlib.import_module(package), name)


class MLBlock(object):

    @classmethod
    def _load_metadata(cls, name):
        """Locate and load the corresponding JSON file."""

        # json_path = cls._get_path(name)
        json_path = get_primitive_path(name)
        with open(json_path, 'r') as f:
            return json.load(f), json_path

    def __new__(cls, name, **kwargs):
        if cls is not MLBlock:
            return super().__new__(cls)

        else:
            metadata, json_path = cls._load_metadata(name)

            primitive = import_object(metadata['primitive'])
            metadata['primitive'] = primitive

            if inspect.isclass(primitive):
                subcls = ClassBlock
            else:
                subcls = FunctionBlock

            instance = super(MLBlock, cls).__new__(subcls)
            instance.metadata = metadata
            instance.json_path = json_path

            return instance

    def _get_fixed_hyperparams(self, kwargs, hyperparameters):
        arguments = dict()
        for param in hyperparameters.get('fixed', list()):
            name = param['name']
            if name in kwargs:
                value = kwargs.pop(name)
            elif 'default' in param:
                value = param['default']
            else:
                raise TypeError("Required argument '{}' not found".format(name))

            arguments[name] = value

        if kwargs:
            error = "Unexpected hyperparameters '{}'".format(', '.join(kwargs.keys()))
            raise TypeError(error)

        return arguments

    def __init__(self, name, **kwargs):
        self.name = self.metadata['name']

        self.primitive = self.metadata['primitive']

        hyperparameters = self.metadata.get('hyperparameters', dict())

        fixed_hyperparameters = self._get_fixed_hyperparams(kwargs, hyperparameters)
        self._hyperparamters = fixed_hyperparameters

        self._tunable = {
            name: param
            for name, param in hyperparameters.get('tunable', dict()).items()
            if name not in fixed_hyperparameters
            # TODO: filter conditionals
        }

        default_hyperparameters = {
            name: param['default']
            for name, param in self._tunable.items()
            # TODO: support undefined defaults
        }

        self.set_hyperparameters(default_hyperparameters)

        self._produce = self.metadata['produce']
        self.produce_args = self._produce['args']
        self.produce_output = self._produce['output']

    def get_tunable_hyperparameters(self):
        return self._tunable

    def get_hyperparameters(self):
        return self._hyperparamters

    def fit(self, *args, **kwargs):
        pass

    # ################ #
    # Abstract methods #
    # ################ #

    def set_hyperparameters(self, hyperparameters):
        raise NotImplementedError()

    def produce(self, *args, **kwargs):
        raise NotImplementedError()


class FunctionBlock(MLBlock):

    def set_hyperparameters(self, hyperparameters):
        self._hyperparamters.update(hyperparameters)

    def produce(self, **kwargs):
        kwargs.update(self._hyperparamters)
        return self.primitive(**kwargs)


class ClassBlock(MLBlock):

    def __init__(self, name, **kwargs):
        super(ClassBlock, self).__init__(name, **kwargs)

        self.produce_method = self._produce['method']

        fit = self.metadata.get('fit')
        if fit is not None:
            self.fit_method = fit['method']
            self.fit_args = fit['args']

        else:
            self.fit_method = None
            self.fit_args = dict()

    def set_hyperparameters(self, hyperparameters):
        self._hyperparamters.update(hyperparameters)

        self.instance = self.primitive(**self._hyperparamters)

    def produce(self, **kwargs):
        return getattr(self.instance, self.produce_method)(**kwargs)

    def fit(self, **kwargs):
        if self.fit_method is not None:
            getattr(self.instance, self.fit_method)(**kwargs)
