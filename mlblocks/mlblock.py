# -*- coding: utf-8 -*-

import importlib
import json

import mlblocks


def import_object(object_name):
    """Import an object from its Fully Qualified Name."""
    package, name = object_name.rsplit('.', 1)
    return getattr(importlib.import_module(package), name)


class MLBlock(object):

    @classmethod
    def _load_metadata(cls, name):
        """Locate and load the corresponding JSON file."""

        json_path = mlblocks.get_primitive_path(name)
        with open(json_path, 'r') as f:
            return json.load(f), json_path

    def _extract_params(self, kwargs, hyperparameters):
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

        for name in kwargs.keys():
            if name in self.fit_args.keys():
                fit_params[name] = kwargs.pop(name)

            elif name in self.produce_args.keys():
                produce_params[name] = kwargs.pop(name)

        if kwargs:
            error = "Unexpected hyperparameters '{}'".format(', '.join(kwargs.keys()))
            raise TypeError(error)

        return init_params, fit_params, produce_params

    def __init__(self, name, **kwargs):
        self.name = name

        metadata, json_path = self._load_metadata(name)
        self.metadata = metadata
        self.json_path = json_path

        self.primitive = import_object(metadata['primitive'])

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

    def get_tunable_hyperparameters(self):
        return self._tunable

    def get_hyperparameters(self):
        return self._hyperparamters

    def set_hyperparameters(self, hyperparameters):
        self._hyperparamters.update(hyperparameters)

        if self._class:
            self.instance = self.primitive(**self._hyperparamters)

    def fit(self, **kwargs):
        fit_args = self._fit_params.copy()
        fit_args.update(kwargs)
        if self.fit_method is not None:
            getattr(self.instance, self.fit_method)(**fit_args)

    def produce(self, **kwargs):
        produce_args = self._produce_params.copy()
        produce_args.update(kwargs)
        if self._class:
            return getattr(self.instance, self.produce_method)(**produce_args)

        else:
            produce_args.update(self._hyperparamters)
            return self.primitive(**produce_args)
