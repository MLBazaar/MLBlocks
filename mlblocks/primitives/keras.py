# -*- coding: utf-8 -*-

import logging
import tempfile

import keras
import numpy as np

from mlblocks import import_object

LOGGER = logging.getLogger(__name__)


class Sequential(object):
    """A Wrapper around Sequential Keras models with a simpler interface."""

    def __getstate__(self):
        state = self.__dict__.copy()

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(state.pop('model'), fd.name, overwrite=True)
            state['model_str'] = fd.read()

        return state

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state.pop('model_str'))
            fd.flush()

            state['model'] = keras.models.load_model(fd.name)

        self.__dict__ = state

    def __init__(self, layers, loss, optimizer, metrics=None, epochs=10, **hyperparameters):

        self.epochs = epochs
        self.is_classification = hyperparameters['dense_units'] > 1

        self.model = keras.models.Sequential()

        for layer in layers:
            layer_class = import_object(layer['class'])
            layer_kwargs = layer['parameters']

            for key, value in layer_kwargs.items():
                if isinstance(value, str):
                    layer_kwargs[key] = hyperparameters.get(value, value)

            self.model.add(layer_class(**layer_kwargs))

        optimizer = import_object(optimizer)()
        loss = import_object(loss)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X, y):
        if self.is_classification:
            y = keras.utils.to_categorical(y)

        self.model.fit(X, y, epochs=self.epochs)

    def predict(self, X):
        y = self.model.predict(X)

        if self.is_classification:
            y = np.argmax(y, axis=1)

        return y
