import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


class TrivialPredictor(object):

    def __init__(self, default=0):
        self.prediction = default

    def fit(self, X, y):
        pass

    def predict(self, X):
        return pd.Series(self.prediction, index=X.index)


class TrivialModePredictor(TrivialPredictor):

    def fit(self, X, y):
        try:
            self.prediction = y.mode().iloc[0]
        except Exception:
            LOGGER.error("Could not compute y.mode(). Using default.")


class TrivialMedianPredictor(object):

    def fit(self, X, y):
        try:
            self.prediction = y.median()
        except Exception:
            LOGGER.error("Could not compute y.median(). Using default.")
