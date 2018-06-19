import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class OneHotLabelEncoder(object):
    """Combination of LabelEncoder + OneHotEncoder.

    >>> df = pd.DataFrame([
    ... {'a': 'a', 'b': 1, 'c': 1},
    ... {'a': 'a', 'b': 2, 'c': 2},
    ... {'a': 'b', 'b': 2, 'c': 1},
    ... ])
    >>> OneHotLabelEncoder().fit_transform(df.a)
       a=a  a=b
    0    1    0
    1    1    0
    2    0    1
    >>> OneHotLabelEncoder(max_labels=1).fit_transform(df.a)
       a=a
    0    1
    1    1
    2    0
    >>> OneHotLabelEncoder(name='a_name').fit_transform(df.a)
       a_name=a  a_name=b
    0         1         0
    1         1         0
    2         0         1
    """

    def __init__(self, name=None, max_labels=None):
        self.name = name
        self.max_labels = max_labels

    def fit(self, feature):
        self.dummies = pd.Series(feature.value_counts().index).astype(str)
        if self.max_labels:
            self.dummies = self.dummies[:self.max_labels]

    def transform(self, feature):
        name = self.name or feature.name
        dummies = pd.get_dummies(feature.astype(str))
        dummies = dummies.reindex(columns=self.dummies, fill_value=0)
        dummies.columns = ['{}={}'.format(name, c) for c in self.dummies]
        return dummies

    def fit_transform(self, feature):
        self.fit(feature)
        return self.transform(feature)


class CategoricalEncoder(object):
    """Use the OneHotLabelEncoder only on categorical features.

    NOTE: At the moment of this release, sklearn.preprocessing.data.CategoricalEncoder
    has not been released yet, this is why we write our own version of it.

    >>> df = pd.DataFrame([
    ... {'a': 'a', 'b': 1, 'c': 1},
    ... {'a': 'a', 'b': 2, 'c': 2},
    ... {'a': 'b', 'b': 2, 'c': 1},
    ... ])
    >>> ce = CategoricalEncoder()
    >>> ce.fit_transform(df, categorical_features=['a', 'c'])
       b  a=a  a=b  c=1  c=2
    0  1    1    0    1    0
    1  2    1    0    0    1
    2  2    0    1    1    0
    """

    def __init__(self, max_labels=None, copy=True, categorical_features=None):
        self.max_labels = max_labels
        self.copy = copy
        self.features = categorical_features

    def detect_features(self, X):
        features = []
        for column in X.columns:
            if not np.issubdtype(X[column].dtype, np.number):
                features.append(column)

        return features

    def fit(self, X, y=None, categorical_features=None):
        if not self.features:
            self.features = categorical_features or self.detect_features(X)

        self.encoders = dict()
        for feature in self.features:
            encoder = OneHotLabelEncoder(feature, self.max_labels)
            encoder.fit(X[feature])
            self.encoders[feature] = encoder

    def transform(self, X):
        if self.copy and self.encoders:
            X = X.copy()

        for name, encoder in self.encoders.items():
            LOGGER.debug("Encoding feature %s", name)
            feature = X.pop(name)
            encoded = encoder.transform(feature)
            X = pd.concat([X, encoded], axis=1)

        return X

    def fit_transform(self, X, y=None, categorical_features=None):
        self.fit(X, y, categorical_features)
        return self.transform(X)
