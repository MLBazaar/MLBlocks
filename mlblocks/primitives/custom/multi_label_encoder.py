import numpy as np
from sklearn.preprocessing import LabelEncoder


class MultiLabelEncoder(object):
    """Multi column LabelEncoder.

    Codifies categorical features as Integer vectors.
    """

    def __init__(self, categorical_features=None):
        self.features = categorical_features

    def detect_features(self, X):
        features = []
        for column in X.columns:
            if not np.issubdtype(X[column].dtype, np.number):
                features.append(column)

        return features

    def encode(self, x):
        encoder = self.encoders.get(x.name)
        if encoder is not None:
            return encoder.transform(x)

        else:
            return x

    def fit_encoder(self, x):
        encoder = self.encoders.get(x.name)
        if encoder is not None:
            return encoder.fit(x)

    def fit(self, X, y=None, categorical_features=None):
        features = categorical_features or self.features
        if not features:
            features = self.detect_features(X)

        self.encoders = {feature: LabelEncoder() for feature in features}

        X[features].apply(self.fit_encoder)

    def produce(self, X):
        return X.apply(self.encode)
