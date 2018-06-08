import featuretools as ft


class DeepFeatureSynthesis(object):
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.features = None

    def fit(self, X, **kwargs):
        self.features = ft.dfs(
            cutoff_time=X,
            features_only=True,
            max_depth=self.max_depth,
            **kwargs
        )

    def produce(self, X, **kwargs):
        feature_matrix = ft.calculate_feature_matrix(
            self.features, cutoff_time=X, **kwargs)

        fm_encoded, features_encoded = ft.encode_features(
            feature_matrix, self.features)

        return fm_encoded.fillna(0)
