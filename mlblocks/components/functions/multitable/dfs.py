import featuretools as ft


class DeepFeatureSynthesis(object):
    def __init__(self, target_entity, training_window, cutoff_time, max_depth, verbose):
        self.target_entity = target_entity
        self.training_window = training_window
        self.cutoff_time = cutoff_time
        self.max_depth = max_depth
        self.verbose = verbose

        self.__features = None

    def dfs(self, entity_set):
        feature_matrix, features = ft.dfs(
            target_entity=self.target_entity,
            cutoff_time=self.cutoff_time,
            training_window=self.training_window,
            entityset=entity_set,
            max_depth=self.max_depth,
            verbose=self.verbose)
        fm_encoded, features_encoded = ft.encode_features(
            feature_matrix, features)
        fm_encoded = fm_encoded.fillna(0)
        return fm_encoded

    def fit(self, X, **kwargs):
        feature_matrix = ft.dfs(
            entit
        )

    def produce(self, X, **kwargs):
        feature_matrix = ft.calculate_feature_matrix(self.__features, **kwargs)
