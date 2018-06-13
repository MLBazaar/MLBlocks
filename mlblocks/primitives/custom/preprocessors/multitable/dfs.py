import featuretools as ft
import pandas as pd
from featuretools import variable_types as vtypes
from featuretools.selection import remove_low_information_features


class DeepFeatureSynthesis(object):

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.features = None

    def fit(self, X, y=None, features_only=True, **kwargs):
        self.features = ft.dfs(
            cutoff_time=X,
            features_only=features_only,
            max_depth=self.max_depth,
            **kwargs
        )

    def produce(self, X, instance_ids=None, include_unknown=True,
                remove_low_information=True, **kwargs):

        if instance_ids is not None:
            feature_matrix = ft.calculate_feature_matrix(
                self.features,
                instance_ids=instance_ids,
                **kwargs
            )

            feature_matrix = (feature_matrix.reset_index('time')
                                            .loc[instance_ids, :]
                                            .set_index('time', append=True))

        else:
            feature_matrix = ft.calculate_feature_matrix(
                self.features, cutoff_time=X, **kwargs)

        for f in self.features:
            if issubclass(f.variable_type, vtypes.Discrete):
                feature_matrix[f.get_name()] = feature_matrix[f.get_name()].astype(object)
            elif issubclass(f.variable_type, vtypes.Numeric):
                feature_matrix[f.get_name()] = pd.to_numeric(feature_matrix[f.get_name()])
            elif issubclass(f.variable_type, vtypes.Datetime):
                feature_matrix[f.get_name()] = pd.to_datetime(feature_matrix[f.get_name()])

        encoded_fm, encoded_fl = ft.encode_features(feature_matrix, self.features)

        if remove_low_information:
            encoded_fm, encoded_fl = remove_low_information_features(encoded_fm, encoded_fl)

        encoded_fm.reset_index('time', drop=True, inplace=True)

        return encoded_fm.fillna(0)
