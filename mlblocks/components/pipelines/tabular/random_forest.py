from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class RandomForestClassifier(MLPipeline):
    """Random forest classifier pipeline."""

    BLOCKS = ['random_forest_classifier']


class RandomForestRegressor(MLPipeline):
    """Random forest classifier pipeline."""

    BLOCKS = ['random_forest_regressor']
