from ml_pipeline.ml_pipeline import MLPipeline


class RandomForestRegressor(MLPipeline):
    """
    Random forest pipeline
    """

    def __new__(cls, *args, **kwargs):
        return MLPipeline.from_dm_json(['random_forest_regressor'])
