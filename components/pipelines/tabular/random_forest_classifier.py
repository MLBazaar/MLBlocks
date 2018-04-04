from ml_pipeline.ml_pipeline import MLPipeline


class RandomForestClassifier(MLPipeline):
    """
    Random forest pipeline
    """

    def __new__(cls, *args, **kwargs):
        return MLPipeline.from_dm_json(['random_forest_classifier'])
