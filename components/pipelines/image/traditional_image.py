from ml_pipeline.ml_pipeline import MLPipeline


class TraditionalImagePipeline(MLPipeline):
    """
    Traditional image pipeline using HOG features.
    """

    def __new__(cls, *args, **kwargs):
        return MLPipeline.from_dm_json(['HOG', 'random_forest_classifier'])
