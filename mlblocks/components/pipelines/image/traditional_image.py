from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class TraditionalImagePipeline(MLPipeline):
    """
    Traditional image pipeline using HOG features.
    """

    def __new__(cls, *args, **kwargs):
        return MLPipeline.from_ml_json(['HOG', 'random_forest_classifier'])
