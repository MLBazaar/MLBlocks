from dm_pipeline.dm_pipeline import DmPipeline


class TraditionalImagePipeline(DmPipeline):
    """
    Traditional image pipeline using HOG features.
    """

    def __new__(cls, *args, **kwargs):
        return DmPipeline.from_dm_json(['HOG', 'random_forest_classifier'])
