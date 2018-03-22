from dm_pipeline.dm_pipeline import DmPipeline


class RandomForestClassifier(DmPipeline):
    """
    Random forest pipeline
    """

    def __new__(cls, *args, **kwargs):
        return DmPipeline.from_dm_json(['random_forest_classifier'])
