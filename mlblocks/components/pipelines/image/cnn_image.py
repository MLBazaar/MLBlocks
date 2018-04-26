from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class SimpleCNN(MLPipeline):
    """
    CNN image pipeline.

    Layers: Conv2D + MaxPooling2D + Dropout + Dense
    """

    def __new__(cls, *args, **kwargs):
        return MLPipeline.from_dm_json(['simple_cnn'])
