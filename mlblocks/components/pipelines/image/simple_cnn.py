from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class SimpleCNN(MLPipeline):
    """
    CNN image pipeline.

    Layers: Conv2D + MaxPooling2D + Dropout + Dense
    """

    def __new__(cls, num_classes):
        simple_cnn = MLPipeline.from_dm_json(['simple_cnn'])
        simple_cnn.update_fixed_hyperparams({
            ('simple_cnn', 'dense2_units'):
            num_classes
        })
        return simple_cnn
