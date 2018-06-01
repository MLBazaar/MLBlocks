from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class SimpleCnnClassifier(MLPipeline):
    """CNN image pipeline.

    Based on:
    https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

    Layers:
        Conv2D
        Conv2D
        MaxPooling2D
        Dropout
        Flatten
        Dense
        Dropout
        Dense
    """
    BLOCKS = ['simple_cnn', 'convert_class_probs']

    def __init__(self, num_classes, optimizer=None, loss=None):
        super(SimpleCnnClassifier, self).__init__()

        update_params = {
            ('simple_cnn', 'dense2_units'): num_classes,
            ('simple_cnn', 'dense2_activation'): 'softmax',
            ('simple_cnn', 'optimizer'): 'keras.optimizers.Adadelta',
            ('simple_cnn', 'loss'): 'keras.losses.categorical_crossentropy'
        }
        if optimizer is not None:
            update_params[('simple_cnn', 'optimizer')] = optimizer

        if loss is not None:
            update_params[('simple_cnn', 'loss')] = loss

        self.update_fixed_hyperparams(update_params)


class SimpleCnnRegressor(MLPipeline):

    BLOCKS = ['simple_cnn']

    def __init__(self, optimizer=None, loss=None):
        super(SimpleCnnRegressor, self).__init__()

        update_params = {}
        if optimizer is not None:
            update_params[('simple_cnn', 'optimizer')] = optimizer

        if loss is not None:
            update_params[('simple_cnn', 'loss')] = loss

        self.update_fixed_hyperparams(update_params)
