from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class SimpleCnnClassifier(MLPipeline):
    """
    CNN image pipeline.

    Based on:
    github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

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

    def __new__(cls, num_classes, optimizer=None, loss=None):
        simple_cnn = MLPipeline.from_dm_json(
            ['simple_cnn', 'convert_class_probs'])

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
        simple_cnn.update_fixed_hyperparams(update_params)

        return simple_cnn


class SimpleCnnRegressor(MLPipeline):
    def __new__(cls, optimizer=None, loss=None):
        simple_cnn = MLPipeline.from_dm_json(['simple_cnn'])

        update_params = {}
        if optimizer is not None:
            update_params[('simple_cnn', 'optimizer')] = optimizer
        if loss is not None:
            update_params[('simple_cnn', 'loss')] = loss
        simple_cnn.update_fixed_hyperparams(update_params)

        return simple_cnn
