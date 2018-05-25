from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class LstmTextClassifier(MLPipeline):
    """LSTM text pipeline via Keras.

    From:
    http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/
    """  # noqa

    def __new__(cls, num_classes, pad_length=None, optimizer=None, loss=None):
        lstm = MLPipeline.from_ml_json([
            'tokenizer', 'sequence_padder', 'lstm_text', 'convert_class_probs'
        ])

        update_params = {
            ('lstm_text', 'dense_units'): num_classes,
            ('lstm_text', 'dense_activation'): 'softmax',
            ('lstm_text', 'optimizer'): 'keras.optimizers.Adadelta',
            ('lstm_text', 'loss'): 'keras.losses.categorical_crossentropy'
        }
        if optimizer is not None:
            update_params[('lstm_text', 'optimizer')] = optimizer
        if loss is not None:
            update_params[('lstm_text', 'loss')] = loss
        if pad_length is not None:
            update_params[('sequence_padder', 'pad_length')] = pad_length
            update_params[('lstm_text', 'pad_length')] = pad_length
        lstm.update_fixed_hyperparams(update_params)

        return lstm


class LstmTextRegressor(MLPipeline):
    def __new__(cls, pad_length=None, optimizer=None, loss=None):
        lstm = MLPipeline.from_ml_json(
            ['tokenizer', 'sequence_padder', 'lstm_text'])

        update_params = {}
        if optimizer is not None:
            update_params[('lstm_text', 'optimizer')] = optimizer
        if loss is not None:
            update_params[('lstm_text', 'loss')] = loss
        if pad_length is not None:
            update_params[('sequence_padder', 'pad_length')] = pad_length
            update_params[('lstm_text', 'pad_length')] = pad_length
        lstm.update_fixed_hyperparams(update_params)

        return lstm
