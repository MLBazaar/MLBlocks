from mlblocks.ml_pipeline.ml_pipeline import MLPipeline


class LstmTextClassifier(MLPipeline):
    """LSTM text pipeline via Keras.

    From:
    http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/
    """  # noqa

    BLOCKS = ['tokenizer', 'sequence_padder', 'lstm_text', 'convert_class_probs']

    def __init__(self, num_classes, pad_length=None, optimizer=None, loss=None):
        super(LstmTextClassifier, self).__init__()

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
            update_params[('text_padder', 'pad_length')] = pad_length
            update_params[('lstm_text', 'pad_length')] = pad_length

        self.update_fixed_hyperparams(update_params)


class LstmTextRegressor(MLPipeline):

    BLOCKS = ['tokenizer', 'sequence_padder', 'lstm_text']

    def __init__(self, optimizer=None, loss=None):
        super(LstmTextRegressor, self).__init__()

        update_params = dict()
        if optimizer is not None:
            update_params[('lstm_text', 'optimizer')] = optimizer

        if loss is not None:
            update_params[('lstm_text', 'loss')] = loss

        self.update_fixed_hyperparams(update_params)
