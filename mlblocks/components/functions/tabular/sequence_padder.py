from keras.preprocessing.sequence import pad_sequences


class SequencePadder(object):
    """
    Wrapper for keras.preprocessing.sequence.pad_sequences.

    Has applications in timeseries, text, etc.
    """
    def __init__(self, pad_length):
        self.pad_length = pad_length

    def pad_sequences(self, X):
        """Pads a sequence of data to a certain length.
        """
        return pad_sequences(X, maxlen=self.pad_length)
