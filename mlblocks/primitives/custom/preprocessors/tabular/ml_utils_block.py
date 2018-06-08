import numpy as np


class MLUtilsBlock(object):
    """Utils for transforming general data in a pipeline."""

    def to_array(self, X):
        return X.toarray()

    def convert_class_probs(self, X):
        """Convert a list of class probabilities to categorical values.

        Categorical values are returned as values from 0 to n, where n
        is the number of classes.

        Args:
            X: A list of class probabilities.

        Returns:
            Categorical values.
        """
        return np.argmax(X, axis=1)
