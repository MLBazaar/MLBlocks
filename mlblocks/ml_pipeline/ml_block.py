class MLBlock(object):
    """A single primitive in an MLPipeline.
    Simply defines the interface. Implementation should be done in an
    MLParser.

    Attributes:
        name: The name of this primitive.
        model: The actual model of this primitive that acts on data.
        tunable_hyperparams: A dictionary mapping this primitive's
            hyperparameter names to corresponding Hyperparam objects.
    """

    def __init__(self, name, model, tunable_hyperparams):
        """ Initializes this MLBlock.

        Args:
            name: The name of this step primitive.
            model: The actual model of this primitive that acts on
                data.
            tunable_hyperparams: A dictionary mapping this
                primitive's hyperparameter names to corresponding
                MLHyperparam objects.
        """
        self.name = name
        self.model = model
        self.tunable_hyperparams = tunable_hyperparams

    def fit(self):
        raise NotImplementedError

    def produce(self):
        raise NotImplementedError
