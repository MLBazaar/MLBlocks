class MLBlock(object):
    """A single primitive in an MLPipeline.
    Simply defines the interface. Implementation should be done in an
    MLParser.

    Attributes:
        name: The name of this primitive.
        model: The actual model of this primitive that acts on data.
        fixed_hyperparams: A dictionary mapping this primitive's
            non-tunable hyperparameter names to their corresponding
            values.
        tunable_hyperparams: A dictionary mapping this primitive's
            hyperparameter names to corresponding Hyperparam objects.
    """

    def __init__(self, name, model, fixed_hyperparams, tunable_hyperparams):
        """Initialize this MLBlock.

        Args:
            name: The name of this block primitive.
            model: The actual model of this primitive that acts on
                data.
            fixed_hyperparams: A dictionary mapping this primitive's
                non-tunable hyperparameter names to their corresponding
                values.
            tunable_hyperparams: A dictionary mapping this
                primitive's hyperparameter names to corresponding
                MLHyperparam objects.
        """
        self.name = name
        self.model = model
        self.fixed_hyperparams = fixed_hyperparams
        self.tunable_hyperparams = tunable_hyperparams

    def update_model(self, fixed_hyperparams, tunable_hyperparams):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError

    def produce(self, x, y):
        raise NotImplementedError
