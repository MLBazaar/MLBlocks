from mlblocks import import_object
from mlblocks.mlblock import MLBlock
from mlblocks.mlhyperparam import MLHyperparam


class MLJsonParser(object):
    """A basic JSON primitive parser.

    Supports loading JSONS in the format shown in:
        mlblocks/primitives/random_forest_classifier.json
    """

    def __init__(self, metadata, init_params):
        """Initialize a basic JSON parser for a given JSON.

        Args:
            metadata: A JSON dict to parse into an MLBlock.
                See mlblocks/primitives for JSON examples.
        """
        self.metadata = metadata
        self.init_params = init_params[self.metadata['name']]

    def build_mlblock(self):
        block_name = self.metadata['name']
        fixed_hyperparams = self.metadata['fixed_hyperparameters']
        fixed_hyperparams.update(self.init_params)

        tunable_hyperparams = self.get_mlhyperparams(block_name)
        model = self.build_mlblock_model(fixed_hyperparams,
                                         tunable_hyperparams)

        instance = MLBlock(
            name=block_name,
            model=model,
            fixed_hyperparams=fixed_hyperparams,
            tunable_hyperparams=tunable_hyperparams
        )

        self.replace_instance_methods(instance)

        return instance

    def build_mlblock_model(self, fixed_hyperparameters, tunable_hyperparameters):
        """Build the model for this primitive block.

        Args:
            fixed_hyperparameters: The fixed hyperparameters to build
                this model with. Should be specified as a dict mapping
                fixed hyperparameter names to their corresponding
                values.
            tunable_hyperparameters: The tunable hyperparameters to
                build this model with. Should be specified as a dict
                mapping hyperparameter name to MLHyperparam object.

        Returns:
            The model instance of this primitive block.
        """
        block_class = import_object(self.metadata['class'])

        model_kwargs = fixed_hyperparameters.copy()
        model_kwargs.update({
            hp_name: tunable_hyperparameters[hp_name].value
            for hp_name in tunable_hyperparameters
        })

        return block_class(**model_kwargs)

    def get_mlhyperparams(self, block_name):
        """Get the hyperparameters belonging to this primitive block.

        Args:
            block_name: The name of this primitive block.

        Returns:
            A dict mapping hyperparameter names to MLHyperparam
            objects.
        """
        tunable_hyperparams = {}
        tunable_hps = self.metadata['tunable_hyperparameters']
        root_hps = set(self.metadata['root_hyperparameters'])

        for hp_name, hp_info in tunable_hps.items():
            hp_type = hp_info['type']
            hp_range = hp_info.get('range', hp_info.get('values'))
            hp_val = hp_info.get('default')
            hp_is_cond = hp_name not in root_hps

            hyperparam = MLHyperparam(
                hp_name, hp_type, hp_range, hp_is_cond, hp_val)
            hyperparam.block_name = block_name

            tunable_hyperparams[hp_name] = hyperparam

        return tunable_hyperparams

    def replace_instance_methods(self, instance):
        """Replace the instance methods of the specified MLBlock instance.

        See the MLBlock class for instance method details.

        fit and produce are replaced with the fit and produce methods specified
        in the JSON.

        update_model is replaced with a function that rebuilds the model via
            build_mlblock_method and updates the model.

        Args:
            instance: The MLBlock instance to replace methods for.
        """
        # Declare fit and predict methods in this way so that they
        # remain bound to the MLBlock instance's model.
        fit_method_name = self.metadata['fit']
        produce_method_name = self.metadata['produce']
        build_method = self.build_mlblock_model

        def fit(self, *args, **kwargs):
            # Only fit if fit method provided.
            if fit_method_name:
                getattr(self.model, fit_method_name)(*args, **kwargs)

        instance.fit = fit.__get__(instance, MLBlock)

        def produce(self, *args, **kwargs):
            # Every MLBlock needs a produce method.
            return getattr(self.model, produce_method_name)(*args, **kwargs)

        instance.produce = produce.__get__(instance, MLBlock)

        def update_model(self, fixed_hyperparams, tunable_hyperparams):
            self.model = build_method(fixed_hyperparams, tunable_hyperparams)

        instance.update_model = update_model.__get__(instance, MLBlock)
