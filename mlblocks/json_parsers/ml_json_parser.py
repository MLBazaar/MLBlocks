import importlib

from mlblocks.ml_pipeline.ml_block import MLBlock
from mlblocks.ml_pipeline.ml_hyperparam import MLHyperparam


class MLJsonParser(object):
    """A basic JSON primitive parser.

    Supports loading JSONS in the format shown in:
        components/primitive_jsons/random_forest_classifier.json
    """

    def __init__(self, block_json):
        """Initializes a basic JSON parser for a given JSON.

        Args:
            block_json: A JSON dict to parse into an MLBlock.
                See components/primitive_jsons for JSON examples.
        """
        self.block_json = block_json

    def build_mlblock(self):
        block_name = self.block_json['name']
        fixed_hyperparams = self.block_json['fixed_hyperparameters']
        tunable_hyperparams = self.get_mlhyperparams(block_name)
        model = self.build_mlblock_model(fixed_hyperparams,
                                         tunable_hyperparams)

        ml_block_instance = MLBlock(
            name=block_name,
            model=model,
            fixed_hyperparams=fixed_hyperparams,
            tunable_hyperparams=tunable_hyperparams)
        self.default_update_ml_block_instance_methods(ml_block_instance)

        return ml_block_instance

    def build_mlblock_model(self, fixed_hyperparameters,
                            tunable_hyperparameters):
        """Builds the model for this primitive block.

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
        # Load the class for this primitive step.
        full_module_class = self.block_json['class']
        module_name = '.'.join(full_module_class.split('.')[:-1])
        class_name = full_module_class.split('.')[-1]
        block_class = getattr(importlib.import_module(module_name), class_name)
        model_kwargs = fixed_hyperparameters.copy()
        model_kwargs.update({
            hp_name: tunable_hyperparameters[hp_name].value
            for hp_name in tunable_hyperparameters
        })
        block_model = block_class(**model_kwargs)

        return block_model

    def get_mlhyperparams(self, block_name):
        """Gets the hyperparameters belonging to this primitive block.

        Args:
            block_name: The name of this primitive block.

        Returns:
            A dict mapping hyperparameter names to MLHyperparam
            objects.
        """
        tunable_hyperparams = {}
        for hp_name in self.block_json['tunable_hyperparameters'].keys():
            hp_info = self.block_json['tunable_hyperparameters'][hp_name]
            hp_type = hp_info['type']
            hp_range = hp_info['range'] if 'range' in hp_info else hp_info[
                'values']
            hp_val = hp_info['default'] if 'default' in hp_info else None
            hp_is_cond = hp_name not in set(
                self.block_json['root_hyperparameters'])
            hyperparam = MLHyperparam(hp_name, hp_type, hp_range, hp_is_cond,
                                      hp_val)
            hyperparam.step_name = block_name
            tunable_hyperparams[hp_name] = hyperparam

        return tunable_hyperparams

    def default_update_ml_block_instance_methods(self, ml_block_instance):
        """Updates the instance methods of the specified MLBlock instance.

        See the MLBlock class for instance method details.

        fit and produce are updated with the fit and produce methods specified
        in the JSON.

        update_model is updated with a function that rebuilds the model via
            build_mlblock_method and updates the model.

        Args:
            ml_block_instance: The MLBlock instance to update methods for.
        """
        # Declare fit and predict methods in this way so that they
        # remain bound to the MLBlock instance's model.
        fit_method_name = self.block_json['fit']
        produce_method_name = self.block_json['produce']
        build_method = self.build_mlblock_model

        def mlblock_fit(self, *args, **kwargs):
            getattr(self.model, fit_method_name)(*args, **kwargs)

        ml_block_instance.fit = mlblock_fit.__get__(ml_block_instance, MLBlock)

        def mlblock_produce(self, *args, **kwargs):
            return getattr(self.model, produce_method_name)(*args, **kwargs)

        ml_block_instance.produce = mlblock_produce.__get__(
            ml_block_instance, MLBlock)

        def mlblock_update_model(self, fixed_hyperparams, tunable_hyperparams):
            self.model = build_method(fixed_hyperparams, tunable_hyperparams)

        ml_block_instance.update_model = mlblock_update_model.__get__(
            ml_block_instance, MLBlock)
