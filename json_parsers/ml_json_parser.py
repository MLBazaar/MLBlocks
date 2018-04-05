import importlib

from ml_pipeline.ml_block import MLBlock
from ml_pipeline.ml_hyperparam import MLHyperparam


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
        tunable_hyperparams = self.get_mlhyperparams(block_name)
        model = self.build_mlblock_model(tunable_hyperparams)

        ml_block_instance = MLBlock(
            name=block_name,
            model=model,
            tunable_hyperparams=tunable_hyperparams)
        ml_block_instance.fit = getattr(ml_block_instance.model,
                                        self.block_json['fit'])
        ml_block_instance.produce = getattr(ml_block_instance.model,
                                            self.block_json['produce'])

        return ml_block_instance

    def build_mlblock_model(self, hyperparameters):
        """Builds the model for this primitive block.

        Args:
            hyperparameters: The hyperparameters to build this model
            with. Should be specified as a kwargs dict mapping
            hyperparameter name to MLHyperparam object.

        Returns:
            The model instance of this primitive block.
        """
        # Load the class for this primitive step.
        full_module_class = self.block_json['class']
        module_name = '.'.join(full_module_class.split('.')[:-1])
        class_name = full_module_class.split('.')[-1]
        block_class = getattr(importlib.import_module(module_name), class_name)
        block_model = block_class(**{
            hp_name: hyperparameters[hp_name].value
            for hp_name in hyperparameters
        })

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
        for hp_name in self.block_json['hyperparameters'].keys():
            hp_info = self.block_json['hyperparameters'][hp_name]
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
