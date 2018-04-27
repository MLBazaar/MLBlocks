import importlib

from mlblocks.json_parsers.ml_json_parser import MLJsonParser
from mlblocks.ml_pipeline.ml_hyperparam import MLHyperparam


class KerasJsonParser(MLJsonParser):
    """A JSON primitive parser for Keras models.

    Supports loading JSONS in the format shown in:
        components/primitive_jsons/simple_cnn.json
    """

    def __init__(self, block_json):
        """Initializes a Keras JSON parser for a given JSON.

        Args:
            block_json: A Keras JSON dict to parse into an MLBlock.
                See components/primitive_jsons for JSON examples.
        """
        super(KerasJsonParser, self).__init__(block_json)

    def build_mlblock_model(self, fixed_hyperparameters,
                            tunable_hyperparameters):
        # Load the class for this primitive step.
        full_module_class = self.block_json['class']
        assert (full_module_class == 'keras.models.Sequential')
        sequential_class = self._get_class(full_module_class)
        model = sequential_class()

        layers = self.block_json['layers']
        for layer_metadata in layers:
            layer_module_class = layer_metadata['class']
            layer_class = self._get_class(layer_module_class)
            layer_kwargs = {}
            for param in layer_metadata['parameters']:
                hp_name = layer_metadata['parameters'][param]
                if hp_name in tunable_hyperparameters:
                    layer_kwargs[param] = tunable_hyperparameters[
                        hp_name].value
                else:
                    layer_kwargs[param] = fixed_hyperparameters[hp_name]
            layer = layer_class(**layer_kwargs)
            model.add(layer)

        optimizer = self._get_class(fixed_hyperparameters['optimizer'])()
        loss = self._get_class(fixed_hyperparameters['loss'])
        model.compile(loss=loss, optimizer=optimizer)

        return model

    @staticmethod
    def _get_class(full_module_class):
        # Loads a class given the full module name.
        # e.g. keras.layers.Dense or keras.models.Sequential
        module_name = '.'.join(full_module_class.split('.')[:-1])
        class_name = full_module_class.split('.')[-1]
        actual_class = getattr(
            importlib.import_module(module_name), class_name)
        return actual_class

    def get_mlhyperparams(self, block_name):
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
