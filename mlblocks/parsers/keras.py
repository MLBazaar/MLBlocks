from mlblocks import import_object
from mlblocks.parsers.json import MLJsonParser


class KerasJsonParser(MLJsonParser):
    """A JSON primitive parser for Keras models.

    Supports loading JSONS in the format shown in:
        mlblocks/primitives/simple_cnn.json
    """

    def build_mlblock_model(self, fixed_hyperparameters, tunable_hyperparameters):
        # Load the class for this primitive block.
        full_module_class = self.metadata['class']
        assert (full_module_class == 'keras.models.Sequential')
        sequential_class = import_object(full_module_class)
        model = sequential_class()

        layers = self.metadata['layers']
        for layer_metadata in layers:
            layer_module_class = layer_metadata['class']
            layer_class = import_object(layer_module_class)
            layer_kwargs = {}
            for param in layer_metadata['parameters']:
                hp_name = layer_metadata['parameters'][param]
                if hp_name in tunable_hyperparameters:
                    layer_kwargs[param] = tunable_hyperparameters[hp_name].value
                else:
                    layer_kwargs[param] = fixed_hyperparameters[hp_name]

            layer = layer_class(**layer_kwargs)
            model.add(layer)

        optimizer = import_object(fixed_hyperparameters['optimizer'])()
        loss = import_object(fixed_hyperparameters['loss'])
        metrics = fixed_hyperparameters.get('metrics')
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        return model
