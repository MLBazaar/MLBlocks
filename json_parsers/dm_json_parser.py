import importlib


class DmJsonParser(object):
    """A basic JSON primitive parser.

    Supports loading JSONS in the format shown in:
        components/primitive_jsons/random_forest_classifier.json
    """

    def __init__(self):
        self.step_metadata = None

    def set_json_metadata(self, metadata):
        self.step_metadata = metadata

    def get_name(self):
        return self.step_metadata['name']

    def get_class(self):
        # Load the class for this primitive step.
        full_module_class = self.step_metadata['class']
        module_name = '.'.join(full_module_class.split('.')[:-1])
        class_name = full_module_class.split('.')[-1]
        step_class = getattr(importlib.import_module(module_name), class_name)

        return step_class

    def get_fit_method(self):
        return self.step_metadata['fit']

    def get_produce_method(self):
        return self.step_metadata['produce']

    def get_hyperparam_names(self):
        return self.step_metadata['hyperparameters'].keys()

    def get_root_hyperparams(self):
        return self.step_metadata['root_hyperparameters']

    def get_hyperparam_type(self, hyperparam_name):
        return self.step_metadata['hyperparameters'][hyperparam_name]['type']

    def get_hyperparam_range(self, hyperparam_name):
        hyperparam_info = self.step_metadata['hyperparameters'][
            hyperparam_name]
        return hyperparam_info[
            'range'] if 'range' in hyperparam_info else hyperparam_info[
                'values']

    def get_hyperparam_default_val(self, hyperparam_name):
        hyperparam_info = self.step_metadata['hyperparameters'][
            hyperparam_name]
        return hyperparam_info[
            'default'] if 'default' in hyperparam_info else None
