import json
import os
import numpy as np

from json_parsers.ml_json_parser import MLJsonParser


class MLPipeline(object):
    """A pipeline that the DeepMining system can operate on.

    Attributes:
        steps_dict: A dictionary mapping this pipeline's step names to
            DMSteps.
    """

    def __init__(self, steps, dataflow=None):
        """Initializes a DmPipeline with a list of corresponding
        DmSteps.

        Args:
            steps: A list of DmSteps composing this pipeline.
        """
        # Contains the actual primitives.
        self.steps_dict = {
            k: v
            for (k, v) in [(step.name, step) for step in steps]
        }

        # For now, just use a list to order the steps.
        self.dataflow = dataflow if dataflow is not None else [
            step.name for step in steps
        ]

    @classmethod
    def from_json_metadata(cls, json_metadata):
        """Initializes a DmPipeline with a list of JSON metadata
        defining DmSteps.

        Args:
            json_metadata: A list of JSON objects representing the
                DmSteps composing this pipeline.

        Returns:
            A DmPipeline defined by the JSON steps.
        """
        block_steps = []
        for json_md in json_metadata:
            parser = MLJsonParser(json_md)
            parser.block_json = json_md
            block_steps.append(parser.build_mlblock())

        return cls(block_steps)

    @classmethod
    def from_json_filepaths(cls, json_filepath_list):
        """Initializes a DmPipeline with a list of paths to JSON files
        defining DmSteps.

        Args:
            json_filepath_list: A list of paths to JSON files
                representing the DmSteps composing this pipeline.

        Returns:
            A DmPipeline defined by the JSON files.
        """
        loaded_json_metadata = [
            json.load(open(json_filepath))
            for json_filepath in json_filepath_list
        ]
        return cls.from_json_metadata(loaded_json_metadata)

    @classmethod
    def from_dm_json(cls, json_names):
        """Initializes a DmPipeline with a list of step names.

        These step names should correspond to the JSON file names
        present in the components/primitive_jsons directory.

        Args:
            json_names: A list of primitive names corresponding to
                JSON files in components/primitive_jsons.

        Returns:
            A DmPipeline defined by the JSON primitive names.
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        json_dir = os.path.join(current_dir, '../components/primitive_jsons')

        json_filepaths = []
        for json_name in json_names:
            path_to_json = os.path.join(json_dir, '%s.%s' % (json_name,
                                                             'json'))
            if not os.path.isfile(path_to_json):
                raise ValueError(
                    "No JSON corresponding to the specified name (%s) exists."
                    % json_name)
            json_filepaths.append(path_to_json)

        return cls.from_json_filepaths(json_filepaths)

    def update_hyperparams(self, hyperparams):
        """Updates the specified hyperparameters of this pipeline.

        Unspecified hyperparameters are not affected.

        Args:
            hyperparams: A list of MLHyperparams to update.
        """
        for hyperparam in hyperparams:
            step_name = hyperparam.step_name
            step = self.steps_dict[step_name]
            step.tunable_hyperparams[hyperparam.param_name] = hyperparam
        
        # Update the hyperparams in the actual model as well.
        for step in self.steps_dict.values():
            step.model = step.model.__class__(**step.tunable_hyperparams)

    def get_tunable_hyperparams(self):
        """Gets all tunable hyperparameters belonging to this pipeline.

        Returns:
            A list of tunable hyperparameters belonging to this
            pipeline.
        """
        tunable_hyperparams = []
        for step in self.steps_dict.values():
            tunable_hyperparams += list(step.tunable_hyperparams.values())
        return tunable_hyperparams

    def set_from_hyperparam_dict(self, hyperparam_dict):
        """Sets the hyperparameters of this pipeline from a name: value
        mapping.

        Args:
            hyperparam_dict: A dict mapping hyperparam names to values.
        """
        all_tunable_hyperparams = self.get_tunable_hyperparams()
        for hp in all_tunable_hyperparams:
            if hp.param_name in hyperparam_dict:
                param_value = hyperparam_dict[hp.param_name]
                if hp.param_type in ("int_cat", "float_cat", "string"):
                    param_value = np.asscalar(param_value)
                hp.value = param_value
        self.update_hyperparams(all_tunable_hyperparams)

    def fit(self, x, y):
        """Fits this pipeline to the specified training data.

        Args:
            x: Training data. Must fulfill input requirements of the
                first step of the pipeline.
            y: Training targets. Must fulfill label requirements for
                all steps of the pipeline.
        """
        # Initially our transformed data is simply our input data.
        transformed_data = x
        for step_name in self.dataflow:
            step = self.steps_dict[step_name]
            step.fit(transformed_data, y)
            transformed_data = step.produce(transformed_data)

    def predict(self, x):
        """Makes predictions with this pipeline on the specified input
        data.

        fit() must be called at least once before predict().

        Args:
            x: Input data. Must fulfill input requirements of the first
                step of the pipeline.

        Returns:
            The predicted values.
        """
        transformed_data = x
        for step_name in self.dataflow:
            step = self.steps_dict[step_name]
            transformed_data = step.produce(transformed_data)

        # The last value stored in transformed_data is our final output value.
        return transformed_data

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        all_tunable_hyperparams = self.get_tunable_hyperparams()
        return {
            '{0}__{1}'.format(hyperparam.step_name, hyperparam.param_name):
            hyperparam.value
            for hyperparam in all_tunable_hyperparams
        }
