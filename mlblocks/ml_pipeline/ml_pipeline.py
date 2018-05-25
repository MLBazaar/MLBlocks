import json
import os

from mlblocks.json_parsers import keras_json_parser, ml_json_parser


class MLPipeline(object):
    """A pipeline that the DeepMining system can operate on.

    Attributes:
        steps_dict: A dictionary mapping this pipeline's step names to
            DMSteps.
    """

    def __init__(self, steps, dataflow=None):
        """Initialize a DmPipeline with a list of corresponding DmSteps.

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
    def _get_parser(cls, json_block_metadata):
        # TODO: Implement better logic for deciding what parser to
        # use. Maybe some sort of config mapping parser to modules?
        parser = ml_json_parser.MLJsonParser(json_block_metadata)
        full_module_class = json_block_metadata['class']

        # For now, hardcode this logic for Keras.
        if full_module_class.startswith('keras.models.Sequential'):
            parser = keras_json_parser.KerasJsonParser(json_block_metadata)

        return parser

    @classmethod
    def from_json_metadata(cls, json_metadata):
        """Initialize a DmPipeline with a list of dicts defining DmSteps.

        Args:
            json_metadata: A list of dicts representing the
                DmSteps composing this pipeline.

        Returns:
            A DmPipeline defined by the provided dicts.
        """
        block_steps = []
        for json_md in json_metadata:
            parser = cls._get_parser(json_md)
            block_steps.append(parser.build_mlblock())

        return cls(block_steps)

    @classmethod
    def from_json_filepaths(cls, json_filepath_list):
        """Initialize a DmPipeline with a list of paths to JSON files.

        Args:
            json_filepath_list: A list of paths to JSON files
                representing the DmSteps composing this pipeline.

        Returns:
            A DmPipeline defined by the JSON files.
        """
        loaded_json_metadata = []
        for json_filepath in json_filepath_list:
            with open(json_filepath, 'r') as f:
                json_metadata = json.load(f)
                loaded_json_metadata.append(json_metadata)

        return cls.from_json_metadata(loaded_json_metadata)

    @classmethod
    def from_ml_json(cls, json_names):
        """Initialize a DmPipeline with a list of step names.

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
            json_filename = '{}.{}'.format(json_name, 'json')
            path_to_json = os.path.join(json_dir, json_filename)
            if not os.path.isfile(path_to_json):
                error = ("No JSON corresponding to the specified "
                         "name ({}) exists.".format(json_name))
                raise ValueError(error)

            json_filepaths.append(path_to_json)

        return cls.from_json_filepaths(json_filepaths)

    def update_fixed_hyperparams(self, fixed_hyperparams):
        """Update the specified fixed hyperparameters of this pipeline.

        Args:
            fixed_hyperparams: A dict mapping
                (step name, fixed hyperparam name) pairs to their
                corresponding values.
        """
        for step_name, hyperparam_name in fixed_hyperparams:
            step = self.steps_dict[step_name]
            hyperparam = fixed_hyperparams[(step_name, hyperparam_name)]
            step.fixed_hyperparams[hyperparam_name] = hyperparam

            # Update the hyperparams in the actual model as well.
            step.update_model(step.fixed_hyperparams, step.tunable_hyperparams)

    def update_tunable_hyperparams(self, tunable_hyperparams):
        """Update the specified tunable hyperparameters of this pipeline.

        Unspecified hyperparameters are not affected.

        Args:
            tunable_hyperparams: A list of MLHyperparams to update.
        """
        for hyperparam in tunable_hyperparams:
            step_name = hyperparam.step_name
            step = self.steps_dict[step_name]
            step.tunable_hyperparams[hyperparam.param_name] = hyperparam
            # Update the hyperparams in the actual model as well.
            step.update_model(step.fixed_hyperparams, step.tunable_hyperparams)

    def get_fixed_hyperparams(self):
        """Get all the fixed hyperparameters belonging to this pipeline.

        Returns:
            A dict mapping (step name, fixed hyperparam name) pairs to
            fixed hyperparam values.
        """
        fixed_hyperparams = {}
        for step in self.steps_dict.values():
            for hp_name in step.fixed_hyperparams:
                hyperparam = step.fixed_hyperparams[hp_name]
                fixed_hyperparams[(step.name, hp_name)] = hyperparam

        return fixed_hyperparams

    def get_tunable_hyperparams(self):
        """Get all tunable hyperparameters belonging to this pipeline.

        Returns:
            A list of tunable hyperparameters belonging to this
            pipeline.
        """
        tunable_hyperparams = []
        for step in self.steps_dict.values():
            tunable_hyperparams += list(step.tunable_hyperparams.values())

        return tunable_hyperparams

    def set_from_hyperparam_dict(self, hyperparam_dict):
        """Set the hyperparameters of this pipeline from a dict.

        This dict maps as follows:
            (step name, hyperparam name): value

        Args:
            hyperparam_dict: A dict mapping (step name, hyperparam name)
                tuples to hyperparam values.
        """
        all_tunable_hyperparams = self.get_tunable_hyperparams()
        for hp in all_tunable_hyperparams:
            if (hp.step_name, hp.param_name) in hyperparam_dict:
                hp.value = hyperparam_dict[(hp.step_name, hp.param_name)]

        self.update_tunable_hyperparams(all_tunable_hyperparams)

    def fit(self, x, y, fit_params=None, produce_params=None):
        """Fit this pipeline to the specified training data.

        Args:
            x: Training data. Must fulfill input requirements of the
                first step of the pipeline.
            y: Training targets. Must fulfill label requirements for
                all steps of the pipeline.
            fit_params: Any params to pass into fit.
                In the form {(step name, param name): param value}
        """
        if fit_params is None:
            fit_params = {}
        if produce_params is None:
            produce_params = {}

        fit_param_dict = {step_name: {} for step_name in self.dataflow}
        for key, value in fit_params.items():
            name, param = key
            fit_param_dict[name][param] = fit_params[key]

        produce_param_dict = {step_name: {} for step_name in self.dataflow}
        for key, value in produce_params.items():
            name, param = key
            produce_param_dict[name][param] = produce_params[key]

        # Initially our transformed data is simply our input data.
        transformed_data = x
        for step_name in self.dataflow:
            step = self.steps_dict[step_name]
            try:
                step.fit(transformed_data, y, **fit_param_dict[step_name])
            except TypeError:
                # Some components only fit on an X.
                step.fit(transformed_data, **fit_param_dict[step_name])

            transformed_data = step.produce(transformed_data, **produce_param_dict[step_name])

    def predict(self, x, predict_params=None):
        """Make predictions with this pipeline on the specified input data.

        fit() must be called at least once before predict().

        Args:
            x: Input data. Must fulfill input requirements of the first
                step of the pipeline.

        Returns:
            The predicted values.
        """
        if predict_params is None:
            predict_params = {}

        param_dict = {step_name: {} for step_name in self.dataflow}
        for key, value in predict_params.items():
            name, param = key
            param_dict[name][param] = predict_params[key]

        transformed_data = x
        for step_name in self.dataflow:
            step = self.steps_dict[step_name]
            transformed_data = step.produce(transformed_data, **param_dict[step_name])

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
