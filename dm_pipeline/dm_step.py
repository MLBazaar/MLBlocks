from dm_hyperparam import DmHyperparam
from json_parsers.dm_json_parser import DmJsonParser


class DmStep(object):
    """A single primitive in a pipeline that the DeepMining system can
    act on.

    Attributes:
        name: The name of this step primitive.
        step_model: The class representing this step primitive.
            e.g. sklearn.ensemble.RandomForestClassifier
        step_instance: An instance of the class representing this step
            primitive. Set with the current hyperparameter values for
            this step.
        fit_func: The actual function this step primitive uses to fit.
        produce_func: The actual function this step primitive uses to
            produce new data. This function generally produces
            transformed data or predictions (final step of a pipeline).
        tunable_hyperparams: A dictionary mapping this step's
            hyperparameter names to the corresponding Hyperparameter
            objects.
    """

    def __init__(self, name, module_class, fit_func, produce_func):
        """Initializes this DmStep.

        Note that on initialization, no tunable hyperparameters are
        specified. Add them with the set_tunable_hyperparam method.

        Args:
            name: The name of this DmStep primitive.
            module_class: The full module identifying the class
                representing this step primitive.
                e.g. sklearn.ensemble.RandomForestClassifier
            fit_func: The actual function this step primitive uses to
                fit.
                e.g. for sklearn.ensemble.RandomForestClassifier, "fit"
            produce_func: The actual function this step primitive uses
                to produce new data.
                e.g. for sklearn.ensenble.RandomForestClassifier,
                "predict".
        """
        self.name = name

        self.step_model = module_class
        self.step_instance = None

        self.fit_func = fit_func
        self.produce_func = produce_func

        self.tunable_hyperparams = {}

    @classmethod
    def from_json(cls, json_metadata, parser=None):
        """Initializes a DmStep from a JSON file.

        See components/primitive_jsons for JSON examples.

        Args:
            json_metadata: A loaded JSON object (python dict) to
                initialize a DmStep with.
            parser: The parser to use to load the provided JSON object.
                Defaults to the DmJsonParser if not specified.

        Returns:
            A DmStep as specified by the JSON metadata.
        """
        # Use the base DeepMining JSON parser by default.
        if parser is None:
            parser = DmJsonParser()

        parser.set_json_metadata(json_metadata)
        name = parser.get_name()
        module_class = parser.get_class()
        fit_func = parser.get_fit_method()
        produce_func = parser.get_produce_method()
        dm_step_instance = cls(name, module_class, fit_func, produce_func)

        # Set hyperparameters.
        root_hyperparam_lookup = set(parser.get_root_hyperparams())
        for hp_name in parser.get_hyperparam_names():
            hp_type = parser.get_hyperparam_type(hp_name)
            hp_range = parser.get_hyperparam_range(hp_name)
            hp_val = parser.get_hyperparam_default_val(hp_name)
            hp_is_cond = hp_name not in root_hyperparam_lookup
            hyperparam = DmHyperparam(hp_name, hp_type, hp_range, hp_is_cond,
                                      hp_val)
            hyperparam.step_name = dm_step_instance.name
            dm_step_instance.tunable_hyperparams[hp_name] = hyperparam

        return dm_step_instance
