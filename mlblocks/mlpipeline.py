import json
import os
from collections import OrderedDict, defaultdict

from mlblocks.mlblock import MLBlock
from mlblocks.parsers.json import MLJsonParser
from mlblocks.parsers.keras import KerasJsonParser

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_JSON_DIR = os.path.join(_CURRENT_DIR, 'primitives')


class MLPipeline(object):
    """A pipeline that the DeepMining system can operate on.

    Attributes:
        blocks: A dictionary mapping this pipeline's block names to
                the actual MLBlock objects.
        BLOCKS: A list containing the blocks to load. Used in Subclasses.
    """

    BLOCKS = None

    @classmethod
    def _get_parser(cls, metadata):
        # TODO: Make this a MLBlock method
        # TODO: Implement better logic for deciding what parser to
        # use. Maybe some sort of config mapping parser to modules?
        # IDEA: organize the primitives by libraries, and decide
        # which parser to use depending on the JSON path

        # For now, hardcode this logic for Keras.
        if metadata['class'] == 'keras.models.Sequential':
            parser = KerasJsonParser(metadata)
        else:
            parser = MLJsonParser(metadata)

        return parser

    @classmethod
    def _get_block_path(cls, block_name):
        """Locate the JSON file of the given primitive."""

        if os.path.isfile(block_name):
            return block_name

        json_filename = '{}.{}'.format(block_name, 'json')
        block_path = os.path.join(_JSON_DIR, json_filename)

        if not os.path.isfile(block_path):
            error = ("No JSON corresponding to the specified "
                     "name ({}) exists.".format(block_name))
            raise ValueError(error)

        return block_path

    @classmethod
    def _load_block(cls, block):
        """Build block from either a Block name or a config dict.

        If a string is given, it is used to locate and load the corresponding
        JSON file, and then loaded.
        """
        if isinstance(block, str):
            block_path = cls._get_block_path(block)
            with open(block_path, 'r') as f:
                block = json.load(f)

        parser = cls._get_parser(block)
        return parser.build_mlblock()

    def __init__(self, blocks=None):
        """Initialize a MLPipeline with a list of corresponding MLBlocks.

        Args:
            blocks: A list of MLBlocks composing this pipeline. MLBlocks
                    can be either MLBlock instances or primitive names to
                    load from the configuration JSON files.
        """

        blocks = blocks or self.BLOCKS

        if not blocks:
            raise ValueError("At least one block is needed")

        self.blocks = OrderedDict()
        for block in blocks:
            if not isinstance(block, MLBlock):
                block = self._load_block(block)

            self.blocks[block.name] = block

    def get_fixed_hyperparams(self):
        """Get all the fixed hyperparameters belonging to this pipeline.

        Returns:
            A dict mapping (block name, fixed hyperparam name) pairs to
            fixed hyperparam values.
        """
        fixed_hyperparams = {}
        for block in self.blocks.values():
            for hp_name in block.fixed_hyperparams:
                hyperparam = block.fixed_hyperparams[hp_name]
                fixed_hyperparams[(block.name, hp_name)] = hyperparam

        return fixed_hyperparams

    def update_fixed_hyperparams(self, fixed_hyperparams):
        """Update the specified fixed hyperparameters of this pipeline.

        Args:
            fixed_hyperparams: A dict mapping
                (block name, fixed hyperparam name) pairs to their
                corresponding values.
        """
        for block_name, hyperparam_name in fixed_hyperparams:
            block = self.blocks[block_name]
            hyperparam = fixed_hyperparams[(block_name, hyperparam_name)]
            block.fixed_hyperparams[hyperparam_name] = hyperparam

            # Update the hyperparams in the actual model as well.
            block.update_model(block.fixed_hyperparams, block.tunable_hyperparams)

    def get_tunable_hyperparams(self):
        """Get all tunable hyperparameters belonging to this pipeline.

        Returns:
            A list of tunable hyperparameters belonging to this
            pipeline.
        """
        tunable_hyperparams = []
        for block in self.blocks.values():
            tunable_hyperparams += list(block.tunable_hyperparams.values())

        return tunable_hyperparams

    def update_tunable_hyperparams(self, tunable_hyperparams):
        """Update the specified tunable hyperparameters of this pipeline.

        Unspecified hyperparameters are not affected.

        Args:
            tunable_hyperparams: A list of MLHyperparams to update.
        """
        for hyperparam in tunable_hyperparams:
            block_name = hyperparam.block_name
            block = self.blocks[block_name]
            block.tunable_hyperparams[hyperparam.param_name] = hyperparam
            # Update the hyperparams in the actual model as well.
            block.update_model(block.fixed_hyperparams, block.tunable_hyperparams)

    def set_from_hyperparam_dict(self, hyperparam_dict):
        """Set the hyperparameters of this pipeline from a dict.

        This dict maps as follows:
            (block name, hyperparam name): value

        Args:
            hyperparam_dict: A dict mapping (block name, hyperparam name)
                tuples to hyperparam values.
        """
        all_tunable_hyperparams = self.get_tunable_hyperparams()
        for hp in all_tunable_hyperparams:
            if (hp.block_name, hp.param_name) in hyperparam_dict:
                hp.value = hyperparam_dict[(hp.block_name, hp.param_name)]

        self.update_tunable_hyperparams(all_tunable_hyperparams)

    def fit(self, x, y, fit_params=None, produce_params=None):
        """Fit this pipeline to the specified training data.

        Args:
            x: Training data. Must fulfill input requirements of the
                first block of the pipeline.
            y: Training targets. Must fulfill label requirements for
                all blocks of the pipeline.
            fit_params: Any params to pass into fit.
                In the form {(block name, param name): param value}
        """
        if fit_params is None:
            fit_params = {}
        if produce_params is None:
            produce_params = {}

        fit_param_dict = defaultdict(dict)
        for key, value in fit_params.items():
            name, param = key
            fit_param_dict[name][param] = fit_params[key]

        produce_param_dict = defaultdict(dict)
        for key, value in produce_params.items():
            name, param = key
            produce_param_dict[name][param] = produce_params[key]

        # Initially our transformed data is simply our input data.
        transformed_data = x
        for block_name, block in self.blocks.items():
            try:
                block.fit(transformed_data, y, **fit_param_dict[block_name])
            except TypeError:
                # Some blocks only fit on an X.
                block.fit(transformed_data, **fit_param_dict[block_name])

            transformed_data = block.produce(
                transformed_data, **produce_param_dict[block_name])

    def predict(self, x, predict_params=None):
        """Make predictions with this pipeline on the specified input data.

        fit() must be called at least once before predict().

        Args:
            x: Input data. Must fulfill input requirements of the first
                block of the pipeline.

        Returns:
            The predicted values.
        """
        if predict_params is None:
            predict_params = {}

        param_dict = defaultdict(dict)
        for key, value in predict_params.items():
            name, param = key
            param_dict[name][param] = predict_params[key]

        transformed_data = x
        for block_name, block in self.blocks.items():
            transformed_data = block.produce(
                transformed_data, **param_dict[block_name])

        # The last value stored in transformed_data is our final output value.
        return transformed_data

    def to_dict(self):
        all_tunable_hyperparams = self.get_tunable_hyperparams()
        return {
            '{0}__{1}'.format(hyperparam.block_name, hyperparam.param_name):
            hyperparam.value
            for hyperparam in all_tunable_hyperparams
        }

    def __str__(self):
        return str(self.to_dict())
