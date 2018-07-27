import json
import logging
import os
from collections import OrderedDict, defaultdict

from mlblocks.mlblock import MLBlock
from mlblocks.parsers.json import MLJsonParser
from mlblocks.parsers.keras import KerasJsonParser

LOGGER = logging.getLogger(__name__)

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_JSON_DIR = os.path.join(_CURRENT_DIR, 'primitives/jsons')


class MLPipeline(object):
    """A pipeline that the DeepMining system can operate on.

    Attributes:
        blocks: A dictionary mapping this pipeline's block names to
                the actual MLBlock objects.
        BLOCKS: A list containing the blocks to load. Used in Subclasses.
    """

    BLOCKS = None

    @classmethod
    def _get_parser(cls, metadata, init_params):
        # TODO: Make this a MLBlock method
        # TODO: Implement better logic for deciding what parser to
        # use. Maybe some sort of config mapping parser to modules?
        # IDEA: organize the primitives by libraries, and decide
        # which parser to use depending on the JSON path

        # For now, hardcode this logic for Keras.
        if metadata['class'] == 'keras.models.Sequential':
            parser_class = KerasJsonParser
        else:
            parser_class = MLJsonParser

        return parser_class(metadata, init_params)

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
    def _load_block(cls, block, init_params):
        """Build block from either a Block name or a config dict.

        If a string is given, it is used to locate and load the corresponding
        JSON file, and then loaded.
        """
        if isinstance(block, str):
            block_path = cls._get_block_path(block)
            with open(block_path, 'r') as f:
                block = json.load(f)

        parser = cls._get_parser(block, init_params)
        return parser.build_mlblock()

    @staticmethod
    def get_nested(params):
        nested_params = defaultdict(dict)

        if params:
            for key, value in params.items():
                if isinstance(key, tuple):
                    name, param = key
                    nested_params[name][param] = value
                else:
                    # already nested
                    nested_params[key] = value

        return nested_params

    @classmethod
    def get_nested_hyperparams(cls, hyperparams):
        if isinstance(hyperparams, dict):
            return cls.get_nested(hyperparams)

        else:
            nested_hyperparams = defaultdict(dict)
            for hyperparam in hyperparams:
                block_name = hyperparam.block_name
                param_name = hyperparam.param_name
                nested_hyperparams[block_name][param_name] = hyperparam

            return nested_hyperparams

    def __init__(self, blocks=None, init_params=None):
        """Initialize a MLPipeline with a list of corresponding MLBlocks.

        Args:
            blocks: A list of MLBlocks composing this pipeline. MLBlocks
                    can be either MLBlock instances or primitive names to
                    load from the configuration JSON files.
        """

        blocks = blocks or self.BLOCKS
        init_params = self.get_nested(init_params)

        if not blocks:
            raise ValueError("At least one block is needed")

        self.blocks = OrderedDict()
        for block in blocks:
            if not isinstance(block, MLBlock):
                block = self._load_block(block, init_params)

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
        # group by block
        hyperparams = self.get_nested_hyperparams(tunable_hyperparams)

        # update each block in one shot
        for block_name, block_params in hyperparams.items():
            block = self.blocks[block_name]
            block.tunable_hyperparams.update(block_params)

            # Update the hyperparams in the actual model as well.
            block.update_model(block.fixed_hyperparams, block.tunable_hyperparams)

    def set_from_hyperparam_dict(self, hyperparams_dict):
        """Set the hyperparameters of this pipeline from a dict.

        This dict maps as follows:
            (block name, hyperparam name): value

        Args:
            hyperparam_dict: A dict mapping (block name, hyperparam name)
                tuples to hyperparam values.
        """
        hyperparams = self.get_nested_hyperparams(self.get_tunable_hyperparams())
        hyperparams_dict = self.get_nested(hyperparams_dict)

        for block, params in hyperparams_dict.items():
            for param, value in params.items():
                hyperparams[block][param].value = value

        self.update_tunable_hyperparams(hyperparams)

    def fit(self, x, y, fit_params=None, predict_params=None):
        """Fit this pipeline to the specified training data.

        Args:
            x: Training data. Must fulfill input requirements of the
                first block of the pipeline.
            y: Training targets. Must fulfill label requirements for
                all blocks of the pipeline.
            fit_params: Any params to pass into fit.
                In the form {(block name, param name): param value}
        """
        fit_params = self.get_nested(fit_params)
        predict_params = self.get_nested(predict_params)

        # Initially our transformed data is simply our input data.
        transformed_data = x
        for index, (block_name, block) in enumerate(self.blocks.items()):
            block_fit_params = fit_params[block_name]
            try:
                LOGGER.debug("Fitting block %s", block_name)
                block.fit(transformed_data, y, **block_fit_params)
            except TypeError:
                # Some blocks only fit on an X.
                LOGGER.debug("TypeError on block %s. Retrying without `y`", block_name)
                block.fit(transformed_data, **block_fit_params)

            LOGGER.debug("Producing block %s", block_name)
            if len(self.blocks) > index:
                transformed_data = block.produce(transformed_data, **predict_params[block_name])

    def predict(self, x, predict_params=None):
        """Make predictions with this pipeline on the specified input data.

        fit() must be called at least once before predict().

        Args:
            x: Input data. Must fulfill input requirements of the first
                block of the pipeline.

        Returns:
            The predicted values.
        """
        predict_params = self.get_nested(predict_params)

        transformed_data = x
        for block_name, block in self.blocks.items():
            LOGGER.debug("Producing block %s", block_name)
            transformed_data = block.produce(transformed_data, **predict_params[block_name])

        # The last value stored in transformed_data is our final output value.
        return transformed_data

    def to_nested_dicts(self):
        return {
            block.name: block.tunable_hyperparams
            for block in self.blocks.values()
        }

    def to_dict(self):
        all_tunable_hyperparams = self.get_tunable_hyperparams()
        return {
            '{0}__{1}'.format(hyperparam.block_name, hyperparam.param_name):
            hyperparam.value
            for hyperparam in all_tunable_hyperparams
        }

    def __str__(self):
        return str(self.to_dict())
