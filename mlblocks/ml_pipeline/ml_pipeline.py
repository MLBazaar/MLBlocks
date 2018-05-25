import json
import os
from collections import OrderedDict, defaultdict

from mlblocks.json_parsers import keras_json_parser, ml_json_parser
from mlblocks.ml_pipeline.ml_block import MLBlock

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_JSON_DIR = os.path.join(_CURRENT_DIR, '../components/primitive_jsons')


class MLPipeline(object):
    """A pipeline that the DeepMining system can operate on.

    Attributes:
        blocks: A dictionary mapping this pipeline's block names to
                the actual MLBlock objects.
        BLOCKS: A list containing the blocks to load. Used in Subclasses.
    """

    BLOCKS = None

    @classmethod
    def _get_parser(cls, json_block_metadata):
        # TODO: Make this a MLBlock method
        # TODO: Implement better logic for deciding what parser to
        # use. Maybe some sort of config mapping parser to modules?
        # IDEA: organize the primitives by libraries, and decide
        # which parser to use depending on the JSON path
        parser = ml_json_parser.MLJsonParser(json_block_metadata)

        # For now, hardcode this logic for Keras.
        full_module_class = json_block_metadata['class']
        if full_module_class.startswith('keras.models.Sequential'):
            parser = keras_json_parser.KerasJsonParser(json_block_metadata)

        return parser

    @classmethod
    def get_block_path(cls, block_name):
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
    def load_block(cls, block):
        """Build block from either a Block name or a config dict.

        If a string is given, it is used to locate and load the corresponding
        JSON file, and then loaded.
        """
        if isinstance(block, str):
            block_path = cls.get_block_path(block)
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
                block = self.load_block(block)

            self.blocks[block.name] = block

    # @classmethod
    # def from_json_filepaths(cls, json_filepath_list):
    #     """Initialize a MLPipeline with a list of paths to JSON files.

    #     Args:
    #         json_filepath_list: A list of paths to JSON files
    #             representing the MLBlocks composing this pipeline.

    #     Returns:
    #         A MLPipeline defined by the JSON files.
    #     """
    #     loaded_json_metadata = []
    #     for json_filepath in json_filepath_list:
    #         with open(json_filepath, 'r') as f:
    #             json_metadata = json.load(f)
    #             loaded_json_metadata.append(json_metadata)

    #     return cls(loaded_json_metadata)

    # @classmethod
    # def from_ml_json(cls, json_names):
    #     """Initialize a MLPipeline with a list of block names.

    #     These block names should correspond to the JSON file names
    #     present in the components/primitive_jsons directory.

    #     Args:
    #         json_names: A list of primitive names corresponding to
    #             JSON files in components/primitive_jsons.

    #     Returns:
    #         A MLPipeline defined by the JSON primitive names.
    #     """
    #     json_filepaths = []
    #     for json_name in json_names:
    #         json_filename = '{}.{}'.format(json_name, 'json')
    #         path_to_json = os.path.join(_JSON_DIR, json_filename)
    #         if not os.path.isfile(path_to_json):
    #             error = ("No JSON corresponding to the specified "
    #                      "name ({}) exists.".format(json_name))
    #             raise ValueError(error)

    #         json_filepaths.append(path_to_json)

    #     return cls.from_json_filepaths(json_filepaths)

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

    def fit(self, x, y, fit_params=None):
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

        param_dict = defaultdict(dict)
        for key, value in fit_params.items():
            name, param = key
            param_dict[name][param] = fit_params[key]

        # Initially our transformed data is simply our input data.
        transformed_data = x
        for block_name, block in self.blocks.items():
            try:
                block.fit(transformed_data, y, **param_dict[block_name])
            except TypeError:
                # Some components only fit on an X.
                block.fit(transformed_data, **param_dict[block_name])

            transformed_data = block.produce(transformed_data)

    def predict(self, x):
        """Make predictions with this pipeline on the specified input data.

        fit() must be called at least once before predict().

        Args:
            x: Input data. Must fulfill input requirements of the first
                block of the pipeline.

        Returns:
            The predicted values.
        """
        transformed_data = x
        for block in self.blocks.values():
            transformed_data = block.produce(transformed_data)

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
