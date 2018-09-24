# -*- coding: utf-8 -*-

"""
MLBlocks is a simple framework for composing end-to-end tunable Machine Learning Pipelines by
seamlessly combining tools from any python library with a simple, common and uniform interface.

* Free software: MIT license
* Documentation: https://HDI-Project.github.io/MLBlocks
"""

import json
import os
import sys

from mlblocks.mlblock import MLBlock
from mlblocks.mlpipeline import MLPipeline

__author__ = 'MIT Data To AI Lab'
__copyright__ = 'Copyright (c) 2018, MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__license__ = 'MIT'
__version__ = '0.2.1'

__all__ = [
    'MLBlock', 'MLPipeline', 'add_primitives_path',
    'get_primitives_paths', 'load_primitive'
]


_PRIMITIVES_PATHS = [
    os.path.join(os.getcwd(), 'mlblocks_primitives'),
    os.path.join(sys.prefix, 'mlblocks_primitives'),
]


def add_primitives_path(path):
    """Add a new path to look for primitives.

    The new path will be inserted in the first place of the list,
    so any primitive found in this new folder will take precedence
    over any other primitive with the same name that existed in the
    system before.

    Args:
        path (str): path to add

    Raises:
        ValueError: A `ValueError` will be raised if the path is not valid.
    """
    if path not in _PRIMITIVES_PATHS:
        if not os.path.isdir(path):
            raise ValueError('Invalid path: {}'.format(path))

        _PRIMITIVES_PATHS.insert(0, os.path.abspath(path))


def get_primitives_paths():
    """Get the list of folders where the primitives will be looked for.

    Returns:
        list: The list of folders.
    """
    return _PRIMITIVES_PATHS


def load_primitive(name):
    """Locate and load the JSON annotation of the given primitive.

    All the paths found in PRIMTIVE_PATHS will be scanned to find a JSON file
    with the given name, and as soon as a JSON with the given name is found it
    is returned.

    Args:
        name (str): name of the primitive to look for. The name should
                    correspond to the primitive, not to the filename, as the
                    `.json` extension will be added dynamically.

    Returns:
        dict: The content of the JSON annotation file loaded into a dict.

    Raises:
        ValueError: A `ValueError` will be raised if the primitive cannot be
                    found.
    """

    for base_path in _PRIMITIVES_PATHS:
        json_path = os.path.join(base_path, name + '.json')
        if os.path.isfile(json_path):
            with open(json_path, 'r') as json_file:
                return json.load(json_file)

    raise ValueError("Unknown primitive: {}".format(name))
