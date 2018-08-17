# -*- coding: utf-8 -*-

"""Top-level package for MLBlocks."""

import os
import sys

from mlblocks.mlblock import MLBlock
from mlblocks.mlpipeline import MLPipeline

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com',
__version__ = '0.2.0'

__all__ = [
    'MLBlock', 'MLPipeline', 'PRIMITIVES_PATHS',
    'add_primitives_path', 'get_primitives_paths'
]


PRIMITIVES_PATHS = [
    os.path.join(os.getcwd(), 'mlblocks_primitives'),
    os.path.join(sys.prefix, 'mlblocks_primitives'),
]


def add_primitives_path(path):
    if path not in PRIMITIVES_PATHS:
        PRIMITIVES_PATHS.insert(0, path)


def get_primitive_path(name):
    """Locate the JSON definition of the given primitive."""

    for base_path in PRIMITIVES_PATHS:
        json_path = os.path.join(base_path, name + '.json')
        if os.path.isfile(json_path):
            return json_path

    raise ValueError("Unknown primitive: {}".format(name))
