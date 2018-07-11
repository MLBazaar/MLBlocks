# -*- coding: utf-8 -*-

"""Top-level package for MLBlocks."""

import os
import sys


__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com',
__version__ = '0.1.7-dev'


PRIMITIVES_PATHS = [
    os.path.join(os.getcwd(), 'mlprimitives'),
    os.path.join(sys.prefix, 'mlprimitives'),
]


def add_primitives_path(path):
    PRIMITIVES_PATHS.insert(0, path)


def get_primitive_path(name):
    """Locate the JSON definition of the given primitive."""

    parts = name.split('.')
    parts[-1] = '{}.{}'.format(parts[-1], 'json')
    for base_path in PRIMITIVES_PATHS:
        json_path = os.path.join(base_path, *parts)
        if os.path.isfile(json_path):
            return json_path

    raise ValueError("Unknown primitive: {}".format(name))
