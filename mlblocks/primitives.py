# -*- coding: utf-8 -*-

"""
Primitives module.

This module contains functions to load primitive annotations,
as well as to configure how MLBlocks finds the primitives.
"""

import json
import os
import sys

import pkg_resources

_PRIMITIVES_FOLDER_NAME = 'mlprimitives'
_OLD_PRIMITIVES_FOLDER_NAME = 'mlblocks_primitives'
_PRIMITIVES_PATHS = [
    os.path.join(os.getcwd(), _PRIMITIVES_FOLDER_NAME),
    os.path.join(os.getcwd(), _OLD_PRIMITIVES_FOLDER_NAME),    # legacy
    os.path.join(sys.prefix, _OLD_PRIMITIVES_FOLDER_NAME),    # legacy
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
        list:
            The list of folders.
    """

    primitives_paths = list()
    for entry_point in pkg_resources.iter_entry_points(_PRIMITIVES_FOLDER_NAME):
        module_path = os.path.join(*entry_point.module_name.split('.'))
        path = pkg_resources.resource_filename(entry_point.name, module_path)
        primitives_paths.append(path)

    return _PRIMITIVES_PATHS + primitives_paths


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
        dict:
            The content of the JSON annotation file loaded into a dict.

    Raises:
        ValueError: A `ValueError` will be raised if the primitive cannot be
                    found.
    """

    for base_path in get_primitives_paths():
        parts = name.split('.')
        number_of_parts = len(parts)

        for folder_parts in range(number_of_parts):
            folder = os.path.join(base_path, *parts[:folder_parts])
            filename = '.'.join(parts[folder_parts:]) + '.json'
            json_path = os.path.join(folder, filename)

            if os.path.isfile(json_path):
                with open(json_path, 'r') as json_file:
                    return json.load(json_file)

    raise ValueError("Unknown primitive: {}".format(name))
