# -*- coding: utf-8 -*-

"""
Primitives module.

This module contains functions to load primitive annotations,
as well as to configure how MLBlocks finds the primitives.
"""

import json
import logging
import os
import sys

import pkg_resources

LOGGER = logging.getLogger(__name__)

_PRIMITIVES_PATHS = [
    os.path.join(os.getcwd(), 'mlprimitives'),
    os.path.join(sys.prefix, 'mlprimitives'),
    os.path.join(os.getcwd(), 'mlblocks_primitives'),    # legacy
    os.path.join(sys.prefix, 'mlblocks_primitives'),    # legacy
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

        LOGGER.debug('Adding new primitives path %s', path)
        _PRIMITIVES_PATHS.insert(0, os.path.abspath(path))


def get_primitives_paths():
    """Get the list of folders where the primitives will be looked for.

    This list will include the value of any `entry_point` named `jsons_path` published under
    the name `mlprimitives`.

    An example of such an entry point would be::

        entry_points = {
            'mlprimitives': [
                'jsons_path=some_module:SOME_VARIABLE'
            ]
        }

    where the module `some_module` contains a variable such as::

        SOME_VARIABLE = os.path.join(os.path.dirname(__file__), 'jsons')

    Returns:
        list:
            The list of folders.
    """

    primitives_paths = list()
    entry_points = pkg_resources.iter_entry_points('mlprimitives')
    for entry_point in entry_points:
        if entry_point.name == 'jsons_path':
            path = entry_point.load()
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
                    LOGGER.debug('Loading primitive %s from %s', name, json_path)
                    return json.load(json_file)

    raise ValueError("Unknown primitive: {}".format(name))
