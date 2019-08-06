# -*- coding: utf-8 -*-

"""
Primitives and Pipelines discovery module.

This module contains functions to load primitive and pipeline
annotations, as well as to configure how MLBlocks finds the
primitives and pipelines.
"""

import json
import logging
import os
import re
import sys

import pkg_resources

LOGGER = logging.getLogger(__name__)

_PRIMITIVES_PATHS = [
    os.path.join(os.getcwd(), 'mlprimitives'),
    os.path.join(sys.prefix, 'mlprimitives'),
    os.path.join(os.getcwd(), 'mlblocks_primitives'),    # legacy
    os.path.join(sys.prefix, 'mlblocks_primitives'),    # legacy
]

_PIPELINES_PATHS = [
    os.path.join(os.getcwd(), 'mlpipelines'),
]


def _add_lookup_path(path, paths):
    """Add a new path to lookup.

    The new path will be inserted in the first place of the list,
    so any element found in this new folder will take precedence
    over any other element with the same name that existed in the
    system before.

    Args:
        path (str):
            path to add
        paths (list):
            list where the new path will be added.

    Raises:
        ValueError:
            A ``ValueError`` will be raised if the path is not valid.

    Returns:
        bool:
            Whether the new path was added or not.
    """
    if path not in paths:
        if not os.path.isdir(path):
            raise ValueError('Invalid path: {}'.format(path))

        paths.insert(0, os.path.abspath(path))
        return True

    return False


def add_primitives_path(path):
    """Add a new path to look for primitives.

    The new path will be inserted in the first place of the list,
    so any primitive found in this new folder will take precedence
    over any other primitive with the same name that existed in the
    system before.

    Args:
        path (str):
            path to add

    Raises:
        ValueError:
            A ``ValueError`` will be raised if the path is not valid.
    """
    added = _add_lookup_path(path, _PRIMITIVES_PATHS)
    if added:
        LOGGER.debug('New primitives path added: %s', path)


def add_pipelines_path(path):
    """Add a new path to look for pipelines.

    The new path will be inserted in the first place of the list,
    so any primitive found in this new folder will take precedence
    over any other pipeline with the same name that existed in the
    system before.

    Args:
        path (str):
            path to add

    Raises:
        ValueError:
            A ``ValueError`` will be raised if the path is not valid.
    """
    added = _add_lookup_path(path, _PIPELINES_PATHS)
    if added:
        LOGGER.debug('New pipelines path added: %s', path)


def _load_entry_points(entry_point_name, entry_point_group='mlblocks'):
    """Get a list of folders from entry points.

    This list will include the value of any entry point named after the given
    ``entry_point_name`` published under the given ``entry_point_group``.

    An example of such an entry point would be::

        entry_points = {
            'mlblocks': [
                'primitives=some_module:SOME_VARIABLE'
            ]
        }

    where the module ``some_module`` contains a variable such as::

        SOME_VARIABLE = os.path.join(os.path.dirname(__file__), 'jsons')

    Args:
        entry_point:
            The name of the ``entry_point`` to look for.

    Returns:
        list:
            The list of folders.
    """
    lookup_paths = list()
    entry_points = pkg_resources.iter_entry_points(entry_point_group)
    for entry_point in entry_points:
        if entry_point.name == entry_point_name:
            paths = entry_point.load()
            if isinstance(paths, str):
                lookup_paths.append(paths)
            elif isinstance(paths, (list, tuple)):
                lookup_paths.extend(paths)

    return lookup_paths


def get_primitives_paths():
    """Get the list of folders where primitives will be looked for.

    This list will include the values of all the entry points named ``primitives``
    published under the entry point group ``mlblocks``.

    Also, for backwards compatibility reasons, the paths from the entry points
    named ``jsons_path`` published under the ``mlprimitives`` group will also
    be included.

    An example of such an entry point would be::

        entry_points = {
            'mlblocks': [
                'primitives=some_module:SOME_VARIABLE'
            ]
        }

    where the module ``some_module`` contains a variable such as::

        SOME_VARIABLE = os.path.join(os.path.dirname(__file__), 'jsons')

    Returns:
        list:
            The list of folders.
    """
    paths = _load_entry_points('primitives') + _load_entry_points('jsons_path', 'mlprimitives')
    return _PRIMITIVES_PATHS + list(set(paths))


def get_pipelines_paths():
    """Get the list of folders where pipelines will be looked for.

    This list will include the values of all the entry points named ``pipelines``
    published under the entry point group ``mlblocks``.

    An example of such an entry point would be::

        entry_points = {
            'mlblocks': [
                'pipelines=some_module:SOME_VARIABLE'
            ]
        }

    where the module ``some_module`` contains a variable such as::

        SOME_VARIABLE = os.path.join(os.path.dirname(__file__), 'jsons')

    Returns:
        list:
            The list of folders.
    """
    return _PIPELINES_PATHS + _load_entry_points('pipelines')


def _load(name, paths):
    """Locate and load the JSON annotation in any of the given paths.

    All the given paths will be scanned to find a JSON file with the given name,
    and as soon as a JSON with the given name is found it is returned.

    Args:
        name (str):
            name of the JSON to look for. The name should not contain the
            ``.json`` extension, as it will be added dynamically.
        paths (list):
            list of paths where the primitives will be looked for.

    Returns:
        dict:
            The content of the JSON annotation file loaded into a dict.
    """
    for base_path in paths:
        parts = name.split('.')
        number_of_parts = len(parts)

        for folder_parts in range(number_of_parts):
            folder = os.path.join(base_path, *parts[:folder_parts])
            filename = '.'.join(parts[folder_parts:]) + '.json'
            json_path = os.path.join(folder, filename)

            if os.path.isfile(json_path):
                with open(json_path, 'r') as json_file:
                    LOGGER.debug('Loading %s from %s', name, json_path)
                    return json.load(json_file)


_PRIMITIVES = dict()


def load_primitive(name):
    """Locate and load the primitive JSON annotation.

    All the primitive paths will be scanned to find a JSON file with the given name,
    and as soon as a JSON with the given name is found it is returned.

    Args:
        name (str):
            name of the JSON to look for. The name should not contain the
            ``.json`` extension, as it will be added dynamically.

    Returns:
        dict:
            The content of the JSON annotation file loaded into a dict.

    Raises:
        ValueError:
            A ``ValueError`` will be raised if the primitive cannot be found.
    """
    primitive = _PRIMITIVES.get(name)
    if primitive is None:
        primitive = _load(name, get_primitives_paths())
        if primitive is None:
            raise ValueError("Unknown primitive: {}".format(name))

        _PRIMITIVES[name] = primitive

    return primitive


_PIPELINES = dict()


def load_pipeline(name):
    """Locate and load the pipeline JSON annotation.

    All the pipeline paths will be scanned to find a JSON file with the given name,
    and as soon as a JSON with the given name is found it is returned.

    Args:
        name (str):
            name of the JSON to look for. The name should not contain the
            ``.json`` extension, as it will be added dynamically.

    Returns:
        dict:
            The content of the JSON annotation file loaded into a dict.

    Raises:
        ValueError:
            A ``ValueError`` will be raised if the pipeline cannot be found.
    """
    pipeline = _PIPELINES.get(name)
    if pipeline is None:
        pipeline = _load(name, get_pipelines_paths())
        if pipeline is None:
            raise ValueError("Unknown pipeline: {}".format(name))

        _PIPELINES[name] = pipeline

    return pipeline


def _search_annotations(base_path, pattern, parts=None):
    annotations = dict()
    parts = parts or list()
    if os.path.exists(base_path):
        for name in os.listdir(base_path):
            path = os.path.abspath(os.path.join(base_path, name))
            if os.path.isdir(path):
                annotations.update(_search_annotations(path, pattern, parts + [name]))
            elif path not in annotations:
                name = '.'.join(parts + [name])
                if pattern.search(name) and name.endswith('.json'):
                    annotations[path] = name[:-5]

    return annotations


def _match_filter(annotation, key, value):
    if '.' in key:
        name, key = key.split('.', 1)
        part = annotation.get(name) or dict()
        return _match_filter(part, key, value)

    annotation_value = annotation.get(key)
    if not isinstance(annotation_value, type(value)):
        if isinstance(annotation_value, (list, dict)):
            return value in annotation_value
        elif isinstance(value, (list, dict)):
            return annotation_value in value

    return annotation_value == value


def _get_annotations_list(paths, loader, pattern, filters):
    pattern = re.compile(pattern)
    annotations = dict()
    for base_path in paths:
        annotations.update(_search_annotations(base_path, pattern))

    matching = list()
    for name in sorted(annotations.values()):
        annotation = loader(name)
        for key, value in filters.items():
            if not _match_filter(annotation, key, value):
                break

        else:
            matching.append(name)

    return matching


def get_primitives_list(pattern='', filters=None):
    filters = filters or dict()
    return _get_annotations_list(get_primitives_paths(), load_primitive, pattern, filters)


def get_pipelines_list(pattern='', filters=None):
    filters = filters or dict()
    return _get_annotations_list(get_pipelines_paths(), load_pipeline, pattern, filters)
