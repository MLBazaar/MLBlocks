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


def _load_json(json_path):
    with open(json_path, 'r') as json_file:
        LOGGER.debug('Loading %s', json_path)
        return json.load(json_file)


def _load(name, paths):
    """Locate and load the JSON annotation in any of the given paths.

    All the given paths will be scanned to find a JSON file with the given name,
    and as soon as a JSON with the given name is found it is returned.

    Args:
        name (str):
            Path to a JSON file or name of the JSON to look for withouth the ``.json`` extension.
        paths (list):
            list of paths where the primitives will be looked for.

    Returns:
        dict:
            The content of the JSON annotation file loaded into a dict.
    """
    if os.path.isfile(name):
        return _load_json(name)

    for base_path in paths:
        parts = name.split('.')
        number_of_parts = len(parts)

        for folder_parts in range(number_of_parts):
            folder = os.path.join(base_path, *parts[:folder_parts])
            filename = '.'.join(parts[folder_parts:]) + '.json'
            json_path = os.path.join(folder, filename)

            if os.path.isfile(json_path):
                return _load_json(json_path)


def load_primitive(name):
    """Locate and load the primitive JSON annotation.

    All the primitive paths will be scanned to find a JSON file with the given name,
    and as soon as a JSON with the given name is found it is returned.

    Args:
        name (str):
            Path to a JSON file or name of the JSON to look for withouth the ``.json`` extension.

    Returns:
        dict:
            The content of the JSON annotation file loaded into a dict.

    Raises:
        ValueError:
            A ``ValueError`` will be raised if the primitive cannot be found.
    """
    primitive = _load(name, get_primitives_paths())
    if primitive is None:
        raise ValueError("Unknown primitive: {}".format(name))

    return primitive


def load_pipeline(name):
    """Locate and load the pipeline JSON annotation.

    All the pipeline paths will be scanned to find a JSON file with the given name,
    and as soon as a JSON with the given name is found it is returned.

    Args:
        name (str):
            Path to a JSON file or name of the JSON to look for withouth the ``.json`` extension.

    Returns:
        dict:
            The content of the JSON annotation file loaded into a dict.

    Raises:
        ValueError:
            A ``ValueError`` will be raised if the pipeline cannot be found.
    """
    pipeline = _load(name, get_pipelines_paths())
    if pipeline is None:
        raise ValueError("Unknown pipeline: {}".format(name))

    return pipeline


def _search_annotations(base_path, pattern, parts=None):
    """Search for annotations within the given path.

    If the indicated path has subfolders, search recursively within them.

    If a pattern is given, return only the annotations whose name
    matches the pattern.

    Args:
        base_path (str):
            path to the folder to be searched for annotations.
        pattern (str):
            Regular expression to search in the annotation names.
        parts (list):
            Optional. List containing the parent folders that are also part
            of the annotation name. Used during recursion to be able to
            build the final annotation name before returning it.

    Returns:
        dict:
            dictionary containing paths as keys and annotation names as
            values.
    """
    pattern = re.compile(pattern)
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


def _match(annotation, key, values):
    """Check if the anotation has the key and it matches any of the values.

    If the given key is not found but it contains dots, split by the dots
    and consider each part a sublevel in the annotation.

    If the key value within the annotation is a list or a dict, check
    whether any of the given values is contained within it instead of
    checking for equality.

    Args:
        annotation (dict):
            Dictionary annotation.
        key (str):
            Key to search within the annoation. It can contain dots to
            separated nested subdictionary levels within the annotation.
        values (object or list):
            Value or list of values to search for.

    Returns:
        bool:
            whether there is a match or not.
    """
    if not isinstance(values, list):
        values = [values]

    if key not in annotation:
        if '.' in key:
            name, key = key.split('.', 1)
            part = annotation.get(name) or dict()
            return _match(part, key, values)
        else:
            return False

    annotation_value = annotation[key]

    for value in values:
        if isinstance(annotation_value, (list, dict)):
            return value in annotation_value
        elif annotation_value == value:
            return True

    return False


def _find_annotations(paths, loader, pattern, filters):
    """Find matching annotations within the given paths.

    Math annotations by both name pattern and filters.

    Args:
        paths (list):
            List of paths to search annotations in.
        loader (callable):
            Function to use to load the annotation contents.
        pattern (str):
            Pattern to match against the annotation name.
        filters (dict):
            Dictionary containing key/value filters.

    Returns:
        list:
            names of the matching annotations.
    """
    annotations = dict()
    for base_path in paths:
        annotations.update(_search_annotations(base_path, pattern))

    matching = list()
    for name in sorted(annotations.values()):
        annotation = loader(name)
        for key, value in filters.items():
            if not _match(annotation, key, value):
                break

        else:
            matching.append(name)

    return matching


def find_primitives(pattern='', filters=None):
    """Find primitives by name and filters.

    If a patter is given, only the primitives whose name matches
    the pattern will be returned.

    If filters are given, they should be a dictionary containing key/value
    filters that will have to be matched within the primitive annotation
    for it to be included in the results.

    If the given key is not found but it contains dots, split by the dots
    and consider each part a sublevel in the annotation.

    If the key value within the annotation is a list or a dict, check
    whether any of the given values is contained within it instead of
    checking for equality.

    Args:
        pattern (str):
            Regular expression to match agains the primitive names.
        filters (dict):
            Dictionary containing the filters to apply over the matchin
            primitives.

    Returns:
        list:
            Names of the matching primitives.
    """
    filters = filters or dict()
    return _find_annotations(get_primitives_paths(), load_primitive, pattern, filters)


def find_pipelines(pattern='', filters=None):
    """Find pipelines by name and filters.

    If a patter is given, only the pipelines whose name matches
    the pattern will be returned.

    If filters are given, they should be a dictionary containing key/value
    filters that will have to be matched within the pipeline annotation
    for it to be included in the results.

    If the given key is not found but it contains dots, split by the dots
    and consider each part a sublevel in the annotation.

    If the key value within the annotation is a list or a dict, check
    whether any of the given values is contained within it instead of
    checking for equality.

    Args:
        pattern (str):
            Regular expression to match agains the pipeline names.
        filters (dict):
            Dictionary containing the filters to apply over the matchin
            pipelines.

    Returns:
        list:
            Names of the matching pipelines.
    """
    filters = filters or dict()
    return _find_annotations(get_pipelines_paths(), load_pipeline, pattern, filters)
