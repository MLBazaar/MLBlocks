# -*- coding: utf-8 -*-

import json
import os
import tempfile
import uuid
from unittest.mock import patch

import pytest
from pkg_resources import Distribution, EntryPoint

from mlblocks import primitives

FAKE_MLPRIMITIVES_PATH = 'this/is/a/fake'


@patch('mlblocks.primitives._PRIMITIVES_PATHS', new=['a', 'b'])
def test_add_primitives_path_do_nothing():
    primitives.add_primitives_path('a')

    assert primitives._PRIMITIVES_PATHS == ['a', 'b']


@patch('mlblocks.primitives._PRIMITIVES_PATHS', new=['a', 'b'])
def test_add_primitives_path_exception():
    invalid_path = str(uuid.uuid4())

    with pytest.raises(ValueError):
        primitives.add_primitives_path(invalid_path)


@patch('mlblocks.primitives._PRIMITIVES_PATHS', new=['a', 'b'])
def test_add_primitives_path():
    primitives.add_primitives_path('tests')

    expected_path = os.path.abspath('tests')

    assert primitives._PRIMITIVES_PATHS == [expected_path, 'a', 'b']


@patch('mlblocks.primitives._PRIMITIVES_PATHS', new=['a', 'b'])
@patch('mlblocks.primitives.pkg_resources.iter_entry_points')
def test_get_primitives_paths_no_entry_points(iep_mock):
    # setup
    iep_mock.return_value == []

    # run
    paths = primitives.get_primitives_paths()

    # assert
    assert paths == ['a', 'b']
    iep_mock.assert_called_once_with('mlprimitives')


@patch('mlblocks.primitives._PRIMITIVES_PATHS', new=['a', 'b'])
@patch('mlblocks.primitives.pkg_resources.iter_entry_points')
def test_get_primitives_paths_entry_points(iep_mock):
    # setup
    something_else_ep = EntryPoint('something_else', 'mlblocks.__version__')
    jsons_path_ep = EntryPoint(
        'jsons_path',
        'tests.test_primitives',
        attrs=['FAKE_MLPRIMITIVES_PATH'],
        dist=Distribution()
    )
    iep_mock.return_value = [
        something_else_ep,
        jsons_path_ep
    ]

    # run
    paths = primitives.get_primitives_paths()

    # assert
    expected = [
        'a',
        'b',
        'this/is/a/fake'
    ]
    assert paths == expected

    iep_mock.assert_called_once_with('mlprimitives')


@patch('mlblocks.primitives._PRIMITIVES_PATHS', new=['a', 'b'])
def test_load_primitive_value_error():
    with pytest.raises(ValueError):
        primitives.load_primitive('invalid.primitive')


def test_load_primitive_success():
    primitive = {
        'name': 'temp.primitive',
        'primitive': 'temp.primitive'
    }

    with tempfile.TemporaryDirectory() as tempdir:
        primitives.add_primitives_path(tempdir)
        primitive_path = os.path.join(tempdir, 'temp.primitive.json')
        with open(primitive_path, 'w') as primitive_file:
            json.dump(primitive, primitive_file, indent=4)

        loaded = primitives.load_primitive('temp.primitive')

        assert primitive == loaded
