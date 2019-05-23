# -*- coding: utf-8 -*-

import json
import os
import tempfile
import uuid
from unittest.mock import call, patch

import pytest
from pkg_resources import Distribution, EntryPoint

from mlblocks import discovery

FAKE_PRIMITIVES_PATH = 'this/is/a/fake'


def test__add_lookup_path_do_nothing():
    paths = ['a', 'b']
    discovery._add_lookup_path('a', paths)

    assert paths == ['a', 'b']


def test__add_lookup_path_exception():
    paths = ['a', 'b']
    invalid_path = str(uuid.uuid4())

    with pytest.raises(ValueError):
        discovery._add_lookup_path(invalid_path, paths)


def test__add_lookup_path():
    paths = ['a', 'b']
    discovery._add_lookup_path('tests', paths)

    expected_path = os.path.abspath('tests')

    assert paths == [expected_path, 'a', 'b']


@patch('mlblocks.discovery._PRIMITIVES_PATHS', new=['a', 'b'])
def test_add_primitives_path():
    discovery.add_primitives_path(os.path.abspath('tests'))

    expected_path = os.path.abspath('tests')
    assert discovery._PRIMITIVES_PATHS == [expected_path, 'a', 'b']


@patch('mlblocks.discovery._PIPELINES_PATHS', new=['a', 'b'])
def test_add_pipelines_path():
    discovery.add_pipelines_path('tests')

    expected_path = os.path.abspath('tests')
    assert discovery._PIPELINES_PATHS == [expected_path, 'a', 'b']


@patch('mlblocks.discovery._PRIMITIVES_PATHS', new=['a', 'b'])
@patch('mlblocks.discovery.pkg_resources.iter_entry_points')
def test__load_entry_points_no_entry_points(iep_mock):
    # setup
    iep_mock.return_value == []

    # run
    paths = discovery._load_entry_points('jsons_path', 'mlprimitives')

    # assert
    assert paths == []
    expected_calls = [
        call('mlprimitives'),
    ]
    assert iep_mock.call_args_list == expected_calls


@patch('mlblocks.discovery.pkg_resources.iter_entry_points')
def test__load_entry_points_entry_points(iep_mock):
    # setup
    something_else_ep = EntryPoint('something_else', 'mlblocks.__version__')
    primitives_ep = EntryPoint(
        'primitives',
        'tests.test_discovery',
        attrs=['FAKE_PRIMITIVES_PATH'],
        dist=Distribution()
    )
    iep_mock.return_value = [
        something_else_ep,
        primitives_ep
    ]

    # run
    paths = discovery._load_entry_points('primitives')

    # assert
    expected = [
        'this/is/a/fake'
    ]
    assert paths == expected

    expected_calls = [
        call('mlblocks'),
    ]
    assert iep_mock.call_args_list == expected_calls


@patch('mlblocks.discovery._PRIMITIVES_PATHS', new=['a', 'b'])
@patch('mlblocks.discovery._load_entry_points')
def test_get_primitives_paths(lep_mock):
    lep_mock.side_effect = [['c'], []]

    paths = discovery.get_primitives_paths()

    assert paths == ['a', 'b', 'c']
    expected_calls = [
        call('primitives'),
        call('jsons_path', 'mlprimitives'),
    ]
    assert lep_mock.call_args_list == expected_calls


@patch('mlblocks.discovery._PIPELINES_PATHS', new=['a', 'b'])
@patch('mlblocks.discovery._load_entry_points')
def test_get_pipelines_paths(lep_mock):
    lep_mock.return_value = ['c']

    paths = discovery.get_pipelines_paths()

    assert paths == ['a', 'b', 'c']
    lep_mock.assert_called_once_with('pipelines')


def test__load_value_error():
    primitive = discovery._load('invalid.primitive', ['a', 'b'])

    assert primitive is None


def test__load_success():
    primitive = {
        'name': 'temp.primitive',
        'primitive': 'temp.primitive'
    }

    with tempfile.TemporaryDirectory() as tempdir:
        paths = [tempdir]
        primitive_path = os.path.join(tempdir, 'temp.primitive.json')
        with open(primitive_path, 'w') as primitive_file:
            json.dump(primitive, primitive_file, indent=4)

        loaded = discovery._load('temp.primitive', paths)

        assert primitive == loaded


@patch('mlblocks.discovery.get_primitives_paths')
@patch('mlblocks.discovery._load')
def test__load_primitive_value_error(load_mock, gpp_mock):
    load_mock.return_value = None
    gpp_mock.return_value = ['a', 'b']

    with pytest.raises(ValueError):
        discovery.load_primitive('invalid.primitive')

    load_mock.assert_called_once_with('invalid.primitive', ['a', 'b'])


@patch('mlblocks.discovery.get_primitives_paths')
@patch('mlblocks.discovery._load')
def test__load_primitive_success(load_mock, gpp_mock):
    gpp_mock.return_value = ['a', 'b']

    primitive = discovery.load_primitive('valid.primitive')

    load_mock.assert_called_once_with('valid.primitive', ['a', 'b'])

    assert primitive == load_mock.return_value


@patch('mlblocks.discovery.get_pipelines_paths')
@patch('mlblocks.discovery._load')
def test__load_pipeline_value_error(load_mock, gpp_mock):
    load_mock.return_value = None
    gpp_mock.return_value = ['a', 'b']

    with pytest.raises(ValueError):
        discovery.load_pipeline('invalid.pipeline')

    load_mock.assert_called_once_with('invalid.pipeline', ['a', 'b'])


@patch('mlblocks.discovery.get_pipelines_paths')
@patch('mlblocks.discovery._load')
def test__load_pipeline_success(load_mock, gpp_mock):
    gpp_mock.return_value = ['a', 'b']

    pipeline = discovery.load_pipeline('valid.pipeline')

    load_mock.assert_called_once_with('valid.pipeline', ['a', 'b'])

    assert pipeline == load_mock.return_value
