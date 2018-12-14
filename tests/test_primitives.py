# -*- coding: utf-8 -*-

import os
import uuid
from unittest.mock import patch

import pytest

from mlblocks import primitives


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
def test_get_primitives_paths():
    paths = primitives.get_primitives_paths()

    assert paths == ['a', 'b']
