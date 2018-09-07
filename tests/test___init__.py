# -*- coding: utf-8 -*-

import os

from mock import patch

import mlblocks


@patch('mlblocks._PRIMITIVES_PATHS', new=['a', 'b'])
def test_add_primitives_path():
    mlblocks.add_primitives_path('tests')

    expected_path = os.path.abspath('tests')

    assert mlblocks._PRIMITIVES_PATHS == [expected_path, 'a', 'b']
