# -*- coding: utf-8 -*-

import os

from mock import patch

from mlblocks import primitives


@patch('mlblocks.primitives._PRIMITIVES_PATHS', new=['a', 'b'])
def test_add_primitives_path():
    primitives.add_primitives_path('tests')

    expected_path = os.path.abspath('tests')

    assert primitives._PRIMITIVES_PATHS == [expected_path, 'a', 'b']
