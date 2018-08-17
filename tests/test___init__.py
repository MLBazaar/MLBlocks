# -*- coding: utf-8 -*-

from mock import patch

import mlblocks


@patch('mlblocks.PRIMITIVES_PATHS', new=['a', 'b'])
def test_add_primitives_path():
    mlblocks.add_primitives_path('c')

    assert mlblocks.PRIMITIVES_PATHS == ['c', 'a', 'b']
