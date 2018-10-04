# -*- coding: utf-8 -*-

"""
MLBlocks top module.

MLBlocks is a simple framework for composing end-to-end tunable Machine Learning Pipelines by
seamlessly combining tools from any python library with a simple, common and uniform interface.

* Free software: MIT license
* Documentation: https://HDI-Project.github.io/MLBlocks
"""

from mlblocks.mlblock import MLBlock  # noqa
from mlblocks.mlpipeline import MLPipeline  # noqa
from mlblocks.primitives import add_primitives_path, get_primitives_paths, load_primitive  # noqa

__author__ = 'MIT Data To AI Lab'
__copyright__ = 'Copyright (c) 2018, MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__license__ = 'MIT'
__version__ = '0.2.3-dev'

__all__ = [
    'MLBlock', 'MLPipeline', 'add_primitives_path',
    'get_primitives_paths', 'load_primitive'
]
