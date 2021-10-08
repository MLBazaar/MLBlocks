# -*- coding: utf-8 -*-

"""
MLBlocks top module.

MLBlocks is a simple framework for composing end-to-end tunable Machine Learning Pipelines by
seamlessly combining tools from any python library with a simple, common and uniform interface.

* Free software: MIT license
* Documentation: https://MLBazaar.github.io/MLBlocks
"""

from mlblocks.discovery import (
    add_pipelines_path, add_primitives_path, find_pipelines, find_primitives, get_pipelines_paths,
    get_primitives_paths, load_pipeline, load_primitive)
from mlblocks.mlblock import MLBlock
from mlblocks.mlpipeline import MLPipeline

__author__ = 'MIT Data To AI Lab'
__copyright__ = 'Copyright (c) 2018, MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__license__ = 'MIT'
__version__ = '0.4.2.dev0'

__all__ = [
    'MLBlock',
    'MLPipeline',
    'add_pipelines_path',
    'add_primitives_path',
    'find_pipelines',
    'find_primitives',
    'get_pipelines_paths',
    'get_primitives_paths',
    'load_pipeline',
    'load_primitive'
]
