<p align="center">
<img width=30% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/mlblocks-icon.png" alt=“MLBlocks” />
</p>

<p align="center">
<i>
Pipelines and Primitives for Machine Learning and Data Science.
</i>
</p>

[![PyPi][pypi-img]][pypi-url]
[![Travis][travis-img]][travis-url]
[![CodeCov][codecov-img]][codecov-url]

[pypi-img]: https://img.shields.io/pypi/v/mlblocks.svg
[pypi-url]: https://pypi.python.org/pypi/mlblocks
[travis-img]: https://travis-ci.org/HDI-Project/MLBlocks.svg?branch=master
[travis-url]: https://travis-ci.org/HDI-Project/MLBlocks
[codecov-img]: https://codecov.io/gh/HDI-Project/MLBlocks/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/HDI-Project/MLBlocks

* Free software: MIT license
* Documentation: https://HDI-Project.github.io/MLBlocks

# Overview

MLBlocks is a simple framework for composing end-to-end tunable Machine Learning Pipelines by
seamlessly combining tools from any python library with a simple, common and uniform interface.

Features include:

* Build Machine Learning Pipelines combining **any Machine Learning Library in Python**.
* Access a repository with hundreds of primitives and pipelines ready to be used with little to
  no python code to write, carefully curated by Machine Learning and Domain experts.
* Extract machine-readable information about which hyperparameters can be tuned and within
  which ranges, allowing automated integration with Hyperparameter Optimization tools like
  [BTB](https://github.com/HDI-Project/BTB).
* Complex multi-branch pipelines and DAG configurations, with unlimited number of inputs and
  outputs per primitive.
* Easy save and load Pipelines using JSON Annotations.

# Installation

The simplest and recommended way to install MLBlocks is using `pip`:

```bash
pip install mlblocks
```

Alternatively, you can also clone the repository and install it from sources

```bash
git clone git@github.com:HDI-Project/MLBlocks.git
cd MLBlocks
make install
```

For development, you can use `make install-develop` instead in order to install all
the required dependencies for testing and code linting.

# Usage Example

Below there is a short example about how to use MLBlocks to create a simple pipeline, fit it
using demo data and use it to make predictions.

For advance usage and more detailed explanation about each component, please have a look
at the [documentation](https://HDI-Project.github.io/MLBlocks)

## Creating a pipeline

With MLBlocks, creating a pipeline is as simple as specifying a list of primitives and passing
them to the `MLPipeline` class.

```python
>>> from mlblocks import MLPipeline
... primitives = [
...     'cv2.GaussianBlur',
...     'skimage.feature.hog',
...     'sklearn.ensemble.RandomForestClassifier'
... ]
>>> pipeline = MLPipeline(primitives)
```

Optionally, specific hyperparameters can be also set by specifying them in a dictionary:

```python
>>> hyperparameters = {
...    'skimage.feature.hog': {
...        'multichannel': True,
...        'visualize': False
...    },
...    'sklearn.ensemble.RandomForestClassifier': {
...         'n_estimators': 100,
...    }
... }
>>> pipeline = MLPipeline(primitives, hyperparameters)
```

If you can see which hyperparameters a particular pipeline is using, you can do so by calling
its `get_hyperparameters` method:

```python
>>> import json
>>> hyperparameters = pipeline.get_hyperparameters()
>>> print(json.dumps(hyperparameters, indent=4))
{
    "cv2.GaussianBlur#1": {
        "ksize_width": 3,
        "ksize_height": 3,
        "sigma_x": 0,
        "sigma_y": 0
    },
    "skimage.feature.hog#1": {
        "multichannel": true,
        "visualize": false,
        "orientations": 9,
        "pixels_per_cell_x": 8,
        "pixels_per_cell_y": 8,
        "cells_per_block_x": 3,
        "cells_per_block_y": 3,
        "block_norm": null
    },
    "sklearn.ensemble.RandomForestClassifier#1": {
        "n_jobs": -1,
        "n_estimators": 100,
        "criterion": "entropy",
        "max_features": null,
        "max_depth": 10,
        "min_samples_split": 0.1,
        "min_samples_leaf": 0.1,
        "class_weight": null
    }
}
```

### Making predictions

Once we have created the pipeline with the desired hyperparameters we can fit it
and then use it to make predictions on new data.

To do this, we first call the `fit` method passing the training data and the corresponding labels.

In this case in particular, we will be loading the handwritten digit classification dataset
from USPS using the `mlblocks.datasets.load_usps` method, which returns a dataset object
ready to be played with.

```python
>>> from mlblocks.datasets import load_usps
>>> dataset = load_usps()
>>> X_train, X_test, y_train, y_test = dataset.get_splits(1)
>>> pipeline.fit(X_train, y_train)
```

Once we have fitted our model to our data, we can call the `predict` method passing new data
to obtain predictions from the pipeline.

```python
>>> predictions = pipeline.predict(X_test)
>>> predictions
array([3, 2, 1, ..., 1, 1, 2])
```

## What's Next?

If you want to learn more about how to tune the pipeline hyperparameters, save and load
the pipelines using JSON annotations or build complex multi-branched pipelines, please
check our [documentation](https://HDI-Project.github.io/MLBlocks).

# History

In its first iteration in 2015, MLBlocks was designed for only multi table, multi entity temporal
data. A good reference to see our design rationale at that time is Bryan Collazo’s thesis:
* [Machine learning blocks](https://dai.lids.mit.edu/wp-content/uploads/2018/06/Mlblocks_Bryan.pdf).
  Bryan Collazo. Masters thesis, MIT EECS, 2015.

With recent availability of a multitude of libraries and tools, we decided it was time to integrate
them and expand the library to address other data types: images, text, graph, time series and
integrate with deep learning libraries.
