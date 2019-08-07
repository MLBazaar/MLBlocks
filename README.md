<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“MLBlocksr” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<p align="left">
<img width=20% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/mlblocks-icon.png" alt=“MLBlocks” />
</p>

<p align="left">
Pipelines and Primitives for Machine Learning and Data Science.
</p>

[![PyPi](https://img.shields.io/pypi/v/mlblocks.svg)](https://pypi.python.org/pypi/mlblocks)
[![Travis](https://travis-ci.org/HDI-Project/MLBlocks.svg?branch=master)](https://travis-ci.org/HDI-Project/MLBlocks)
[![CodeCov](https://codecov.io/gh/HDI-Project/MLBlocks/branch/master/graph/badge.svg)](https://codecov.io/gh/HDI-Project/MLBlocks)
[![Downloads](https://pepy.tech/badge/mlblocks)](https://pepy.tech/project/mlblocks)

* Free software: MIT license
* Documentation: https://HDI-Project.github.io/MLBlocks
- Homepage: https://github.com/HDI-Project/MLBlocks

# MLBlocks

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

# Install

## Requirements

**MLBlocks** has been developed and tested on [Python 3.5 and 3.6](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **MLBlocks** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **MLBlocks**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) mlblocks-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source mlblocks-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **MLBlocks**!

## Install with pip

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **MLBlocks**:

```bash
pip install mlblocks
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from source

Alternatively, with your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:HDI-Project/MLBlocks.git
cd MLBlocks
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

First, please head to [the GitHub page of the project](https://github.com/HDI-Project/MLBlocks)
and make a fork of the project under you own username by clicking on the **fork** button on the
upper right corner of the page.

Afterwards, clone your fork and create a branch from master with a descriptive name that includes
the number of the issue that you are going to work on:

```bash
git clone git@github.com:{your username}/MLBlocks.git
cd MLBlocks
git branch issue-xx-cool-new-feature master
git checkout issue-xx-cool-new-feature
```

Finally, install the project with the following command, which will install some additional
dependencies for code linting and testing.

```bash
make install-develop
```

Make sure to use them regularly while developing by running the commands `make lint` and `make test`.


## MLPrimitives

In order to be usable, MLBlocks requires a compatible primitives library.

The official library, required in order to follow the following MLBlocks tutorial,
is [MLPrimitives](https://github.com/HDI-Project/MLPrimitives), which you can install
with this command:

```bash
pip install mlprimitives
```

# Quickstart

Below there is a short example about how to use MLBlocks to create a simple pipeline, fit it
using demo data and use it to make predictions.

Please make sure to also having installed [MLPrimitives](https://github.com/HDI-Project/MLPrimitives)
before following it.

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

Optionally, specific initialization arguments can be also set by specifying them in a dictionary:

```python
>>> init_params = {
...    'skimage.feature.hog': {
...        'multichannel': True,
...        'visualize': False
...    },
...    'sklearn.ensemble.RandomForestClassifier': {
...         'n_estimators': 100,
...    }
... }
>>> pipeline = MLPipeline(primitives, init_params=init_params)
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

## Making predictions

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

# What's Next?

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
