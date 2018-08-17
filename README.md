<p align="center">
<img width=30% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/mlblocks-icon.png" alt=“MLBlocks” />
</p>

<p align="center">
<i>MLBlocks is a simple framework for composing end-to-end tunable machine learning pipelines</i>
</p>


[![PyPi][pypi-img]][pypi-url]
[![CircleCI][circleci-img]][circleci-url]
[![Travis][travis-img]][travis-url]

[travis-img]: https://travis-ci.org/HDI-Project/MLBlocks.svg?branch=master
[travis-url]: https://travis-ci.org/HDI-Project/MLBlocks
[pypi-img]: https://img.shields.io/pypi/v/mlblocks.svg
[pypi-url]: https://pypi.python.org/pypi/mlblocks
[circleci-img]: https://circleci.com/gh/HDI-Project/MLBlocks.svg?style=shield
[circleci-url]: https://circleci.com/gh/HDI-Project/MLBlocks

Pipelines and primitives for machine learning and data science.

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/MLBlocks

## Overview

At a high level:
 * Machine learning primitives are specified using standardized JSONs
 * User (or an external automated engine) specifies a list of primitives
 * The library transforms JSON specifications of machine learning primitives (blocks) into
   MLBlock instances, which expose tunable hyperparameters via MLHyperparams and composes
   a MLPipeline
 * The pipeline.fit and pipeline.predict functions then allow user to fit the pipeline to
   data and predict on a new set of data.

## Project Structure

The MLBlocks consists of the following modules and folders:

* `mlblocks.mlblocks`: Defines the `MLBlock` core class of the library.
* `mlblocks.mlpipeline`: Defines the `MLPipeline` class that allows combining multiple MLBlock
  instances.
* `mlblocks_primitives`: folder that contains the collection of JSON primitives. This folder
  can either be provided by the user or installed via the MLPrimitives subproject.

### Primitive JSONS

The primitive JSONs are the main component of our library.
The contents of said JSON files varies slightly depending on the model source library,
but they all have a common structure.

Examples of such JSON files can be found inside the `examples` folder.

## Installation

### Install with pip

The simplest and recommended way to install MLBlocks is using `pip`:

	pip install mlblocks

### Install from sources

You can also clone the repository and install it from sources

    git clone git@github.com:HDI-Project/MLBlocks.git
    cd MLBlocks
    pip install -e .

## Usage

The following points cover the most basic usage of the MLBlocks library.

Note that in order to be able to execute the given code snippets, you will
need to install a couple of additional libraries, which you can do by running:

```
pip install mlblocks[demo]
```

if you installed the library from PyPi or

```
pip install -e .[demo]
```

If you installed from sources.

### Initializing a pipeline

With MLBlocks, we can simply initialize a pipeline by passing it the list
of MLBlocks that will compose it.

```
>>> from mlblocks import MLPipeline
>>> pipeline = MLPipeline(['sklearn.ensemble.RandomForestClassifier'])
```

### Obtaining and updating hyperparameters

Upon initialization, a pipeline has a set of default hyperparamters. For a
particular data science problem, we may want to set or view the values and
attributes of particular hyperparameters. For example, we may need to pass in
the current hyperparameter values of our pipeline into a third party tuner.

To obtain the list of tunable hyperparameters can be obtained by calling the pipeline
method `get_tunable_hyperparameters`.

```
>>> tunable_hp = pipeline.get_tunable_hyperparameters()
>>> import json
>>> print(json.dumps(tunable_hp, indent=4))
{
    "sklearn.ensemble.RandomForestClassifier#1": {
        "criterion": {
            "type": "str",
            "default": "entropy",
            "values": [
                "entropy",
                "gini"
            ]
        },
        "max_features": {
            "type": "str",
            "default": null,
            "range": [
                null,
                "auto",
                "log2"
            ]
        },
        "max_depth": {
            "type": "int",
            "default": 10,
            "range": [
                1,
                30
            ]
        },
        "min_samples_split": {
            "type": "float",
            "default": 0.1,
            "range": [
                0.0001,
                0.5
            ]
        },
        "min_samples_leaf": {
            "type": "float",
            "default": 0.1,
            "range": [
                0.0001,
                0.5
            ]
        },
        "n_estimators": {
            "type": "int",
            "default": 30,
            "values": [
                2,
                500
            ]
        },
        "class_weight": {
            "type": "str",
            "default": null,
            "range": [
                null,
                "balanced"
            ]
        }
    }
}
```

To obtain the values that the hyperparmeters of our pipeline currently has,
the method `get_hyperparameters` can be used.

```
>>> current_hp = pipeline.get_hyperparameters()
>>> print(json.dumps(current_hp, indent=4))
{
    "sklearn.ensemble.RandomForestClassifier#1": {
        "n_jobs": -1,
        "criterion": "entropy",
        "max_features": null,
        "max_depth": 10,
        "min_samples_split": 0.1,
        "min_samples_leaf": 0.1,
        "n_estimators": 30,
        "class_weight": null
    }
}
```

Similarly, to set different hyperparameter values, the method `set_hyperparameters`
can be used.

```
>>> new_hyperparameters = {'sklearn.ensemble.RandomForestClassifier#1': {'max_depth': 20}}
>>> pipeline.set_hyperparameters(new_hyperparameters)
>>> pipeline.get_hyperparameters()['sklearn.ensemble.RandomForestClassifier#1']['max_depth']
20
```

### Making predictions

Once we have set the appropriate hyperparameters for our pipeline, we can make
predictions on a dataset.

To do this, we first call the `fit` method if necessary. This takes in training
data and labels as well as any other parameters each individual step may
use during fitting.

```
>>> from sklearn.datasets import load_wine
>>> from sklearn.model_selection import train_test_split
>>> wine = load_wine()
>>> X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target)
>>> pipeline.fit(X_train, y_train)
```

Once we have fit our model to our data, we can simply make predictions. From
these predictions, we can do useful things, such as obtain an accuracy score.

```
>>> y_pred = pipeline.predict(X_test)
>>> from sklearn.metrics import accuracy_score
>>> accuracy_score(y_test, y_pred)
1.0
```

# History

In its first iteration in 2015, MLBlocks was designed for only multi table, multi entity temporal
data. A good reference to see our design rationale at that time is Bryan Collazo’s thesis:
* [Machine learning blocks](https://dai.lids.mit.edu/wp-content/uploads/2018/06/Mlblocks_Bryan.pdf).
  Bryan Collazo. Masters thesis, MIT EECS, 2015.

With recent availability of a multitude of libraries and tools, we decided it was time to integrate
them and expand the library to address other data types: images, text, graph, time series and
integrate with deep learning libraries.
