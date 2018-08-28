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

MLBlocks goal is to seamlessly combine any possible set of Machine Learning tools developed
in Python, whether they are custom developments or belong to third party libraries, and
build Pipelines out of them that can be fitted and then used to make predictions.

This is achieved by providing a simple and intuitive annotation language that allows the
user to specify how to integrate with each tool, here called primitives, in order to provide
a common uniform interface to each one of them.

At a high level:
* Each available primitive has been annotated using a standardized JSON file that specifies its
  native interface, as well as which hyperparameters can be used to tune its behavior.
* A list of primitives that will be combined into a pipeline is provided by the user, optionally
  passing along the hyperparameters to use for each primitive.
* An MLBlock instance is build for each primitive, offering a common interface for all of them.
* The MLBlock instances are then combined into an MLPipeline instance, able to run them all in
  the right order, passing the output from each one as input to the next one.
* The training data is passed to the `MLPipeline.fit` method, which sequentially fits each
  MLBlock instance following the JSON annotation specification.
* The data used to make predictions is passed to the `MLPipeline.predict` method, which uses each
  MLBlock sequentially to obtain the desired predictions.

## Main Elements

Here we provide an overview of the main elements of the MLBlocks library:

### JSON Annotations

Each integrated primitive has an associated JSON file that specifies its methods, their arguments,
their types and, most importantly, any possible hyperparameter that the primitive has, as well
as their types, ranges and conditions, if any.

These JSON annotations can be:

* **Installed** using the **MLPrimitives** related project, which is the recommended approach.
* **Created by the user** and configured for MLBlocks to use them.

And the primitives can be of two types:

* Function Primitives: Simple functions that can be called directly.
* Class Primitives: Class objects that need to be instantiated before they can be used.

Here are some simplified examples of these JSONs, but for more detailed examples, please refer to
the `examples` folder.

#### Function Primitives

The most simple type of primitives are simple functions that can be called directly, without
the need to created any class instance before.

In most cases, if not all, these functions do not have any associated learning process,
and their behavior is always the same both during the fitting and the predicting phases
of the pipeline.

A simple example of such a primitive would be the `numpy.argmax` function, which expects a 2
dimensional array as input, and returns a 1 dimensional array that indicates the index of the
maximum values along an axis.

The simplest JSON annotation for this primitive would look like this:

```
{
    "name": "numpy.argmax",
    "primitive": "numpy.argmax",
    "produce": {
        "args": [
            {
                "name": "y",
                "type": "ndarray"
            }
        ],
        "output": [
            {
                "name": "y",
                "type": "ndarray"
            }
        ]
    },
    "hyperparameters": {
        "fixed": {
            "axis": {
                "type": "int",
                "default": 1
            }
        }
    }
}
```

The main elements of this JSON are:

* **name**: The name that is given to the primitive, and which will be used later on to add this
            primitive to a pipeline.
* **primitive**: The fully qualified, directly importable name of the function to be used.
* **produce**: A nested JSON that specifies the names and types of arguments and the output values
               of the primitive.
* **hyperparameters**: A nested JSON that specifies the hyperparameters of this primitive.
                       Note that multiple types of hyperparameters exist, but that this primitive
                       has only one `fixed` hyperparameter, which mean that this is not tunable
                       and that, even though the user can specify a value different than the
                       default, changes are not expected during the MLBlock intance life cycle.

#### Class Primitives

A more complex type of primitives are classes which need to be instantiated before they can
be used.

In most cases, these classes will have an associated learning process, and they will have some
fit method or equivalent that will be called during the fitting phase but not during the
predicting one.

A simple example of such a primitive would be the `sklearn.preprocessing.StandardScaler` class,
which is used to standardize a set of values by calculating their z-score, which means centering
them around 0 and scaling them to unit variance.

This primitive has an associated learning process, where it calculates the mean and standard
deviation of the training data, to later on use them to transform the prediction data to the
same center and scale.

The simplest JSON annotation for this primitive would look like this:

```
{
    "name": "sklearn.preprocessing.StandardScaler",
    "primitive": "sklearn.preprocessing.StandardScaler",
    "fit": {
        "method": "fit",
        "args": [
            {
                "name": "X",
                "type": "ndarray"
            }
        ]
    },
    "produce": {
        "method": "transform",
        "args": [
            {
                "name": "X",
                "type": "ndarray"
            }
        ],
        "output": [
            {
                "name": "X",
                "type": "ndarray"
            }
        ]
    },
    "hyperparameters": {
        "tunable": {
            "with_mean": {
                "type": "bool",
                "default": true
            },
            "with_std": {
                "type": "bool",
                "default": true
            }
        }
    }
}

```

Note that there are some details of this JSON annotation that make it different from the
Function Primitive one that explained above:

* **primitive**: The fully qualified, directly importable name of the class to be used. This
                 class is the one that will be used to create the actual primitive instance.
* **fit**: A nested JSON that specifies the name of the method to call during the fitting phase,
           which in this case happens to also be `fit`, as well as the names and types of
           arguments that this method expects.
* **produce**: A nested JSON that specifies the name of the method to call during the predicting
               phase, in this case called `transform`, as well as the names and types of
               arguments that this method expects and its outputs.
* **hyperparameters**: A nested JSON that specifies the hyperparameters of this primitive.
                       In this case, only `tunable` hyperparameters are specified, with their
                       names and types. If the type was something other than `bool`, a list or
                       range of valid values would also be specified.

#### Hyperparameters

A very importnat element of both Function and Class primitives are the hyperparameters.

The hyperparameters are arguments that modify the behavior of the primitive and its learning
process, which are set before the learning process starts and are not deduced from the data.
These hyperparameters are usually passed as arguments to the primitive constructor or to the
methods or funcitons that will be called during the fitting or predicting phase.

Two types of hyperparameters exist:

* **fixed**: These hyperparameters do not alter the learning process, and their values modify
             the behavior of the primitive but not its prediction performance. In some cases
             these hyperparameters have a default value, but most of the times their values
             have to be explicitly set by the user.
* **tunable**: These hyperparameters participate directly in the learning process, and their
               values can alter how well the primitive learns and is able to later on predict.
               For this reason, even though these hyperparameters do not alter the behavior
               of the primitive, they can be tuned to improve the prediction performance.

### MLBlock Class

The `mlblocks.MLBlock` class is the core class of the library.

This is used to wrap the annotated primitives, offering a common and uniform interface to
interact with any possible Machine Learning tool implemented in Python.

These are the inputs required to create an `MLBlock` instance:

* `name`: the name of the primitive JSON to load.
* `**hyperparameters`: Hyperparameters of the primitive, passed as keyword arguments.

And it has these available methods:

* `get_tunable_hyperparameters`: Get a dictionary indicating which hyperparameters can be tuned
                                 for this primitive, with their types, available ranges, default
                                 values and documentation.
* `get_hyperparameters`: Get a dictionary with the hyperparameter values that the primitive
                         is using.
* `set_hyperparameters`: Set new hyperparameters for the primitive, recreating any necessary
                         object for the changes to take effect.
* `fit`: Call the method specified in the JSON annotation passing any required arguments.
* `produce`: Call the method specified in the JSON annotation passing any required arguments and
             capture its outputs as variables.

### MLPipeline Class

The `mlblocks.MLPipeline` class is the one responsible for combining multiple `MLBlock` instances,
called `blocks` in this context, to be called in succession for fitting and predicting.

This one is the object which the user should be mostly interacting with.

These are the expected inputs to create the instance:

* `blocks`: A list containing the names of the primitives to load as MLBlock instances.
* `init_params`: Hyperparameters to be used for the MLBlock instances creation, specified as a
                 dictionary that contains the name of the blocks as keys and the set of keyword
                 arguments to pass to each one of them specified as subdicts.

And it has these available methods:
* `get_tunable_hyperparameters`: Get a dictionary indicating which hyperparameters can be tuned
                                 for each primitive, with their types, available ranges, default
                                 values and documentation.
* `get_hyperparameters`: Get a dictionary with the hyperparameter values that each primitive
                         is currently using.
* `set_hyperparameters`: Set new hyperparameters for one or more primitives, passed as a dictionary
                         where keys are the name of the blocks to modify and values
                         are nested dictionaries with the hyperparameters to set.
* `fit`: Call the `fit` method and then the `produce` method of each block, in sequence, passing
         each time as input the output of the `produce` method of the previous block.
* `predict`: Call the `produce` method of each block in sequence passing each time as input the
             output of the `produce` method of the previous block.

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

if you installed it from its sources.

### Initializing a pipeline

With MLBlocks, we can simply initialize a pipeline by passing it the list
of primitives that will compose it.

```
>>> from mlblocks import MLPipeline
>>> pipeline = MLPipeline(['sklearn.ensemble.RandomForestClassifier'])
```

### Obtaining and updating hyperparameters

Upon initialization, a pipeline has a set of default hyperparamters. For a particular data
science problem, we may want to set or view the values and attributes of particular
hyperparameters. For example, we may need to pass in the current hyperparameter values of
our pipeline into a third party tuner.

To obtain the list of tunable hyperparameters can be obtained by calling the pipeline method
`get_tunable_hyperparameters`.

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

To obtain the values that the hyperparmeters of our pipeline currently has, the method
`get_hyperparameters` can be used.

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

Similarly, to set different hyperparameter values, the method `set_hyperparameters` can be used.

```
>>> new_hyperparameters = {'sklearn.ensemble.RandomForestClassifier#1': {'max_depth': 20}}
>>> pipeline.set_hyperparameters(new_hyperparameters)
>>> pipeline.get_hyperparameters()['sklearn.ensemble.RandomForestClassifier#1']['max_depth']
20
```

### Making predictions

Once we have set the appropriate hyperparameters for our pipeline, we can make predictions on
a dataset.

To do this, we first call the `fit` method if necessary. This takes in training data and labels
as well as any other parameters each individual step may use during fitting.

```
>>> from sklearn.datasets import load_wine
>>> from sklearn.model_selection import train_test_split
>>> wine = load_wine()
>>> X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target)
>>> pipeline.fit(X_train, y_train)
```

Once we have fit our model to our data, we can simply make predictions. From these predictions,
we can do useful things, such as obtain an accuracy score.

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
