<p align="center"> 
<img width=30% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/mlblocks-icon.png" alt=“MLBlocks” />
</p>

<p align="center"> 
<i>MLBlocks is a simple framework for composing end-to-end tunable machine learning pipelines</i> 
</p>


[![PyPi][pypi-img]][pypi-url]
[![CircleCI][circleci-img]][circleci-url]

[pypi-img]: https://img.shields.io/pypi/v/mlblocks.svg
[pypi-url]: https://pypi.python.org/pypi/mlblocks
[circleci-img]: https://circleci.com/gh/HDI-Project/MLBlocks.svg?style=shield
[circleci-url]: https://circleci.com/gh/HDI-Project/MLBlocks


Pipelines and primitives for machine learning and data science.

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/mlblocks

## Overview

At a high level:
 * Machine learning primitives are specified using standardized JSONs
 * User (or an external automated engine) specifies a list of primitives
 * The library transforms JSON specifications of machine learning primitives (blocks) into MLBlock instances, which expose tunable hyperparameters via MLHyperparams and composes a MLPipeline
 * The pipeline.fit and pipeline.predict functions then allow user to fit the pipeline to data and predict on a new set of data.

## Project Structure

The MLBlocks consists of several modules and folders:

* `mlblocks.py`: Defines the `MLBlocks` core class of the library.
* `mlpipeline.py`: Defines the `MLPipeline` class that allows combining multiple MLBlocks.
* `mlhyperparam.py`: Defines the MLHyperparam, an abstraction of an MLBlock tunable
  hyperparameter.
* `components`: is a submodule that contains a collection of helper functions used to integrate
  primitives into MLBlocks, as well as some custom primitives.
* `parsers`: defines the parsers: classes that initialize MLBlock instances
  from JSON primitives.
* `primitives`: folder that contains the collection of JSON primitives.

### Primitive JSONS

The primitive JSONs are the main component of our library.
The format of said JSON files varies slightly depending on the model source library,
but generally `random_forest_classifier.json` is a good starting example to look at.
For neural keras primitives, refer to `simple_cnn.json`.

### Components

The components subpackage provides the code for some auxiliary custom functions
that are useful when creating pipelines. Each custom function should also have
a corresponding primitive JSON. A useful example is the HOG featurization step
for image pipelines, defined in `components/functions/image/hog.py` and
`primitives/HOG.json`.

## Parsers

Parsers provide the logic to create MLBlock instances from JSON primitive
specifications. All parsers should extend the MLParser base class, particularly
overriding the `build_mlblock` method. Other quality-of-life helper functions
are provided in the MLParser class as well.

## Installation

### Install with pip

The simplest and recommended way to install MLBlocks is using `pip`:

	pip install mlblocks

### Install from sources

You can also clone the repository and install it from sources

    git clone git@github.com:HDI-Project/MLBlocks.git
    cd MLBlocks
    make install

## Usage

### Initializing a pipeline

With MLBlocks, we can simply initialize a pipeline by passing it the list
of MLBlocks that will compose it.

```
>>> from mlblocks.mlpipeline import MLPipeline
>>> image_pipeline = MLPipeline(['HOG', 'random_forest_classifier'])
```

### Obtaining and updating hyperparameters

Upon initialization, a pipeline has random hyperparameter values. For a
particular data science problem, we may want to set or view the values and
attributes of particular hyperparameters. For example, we may need to pass in
the current hyperparameter values of our pipeline into a third party tuner.

For tunable hyperparameters, we use `get_tunable_hyperparams`
and `update_tunable_hyperparams`, in which we obtain a list of MLHyperparams
and pass in a list of updated MLHyperparams respectively.

```
>>> tunable_hp = image_pipeline.get_tunable_hyperparams()
>>> print('\n'.join(map(str, tunable_hp)))
Hyperparameter: Name: num_orientations, Step Name: HOG, Type: int, Range: [9, 9], Value: 9
Hyperparameter: Name: num_cell_pixels, Step Name: HOG, Type: int, Range: [8, 8], Value: 8
Hyperparameter: Name: num_cells_block, Step Name: HOG, Type: int, Range: [3, 3], Value: 3
Hyperparameter: Name: criterion, Step Name: rf_classifier, Type: string, Range: ['entropy', 'gini'], Value: entropy
Hyperparameter: Name: max_features, Step Name: rf_classifier, Type: float, Range: [0.1, 1.0], Value: 0.9134616693335704
Hyperparameter: Name: max_depth, Step Name: rf_classifier, Type: int, Range: [2, 10], Value: 2
Hyperparameter: Name: min_samples_split, Step Name: rf_classifier, Type: int, Range: [2, 4], Value: 3
Hyperparameter: Name: min_samples_leaf, Step Name: rf_classifier, Type: int, Range: [1, 3], Value: 3
Hyperparameter: Name: n_estimators, Step Name: rf_classifier, Type: int_cat, Range: [100], Value: 100
Hyperparameter: Name: n_jobs, Step Name: rf_classifier, Type: int_cat, Range: [-1], Value: -1
>>> image_pipeline.update_tunable_hyperparams(tunable_hp)
```

If we only want to update the value of certain tunable hyperparameters, we can
use the `set_from_hyperparam_dict` method, in which we provide a mapping of
(step name, hyperparameter name) tuples to values to update to as input.

```
>>> hp_dict = {('rf_classifier', 'max_depth'): 9}
>>> image_pipeline.set_from_hyperparam_dict(hp_dict)
>>> updated_hp = image_pipeline.get_tunable_hyperparams()
>>> print('\n'.join(map(str, updated_hp)))
Hyperparameter: Name: num_orientations, Step Name: HOG, Type: int, Range: [9, 9], Value: 9
Hyperparameter: Name: num_cell_pixels, Step Name: HOG, Type: int, Range: [8, 8], Value: 8
Hyperparameter: Name: num_cells_block, Step Name: HOG, Type: int, Range: [3, 3], Value: 3
Hyperparameter: Name: criterion, Step Name: rf_classifier, Type: string, Range: ['entropy', 'gini'], Value: entropy
Hyperparameter: Name: max_features, Step Name: rf_classifier, Type: float, Range: [0.1, 1.0], Value: 0.9134616693335704
Hyperparameter: Name: max_depth, Step Name: rf_classifier, Type: int, Range: [2, 10], Value: 9
Hyperparameter: Name: min_samples_split, Step Name: rf_classifier, Type: int, Range: [2, 4], Value: 3
Hyperparameter: Name: min_samples_leaf, Step Name: rf_classifier, Type: int, Range: [1, 3], Value: 3
Hyperparameter: Name: n_estimators, Step Name: rf_classifier, Type: int_cat, Range: [100], Value: 100
Hyperparameter: Name: n_jobs, Step Name: rf_classifier, Type: int_cat, Range: [-1], Value: -1
```

Sometimes, we might want to obtain and update fixed hyperparameters. We can
use analogous `get_fixed_hyperparams` and `update_fixed_hyperparams`. Similarly
to the `set_from_hyperparam_dict`, the outputs of and input to these functions
respectively are mappings of (step name, hyperparameter name) tuples to
hyperparameter values.

```
>>> hp_to_update = {('rf_classifier', 'bootstrap'): True}
>>> image_pipeline.update_fixed_hyperparams(hp_to_update)
>>> image_pipeline.get_fixed_hyperparams()
{('rf_classifier', 'bootstrap'): True}
```

### Making predictions

Once we have set the appropriate hyperparameters for our pipeline, we can make
predictions on a dataset.

To do this, we first call the `fit` method if necessary. This takes in training
data and labels as well as any other parameters each individual step may
use during fitting. These are specified as mappings from (step name, fit
parameter name) tuples to fit parameter values.

```
>>> from sklearn.datasets import fetch_mldata
>>> from sklearn.model_selection import train_test_split
>>> mnist = fetch_mldata('MNIST original')
>>> X, X_test, y, y_test = train_test_split(mnist.data, mnist.target, train_size=1000, test_size=300)
>>> optional_fit_params = {('rf_classifier', 'sample_weight'): None}
>>> image_pipeline.fit(X, y, optional_fit_params)
```

Once we have fit our model to our data, we can simply make predictions. From
these predictions, we can do useful things, such as obtain an accuracy score.

```
>>> from sklearn.metrics import accuracy_score
>>> predicted_y_val = image_pipeline.predict(X_test)
>>> score = accuracy_score(y_test, predicted_y_val)
>>> print(score)
0.85
```
