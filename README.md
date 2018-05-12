[![CircleCI][circleci-img]][circleci-url]

[circleci-img]: https://circleci.com/gh/HDI-Project/MLBlocks.svg?style=shield
[circleci-url]: https://circleci.com/gh/HDI-Project/MLBlocks

# MLBlocks

Pipelines and primitives for machine learning and data science.

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/mlblocks

## Overview

MLBlocks is a simple framework for composing end-to-end tunable data science
pipelines.

At a high level, it transforms JSON specifications of data science primitives
into MLBlock instances, which expose tunable hyperparameters via MLHyperparams
and can be composed together to form MLPipelines, which model end-to-end
tunable data science pipelines.

## Submodules

* `components` is a library of various data science functions, primitives, and
  fully-specified pipelines.
* `json_parsers` defines Parsers: classes that initialize MLBlock instances
  from JSON primitives.

## Components

The components library consists of three sublibraries: `primitive_jsons`,
`functions`, and `pipelines`.

### Primitive JSONS

The primitive JSONS sublibrary is the main component of our components library.
It defines primitives as JSON files. The format of said JSON files varies
slightly depending on the model source library, but generally
`random_forest_classifier.json` is a good starting example to look at.
For neural keras primitives, refer to `simple_cnn.json`.

### Functions

The functions sublibrary provides the code for some auxiliary custom functions
that are useful when creating pipelines. Each custom function should also have
a corresponding primitive JSON. A useful example is the HOG featurization step
for image pipelines, defined in `functions/image/hog.py` amd
`primitive_jsons/HOG.json`.

### Pipelines

The pipelines sublibrary provides thin wrappers around useful, fully-specified,
untuned pipelines. `image/traditional_image.py` provides a useful basic
example, while `text/lstm_text.py` is a good starting neural example.

## Parsers

Parsers provide the logic to create MLBlock instances from JSON primitive
specifications. All parsers should extend the MLParser base class, particularly
overriding the `build_mlblock` method. Other quality-of-life helper functions
are provided in the MLParser class as well.

The MLParser class can be found in `ml_json_parser.py`.

## Installation

### Install with pip

MLBlocks is not published in PyPi yet, but you can already install the latest
release using pip

	pip install -e git+https://github.com/HDI-Project/MLBlocks.git#egg=mlblocks

### Install from sources

You can also clone the repository and install it from sources

    git clone git@github.com:HDI-Project/MLBlocks.git
    cd MLBlocks
    make install

## Usage

### Initializing a pipeline

With MLBlocks, we can simply initialize a pipeline consisting of primitives
in our library by passing in a list of JSON names into the MLPipeline
`from_ml_json` method:

```
>>> from mlblocks.ml_pipeline.ml_pipeline import MLPipeline
>>> image_pipeline = MLPipeline.from_ml_json(['HOG', 'random_forest_classifier'])
```

We can also initialize from full path names to JSON files or from
already-loaded JSON dictionaries via the MLPipeline `from_json_filepaths` and
`from_json_metadata` methods respectively.

#### Initialization from pipeline sublibrary

As previously mentioned, we maintain a pipeline sublibrary that contains
several wrappers around specified pipelines for various problem types.
We can simply initialize via the wrapper constructors:

```
>>> from mlblocks.components.pipelines.image.traditional_image import TraditionalImagePipeline
>>> image_pipeline = TraditionalImagePipeline()
```

At this point, we can already fit and predict on data.

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


