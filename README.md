<p align="center">
<img width=30% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/mlblocks-icon.png" alt=“MLBlocks” />
</p>

<p align="center">
<i>
Pipelines and Primitives for Machine Learning and Data Science.
</i>
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

MLBlocks is a simple framework for composing end-to-end tunable Machine Learning Pipelines by
seamlessly combining tools from any python library with a simple, common and uniform interface.

* Free software: MIT license
* Documentation: https://HDI-Project.github.io/MLBlocks

# Installation

The simplest and recommended way to install MLBlocks is using `pip`:

```bash
pip install mlblocks
```

Alternatively, you can also clone the repository and install it from sources

```bash
git clone git@github.com:HDI-Project/MLBlocks.git
cd MLBlocks
pip install -e .
```

# Usage Example

Below there is a short example about how to use MLBlocks to create a simple pipeline, fit it
using demo data and use it to make predictions.

For advance usage and more detailed explanation about each component, please have a look
at the [documentation](https://HDI-Project.github.io/MLBlocks)

## Additional Libraries

In order to be able to execute the given code snippets, you will need to install a couple of
additional libraries, which you can do by running:

```bash
pip install mlblocks[demo]
```

## Creating a pipeline

With MLBlocks, creating a pipeline is as simple as specifying a list of primitives and passing
them to the `MLPipeline` class:

```python
>>> from mlblocks import MLPipeline
>>> primitives = [
...     'sklearn.preprocessing.StandardScaler',
...     'sklearn.ensemble.RandomForestClassifier'
... ]
>>> pipeline = MLPipeline(primitives)
```

Optionally, specific hyperparameters can be also set by specifying them in a dictionary:

```python
>>> hyperparameters = {
...     'sklearn.ensemble.RandomForestClassifier': {
...         'n_estimators': 100
...     }
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
    "sklearn.preprocessing.StandardScaler#1": {
        "with_mean": true,
        "with_std": true
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

```python
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> iris = load_iris()
>>> X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
>>> pipeline.fit(X_train, y_train)
```

Once we have fit our model to our data, we can simply make predictions. From these predictions,
we can do useful things, such as obtain an accuracy score.

```python
>>> from sklearn.metrics import accuracy_score as score
>>> y_pred = pipeline.predict(X_test)
>>> score(y_test, y_pred)
0.9473684210526315
```

# History

In its first iteration in 2015, MLBlocks was designed for only multi table, multi entity temporal
data. A good reference to see our design rationale at that time is Bryan Collazo’s thesis:
* [Machine learning blocks](https://dai.lids.mit.edu/wp-content/uploads/2018/06/Mlblocks_Bryan.pdf).
  Bryan Collazo. Masters thesis, MIT EECS, 2015.

With recent availability of a multitude of libraries and tools, we decided it was time to integrate
them and expand the library to address other data types: images, text, graph, time series and
integrate with deep learning libraries.
