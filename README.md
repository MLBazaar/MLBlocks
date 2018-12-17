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
...     'xgboost.XGBClassifier'
... ]
>>> pipeline = MLPipeline(primitives)
```

Optionally, specific hyperparameters can be also set by specifying them in a dictionary:

```python
>>> hyperparameters = {
...     'xgboost.XGBClassifier': {
...         'learning_rate': 0.1
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
    "xgboost.XGBClassifier#1": {
        "n_jobs": -1,
        "learning_rate": 0.1,
        "n_estimators": 10,
        "max_depth": 3,
        "gamma": 0,
        "min_child_weight": 1
    }
}
```

### Making predictions

Once we have created the pipeline with the desired hyperparameters we can fit it
and then use it to make predictions on new data.

To do this, we first call the `fit` method passing the training data and the corresponding labels.

```python
>>> from mlblocks.datasets import load_iris
>>> dataset = load_iris()
>>> pipeline.fit(dataset.train_data, dataset.train_target)
```

Once we have fitted our model to our data, we can call the `predict` method passing new data
to obtain predictions from the pipeline.

```python
>>> predictions = pipeline.predict(dataset.test_data)
>>> predictions
array([2, 0, 1, 0, 1, 0, 0, 1, 2, 1, 2, 1, 2, 2, 0, 1, 0, 2, 1, 1, 0, 1,
       0, 2, 0, 1, 0, 0, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2])
>>> dataset.score(dataset.test_target, predictions)
0.9736842105263158
```

# History

In its first iteration in 2015, MLBlocks was designed for only multi table, multi entity temporal
data. A good reference to see our design rationale at that time is Bryan Collazo’s thesis:
* [Machine learning blocks](https://dai.lids.mit.edu/wp-content/uploads/2018/06/Mlblocks_Bryan.pdf).
  Bryan Collazo. Masters thesis, MIT EECS, 2015.

With recent availability of a multitude of libraries and tools, we decided it was time to integrate
them and expand the library to address other data types: images, text, graph, time series and
integrate with deep learning libraries.
