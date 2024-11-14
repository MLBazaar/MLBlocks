<p align="left">
  <a href="https://dai.lids.mit.edu">
    <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
  </a>
  <i>An Open Source Project from the <a href="https://dai.lids.mit.edu">Data to AI Lab, at MIT</a></i>
</p>

<p align="left">
<img width=20% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/mlblocks-icon.png" alt=“MLBlocks” />
</p>

<p align="left">
Pipelines and Primitives for Machine Learning and Data Science.
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi](https://img.shields.io/pypi/v/mlblocks.svg)](https://pypi.python.org/pypi/mlblocks)
[![Tests](https://github.com/MLBazaar/MLBlocks/workflows/Run%20Tests/badge.svg)](https://github.com/MLBazaar/MLBlocks/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster)
[![CodeCov](https://codecov.io/gh/MLBazaar/MLBlocks/branch/master/graph/badge.svg)](https://codecov.io/gh/MLBazaar/MLBlocks)
[![Downloads](https://pepy.tech/badge/mlblocks)](https://pepy.tech/project/mlblocks)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MLBazaar/MLBlocks/master?filepath=examples/tutorials)

<br>

# MLBlocks

* Documentation: https://mlbazaar.github.io/MLBlocks
* Github: https://github.com/MLBazaar/MLBlocks
* License: [MIT](https://github.com/MLBazaar/MLBlocks/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)

## Overview

MLBlocks is a simple framework for composing end-to-end tunable Machine Learning Pipelines by
seamlessly combining tools from any python library with a simple, common and uniform interface.

Features include:

* Build Machine Learning Pipelines combining **any Machine Learning Library in Python**.
* Access a repository with hundreds of primitives and pipelines ready to be used with little to
  no python code to write, carefully curated by Machine Learning and Domain experts.
* Extract machine-readable information about which hyperparameters can be tuned and within
  which ranges, allowing automated integration with Hyperparameter Optimization tools like
  [BTB](https://github.com/MLBazaar/BTB).
* Complex multi-branch pipelines and DAG configurations, with unlimited number of inputs and
  outputs per primitive.
* Easy save and load Pipelines using JSON Annotations.

# Install

## Requirements

**MLBlocks** has been developed and tested on [Python 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13](https://www.python.org/downloads/)

## Install with `pip`

The easiest and recommended way to install **MLBlocks** is using [pip](
https://pip.pypa.io/en/stable/):

```bash
pip install mlblocks
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://mlbazaar.github.io/MLBlocks/contributing.html#get-started).

## MLPrimitives

In order to be usable, MLBlocks requires a compatible primitives library.

The official library, required in order to follow the following MLBlocks tutorial,
is [MLPrimitives](https://github.com/MLBazaar/MLPrimitives), which you can install
with this command:

```bash
pip install mlprimitives
```

# Quickstart

Below there is a short example about how to use **MLBlocks** to solve the [Adult Census
Dataset](https://archive.ics.uci.edu/ml/datasets/Adult) classification problem using a
pipeline which combines primitives from [MLPrimitives](https://github.com/MLBazaar/MLPrimitives),
[scikit-learn](https://scikit-learn.org/) and [xgboost](https://xgboost.readthedocs.io/).

```python3
import pandas as pd
from mlblocks import MLPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('http://mlblocks.s3.amazonaws.com/census.csv')
label = dataset.pop('label')

X_train, X_test, y_train, y_test = train_test_split(dataset, label, stratify=label)

primitives = [
    'mlprimitives.custom.preprocessing.ClassEncoder',
    'mlprimitives.custom.feature_extraction.CategoricalEncoder',
    # 'sklearn.impute.SimpleImputer',
    'xgboost.XGBClassifier',
    'mlprimitives.custom.preprocessing.ClassDecoder'
]
pipeline = MLPipeline(primitives)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

accuracy_score(y_test, predictions)
```

# What's Next?

If you want to learn more about how to tune the pipeline hyperparameters, save and load
the pipelines using JSON annotations or build complex multi-branched pipelines, please
check our [documentation site](https://mlbazaar.github.io/MLBlocks).

Also do not forget to have a look at the [notebook tutorials](
https://github.com/MLBazaar/MLBlocks/tree/master/examples/tutorials)!

# Citing MLBlocks

If you use MLBlocks for your research, please consider citing our related papers.

For the current design of MLBlocks and its usage within the larger *Machine Learning Bazaar* project at
the MIT Data To AI Lab, please see:

Micah J. Smith, Carles Sala, James Max Kanter, and Kalyan Veeramachaneni. ["The Machine Learning Bazaar:
Harnessing the ML Ecosystem for Effective System Development."](https://arxiv.org/abs/1905.08942) arXiv
Preprint 1905.08942. 2019.

```bibtex
@article{smith2019mlbazaar,
  author = {Smith, Micah J. and Sala, Carles and Kanter, James Max and Veeramachaneni, Kalyan},
  title = {The Machine Learning Bazaar: Harnessing the ML Ecosystem for Effective System Development},
  journal = {arXiv e-prints},
  year = {2019},
  eid = {arXiv:1905.08942},
  pages = {arXiv:1905.08942},
  archivePrefix = {arXiv},
  eprint = {1905.08942},
}
```

For the first MLBlocks version from 2015, designed for only multi table, multi entity temporal data, please
refer to Bryan Collazo’s thesis:

* [Machine learning blocks](https://dai.lids.mit.edu/wp-content/uploads/2018/06/Mlblocks_Bryan.pdf).
  Bryan Collazo. Masters thesis, MIT EECS, 2015.

With recent availability of a multitude of libraries and tools, we decided it was time to integrate
them and expand the library to address other data types: images, text, graph, time series and
integrate with deep learning libraries.
