Changelog
=========

0.3.2 - 2019-08-12
------------------

* Allow passing fit and produce arguments as `init_params` - [Issue #96](https://github.com/HDI-Project/MLBlocks/issues/96) by @csala
* Support optional fit and produce args and arg defaults - [Issue #95](https://github.com/HDI-Project/MLBlocks/issues/95) by @csala
* Isolate primitives from their hyperparameters dictionary - [Issue #94](https://github.com/HDI-Project/MLBlocks/issues/94) by @csala
* Add functions to explore the available primitives and pipelines - [Issue #90](https://github.com/HDI-Project/MLBlocks/issues/90) by @csala
* Add primitive caching New Feature - [Issue #22](https://github.com/HDI-Project/MLBlocks/issues/22) by @csala

0.3.1 - Pipelines Discovery
---------------------------

* Support flat hyperparameter dictionaries - [Issue #92](https://github.com/HDI-Project/MLBlocks/issues/92) by @csala
* Load pipelines by name and register them as `entry_points` - [Issue #88](https://github.com/HDI-Project/MLBlocks/issues/88) by @csala
* Implement partial re-fit -[Issue #61](https://github.com/HDI-Project/MLBlocks/issues/61) by @csala
* Move argument parsing to MLBlock - [Issue #86](https://github.com/HDI-Project/MLBlocks/issues/86) by @csala
* Allow getting intermediate outputs - [Issue #58](https://github.com/HDI-Project/MLBlocks/issues/58) by @csala

0.3.0 - New Primitives Discovery
--------------------------------

* New primitives discovery system based on `entry_points`.
* Conditional Hyperparameters filtering in MLBlock initialization.
* Improved logging and exception reporting.

0.2.4 - New Datasets and Unit Tests
-----------------------------------

* Add a new multi-table dataset.
* Add Unit Tests up to 50% coverage.
* Improve documentation.
* Fix minor bug in newsgroups dataset.

0.2.3 - Demo Datasets
---------------------

* Add new methods to Dataset class.
* Add documentation for the datasets module.

0.2.2 - MLPipeline Load/Save
----------------------------

* Implement save and load methods for MLPipelines
* Add more datasets

0.2.1 - New Documentation
-------------------------

* Add mlblocks.datasets module with demo data download functions.
* Extensive documentation, including multiple pipeline examples.

0.2.0 - New MLBlocks API
------------------------

A new MLBlocks API and Primitive format.

This is a summary of the changes:

* Primitives JSONs and Python code has been moved to a different repository, called MLPrimitives
* Optional usage of multiple JSON primitive folders.
* JSON format has been changed to allow more flexibility and features:
    * input and output arguments, as well as argument types, can be specified for each method
    * both classes and function as primitives are supported
    * multitype and conditional hyperparameters fully supported
    * data modalities and primitive classifiers introduced
    * metadata such as documentation, description and author fields added
* Parsers are removed, and now the MLBlock class is responsible for loading and reading the
  JSON primitive.
* Multiple blocks of the same primitive are supported within the same pipeline.
* Arbitrary inputs and outputs for both pipelines and blocks are allowed.
* Shared variables during pipeline execution, usable by multiple blocks.

0.1.9 - Bugfix Release
----------------------

* Disable some NetworkX functions for incompatibilities with some types of graphs.

0.1.8 - New primitives and some improvements
--------------------------------------------

* Improve the NetworkX primitives.
* Add String Vectorization and Datetime Featurization primitives.
* Refactor some Keras primitives to work with single dimension `y` arrays and be compatible with `pickle`.
* Add XGBClassifier and XGBRegressor primitives.
* Add some `keras.applications` pretrained networks as preprocessing primitives.
* Add helper class to allow function primitives.

0.1.7 - Nested hyperparams dicts
--------------------------------

* Support passing hyperparams as nested dicts.

0.1.6 - Text and Graph Pipelines
--------------------------------

* Add LSTM classifier and regressor primitives.
* Add OneHotEncoder and MultiLabelEncoder primitives.
* Add several NetworkX graph featurization primitives.
* Add `community.best_partition` primitive.

0.1.5 - Collaborative Filtering Pipelines
-----------------------------------------

* Add LightFM primitive.

0.1.4 - Image pipelines improved
--------------------------------

* Allow passing `init_params` on `MLPipeline` creation.
* Fix bug with MLHyperparam types and Keras.
* Rename `produce_params` as `predict_params`.
* Add SingleCNN Classifier and Regressor primitives.
* Simplify and improve Trivial Predictor

0.1.3 - Multi Table pipelines improved
--------------------------------------

* Improve RandomForest primitive ranges
* Improve DFS primitive
* Add Tree Based Feature Selection primitives
* Fix bugs in TrivialPredictor
* Improved documentation

0.1.2 - Bugfix release
----------------------

* Fix bug in TrivialMedianPredictor
* Fix bug in OneHotLabelEncoder

0.1.1 - Single Table pipelines improved
---------------------------------------

* New project structure and primitives for integration into MIT-TA2.
* MIT-TA2 default pipelines and single table pipelines fully working.

0.1.0
-----

* First release on PyPI.
