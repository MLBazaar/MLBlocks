Changelog
=========

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
