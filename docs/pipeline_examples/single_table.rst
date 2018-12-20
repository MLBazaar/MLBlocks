Single Table Pipelines
======================

In this section we will go over a few pipeline examples to show **MLBlocks** working
in different scenarios and with different types of data.

For each example, we will be using example datasets which can be downloaded using the
various functions found in the ``mlblocks.datasets`` module.

.. note:: Even though the datasets are not especially big, some of the examples might
          use a considerable amount of resources, especially memory, and might take
          several minutes to run.

Regression Pipeline
-------------------

In the most simple example, we will be using a single `RandomForestRegressor`_ primitive over
the numeric data from `The Boston Dataset`_, which we will load using the
``mlblocks.dataset.load_boston`` function.

.. code-block:: python

    from mlblocks import MLPipeline
    from mlblocks.datasets import load_boston

    dataset = load_boston()
    dataset.describe()

    X_train, X_test, y_train, y_test = dataset.get_splits(1)

    primitives = [
        'sklearn.ensemble.RandomForestRegressor'
    ]
    pipeline = MLPipeline(primitives)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    dataset.score(y_test, predictions)

Classification Pipeline
-----------------------

As a Classification example, we will be using `The Iris Dataset`_, which we will load using the
``mlblocks.dataset.load_iris`` function.

Here we will combine the `StandardScaler from scikit-learn`_ with an `XGBClassifier primitive`_.

In this case, we will also be passing some initialization parameters for the XGBClassifier.

.. code-block:: python

    from mlblocks import MLPipeline
    from mlblocks.datasets import load_iris

    dataset = load_iris()
    dataset.describe()

    X_train, X_test, y_train, y_test = dataset.get_splits(1)

    primitives = [
        'sklearn.preprocessing.StandardScaler',
        'xgboost.XGBClassifier'
    ]
    init_params = {
        'xgboost.XGBClassifier': {
            'learning_rate': 0.1
        }
    }
    pipeline = MLPipeline(primitives, init_params)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    dataset.score(y_test, predictions)


.. _The Boston Dataset: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html
.. _RandomForestRegressor: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
.. _XGBRegressor: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
.. _The Iris Dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set
.. _StandardScaler from scikit-learn: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
.. _XGBClassifier primitive: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
