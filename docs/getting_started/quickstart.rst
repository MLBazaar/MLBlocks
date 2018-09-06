Quickstart
==========

Below is a short tutorial that will show you how to get started using **MLBlocks**.

In this tutorial we will learn how to:

* Create a pipeline using multiple primitives
* Specify hyperparameters for each primitive in the pipeline
* Fit the pipeline using training data
* Use the pipeline to make predictions from new data

Creating a pipeline
-------------------

With MLBlocks, creating a pipeline is as simple as specifying a list of primitives and passing
them to the `MLPipeline` class:

.. ipython:: python

    from mlblocks import MLPipeline
    primitives = [
        'sklearn.preprocessing.StandardScaler',
        'sklearn.ensemble.RandomForestClassifier'
    ]
    pipeline = MLPipeline(primitives)

Optionally, specific hyperparameters can be also set by specifying them in a dictionary:

.. ipython:: python

    hyperparameters = {
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': 100
        }
    }
    pipeline = MLPipeline(primitives, hyperparameters)

If you can see which hyperparameters a particular pipeline is using, you can do so by calling
its ``get_hyperparameters`` method:

.. ipython:: python

    pipeline.get_hyperparameters()

Making predictions
------------------

Once we have created the pipeline with the desired hyperparameters we can fit it
and then use it to make predictions on new data.

To do this, we first call the ``fit`` method passing the training data and the corresponding labels.

.. ipython:: python

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
    pipeline.fit(X_train, y_train)

Once we have fitted our model to our data, we can call the ``predict`` method passing new data
to obtain predictions from the pipeline.

.. ipython:: python

    from sklearn.metrics import accuracy_score as score
    y_pred = pipeline.predict(X_test)
    y_pred
    score(y_test, y_pred)
