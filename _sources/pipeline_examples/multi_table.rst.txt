Multi Table Pipelines
=====================

In the previous section we explored the most simple use cases, where the datasets
consisted in a single table.

In this section we will cover cases where the dataset consist on multiple tables
related by foreign keys.

Multi Table Classification Pipeline
-----------------------------------

In this example, we will be using the `WikiQA dataset`_, which contains 4 different tables
with simple parent/child relationships, and which we will load using the
``mlblocks.dataset.load_wikiqa`` function.

In our pipeline, we will be using the `DeepFeatureSynthesis`_ primitive from `featuretools`_
for feature extraction over the various tables that we have and later on apply an
`XGBClassifier`_ on the resulting feature matrix.

Note how in this example we need to pass some additional information to the pipeline
for the DFS primitive for it to know what the relationships between the multiple
tables are.

.. code-block:: python

    from mlblocks import MLPipeline
    from mlblocks.datasets import load_wikiqa

    dataset = load_wikiqa()
    dataset.describe()

    X_train, X_test, y_train, y_test = dataset.get_splits(1)

    primitives = [
        'featuretools.dfs',
        'xgboost.XGBClassifier'
    ]
    pipeline = MLPipeline(primitives)

    pipeline.fit(X_train, y_train, entities=dataset.entities,
                 relationships=dataset.relationships, target_entity='data')

    predictions = pipeline.predict(X_test, entities=dataset.entities,
                  relationships=dataset.relationships, target_entity='data')

    dataset.score(y_test, predictions)


.. _WikiQA dataset: https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/
.. _XGBClassifier: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
.. _DeepFeatureSynthesis: https://github.com/HDI-Project/MLPrimitives/blob/master/mlblocks_primitives/featuretools.dfs.json
.. _featuretools: https://www.featuretools.com/
