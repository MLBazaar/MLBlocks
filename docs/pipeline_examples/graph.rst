Graph Pipelines
===============

Here we will be showing some examples using **MLBlocks** to resolve graph problems.

Link Prediction
---------------

For the Graph Link Prediction  examples we will be using the UMLS biomedical ontology dataset,
which we will load using the ``mlblocks.dataset.load_umls`` function.

The data consists of information about a 135 Graph and the relations between their nodes given
as a DataFrame with three columns, `source`, `target` and `type`, indicating which nodes are
related and with which type of link.
The target is a 1d numpy binary integer array indicating whether the indicated link exists or not.


NetworkX + MLPrimitives + Scikit-learn + XGBoost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will go use some `NetworkX Link Prediction` functions to extract attributes
from the Graph, to later on encode the categorical features with the `CategoricalEncoder from
MLPrimitives`_, scale the data using the `StandardScaler from scikit-learn`_ and finally go into
an `XGBClassifier`.

Note how in this example, the Graph objects and the names of the node columns are passed as
additional variables to be added to the context, as the NetworkX primitive will need some
additional information not found inside `X`.

.. code-block:: python

    from mlblocks import MLPipeline
    from mlblocks.datasets import load_umls

    dataset = load_umls()
    dataset.describe()

    X_train, X_test, y_train, y_test = dataset.get_splits(1)

    primitives = [
        'networkx.link_prediction_feature_extraction',
        'mlprimitives.custom.feature_extraction.CategoricalEncoder',
        'sklearn.preprocessing.StandardScaler',
        'xgboost.XGBClassifier'
    ]
    init_params = {
        'xgboost.XGBClassifier': {
            'n_estimators': 300,
            'learning_rate': 0.1
        }
    }
    pipeline = MLPipeline(primitives)

    node_columns = ['source', 'target']
    pipeline.fit(
        X_train,
        y_train,
        graph=dataset.graph,       # These will be set in the pipeline Context
        node_columns=node_columns  # and made available for the networkx primitive
    )

    predictions = pipeline.predict(
        X_test,
        graph=dataset.graph,       # These will be set in the pipeline Context
        node_columns=node_columns  # and made available for the networkx primitive
    )

    dataset.score(y_test, predictions)


.. _NetworkX Link Prediction: https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.link_prediction.html
.. _CategoricalEncoder from MLPrimitives: https://github.com/HDI-Project/MLPrimitives/blob/master/mlblocks_primitives/mlprimitives.custom.feature_extraction.CategoricalEncoder.json
.. _StandardScaler from scikit-learn: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
.. _XGBClassifier: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
