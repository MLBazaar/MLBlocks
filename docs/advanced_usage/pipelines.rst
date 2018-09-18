Pipelines
=========

The overall goal of **MLBlocks** is to be able to build **Pipelines**.

What is a Pipeline?
-------------------

A **Pipeline** is a sequence of `primitives`_ working together to learn from training data and
later on make predictions on new data as if they were a single object.

.. graphviz::

    digraph {
        rankdir=LR;
        "Input Data" -> "Pipeline";
        "Pipeline" -> "Predictions";
    }

Pipelines can have any possible composition, but the typical setup includes data cleanup,
feature extraction and feature selection `primitives`_, and ends up with one or more estimator
`primitives`_ that generate the final predictions.

.. graphviz::

    digraph {
        rankdir=LR;
        subgraph cluster {
            "Data Cleanup" -> "Feature Extraction" -> "Feature Selection" -> "Estimator";
            label = "pipeline";
            color=blue
        }
    }

Also, just like `primitives`_, pipelines have `hyperparameters`_, which are nothing but the list
of hyperparameters that the `primitives`_ that compose them expect, and which can be also tuned to
improve their prediction performance.

The MLPipeline Class
--------------------

In **MLBlocks**, a pipeline is represented by the `mlblocks.MLPipeline`_ class, which combines
multiple `mlblock.MLBlock`_ instances, called ``blocks`` in this context, and calls them in
succession for fitting and predicting.

As you have seen in the `quickstart tutorial`_, You can create an **MLPipeline** by simply
passing it the list of primitives that will compose it:

.. ipython:: python

    from mlblocks import MLPipeline

    primitives = [
        'sklearn.preprocessing.StandardScaler',
        'sklearn.ensemble.RandomForestClassifier'
    ]
    pipeline = MLPipeline(primitives)

Block Names
~~~~~~~~~~~

When the MLPipeline is created, a list of MLBlock instances has been created inside it.

Each of this blocks is given a unique name which is composed by the primitive name and a counter,
separated by a hash symbol ``#``, allowing multiple blocks for the same primitive to exist
within a single MLPipeline:

.. ipython:: python

    dict(pipeline.blocks)

Init Arguments
~~~~~~~~~~~~~~

Sometimes, additional arguments need to be passed to the blocks during instatiation.

This can be done by passing an extra dictionary to the MLPipeline when it is created:

.. ipython:: python

    init_params = {
        'sklearn.preprocessing.StandardScaler': {
            'with_mean': False
        },
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': 100
        }
    }
    pipeline = MLPipeline(primitives, init_params)

This dictionary must have as keys the name of the blocks that the arguments belong to, and
as values the dictionary that contains the argument names and their values.

.. note:: If only one block of a specific primitive exists in the pipeline, the counter
          appended to its name can be skipped when specifying the arguments, as shown in the
          example.

Context
-------

One element that plays an important role during the execution of the `fit`_ and `predict`_
methods of a pipeline is the **Context dictionary**.

Each time any of these methods is called, a context dictionary is internally created and all
the variables passed to the method are stored in it.

Then, the following happens for each block:

* The list of arguments that the method expects is retrieved from the the block config.
* The correspoding values are read from the **Context** and passed to the method.
* The list of outputs that the method returns is retrieved from the the block.
* The indicated outputs are captured in order and put back to the **Context** dictionary
  using the name specified.

Context Usage Examples
~~~~~~~~~~~~~~~~~~~~~~

The most simple version of this schema is one where all the blocks expect a single feature
matrix as input, called `X`, and output another feature matrix, also called `X`, until the last
one, which outputs the final prediction, called `y`.

In this case, supposing that we only have 3 blocks, the sequence when the ``pipeline.predict(X)``
call is issued would be:

1. The value of ``X`` is stored in the **Context**.
2. The value of ``X`` is pulled from the **Context** and passed to `block1`.
3. The output from `block1` is put back into the **Context**, overwritting the old value.
4. The value of ``X`` is pulled again from the **Context** and passed to `block2`.
5. The output from `block2` is put back into the **Context**, overwritting again the old value.
6. The value of ``X`` is pulled for the last time from the **Context** and passed to `block3`.
7. The output from `block3`, since it is the last one, is returned.

.. graphviz::

    digraph G {
        subgraph cluster_0 {
            label = "pipeline.predict(X)";

            b1 [label="block1.produce(X)"];
            b2 [label="block2.produce(X)"];
            b3 [label="block3.produce(X)"];

            b1 -> b2 -> b3 [style=invis];

            subgraph cluster_1 {
                X1 [label=X];
                X2 [label=X];
                X3 [label=X];
                X1 -> X2 -> X3 [style="dashed"];
                label = "Context";
            }

        }

        X -> X1;
        X1 -> b1 [constraint=false];
        b1 -> X2 [label=modified];
        X2 -> b2 [constraint=false]
        b2 -> X3 [label=modified];
        X3 -> b3 [constraint=false]
        b3 -> y
    }

Another schema with some more compelexity would be one where there is one primitive that
needs to be passed an additional argument that provides information about the data.

Suppose, for example, that there is a primitive that encodes categorical features, but
it needs to be given the list of features that it needs to encode in a variable called
``features``. Suppose also that this primitive is followed directly by an estimator primitive.

In this case, the call would be ``pipeline.predict(X, features=features)``, and the sequence
of actions would be:

1. The value of ``X`` and ``features`` is stored in the **Context**.
2. The value of ``X`` and ``features`` is pulled from the **Context** and passed to the
   `encoder` block.
3. The output from `encoder` is put back into the **Context** as ``X``, overwritting the old value.
4. The value of ``X`` is pulled again from the **Context** and passed to the `estimator` block.
5. The output from the `estimator` block is returned.


.. graphviz::

    digraph G {
        subgraph cluster_0 {
            label = "pipeline.predict(X, features=features)";

            b1 [label="encoder.produce(X, features=features)"];
            b2 [label="estimator.produce(X)"];

            b1 -> b2 [style=invis];

            subgraph cluster_1 {
                X1 [label=X];
                f1 [label=features];
                X2 [label=X];
                f1 -> X1 [style=invis];
                X1 -> X2 [style=dashed];
                label = "Context";
            }

        }

        {rank=same X features}
        features -> f1;
        X -> X1;
        {X1 f1} -> b1 [constraint=false];
        b1 -> X2 [label=encoded];
        X2 -> b2 [constraint=false]
        b2 -> y
    }


But, what if we also have a primitive, which we will call `detector`, that detects which features
are categorical and want to use it instead of passing the a manually crafted list of features?

We can also achieve it using the **Context**!

In this case, we go back to the ``pipeline.predict(X)`` call, and let the `detector` primitive
do its job:

1. The value of ``X`` is stored in the **Context**.
2. The value of ``X`` is pulled from the **Context** and passed to the `detector` block.
3. The output from the `detector` block is stored in the **Context** as the `features` variable.
4. The value of ``X`` and ``features`` is pulled from the **Context** and passed to the
   `encoder` block.
5. The output from `encoder` is put back into the **Context** as ``X``, overwritting the old value.
6. The value of ``X`` is pulled again from the **Context** and passed to the `estimator` block.
7. The output from the `estimator` block is returned.


.. graphviz::

    digraph G {
        subgraph cluster_0 {
            label = "pipeline.predict(X)";

            b0 [label="detector.produce(X)"];
            b1 [label="encoder.produce(X, features=features)"];
            b2 [label="estimator.produce(X)"];

            b0 -> b1 -> b2 [style=invis];

            subgraph cluster_1 {
                X1 [label=X];
                f1 [label=features];
                X2 [label=X];
                X1 -> f1 -> X2 [style=invis];
                X1 -> X2 [style=dashed];
                label = "Context";
            }

        }

        X -> X1;
        X1 -> b0 [constraint=false];
        b0 -> f1;
        {X1 f1} -> b1 [constraint=false];
        b1 -> X2 [label=encoded];
        X2 -> b2 [constraint=false]
        b2 -> y
    }


.. _API Reference: ../api_reference.html
.. _primitives: ../primitives.html
.. _MLPipeline: ../api_reference.html#mlblocks.MLPipeline
.. _fit: ../api_reference.html#mlblocks.MLPipeline.fit
.. _predict: ../api_reference.html#mlblocks.MLPipeline.predict
.. _MLBlock: ../api_reference.html#mlblocks.MLBlock
.. _hyperparameters: hyperparameters.html
