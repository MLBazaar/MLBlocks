Pipelines
=========

The overall goal of **MLBlocks** is to be able to build **Pipelines**.

A **Pipeline** is a sequence of `primitives`_ working together to learn from training data and
later on make predictions on new data as if they were a single object.

Pipelines can have multiple compositions, but they usually start with a set of data cleanup,
feature extraction and feature selection primitives, and end up with one or more estimator
primitives that generate the final predictions.

.. graphviz::

   digraph{
      rankdir=LR;
      "Data Cleanup" -> "Feature Extraction" -> "Feature Selection" -> "Estimator";
   }

Just like primitives, pipelines have `hyperparameters`_, which correspond to the hyperparameters
of the primitives that compose them, and which can be tuned to improve the goodness of their
fitting, increasing the performance of the predictions that they make.

MLPipeline Class
----------------

In **MLBlocks**, a pipeline is represented by the `MLPipeline`_ class, which combines multiple
`MLBlock`_ instances, called ``blocks`` in this context, which then calls in succession for
fitting and predicting.

This one is the object which the user should be mostly interacting with.

These are the expected inputs to create the instance:

* ``blocks``: A list containing the names of the primitives to load as MLBlock instances.
* ``init_params``: Hyperparameters to be used for the MLBlock instances creation, specified as a
  dictionary that contains the name of the blocks as keys and the set of keyword
  arguments to pass to each one of them specified as subdicts.

And it has these available methods:

* ``get_tunable_hyperparameters``: Get a dictionary indicating which hyperparameters can be tuned
  for each primitive, with their types, available ranges, default values and documentation.
* ``get_hyperparameters``: Get a dictionary with the hyperparameter values that each primitive
  is currently using.
* ``set_hyperparameters``: Set new hyperparameters for one or more primitives, passed as a
  dictionary where keys are the name of the blocks to modify and values are nested dictionaries
  with the hyperparameters to set.
* ``fit``: Call the ``fit`` method and then the ``produce`` method of each block, in sequence,
  passing each time as input the output of the ``produce`` method of the previous block.
* ``predict``: Call the ``produce`` method of each block in sequence passing each time as input
  the output of the ``produce`` method of the previous block.

For a more detailed description of the methods and their arguments, please check the corresponding
section in the `API Reference`_ documentation.

Context
-------

One element that plays an important role during the execution of the ``fit`` and ``predict``
methods of a pipeline is the **Context dictionary**.

Each time any of these methods is called, a context dictionary is internally created and all
the variables passed to the method are stored in it.

Then, the following happens for each block:

* The list of arguments that the method expects is retrieved from the the block.
* The correspoding values are read from the context and passed to the method.
* The list of outputs that the method returns is retrieved from the the block.
* The indicated outputs are captured in order and put back to the context dictionary
  using the name specified.

Let's go through an example to develop a better understanding of this:

Additional Arguments
--------------------

TODO: Work in Progress

.. _API Reference: ../api_reference.html
.. _primitives: ../primitives.html
.. _MLPipeline: ../api_reference.html#mlblocks.MLPipeline
.. _MLBlock: ../api_reference.html#mlblocks.MLBlock
.. _hyperparameters: hyperparameters.html
