Pipelines
=========

MLPipeline Class
----------------

The ``mlblocks.MLPipeline`` class is the representation of a **pipeline**  within the **MLBlocks**
library, and it works by combining multiple ``MLBlock`` instances, called ``blocks`` in this
context, which then calls in succession for fitting and predicting.

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

For a more detailed description of the methods and their arguments, please check the API Reference
section of the documentation.
