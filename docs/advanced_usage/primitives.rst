Primitives
==========

MLBlocks goal is to seamlessly combine any possible set of Machine Learning tools developed
in Python, whether they are custom developments or belong to third party libraries, and
build Pipelines out of them that can be fitted and then used to make predictions.

We call each one of these Machine Learning tools a **primitive**.

A valid MLBlocks primitive is an importable Python tool that:

* Might be a function or a class.
* If it is a class, it **might** have a `fitting` stage, where the primitive is passed some
  training data and it `learns` from it, and which can be executed with a single method call.
  Function primitives have no `fitting` stage.
* **Must** have a `producing` stage, where the primitive is passed some data and it returns some
  other data, whether it is a transformation of the input data or some new data derived from it,
  such as a set of predictions. This `producing` stage must be executed with a single function or
  method call.
* Might have hyperparameters, additional arguments to be passed to either the function call or
  the class constructor in order to alter or control the way the fitting and producing stages work.

JSON Annotations
----------------

Each integrated primitive has an associated JSON file that specifies its methods, their arguments,
their types and, most importantly, any possible hyperparameter that the primitive has, as well
as their types, ranges and conditions, if any.

These JSON annotations can be:

* **Installed** using the `MLPrimitives`_ related project, which is the recommended approach.
* **Created by the user** and configured for MLBlocks to use them.

And the primitives can be of two types:

* Function Primitives: Simple functions that can be called directly.
* Class Primitives: Class objects that need to be instantiated before they can be used.

Here are some simplified examples of these JSONs, but for more detailed examples, please refer to
the ``examples`` folder.

.. _MLPrimitives: https://github.com/HDI-Project/MLPrimitives

Function Primitives
~~~~~~~~~~~~~~~~~~~

The most simple type of primitives are simple functions that can be called directly, without
the need to created any class instance before.

In most cases, if not all, these functions do not have any associated learning process,
and their behavior is always the same both during the fitting and the predicting phases
of the pipeline.

A simple example of such a primitive would be the ``numpy.argmax`` function, which expects a 2
dimensional array as input, and returns a 1 dimensional array that indicates the index of the
maximum values along an axis.

The simplest JSON annotation for this primitive would look like this::

    {
        "primitive": "numpy.argmax",
        "produce": {
            "args": [
                {
                    "name": "y",
                    "type": "ndarray"
                }
            ],
            "output": [
                {
                    "name": "y",
                    "type": "ndarray"
                }
            ]
        },
        "hyperparameters": {
            "fixed": {
                "axis": {
                    "type": "int",
                    "default": 1
                }
            }
        }
    }

The main elements of this JSON are:

* **primitive**: The fully qualified, directly importable name of the function to be used.
* **produce**: A nested JSON that specifies the names and types of arguments and the output values
  of the primitive.
* **hyperparameters**: A nested JSON that specifies the hyperparameters of this primitive.
  Note that multiple types of hyperparameters exist, but that this primitive has only one ``fixed``
  hyperparameter, which mean that this is not tunable and that, even though the user can specify
  a value different than the default, changes are not expected during the MLBlock intance life cycle.

Class Primitives
~~~~~~~~~~~~~~~~

A more complex type of primitives are classes which need to be instantiated before they can
be used.

In most cases, these classes will have an associated learning process, and they will have some
fit method or equivalent that will be called during the fitting phase but not during the
predicting one.

A simple example of such a primitive would be the ``sklearn.preprocessing.StandardScaler`` class,
which is used to standardize a set of values by calculating their z-score, which means centering
them around 0 and scaling them to unit variance.

This primitive has an associated learning process, where it calculates the mean and standard
deviation of the training data, to later on use them to transform the prediction data to the
same center and scale.

The simplest JSON annotation for this primitive would look like this::

    {
        "primitive": "sklearn.preprocessing.StandardScaler",
        "fit": {
            "method": "fit",
            "args": [
                {
                    "name": "X",
                    "type": "ndarray"
                }
            ]
        },
        "produce": {
            "method": "transform",
            "args": [
                {
                    "name": "X",
                    "type": "ndarray"
                }
            ],
            "output": [
                {
                    "name": "X",
                    "type": "ndarray"
                }
            ]
        },
        "hyperparameters": {
            "tunable": {
                "with_mean": {
                    "type": "bool",
                    "default": true
                },
                "with_std": {
                    "type": "bool",
                    "default": true
                }
            }
        }
    }

Note that there are some details of this JSON annotation that make it different from the
Function Primitive one that explained above:

* **primitive**: The fully qualified, directly importable name of the class to be used. This
  class is the one that will be used to create the actual primitive instance.
* **fit**: A nested JSON that specifies the name of the method to call during the fitting phase,
  which in this case happens to also be ``fit``, as well as the names and types of
  arguments that this method expects.
* **produce**: A nested JSON that specifies the name of the method to call during the predicting
  phase, in this case called ``transform``, as well as the names and types of
  arguments that this method expects and its outputs.
* **hyperparameters**: A nested JSON that specifies the hyperparameters of this primitive.
  In this case, only ``tunable`` hyperparameters are specified, with their
  names and types. If the type was something other than ``bool``, a list or
  range of valid values would also be specified.

MLBlock Class
~~~~~~~~~~~~~

The ``mlblocks.MLBlock`` class is the representation of a primitive within the **MLBlocks**
library.

This is used to wrap the annotated primitives, offering a common and uniform interface to
interact with any possible Machine Learning tool implemented in Python.

These are the inputs required to create an ``MLBlock`` instance:

* ``name``: the name of the primitive JSON to load.
* ``**hyperparameters``: Hyperparameters of the primitive, passed as keyword arguments.

And it has these available methods:

* ``get_tunable_hyperparameters``: Get a dictionary indicating which hyperparameters can be tuned
  for this primitive, with their types, available ranges, default
  values and documentation.
* ``get_hyperparameters``: Get a dictionary with the hyperparameter values that the primitive is using.
* ``set_hyperparameters``: Set new hyperparameters for the primitive, recreating any necessary
  object for the changes to take effect.
* ``fit``: Call the method specified in the JSON annotation passing any required arguments.
* ``produce``: Call the method specified in the JSON annotation passing any required arguments and
  capture its outputs as variables.

For a more detailed description of the methods and their arguments, please check the API Reference
section of the documentation.
