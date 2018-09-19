Primitives
==========

MLBlocks goal is to seamlessly combine any possible set of Machine Learning tools developed
in Python, whether they are custom developments or belong to third party libraries, and
build `pipelines`_ out of them that can be fitted and then used to make predictions.

We call each one of these Machine Learning tools a **primitive**.

What is a Primitive?
--------------------

A valid MLBlocks primitive is an importable Python object that:

* Must be either a function or a class.
* If it is a class, it **might** have a `fitting` stage, where the primitive is passed some
  training data and it `learns` from it, and which can be executed with a single method call.
  Function primitives have no `fitting` stage.
* **Must** have a `producing` stage, where the primitive is passed some data and it returns some
  other data, whether it is a transformation of the input data or some new data derived from it,
  such as a set of predictions. This `producing` stage must be executed with a single function or
  method call.
* Might have `hyperparameters`_, additional arguments to be passed to either the function call or
  the class constructor in order to alter or control the way the fitting and producing stages work.

Here are some examples of primitives:

+-----------------------------------------------+-----------+--------------+--------------------+
| primitive                                     | type      | fit          | produce            |
+===============================================+===========+==============+====================+
| sklearn.preprocessing.StandardScaler          | class     | fit          | transform          |
+-----------------------------------------------+-----------+--------------+--------------------+
| sklearn.ensemble.RandomForestClassifier       | class     | fit          | predict            |
+-----------------------------------------------+-----------+--------------+--------------------+
| skimage.feature.hog                           | function  | --           | --                 |
+-----------------------------------------------+-----------+--------------+--------------------+
| xgboost.XGBRegressor                          | class     | fit          | predict            |
+-----------------------------------------------+-----------+--------------+--------------------+
| keras.applications.resnet50.preprocess_input  | function  | --           | --                 |
+-----------------------------------------------+-----------+--------------+--------------------+
| keras.applications.resnet50.ResNet50          | class     | --           | predict            |
+-----------------------------------------------+-----------+--------------+--------------------+
| keras.preprocessing.sequence.pad_sequences    | function  | --           | --                 |
+-----------------------------------------------+-----------+--------------+--------------------+
| keras.preprocessing.text.Tokenizer            | class     | fit_on_texts | texts_to_sequences |
+-----------------------------------------------+-----------+--------------+--------------------+
| lightfm.LightFM                               | class     | fit          | predict            |
+-----------------------------------------------+-----------+--------------+--------------------+

JSON Annotations
----------------

Each integrated primitive has an associated JSON file that specifies its methods, their arguments,
their types and, most importantly, any possible `hyperparameters`_ that the primitive has, as well
as their types, ranges and conditions, if any.

These JSON annotations can be:

* **Installed** using the `MLPrimitives`_ related project, which is the recommended approach.
* **Created by the user** and configured for MLBlocks to use them.

And the primitives can be of two types:

* Function Primitives: Simple functions that can be called directly.
* Class Primitives: Class objects that need to be instantiated before they can be used.

Here are some simplified examples of these JSONs, but for more detailed examples, please refer to
the `examples folder`_ of the project.

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

* **primitive**: The fully qualified, directly importable name of the function to be used::

    "primitive": "numpy.argmax",

* **produce**: A nested JSON that specifies the names and types of arguments and the output values
  of the primitive::

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
    }

* **hyperparameters**: A nested JSON that specifies the `hyperparameters`_ of this primitive.
  Note that multiple types of hyperparameters exist, but that this primitive has only one ``fixed``
  hyperparameter, which mean that this is not tunable and that, even though the user can specify
  a value different than the default, changes are not expected during the MLBlock instance life
  cycle::

    "hyperparameters": {
        "fixed": {
            "axis": {
                "type": "int",
                "default": 1
            }
        }
    }

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
  class is the one that will be used to create the actual primitive instance::

    "primitive": "sklearn.preprocessing.StandardScaler",

* **fit**: A nested JSON that specifies the name of the method to call during the fitting phase,
  which in this case happens to also be ``fit``, as well as the names and types of
  arguments that this method expects::

    "fit": {
        "method": "fit",
        "args": [
            {
                "name": "X",
                "type": "ndarray"
            }
        ]
    }

* **produce**: A nested JSON that specifies the name of the method to call during the predicting
  phase, in this case called ``transform``, as well as the names and types of
  arguments that this method expects and its outputs::

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
    }

* **hyperparameters**: A nested JSON that specifies the hyperparameters of this primitive.
  In this case, only ``tunable`` hyperparameters are specified, with their
  names and types. If the type was something other than ``bool``, a list or
  range of valid values would also be specified::

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

The MLBlock Class
-----------------

Within the **MLBlocks** library, a primitive is represented through the `mlblocks.MLBlock`_ class.

This is used to wrap around the annotated primitives, offering a common and uniform interface to
all of them.

More specifically, the `mlblocks.MLBlock`_ class offers two public methods, `fit`_ and `produce`_,
which are directly linked to the methods specified in the JSON Annotation:

For example, we can look at the `keras.preprocessing.text.Tokenizer`_ primitive from
`MLPrimitives`_, which calls the method ``fit_on_texts`` when ``fit`` is called, and
``tests_to_sequences`` when ``produce`` is called:

.. graphviz::

    digraph {
        {
            node [shape=box]
            fit_on_texts;
            texts_to_sequences;
            fit;
            produce;
        }
        subgraph cluster_1 {
            {rank=same; fit produce};
            fit -> produce [style=invis];
            fit -> fit_on_texts;
            produce -> texts_to_sequences;
            label = "mlblocks.MLBlock";
            subgraph cluster_2 {
                fit_on_texts;
                texts_to_sequences;
                label = "keras.preprocessing.text.Tokenizer";
            }
        }
    }

For a more detailed description of this class, please check the corresponding
section in the `API Reference`_ documentation.

.. _API Reference: ../api_reference.html
.. _MLPrimitives: https://github.com/HDI-Project/MLPrimitives
.. _keras.preprocessing.text.Tokenizer: https://github.com/HDI-Project/MLPrimitives/blob/master/mlblocks_primitives/keras.preprocessing.text.Tokenizer.json
.. _hyperparameters: hyperparameters.html
.. _mlblocks.MLBlock: ../api_reference.html#mlblocks.MLBlock
.. _pipelines: pipelines.html
.. _examples folder: https://github.com/HDI-Project/MLBlocks/tree/master/examples
.. _fit: ../api_reference.html#mlblocks.MLBlock.fit
.. _produce: ../api_reference.html#mlblocks.MLBlock.produce
