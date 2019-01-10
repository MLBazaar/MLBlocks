Hyperparameters
===============

A very important element of both Function and Class primitives are the hyperparameters.

The hyperparameters are arguments that modify the behavior of the primitive and its learning
process, which are set before the learning process starts and are not deduced from the data.
These hyperparameters are usually passed as arguments to the primitive constructor or to the
methods or functions that will be called during the fitting or predicting phase.

In **MLBlocks**, each primitive has all its hyperparameters and their valid values specified
on their `JSON Annotations`_.

Here, for example, we are looking at the ``hyperparameters`` section of the
``keras.preprocessing.text.Tokenizer`` primitive from `MLPrimitives`_::

    "hyperparameters: {
        "fixed": {
            "filters": {
                "type": "str",
                "default": "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\n"
            },
            "split": {
                "type": "str",
                "default": " "
            },
            "oov_token": {
                "type": "str",
                "default": null
            }
        },
        "tunable": {
            "num_words": {
                "type": "int",
                "default": null,
                "range": [1, 10000]
            },
            "lower": {
                "type": "bool",
                "default": true
            },
            "char_level": {
                "type": "bool",
                "default": false
            }
        }
    }

As can be seen, two types of hyperparameters exist: **fixed** and **tunable**.

Fixed Hyperparameters
---------------------

These hyperparameters do not alter the learning process, and their values modify
the behavior of the primitive but not its prediction performance. In some cases these
hyperparameters have a default value, but most of the times their values have to be explicitly
set by the user.

In the `JSON Annotations`_, these hyperparameters are specified as a JSON that has the argument
name as the keyword and a nested JSON that specifies its details::

    "fixed": {
        "filters": {
            "type": "str",
            "default": "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\n"
        },
        "split": {
            "type": "str",
            "default": " "
        },
        "oov_token": {
            "type": "str",
            "default": null
        }
    }

Each entry in the ``fixed`` hyperparameters contains:

* **default**: This indicates the default value that the argument will take if the user does
  not specify another value when the `MLPipeline`_ is created. This keyword is optional, and
  if it is not specified, the user expected to always provide a value.
* **type**: The type of the argument. This is only informative and is not used by MLBlocks, but
  it is always included in all the `MLPrimitives`_ annotations.

Tunable Hyperparameters
-----------------------

These hyperparameters do not modify the primitive behaviour, but they have a direct
impact on the learning process and on how well the primitive learns from the data.
For this reason, their values can be tuned to improve the prediction performance.

In the `JSON Annotations`_, these hyperparameters are specified as a JSON that has the argument
name as the keyword and a nested JSON that specifies its details::

    "tunable": {
        "num_words": {
            "type": "int",
            "default": null,
            "range": [1, 10000]
        },
        "lower": {
            "type": "bool",
            "default": true
        },
        "char_level": {
            "type": "bool",
            "default": false
        }
    }

Each entry in the ``tunable`` hyperparameters contains:

* **type**: The type of the argument. This can be one of the primitive variable types, ``int``,
  ``float``, ``str`` or ``bool``, or one of the special types, `multitype`_ or `conditional`_.
* **default**: This indicates the default value that the argument will take if the user does
  not specify another value when the `MLPipeline`_ is created.
* **range**: Optional - This is expected to be found in numeric hyperparameters, and specifies
  the minimum and maximum values that this primitive will work well with.
* **values**: Optional - this is expected to be found in categorical hyperparameters, and
  indicates the list of possible values that it can work with.

Special Hyperparameter Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, hyperparameters do not accept only one type of value, or their possible values may
depend on the value of other hyperparameters.

Multitype Hyperparameters
*************************

Some hyperparameters accept more than one type of value.

For example, suppose a primitive expects a hyperparameter called `max_features` that can take
one of three types:

* An integer indicating the absolute number of features to use.
* A float between 0 and 1 indicating the proportion of the maximum possible number of features.
* The strings ``"min"``, ``"max"`` or ``"mean"``, indicating that the number needs to be computed
  by the primitive itself in some way.

In this case, the ``type`` of this hyperparameter is ``multitype``, and its specification could
be as follows::

    "max_features": {
        "type": "multitype",
        "default": "mean",
        "types": {
            "int": {
                "range": [1, 100]
            },
            "float": {
                "range": [0.1, 0.9]
            },
            "string": {
                "values": ["mean", "min", "max"]
            }
        }
    }

Note how a new keyword ``types`` exist, that holds the possible values for each one of the
possible types that this hyperparameter can have.

Conditional Hyperparameters
***************************

In some other cases, the values that a hyperparameter can take depend on the value of another
one.
For example, sometimes a primitive has a hyperparameter that specifies a kernel, and depending
on the kernel used some other hyperparameters may be or not be used, or they might be able
to take only some specific values.

In this case, the ``type`` of the hyperparameter whose values depend on the other is specified
as ``conditional``.
In this case, two additional entries are required:

* an entry called ``condition``, which specifies the name of the other hyperparameter, the value
  of which is evaluated to decide which values this hyperparameter can take.
* an additional subdictionary called ``values``, which relates the  possible values that the
  `condition` hyperparameter can have with the full specifications of the type and values that
  this hyperparameter can take in each case.

Suppose, for example, that the primitive explained in the previous point does not expect
the ``mean``, ``min`` or ``max`` strings as values for the ``max_features`` hyperparameter,
but as a separated one called ``max_feature_aggregation``, which is only used then the
``max_features`` hyperparameter has been given the value ``auto``.

In this case, the hyperparameters would be annotated like this::

    "max_features": {
        "type": "multitype",
        "default": "auto",
        "types": {
            "int": {
                "range": [1, 100]
            },
            "float": {
                "range": [0.1, 0.9]
            },
            "string": {
                "values": ["auto"]
            }
        }
    }
    "max_features_aggregation": {
        "type": "conditional",
        "condition": "max_features",
        "default": null,
        "values": {
            "auto": {
                "description": "this will be used only if the value of max_features is `auto`",
                "type": "str",
                "default": "mean",
                "range": ["mean", "max", "min"]
            }
        }
    }

.. note:: Just like a regular hyperparameter, if there is no match the default entry is used.
          In this example, the ``null`` value indicates that the hyperparameter needs to be
          disabled if there is no match, but instead of it we could add there a full specification
          of type, range and default value as a nested dictionary to be used by default.

.. _JSON Annotations: primitives.html#json-annotations
.. _MLPrimitives: https://github.com/HDI-Project/MLPrimitives
.. _BTB: https://github.com/HDI-Project/BTB
.. _MLPipeline: ../api_reference.html#mlblocks.MLPipeline
.. _multitype: #multitype-hyperparameters
.. _conditional: #conditional-hyperparameters
