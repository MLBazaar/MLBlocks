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
name as the keyword and a nested JSON that specifies its details:

* **default**: This indicates the default value that the argument will take if the user does
  not specify another value when the `MLPipeline`_ is created. This keyword is optional, and
  if it is not specified, the user expected to always provide a value.
* **type**: The type of the argument. This is only informative and is ignored by MLBlocks, but
  it is always included in all the `MLPrimitives`_ annotations.

Tunable Hyperparameters
-----------------------

These hyperparameters do not modify the primitive behaviour, but they have a direct
impact on the learning process and on how well the primitive learns from the data.
For this reason, their values can be tuned to improve the prediction performance.

In the `JSON Annotations`_, these hyperparameters are specified as a JSON that has the argument
name as the keyword and a nested JSON that specifies its details:

* **type**: The type of the argument.
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

TODO: Work in progress

Conditional Hyperparameters
***************************

TODO: Work in progress


.. _JSON Annotations: primitives.html#json-annotations
.. _MLPrimitives: https://github.com/HDI-Project/MLPrimitives
.. _BTB: https://github.com/HDI-Project/BTB
.. _MLPipeline: ../api_reference.html#mlblocks.MLPipeline
