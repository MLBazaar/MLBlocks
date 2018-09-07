Hyperparameters
===============

A very importnat element of both Function and Class primitives are the hyperparameters.

The hyperparameters are arguments that modify the behavior of the primitive and its learning
process, which are set before the learning process starts and are not deduced from the data.
These hyperparameters are usually passed as arguments to the primitive constructor or to the
methods or funcitons that will be called during the fitting or predicting phase.

Two types of hyperparameters exist:

* **fixed**: These hyperparameters do not alter the learning process, and their values modify
  the behavior of the primitive but not its prediction performance. In some cases these
  hyperparameters have a default value, but most of the times their values have to be explicitly
  set by the user.
* **tunable**: These hyperparameters participate directly in the learning process, and their
  values can alter how well the primitive learns and is able to later on predict. For this reason,
  even though these hyperparameters do not alter the behavior of the primitive, they can be tuned
  to improve the prediction performance.
