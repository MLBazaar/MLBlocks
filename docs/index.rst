What is MLBlocks?
=================

.. image:: images/mlblocks-logo.png
   :width: 300 px
   :alt: MLBlocks
   :align: center

MLBlocks is a simple framework for seamlessly combining any possible set of Machine Learning
tools developed in Python, whether they are custom developments or belong to third party
libraries, and build Pipelines out of them that can be fitted and then used to make predictions.

This is achieved by providing a simple and intuitive annotation language that allows the
user to specify how to integrate with each tool, here called primitives, in order to provide
a common uniform interface to each one of them.

At a high level:

* Each available primitive has been annotated using a standardized JSON file that specifies its
  native interface, as well as which hyperparameters can be used to tune its behavior.
* A list of primitives that will be combined into a pipeline is provided by the user, optionally
  passing along the hyperparameters to use for each primitive.
* An MLBlock instance is build for each primitive, offering a common interface for all of them.
* The MLBlock instances are then combined into an MLPipeline instance, able to run them all in
  the right order, passing the output from each one as input to the next one.
* The training data is passed to the `MLPipeline.fit` method, which sequentially fits each
  MLBlock instance following the JSON annotation specification.
* The data used to make predictions is passed to the `MLPipeline.predict` method, which uses each
  MLBlock sequentially to obtain the desired predictions.

History
-------

In its first iteration in 2015, MLBlocks was designed for only multi table, multi entity temporal
data. A good reference to see our design rationale at that time is Bryan Collazo’s thesis:

* `Machine learning blocks`_.
  Bryan Collazo. Masters thesis, MIT EECS, 2015.

With recent availability of a multitude of libraries and tools, we decided it was time to integrate
them and expand the library to address other data types: images, text, graph, time series and
integrate with deep learning libraries.

.. toctree::
   :caption: Getting Started
   :titlesonly:

   self
   getting_started/install
   getting_started/quickstart

.. toctree::
   :caption: Advanced Usage
   :maxdepth: 1

   advanced_usage/primitives
   advanced_usage/hyperparameters
   advanced_usage/pipelines
   advanced_usage/adding_primitives

.. toctree::
   :caption: Pipeline Examples
   :maxdepth: 1

   pipeline_examples/single_table
   pipeline_examples/multi_table
   pipeline_examples/text
   pipeline_examples/image
   pipeline_examples/graph

.. toctree::
   :caption: API Reference
   :titlesonly:

   api/mlblocks
   api/mlblocks.datasets
   api/mlblocks.discovery

.. toctree::
   :caption: Resources
   :titlesonly:

   contributing
   authors
   changelog

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Machine learning blocks: https://github.com/HDI-Project/mlblocks
.. _tarball: https://github.com/HDI-Project/mlblocks/tarball/master
