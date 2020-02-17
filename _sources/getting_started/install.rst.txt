.. highlight:: shell

Installation
============

From PyPi
---------

The simplest and recommended way to install MLBlocks is using `pip`:

.. code-block:: console

    pip install mlblocks

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Additional dependencies
-----------------------

In order to be usable, MLBlocks requires a compatible primitives library.

The official library, required in order to follow the MLBlocks tutorials and documentation examples,
is `MLPrimitives`_, which you can install with this command:

.. code-block:: console

    pip install mlprimitives

.. _MLPrimitives: https://github.com/HDI-Project/MLPrimitives

Install for development
-----------------------

If you are installing **MLBlocks** in order to modify its code, the installation must be done
from its sources, in the editable mode, and also including some additional dependencies in
order to be able to run the tests and build the documentation. Instructions about this process
can be found in the `Contributing guide`_.

.. _Contributing guide: ../contributing.html#get-started
