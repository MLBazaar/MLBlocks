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

From sources
------------

The sources for MLBlocks can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    git clone git://github.com/HDI-Project/mlblocks

Or download the `tarball`_:

.. code-block:: console

    curl  -OL https://github.com/HDI-Project/mlblocks/tarball/master

Once you have a copy of the source, you can install it running the next command inside the
project folder:

.. code-block:: console

    $ pip install .

.. _Github repo: https://github.com/HDI-Project/mlblocks
.. _tarball: https://github.com/HDI-Project/mlblocks/tarball/master

Additional Dependencies
-----------------------

The previous commands install the bare minimum requirements to make MLBlocks work, but
additional dependencies should be installed in order to run the `quickstart`_ and various
examples found in the documentation.

The most important of these dependencies is the related project `MLPrimitives`_, which
includes a huge list of primitives ready to be used by **MLBlocks**.

Installing these additional dependencies can be achieved by running the command:

.. code-block:: console

    pip install mlblocks[demo]

if **MLBlocks** was installed from PyPi, or:

.. code-block:: console

    pip install .[demo]

if you installed **MLBlocks** from sources.

.. _quickstart: quickstart.html
.. _MLPrimitives: https://github.com/HDI-Project/MLPrimitives

Development
-----------

If you are installing **MLBlocks** in order to modify its code, the installation must be done
from its sources, in the editable mode, and also including some additional dependencies in
order to be able to run the tests and build the documentation:

.. code-block:: console

    pip install -e .[dev]
