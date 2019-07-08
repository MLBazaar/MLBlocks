Adding Primitives
=================

The **MLBlocks** library is only the engine and it has no use without primitives, so here we
explain how to add new primitives for **MLBlocks**.

MLPrimitives
------------

**MLBlocks** has a related project, `MLPrimitives`_, which already includes a huge list of
integrated primitives, so the easiest and recommended way to add primitives for **MLBlocks**
is to install **MLPrimitives**.

This can be achieved by running the commands::

    pip install mlprimitives

For further details, please refer to the `MLPrimitives Documentation`_.

.. _MLPrimitives: https://github.com/HDI-Project/MLPrimitives
.. _MLPrimitives Documentation: https://hdi-project.github.io/MLPrimitives/

Writing Primitives
------------------

Sometimes you will find that you want to use a primitive that is not in the list of
`MLPrimitives integrated primitives`_, so you will have to integrate the primitive yourself
by writing the corresponding `JSON annotation <primitives.html#json-annotations>`_.

.. _MLPrimitives integrated primitives: https://github.com/HDI-Project/MLPrimitives/tree/master/mlblocks_primitives

.. note:: If you create new primitives for MLBlocks, please consider contributing them to the
          **MLPrimitives** project!

The first thing to do when adding a new primitive is making sure that it complies with the
necessary requirements, which depend on whether the primitive is a function or a class.

For `Function Primitives`_, the only requirement is that they have to be a single function.
Calling multiple functions sequentially as part of a single primitive is not supported, and
in order to achieve this you are expected to write a separated primitive for each function.

For `Class Primitives`_, just like the function primitives, the `fit` and `produce` phases
must consist of a single method each. Calling multiple methods sequentially within a single
primitive is not supported either.

`Class Primitives`_ also need to be able to be instantiated at once. Running setup or compiling
calls after the instance creation is not possible.

.. _Function Primitives: primitives.html#function-primitives
.. _Class Primitives: primitives.html#class-primitives

Primitives Lookup
-----------------

Once you have written the JSON annotation for your primitive, you will need to put in it in a
place known to **MLBlocks**.

**MLBlocks** looks for primitives in the following folders, in this order:

1. Any folder specified by the user, starting by the latest one.
2. A folder named ``mlblocks_primitives`` or ``mlprimitives`` in the current working directory.
3. A folder named ``mlblocks_primitives`` or ``mlprimitives`` in the `system prefix`_.

.. _system prefix: https://docs.python.org/3/library/sys.html#sys.prefix

The list of folders where **MLBlocks** will search for primitives can be seen at any time
by calling the method `mlblocks.get_primitives_paths`_.

.. _mlblocks.get_primitives_paths: ../api_reference.html#mlblocks.get_primitives_paths

Adding a Primitives Folder
--------------------------

The simplest option in order to quickly add new primitives is to put their JSON annotations
in a folder called `mlblocks_primitives` in the root of your project, or in your current
working directory.

However, sometimes you will want to add a custom directory.

This can be easily done by using the `mlblocks.add_primitives_path`_ method.

.. _mlblocks.add_primitives_path: ../api_reference.html#mlblocks.add_primitives_path

Developing a Primitives Library
-------------------------------

Another option to add multiple libraries is creating a primitives library, such as
`MLPrimitives`_.

In order to make **MLBLocks** able to find the primitives defined in such a library,
all you need to do is setting up an `Entry Point`_ in your `setup.py` script with the
following specification:

1. It has to be published under the group ``mlblocks``.
2. It has to be named exactly ``primitives``.
3. It has to point at a variable that contains a path or a list of paths to the JSONS folder(s).

An example of such an entry point would be::

    entry_points = {
        'mlblocks': [
            'primitives=some_module:SOME_VARIABLE'
        ]
    }

where the module `some_module` contains a variable such as::

    SOME_VARIABLE = 'path/to/primitives'

or::

    SOME_VARIABLE = [
        'path/to/primitives',
        'path/to/more/primitives'
    ]

.. _Entry Point: https://packaging.python.org/specifications/entry-points/
