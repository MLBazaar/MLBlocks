Adding Primitives
=================

The **MLBlocks** library is only the engine, and it has no use without primitives, so here we
explain how to add new primitives for **MLBlocks**.

MLPrimitives
------------

**MLBlocks** has a cousin project, `MLPrimitives`_,
which already includes a huge list of integrated primitives, so the easiest way to add
primitives for **MLBlocks** is to install **MLPrimitives**.

This can be achieved by running the commands::

    pip install mlprimitives

For further details, please refer to the `MLPrimitives Documentation`_.

.. _MLPrimitives: https://github.com/HDI-Project/MLPrimitives
.. _MLPrimitives Documentation: https://hdi-project.github.io/MLPrimitives/

Writing Primitives
------------------

Sometimes you will find that you want to use a primitive that is not in the list of
`MLPrimitives integrated primitives`_, so you will have to integrate the primitive yourself.

.. _MLPrimitives integrated primitives: https://github.com/HDI-Project/MLPrimitives/tree/master/mlblocks_primitives

.. note:: If you integrate new primitives, please consider contributing them to the **MLPrimitives**
          project!

Writing a primitives library
----------------------------

TODO: explain here the entry_points when implemented.
