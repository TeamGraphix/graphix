Pattern data structure
======================

:mod:`graphix.command` module
+++++++++++++++++++++++++++++

This module defines standard data structure for pattern commands.

.. currentmodule:: graphix.command

.. autoclass:: CommandKind

.. autoclass:: N

.. autoclass:: M

.. autoclass:: E

.. autoclass:: C

.. autoclass:: X

.. autoclass:: Z



:mod:`graphix.pauli` module
+++++++++++++++++++++++++++

This module defines standard data structure for Pauli operators, measurement planes and their transformations.

.. currentmodule:: graphix.pauli

.. autoclass:: Plane
    :members:

.. autoclass:: Pauli
    :members:

.. autoclass:: MeasureUpdate

:mod:`graphix.instruction` module
+++++++++++++++++++++++++++++++++

This module defines standard data structure for gate seqence (circuit model) used for :class:`graphix.transpiler.Circuit`.

.. currentmodule:: graphix.instruction

.. autoclass:: InstructionKind

.. autoclass:: RX

.. autoclass:: RZ

.. autoclass:: RY

.. autoclass:: M

.. autoclass:: X

.. autoclass:: Y

.. autoclass:: Z

.. autoclass:: S

.. autoclass:: H

.. autoclass:: SWAP

.. autoclass:: CNOT

:mod:`graphix.parameter` module
+++++++++++++++++++++++++++++++

This module defines parameter objects and parameterized expressions.
Parameterized expressions can appear in measurement angles in patterns
and rotation angles in circuits, and they can be substituted with
actual values.

The module provides generic interfaces for parameters and expressions,
as well as a simple :class:`Placeholder` class that can be used in
affine expressions (:class:`AffineExpression`). Affine expressions are
sufficient for transpiling and pattern optimizations (such as
standardization, minimization, signal shifting, and Pauli
preprocessing), but they do not support simulation.

Parameter objects that support symbolic simulation with `sympy` are
available in a separate package:
https://github.com/TeamGraphix/graphix-symbolic.

.. currentmodule:: graphix.parameter

.. autoclass:: Expression

.. autoclass:: Parameter

.. autoclass:: AffineExpression

.. autoclass:: Placeholder
