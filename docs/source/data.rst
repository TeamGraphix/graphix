Pattern data structure
======================

:mod:`graphix.command` module
+++++++++++++++++++++++++++++

This module defines standard data structure for pattern commands.

.. automodule:: graphix.command

.. currentmodule:: graphix.command

.. autoclass:: CommandKind

.. autoclass:: N

.. autoclass:: M

.. autoclass:: E

.. autoclass:: C

.. autoclass:: X

.. autoclass:: Z

.. autoclass:: MeasureUpdate


:mod:`graphix.fundamentals` module
++++++++++++++++++++++++++++++++++

This module defines standard data structure for Pauli operators.

.. automodule:: graphix.fundamentals

.. currentmodule:: graphix.fundamentals

.. autoclass:: Axis
    :members:

.. autoclass:: ComplexUnit
    :members:

.. autoclass:: Sign
    :members:

.. autoclass:: IXYZ
    :members:

.. autoclass:: Plane
    :members:

:mod:`graphix.pauli` module
+++++++++++++++++++++++++++

This module defines standard data structure for Pauli operators.

.. automodule:: graphix.pauli

.. currentmodule:: graphix.pauli

.. autoclass:: Pauli

:mod:`graphix.instruction` module
+++++++++++++++++++++++++++++++++

This module defines standard data structure for gate seqence (circuit model) used for :class:`graphix.transpiler.Circuit`.

.. automodule:: graphix.instruction

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

:mod:`graphix.states` module
++++++++++++++++++++++++++++

.. automodule:: graphix.states

.. currentmodule:: graphix.states

.. autoclass:: State
