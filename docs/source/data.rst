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





