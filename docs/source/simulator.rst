Pattern Simulation
==================

:mod:`graphix.simulator` module
+++++++++++++++++++++++++++++++++++

.. currentmodule:: graphix.simulator

.. autoclass:: PatternSimulator

    .. automethod:: __init__

    .. automethod:: run


Simulator backends
++++++++++++++++++

Matrix Product State (MPS)
--------------------------

.. currentmodule:: graphix.sim.mps

.. autoclass:: MPS
    :members:

Statevector
-----------

.. currentmodule:: graphix.sim.statevec

.. autoclass:: StatevectorBackend
    :members:

.. autofunction:: meas_op

.. autoclass:: Statevec
    :members:
