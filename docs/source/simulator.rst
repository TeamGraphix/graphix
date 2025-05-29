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

Tensor Network
--------------------------

.. currentmodule:: graphix.sim.tensornet

.. autoclass:: TensorNetworkBackend
    :members:

.. autofunction:: gen_str

.. autofunction:: outer_product

Statevector
-----------

.. currentmodule:: graphix.sim.statevec

.. autoclass:: StatevectorBackend
    :members:

.. autoclass:: Statevec
    :members:

Density Matrix
--------------

.. currentmodule:: graphix.sim.density_matrix

.. autoclass:: DensityMatrixBackend
    :members:

.. autoclass:: DensityMatrix
    :members:
