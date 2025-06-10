Pattern Simulation
==================

:mod:`graphix.simulator` module
+++++++++++++++++++++++++++++++++++

.. currentmodule:: graphix.simulator

.. autoclass:: PatternSimulator

    .. automethod:: __init__

    .. automethod:: run

Branch Selection (:mod:`graphix.branch_selector` module)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. currentmodule:: graphix.branch_selector

.. autoclass:: BranchSelector

    .. automethod:: measure

.. autoclass:: RandomBranchSelector
    :members:

    .. automethod:: measure

.. autoclass:: FixedBranchSelector
    :members:

    .. automethod:: measure

.. autoclass:: ConstBranchSelector
    :members:

    .. automethod:: measure

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
