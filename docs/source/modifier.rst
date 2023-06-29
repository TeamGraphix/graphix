Pattern Optimization
====================

:mod:`graphix.pattern` module
+++++++++++++++++++++++++++++++++++

.. currentmodule:: graphix.pattern

.. autoclass:: Pattern

    .. automethod:: __init__

    .. automethod:: simulate_pattern

    .. automethod:: run_pattern

    .. automethod:: perform_pauli_measurements

    .. automethod:: print_pattern

    .. automethod:: standardize

    .. automethod:: shift_signals

    .. automethod:: is_standard

    .. automethod:: get_graph

    .. automethod:: parallelize_pattern

    .. automethod:: minimize_space

    .. automethod:: max_space

    .. automethod:: get_layers

    .. automethod:: to_qasm3


.. autoclass:: CommandNode

    .. automethod:: __init__

    .. automethod:: print_pattern


.. autoclass:: LocalPattern

    .. automethod:: __init__

    .. automethod:: standardize

    .. automethod:: shift_signals

    .. automethod:: get_graph

    .. automethod:: get_pattern


.. autofunction:: measure_pauli

