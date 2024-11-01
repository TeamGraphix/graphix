Pattern Manipulation
====================

:mod:`graphix.pattern` module
+++++++++++++++++++++++++++++++++++

.. currentmodule:: graphix.pattern

.. autoclass:: Pattern

    .. automethod:: __init__

    .. automethod:: add

    .. automethod:: extend

    .. automethod:: clear

    .. automethod:: replace

    .. automethod:: reorder_output_nodes

    .. automethod:: reorder_input_nodes

    .. automethod:: simulate_pattern

    .. automethod:: get_max_degree

    .. automethod:: get_angles

    .. automethod:: get_vops

    .. automethod:: connected_nodes

    .. automethod:: run_pattern

    .. automethod:: perform_pauli_measurements

    .. automethod:: print_pattern

    .. automethod:: standardize

    .. automethod:: shift_signals

    .. automethod:: is_standard

    .. automethod:: get_graph

    .. automethod:: parallelize_pattern

    .. automethod:: minimize_space

    .. automethod:: draw_graph

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
