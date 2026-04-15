Optimization passes
===================

:mod:`graphix.optimization` module
++++++++++++++++++++++++++++++++++

This module defines some optimization passes for patterns.

.. currentmodule:: graphix.optimization

.. autofunction:: standardize

.. autoclass:: StandardizedPattern

.. autofunction:: incorporate_pauli_results

.. autofunction:: remove_useless_domains

.. autofunction:: single_qubit_domains

:mod:`graphix.space_minimization` module
++++++++++++++++++++++++++++++++++++++++

This module defines space minimization procedures for patterns.

.. currentmodule:: graphix.space_minimization

.. autofunction:: minimize_space

.. autofunction:: minimization_using_causal_flow

.. autofunction:: keep_measurement_order_unchanged

.. autofunction:: greedy_minimization_by_degree
