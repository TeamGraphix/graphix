Miscellaneous modules
=====================

:mod:`graphix.graphsim` module
+++++++++++++++++++++++++++++++++++

This provides efficient graph state simulator using the decorated graph method.

.. currentmodule:: graphix.graphsim

.. autoclass:: GraphState
    :members:
    :undoc-members:


:mod:`graphix.gflow` module
+++++++++++++++++++++++++++++++++++

.. currentmodule:: graphix.gflow

.. autofunction:: generate_from_graph

.. autofunction:: gflow

.. autofunction:: flow

.. autofunction:: solvebool

:mod:`graphix.clifford` module
+++++++++++++++++++++++++++++++++++

.. currentmodule:: graphix.clifford

.. data:: graphix.clifford.CLIFFORD

    list of 24 unique single-qubit Clifford operators as numpy array.

.. data:: graphix.clifford.CLIFFORD_MUL

   the matrix multiplication of single-qubit Clifford gates, expressed as a mapping of CLIFFORD indices. This is possible because multiplication of two Clifford gates result in a Clifford gate.

.. data:: graphix.clifford.CLIFFORD_MEASURE

    The mapping of Pauli operators under conjugation by single-qubit Clifford gates, expressed as a mapping into Pauli operator indices (in CLIFFORD) and sign.


