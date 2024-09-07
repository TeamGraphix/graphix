
Miscellaneous modules
=====================

:mod:`graphix.clifford` module
++++++++++++++++++++++++++++++

.. automodule:: graphix.clifford

.. currentmodule:: graphix.clifford

.. autoclass:: graphix.clifford.Clifford

.. data:: graphix.cliffford.CLIFFORD

    list of 24 unique single-qubit Clifford operators as numpy array.

.. data:: graphix.clifford.CLIFFORD_MUL

   the matrix multiplication of single-qubit Clifford gates, expressed as a mapping of CLIFFORD indices. This is possible because multiplication of two Clifford gates result in a Clifford gate.

.. data:: graphix.clifford.CLIFFORD_MEASURE

    The mapping of Pauli operators under conjugation by single-qubit Clifford gates, expressed as a mapping into Pauli operator indices (in CLIFFORD) and sign.


