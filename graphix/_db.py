"""Database module for Graphix."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _lock(m: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Create a true immutable view."""
    m.flags.writeable = False
    v = m.view()
    assert not v.flags.writeable
    return v


# 24 Unique 1-qubit Clifford gates
_C0 = _lock(np.array([[1, 0], [0, 1]], dtype=np.complex128))  # identity
_C1 = _lock(np.array([[0, 1], [1, 0]], dtype=np.complex128))  # X
_C2 = _lock(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))  # Y
_C3 = _lock(np.array([[1, 0], [0, -1]], dtype=np.complex128))  # Z
_C4 = _lock(np.array([[1, 0], [0, 1j]], dtype=np.complex128))  # S = \sqrt{Z}
_C5 = _lock(np.array([[1, 0], [0, -1j]], dtype=np.complex128))  # S dagger
_C6 = _lock(np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2))  # Hadamard
_C7 = _lock(np.array([[1, -1j], [-1j, 1]], dtype=np.complex128) / np.sqrt(2))  # \sqrt{iX}
_C8 = _lock(np.array([[1, -1], [1, 1]], dtype=np.complex128) / np.sqrt(2))  # \sqrt{iY}
_C9 = _lock(np.array([[0, 1 - 1j], [-1 - 1j, 0]], dtype=np.complex128) / np.sqrt(2))  # sqrt{I}
_C10 = _lock(np.array([[0, -1 - 1j], [1 - 1j, 0]], dtype=np.complex128) / np.sqrt(2))  # sqrt{-I}
_C11 = _lock(np.array([[1, -1], [-1, -1]], dtype=np.complex128) / np.sqrt(2))  # sqrt{I}
_C12 = _lock(np.array([[-1, -1], [1, -1]], dtype=np.complex128) / np.sqrt(2))  # sqrt{-iY}
_C13 = _lock(np.array([[1j, -1], [1, -1j]], dtype=np.complex128) / np.sqrt(2))  # sqrt{-I}
_C14 = _lock(np.array([[1j, 1], [-1, -1j]], dtype=np.complex128) / np.sqrt(2))  # sqrt{-I}
_C15 = _lock(np.array([[-1, -1j], [-1j, -1]], dtype=np.complex128) / np.sqrt(2))  # sqrt{-iX}
_C16 = _lock(np.array([[-1 + 1j, 1 + 1j], [-1 + 1j, -1 - 1j]], dtype=np.complex128) / 2)  # I^(1/3)
_C17 = _lock(np.array([[-1 + 1j, -1 - 1j], [1 - 1j, -1 - 1j]], dtype=np.complex128) / 2)  # I^(1/3)
_C18 = _lock(np.array([[1 + 1j, 1 - 1j], [-1 - 1j, 1 - 1j]], dtype=np.complex128) / 2)  # I^(1/3)
_C19 = _lock(np.array([[-1 - 1j, 1 - 1j], [-1 - 1j, -1 + 1j]], dtype=np.complex128) / 2)  # I^(1/3)
_C20 = _lock(np.array([[-1 - 1j, -1 - 1j], [1 - 1j, -1 + 1j]], dtype=np.complex128) / 2)  # I^(1/3)
_C21 = _lock(np.array([[-1 + 1j, -1 + 1j], [1 + 1j, -1 - 1j]], dtype=np.complex128) / 2)  # I^(1/3)
_C22 = _lock(np.array([[1 + 1j, -1 - 1j], [1 - 1j, 1 - 1j]], dtype=np.complex128) / 2)  # I^(1/3)
_C23 = _lock(np.array([[-1 + 1j, 1 - 1j], [-1 - 1j, -1 - 1j]], dtype=np.complex128) / 2)  # I^(1/3)


# list of unique 1-qubit Clifford gates
CLIFFORD = (
    _C0,
    _C1,
    _C2,
    _C3,
    _C4,
    _C5,
    _C6,
    _C7,
    _C8,
    _C9,
    _C10,
    _C11,
    _C12,
    _C13,
    _C14,
    _C15,
    _C16,
    _C17,
    _C18,
    _C19,
    _C20,
    _C21,
    _C22,
    _C23,
)

for op in CLIFFORD:
    # Prevent modification
    op.flags.writeable = False

# readable labels for the 1-qubit Clifford
CLIFFORD_LABEL = (
    "I",
    "X",
    "Y",
    "Z",
    "S",
    "Sdagger",
    "H",
    r"\sqrt{iX}",
    r"\sqrt{iY}",
    r"\sqrt{I}",
    r"\sqrt{-I}",
    r"\sqrt{I}",
    r"\sqrt{-iY}",
    r"\sqrt{-I}",
    r"\sqrt{-I}",
    r"\sqrt{-iX}",
    "I^{1/3}",
    "I^{1/3}",
    "I^{1/3}",
    "I^{1/3}",
    "I^{1/3}",
    "I^{1/3}",
    "I^{1/3}",
    "I^{1/3}",
    "I^{1/3}",
)

# Multiplying single-qubit Clifford gates result in a single-qubit Clifford gate.
# CLIFFORD_MUL provides the result of Clifford gate multiplications by Clifford index (see above).
CLIFFORD_MUL = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
    (1, 0, 3, 2, 9, 10, 8, 15, 6, 4, 5, 12, 11, 14, 13, 7, 19, 18, 17, 16, 22, 23, 20, 21),
    (2, 3, 0, 1, 10, 9, 11, 13, 12, 5, 4, 6, 8, 7, 15, 14, 17, 16, 19, 18, 23, 22, 21, 20),
    (3, 2, 1, 0, 5, 4, 12, 14, 11, 10, 9, 8, 6, 15, 7, 13, 18, 19, 16, 17, 21, 20, 23, 22),
    (4, 10, 9, 5, 3, 0, 20, 16, 23, 1, 2, 22, 21, 19, 18, 17, 14, 13, 7, 15, 12, 6, 8, 11),
    (5, 9, 10, 4, 0, 3, 21, 18, 22, 2, 1, 23, 20, 17, 16, 19, 7, 15, 14, 13, 6, 12, 11, 8),
    (6, 12, 11, 8, 19, 16, 0, 20, 3, 17, 18, 2, 1, 23, 22, 21, 5, 9, 10, 4, 7, 15, 14, 13),
    (7, 15, 14, 13, 21, 22, 19, 1, 16, 23, 20, 17, 18, 2, 3, 0, 6, 12, 11, 8, 5, 9, 10, 4),
    (8, 11, 12, 6, 16, 19, 1, 22, 2, 18, 17, 3, 0, 21, 20, 23, 10, 4, 5, 9, 15, 7, 13, 14),
    (9, 5, 4, 10, 2, 1, 22, 19, 21, 0, 3, 20, 23, 16, 17, 18, 13, 14, 15, 7, 11, 8, 6, 12),
    (10, 4, 5, 9, 1, 2, 23, 17, 20, 3, 0, 21, 22, 18, 19, 16, 15, 7, 13, 14, 8, 11, 12, 6),
    (11, 8, 6, 12, 18, 17, 2, 23, 1, 16, 19, 0, 3, 20, 21, 22, 9, 5, 4, 10, 13, 14, 15, 7),
    (12, 6, 8, 11, 17, 18, 3, 21, 0, 19, 16, 1, 2, 22, 23, 20, 4, 10, 9, 5, 14, 13, 7, 15),
    (13, 14, 15, 7, 22, 21, 18, 3, 17, 20, 23, 16, 19, 0, 1, 2, 11, 8, 6, 12, 9, 5, 4, 10),
    (14, 13, 7, 15, 20, 23, 17, 2, 18, 22, 21, 19, 16, 1, 0, 3, 12, 6, 8, 11, 4, 10, 9, 5),
    (15, 7, 13, 14, 23, 20, 16, 0, 19, 21, 22, 18, 17, 3, 2, 1, 8, 11, 12, 6, 10, 4, 5, 9),
    (16, 17, 18, 19, 6, 8, 15, 10, 14, 11, 12, 13, 7, 9, 5, 4, 20, 21, 22, 23, 0, 1, 2, 3),
    (17, 16, 19, 18, 11, 12, 14, 4, 15, 6, 8, 7, 13, 5, 9, 10, 23, 22, 21, 20, 2, 3, 0, 1),
    (18, 19, 16, 17, 12, 11, 13, 9, 7, 8, 6, 15, 14, 10, 4, 5, 21, 20, 23, 22, 3, 2, 1, 0),
    (19, 18, 17, 16, 8, 6, 7, 5, 13, 12, 11, 14, 15, 4, 10, 9, 22, 23, 20, 21, 1, 0, 3, 2),
    (20, 21, 22, 23, 15, 14, 4, 12, 5, 13, 7, 9, 10, 11, 8, 6, 0, 1, 2, 3, 16, 17, 18, 19),
    (21, 20, 23, 22, 13, 7, 5, 6, 4, 15, 14, 10, 9, 8, 11, 12, 3, 2, 1, 0, 18, 19, 16, 17),
    (22, 23, 20, 21, 7, 13, 9, 11, 10, 14, 15, 4, 5, 12, 6, 8, 1, 0, 3, 2, 19, 18, 17, 16),
    (23, 22, 21, 20, 14, 15, 10, 8, 9, 7, 13, 5, 4, 6, 12, 11, 2, 3, 0, 1, 17, 16, 19, 18),
)

# Conjugation of Clifford gates result in a Clifford gate.
# CLIFFORD_CONJ provides the Clifford index of conjugated matrix.
# Example (S and S dagger):  CLIFFORD_CONJ[4] = 5
# WARNING: CLIFFORD[i].conj().T is not necessarily equal to
# CLIFFORD[CLIFFORD_CONJ[i]] in general: the phase may differ.
# For instance, CLIFFORD[7].conj().T = - CLIFFORD[CLIFFORD_CONJ[7]]
CLIFFORD_CONJ = (0, 1, 2, 3, 5, 4, 6, 15, 12, 9, 10, 11, 8, 13, 14, 7, 20, 22, 23, 21, 16, 19, 17, 18)

# Conjugation of Pauli gates P with Clifford gate C,
# i.e. C @ P @ C^dagger result in Pauli group, i.e. {\pm} \times {X, Y, Z}.
# CLIFFORD_MEASURE contains the effect of Clifford conjugation of Pauli gates.
# Example(H gate): CLIFFORD_MEASURE[6] = ((2, 0), (1, 1), (0, 0))
# first item is the result of conjugation of X gate, with first item of the tuple
# being the Clifford index of resulting gate and second item giving sign (+ for 0 and - for 1).
# i.e. HXH = X, HYH = -Y, HZH = X
CLIFFORD_MEASURE = (
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0)),
    ((1, 1), (0, 0), (2, 0)),
    ((1, 0), (0, 1), (2, 0)),
    ((2, 0), (1, 1), (0, 0)),
    ((0, 0), (2, 1), (1, 0)),
    ((2, 0), (1, 0), (0, 1)),
    ((1, 1), (0, 1), (2, 1)),
    ((1, 0), (0, 0), (2, 1)),
    ((2, 1), (1, 1), (0, 1)),
    ((2, 1), (1, 0), (0, 0)),
    ((0, 1), (2, 1), (1, 1)),
    ((0, 1), (2, 0), (1, 0)),
    ((0, 0), (2, 0), (1, 1)),
    ((2, 0), (0, 0), (1, 0)),
    ((2, 1), (0, 0), (1, 1)),
    ((2, 1), (0, 1), (1, 0)),
    ((2, 0), (0, 1), (1, 1)),
    ((1, 0), (2, 0), (0, 0)),
    ((1, 1), (2, 1), (0, 0)),
    ((1, 0), (2, 1), (0, 1)),
    ((1, 1), (2, 0), (0, 1)),
)

# Decomposition of Clifford gates with H, S and Z.
CLIFFORD_HSZ_DECOMPOSITION = (
    (0,),
    (6, 3, 6),
    (6, 3, 6, 3),
    (3,),
    (4,),
    (4, 3),
    (6,),
    (4, 3, 6, 4, 3),
    (6, 3),
    (3, 6, 3, 6, 3, 4),
    (6, 3, 6, 3, 4),
    (6, 6, 3, 6, 3),
    (3, 6),
    (4, 6, 3, 4, 6, 3, 6),
    (4, 6, 3, 4),
    (4, 3, 6, 4, 3, 6, 3, 6),
    (6, 4, 3),
    (6, 3, 6, 3, 6, 4, 3),
    (3, 6, 3, 4),
    (6, 4),
    (4, 6),
    (3, 6, 4, 3, 6, 4, 3),
    (4, 3, 6, 3),
    (4, 6, 3),
)


# OpenQASM3 representation of Clifford gates above.
CLIFFORD_TO_QASM3 = (
    ("id",),
    ("x",),
    ("y",),
    ("z",),
    ("s",),
    ("sdg",),
    ("h",),
    ("sdg", "h", "sdg"),
    ("h", "x"),
    ("sdg", "y"),
    ("sdg", "x"),
    ("h", "y"),
    ("h", "z"),
    ("sdg", "h", "sdg", "y"),
    ("sdg", "h", "s"),
    ("sdg", "h", "sdg", "x"),
    ("sdg", "h"),
    ("sdg", "h", "y"),
    ("sdg", "h", "z"),
    ("sdg", "h", "x"),
    ("h", "s"),
    ("h", "sdg"),
    ("h", "x", "sdg"),
    ("h", "x", "s"),
)
