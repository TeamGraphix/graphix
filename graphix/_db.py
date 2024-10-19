"""Database module for Graphix."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from graphix import utils
from graphix.fundamentals import IXYZ, Sign
from graphix.ops import Ops

# 24 unique 1-qubit Clifford gates
_C0 = Ops.I  # I
_C1 = Ops.X  # X
_C2 = Ops.Y  # Y
_C3 = Ops.Z  # Z
_C4 = Ops.S  # S = \sqrt{Z}
_C5 = Ops.SDG  # SDG = S^{\dagger}
_C6 = Ops.H  # H
_C7 = utils.lock(np.asarray([[1, -1j], [-1j, 1]]) / np.sqrt(2))  # \sqrt{iX}
_C8 = utils.lock(np.asarray([[1, -1], [1, 1]]) / np.sqrt(2))  # \sqrt{iY}
_C9 = utils.lock(np.asarray([[0, 1 - 1j], [-1 - 1j, 0]]) / np.sqrt(2))  # sqrt{I}
_C10 = utils.lock(np.asarray([[0, -1 - 1j], [1 - 1j, 0]]) / np.sqrt(2))  # sqrt{-I}
_C11 = utils.lock(np.asarray([[1, -1], [-1, -1]]) / np.sqrt(2))  # sqrt{I}
_C12 = utils.lock(np.asarray([[-1, -1], [1, -1]]) / np.sqrt(2))  # sqrt{-iY}
_C13 = utils.lock(np.asarray([[1j, -1], [1, -1j]]) / np.sqrt(2))  # sqrt{-I}
_C14 = utils.lock(np.asarray([[1j, 1], [-1, -1j]]) / np.sqrt(2))  # sqrt{-I}
_C15 = utils.lock(np.asarray([[-1, -1j], [-1j, -1]]) / np.sqrt(2))  # sqrt{-iX}
_C16 = utils.lock(np.asarray([[-1 + 1j, 1 + 1j], [-1 + 1j, -1 - 1j]]) / 2)  # I^(1/3)
_C17 = utils.lock(np.asarray([[-1 + 1j, -1 - 1j], [1 - 1j, -1 - 1j]]) / 2)  # I^(1/3)
_C18 = utils.lock(np.asarray([[1 + 1j, 1 - 1j], [-1 - 1j, 1 - 1j]]) / 2)  # I^(1/3)
_C19 = utils.lock(np.asarray([[-1 - 1j, 1 - 1j], [-1 - 1j, -1 + 1j]]) / 2)  # I^(1/3)
_C20 = utils.lock(np.asarray([[-1 - 1j, -1 - 1j], [1 - 1j, -1 + 1j]]) / 2)  # I^(1/3)
_C21 = utils.lock(np.asarray([[-1 + 1j, -1 + 1j], [1 + 1j, -1 - 1j]]) / 2)  # I^(1/3)
_C22 = utils.lock(np.asarray([[1 + 1j, -1 - 1j], [1 - 1j, 1 - 1j]]) / 2)  # I^(1/3)
_C23 = utils.lock(np.asarray([[-1 + 1j, 1 - 1j], [-1 - 1j, -1 - 1j]]) / 2)  # I^(1/3)


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

# Human-readable labels
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

# Clifford(CLIFFORD_MUL[i][j]) ~ CLIFFORD[i] @ CLIFFORD[j] (up to phase)
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

# Clifford(CLIFFORD_CONJ[i]) ~ CLIFFORD[i].H (up to phase)
CLIFFORD_CONJ = (0, 1, 2, 3, 5, 4, 6, 15, 12, 9, 10, 11, 8, 13, 14, 7, 20, 22, 23, 21, 16, 19, 17, 18)


class _CM(NamedTuple):
    """Pauli string and sign."""

    pstr: IXYZ
    sign: Sign


class _CMTuple(NamedTuple):
    x: _CM
    y: _CM
    z: _CM


# Conjugation of Pauli gates P with Clifford gate C,
# i.e. C @ P @ C^dagger result in Pauli group, i.e. {\pm} \times {X, Y, Z}.
# CLIFFORD_MEASURE contains the effect of Clifford conjugation of Pauli gates.
CLIFFORD_MEASURE = (
    _CMTuple(_CM(IXYZ.X, Sign.PLUS), _CM(IXYZ.Y, Sign.PLUS), _CM(IXYZ.Z, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.X, Sign.PLUS), _CM(IXYZ.Y, Sign.MINUS), _CM(IXYZ.Z, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.X, Sign.MINUS), _CM(IXYZ.Y, Sign.PLUS), _CM(IXYZ.Z, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.X, Sign.MINUS), _CM(IXYZ.Y, Sign.MINUS), _CM(IXYZ.Z, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.Y, Sign.MINUS), _CM(IXYZ.X, Sign.PLUS), _CM(IXYZ.Z, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.Y, Sign.PLUS), _CM(IXYZ.X, Sign.MINUS), _CM(IXYZ.Z, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.Z, Sign.PLUS), _CM(IXYZ.Y, Sign.MINUS), _CM(IXYZ.X, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.X, Sign.PLUS), _CM(IXYZ.Z, Sign.MINUS), _CM(IXYZ.Y, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.Z, Sign.PLUS), _CM(IXYZ.Y, Sign.PLUS), _CM(IXYZ.X, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.Y, Sign.MINUS), _CM(IXYZ.X, Sign.MINUS), _CM(IXYZ.Z, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.Y, Sign.PLUS), _CM(IXYZ.X, Sign.PLUS), _CM(IXYZ.Z, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.Z, Sign.MINUS), _CM(IXYZ.Y, Sign.MINUS), _CM(IXYZ.X, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.Z, Sign.MINUS), _CM(IXYZ.Y, Sign.PLUS), _CM(IXYZ.X, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.X, Sign.MINUS), _CM(IXYZ.Z, Sign.MINUS), _CM(IXYZ.Y, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.X, Sign.MINUS), _CM(IXYZ.Z, Sign.PLUS), _CM(IXYZ.Y, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.X, Sign.PLUS), _CM(IXYZ.Z, Sign.PLUS), _CM(IXYZ.Y, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.Z, Sign.PLUS), _CM(IXYZ.X, Sign.PLUS), _CM(IXYZ.Y, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.Z, Sign.MINUS), _CM(IXYZ.X, Sign.PLUS), _CM(IXYZ.Y, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.Z, Sign.MINUS), _CM(IXYZ.X, Sign.MINUS), _CM(IXYZ.Y, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.Z, Sign.PLUS), _CM(IXYZ.X, Sign.MINUS), _CM(IXYZ.Y, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.Y, Sign.PLUS), _CM(IXYZ.Z, Sign.PLUS), _CM(IXYZ.X, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.Y, Sign.MINUS), _CM(IXYZ.Z, Sign.MINUS), _CM(IXYZ.X, Sign.PLUS)),
    _CMTuple(_CM(IXYZ.Y, Sign.PLUS), _CM(IXYZ.Z, Sign.MINUS), _CM(IXYZ.X, Sign.MINUS)),
    _CMTuple(_CM(IXYZ.Y, Sign.MINUS), _CM(IXYZ.Z, Sign.PLUS), _CM(IXYZ.X, Sign.MINUS)),
)

# Decomposition of Clifford gates with H, S and Z (up to phase).
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
