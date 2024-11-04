"""24 Unique single-qubit Clifford gates and their multiplications, conjugations and Pauli conjugations."""

from __future__ import annotations

import copy
import math
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import typing_extensions

from graphix._db import (
    CLIFFORD,
    CLIFFORD_CONJ,
    CLIFFORD_HSZ_DECOMPOSITION,
    CLIFFORD_LABEL,
    CLIFFORD_MEASURE,
    CLIFFORD_MUL,
    CLIFFORD_TO_QASM3,
)
from graphix.fundamentals import IXYZ, ComplexUnit
from graphix.measurements import Domains
from graphix.pauli import Pauli

if TYPE_CHECKING:
    import numpy.typing as npt


class Clifford(Enum):
    """Clifford gate."""

    # MEMO: Cannot use ClassVar here
    I: Clifford
    X: Clifford
    Y: Clifford
    Z: Clifford
    S: Clifford
    SDG: Clifford
    H: Clifford

    _0 = 0
    _1 = 1
    _2 = 2
    _3 = 3
    _4 = 4
    _5 = 5
    _6 = 6
    _7 = 7
    _8 = 8
    _9 = 9
    _10 = 10
    _11 = 11
    _12 = 12
    _13 = 13
    _14 = 14
    _15 = 15
    _16 = 16
    _17 = 17
    _18 = 18
    _19 = 19
    _20 = 20
    _21 = 21
    _22 = 22
    _23 = 23

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """Return the matrix of the Clifford gate."""
        return CLIFFORD[self.value]

    @staticmethod
    def try_from_matrix(mat: npt.NDArray[Any]) -> Clifford | None:
        """Find the Clifford gate from the matrix.

        Return `None` if not found.

        Notes
        -----
        Global phase is ignored.
        """
        if mat.shape != (2, 2):
            return None
        for ci in Clifford:
            mi = ci.matrix
            for piv, piv_ in zip(mat.flat, mi.flat):
                if math.isclose(abs(piv), 0):
                    continue
                if math.isclose(abs(piv_), 0):
                    continue
                if np.allclose(mat / piv, mi / piv_):
                    return ci
        return None

    def __repr__(self) -> str:
        """Return the Clifford expression on the form of HSZ decomposition."""
        formula = " @ ".join([f"Clifford.{gate}" for gate in self.hsz])
        if len(self.hsz) == 1:
            return formula
        return f"({formula})"

    def __str__(self) -> str:
        """Return the name of the Clifford gate."""
        return CLIFFORD_LABEL[self.value]

    @property
    def conj(self) -> Clifford:
        """Return the conjugate of the Clifford gate."""
        return Clifford(CLIFFORD_CONJ[self.value])

    @property
    def hsz(self) -> list[Clifford]:
        """Return a decomposition of the Clifford gate with the gates `H`, `S`, `Z`."""
        return [Clifford(i) for i in CLIFFORD_HSZ_DECOMPOSITION[self.value]]

    @property
    def qasm3(self) -> tuple[str, ...]:
        """Return a decomposition of the Clifford gate as qasm3 gates."""
        return CLIFFORD_TO_QASM3[self.value]

    def __matmul__(self, other: Clifford) -> Clifford:
        """Multiplication within the Clifford group (modulo unit factor)."""
        if isinstance(other, Clifford):
            return Clifford(CLIFFORD_MUL[self.value][other.value])
        return NotImplemented

    def measure(self, pauli: Pauli) -> Pauli:
        """Compute C† P C."""
        if pauli.symbol == IXYZ.I:
            return copy.deepcopy(pauli)
        table = CLIFFORD_MEASURE[self.value]
        if pauli.symbol == IXYZ.X:
            symbol, sign = table.x
        elif pauli.symbol == IXYZ.Y:
            symbol, sign = table.y
        elif pauli.symbol == IXYZ.Z:
            symbol, sign = table.z
        else:
            typing_extensions.assert_never(pauli.symbol)
        return pauli.unit * Pauli(symbol, ComplexUnit.from_properties(sign=sign))

    def commute_domains(self, domains: Domains) -> Domains:
        """
        Commute `X^sZ^t` with `C`.

        Given `X^sZ^t`, return `X^s'Z^t'` such that `X^sZ^tC = CX^s'Z^t'`.

        Note that applying the method to `self.conj` computes the reverse commutation:
        indeed, `C†X^sZ^t = (X^sZ^tC)† = (CX^s'Z^t')† = X^s'Z^t'C†`.
        """
        s_domain = domains.s_domain.copy()
        t_domain = domains.t_domain.copy()
        for gate in self.hsz:
            if gate == Clifford.I:
                pass
            elif gate == Clifford.H:
                t_domain, s_domain = s_domain, t_domain
            elif gate == Clifford.S:
                t_domain ^= s_domain
            elif gate == Clifford.Z:
                pass
            else:  # pragma: no cover
                raise RuntimeError(f"{gate} should be either I, H, S or Z.")
        return Domains(s_domain, t_domain)


Clifford.I = Clifford(0)
Clifford.X = Clifford(1)
Clifford.Y = Clifford(2)
Clifford.Z = Clifford(3)
Clifford.S = Clifford(4)
Clifford.SDG = Clifford(5)
Clifford.H = Clifford(6)
