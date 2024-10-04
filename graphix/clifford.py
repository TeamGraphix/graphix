"""24 Unique single-qubit Clifford gates and their multiplications, conjugations and Pauli conjugations."""

from __future__ import annotations

import copy
import dataclasses
from enum import Enum
from typing import TYPE_CHECKING

from graphix._db import (
    CLIFFORD,
    CLIFFORD_CONJ,
    CLIFFORD_HSZ_DECOMPOSITION,
    CLIFFORD_LABEL,
    CLIFFORD_MEASURE,
    CLIFFORD_MUL,
    CLIFFORD_TO_QASM3,
)
from graphix.pauli import IXYZ, ComplexUnit, Pauli, Sign

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclasses.dataclass
class Domains:
    """
    Represent `X^sZ^t` where s and t are XOR of results from given sets of indices.

    This representation is used in `Clifford.commute_domains`.
    """

    s_domain: set[int]
    t_domain: set[int]


class Clifford(Enum):
    """Clifford gate."""

    # MEMO: Cannot use ClassVar here
    I: Clifford
    X: Clifford
    Y: Clifford
    Z: Clifford
    S: Clifford
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
    def index(self) -> int:
        """Return the index of the Clifford gate."""
        # mypy does not infer variant type (pyright does)
        return self.value  # type: ignore[no-any-return]

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """Return the matrix of the Clifford gate."""
        return CLIFFORD[self.value]

    def __repr__(self) -> str:
        """Return the Clifford expression on the form of HSZ decomposition."""
        return " @ ".join([f"graphix.clifford.{gate}" for gate in self.hsz])

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
        symbol, sign = table[pauli.symbol.value]
        return pauli.unit * Pauli(IXYZ[symbol], ComplexUnit(Sign(sign), False))

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
Clifford.H = Clifford(6)
