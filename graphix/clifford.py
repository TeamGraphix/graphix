"""24 Unique single-qubit Clifford gates and their multiplications, conjugations and Pauli conjugations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import graphix.pauli
from graphix._db import (
    CLIFFORD,
    CLIFFORD_CONJ,
    CLIFFORD_HSZ_DECOMPOSITION,
    CLIFFORD_LABEL,
    CLIFFORD_MEASURE,
    CLIFFORD_MUL,
    CLIFFORD_TO_QASM3,
)

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Domains:
    """
    Represent `X^sZ^t` where s and t are XOR of results from given sets of indices.

    This representation is used in `Clifford.commute_domains`.
    """

    s_domain: set[int]
    t_domain: set[int]


class Clifford:
    """Clifford gate."""

    def __init__(self, index: int):
        self.__index = index

    @property
    def index(self) -> int:
        """Return the index of the Clifford gate (inverse of clifford.get)."""
        return self.__index

    @property
    def matrix(self) -> np.ndarray:
        """Return the matrix of the Clifford gate."""
        return CLIFFORD[self.__index]

    def __repr__(self) -> str:
        """Return the Clifford expression on the form of HSZ decomposition."""
        return " @ ".join([f"graphix.clifford.{gate}" for gate in self.hsz])

    def __str__(self) -> str:
        """Return the name of the Clifford gate."""
        return CLIFFORD_LABEL[self.__index]

    @property
    def conj(self) -> Clifford:
        """Return the conjugate of the Clifford gate."""
        return get(CLIFFORD_CONJ[self.__index])

    @property
    def hsz(self) -> list[Clifford]:
        """Return a decomposition of the Clifford gate with the gates `H`, `S`, `Z`."""
        return list(map(get, CLIFFORD_HSZ_DECOMPOSITION[self.__index]))

    @property
    def qasm3(self) -> tuple[str, ...]:
        """Return a decomposition of the Clifford gate as qasm3 gates."""
        return CLIFFORD_TO_QASM3[self.__index]

    def __matmul__(self, other) -> Clifford:
        """Multiplication within the Clifford group (modulo unit factor)."""
        if isinstance(other, Clifford):
            return get(CLIFFORD_MUL[self.__index][other.__index])
        return NotImplemented

    def measure(self, pauli: graphix.pauli.Pauli) -> graphix.pauli.Pauli:
        """Compute C† P C."""
        if pauli.symbol == graphix.pauli.IXYZ.I:
            return pauli
        table = CLIFFORD_MEASURE[self.__index]
        symbol, sign = table[pauli.symbol.value]
        return pauli.unit * graphix.pauli.TABLE[symbol + 1][sign][False]

    def commute_domains(self, domains: Domains) -> Domains:
        """
        Commute `X^sZ^t` with `C`.

        Given `X^sZ^t`, return `X^s'Z^t'` such that `X^sZ^tC = CX^s'Z^t'`.

        Note that applying the method to `self.conj` computes the reverse commutation:
        indeed, `C†X^sZ^t = (X^sZ^tC)† = (CX^s'Z^t')† = X^s'Z^t'C†`.
        """
        s_domain = domains.s_domain
        t_domain = domains.t_domain
        for gate in self.hsz:
            if gate == graphix.clifford.I:
                pass
            elif gate == graphix.clifford.H:
                t_domain, s_domain = s_domain, t_domain
            elif gate == graphix.clifford.S:
                t_domain ^= s_domain
            elif gate == graphix.clifford.Z:
                pass
            else:
                raise RuntimeError(f"{gate} should be either I, H, S or Z.")
        return Domains(s_domain, t_domain)


TABLE = tuple(Clifford(i) for i in range(len(CLIFFORD)))


def get(index: int) -> Clifford:
    """Return the Clifford gate with given index."""
    return TABLE[index]


I = get(0)
X = get(1)
Y = get(2)
Z = get(3)
S = get(4)
H = get(6)
