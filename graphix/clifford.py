"""24 Unique single-qubit Clifford gates and their multiplications, conjugations and Pauli conjugations."""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, ClassVar

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
    from collections.abc import Iterator

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


class Clifford:
    """Clifford gate."""

    __index: int

    I: ClassVar[Clifford]
    X: ClassVar[Clifford]
    Y: ClassVar[Clifford]
    Z: ClassVar[Clifford]
    S: ClassVar[Clifford]
    H: ClassVar[Clifford]

    def __init__(self, index: int) -> None:
        if not (0 <= index < len(CLIFFORD)):
            raise ValueError("Clifford index out of range.")
        self.__index = index

    @property
    def index(self) -> int:
        """Return the index of the Clifford gate."""
        return self.__index

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """Return the matrix of the Clifford gate."""
        return CLIFFORD[self.__index]

    def __eq__(self, other: object) -> bool:
        """Compare two Clifford gates."""
        if isinstance(other, Clifford):
            return self.__index == other.__index
        return NotImplemented

    def __hash__(self) -> int:
        """Hash the Clifford gate using its index."""
        return hash(self.__index)

    def __repr__(self) -> str:
        """Return the Clifford expression on the form of HSZ decomposition."""
        return " @ ".join([f"graphix.clifford.{gate}" for gate in self.hsz])

    def __str__(self) -> str:
        """Return the name of the Clifford gate."""
        return CLIFFORD_LABEL[self.__index]

    @property
    def conj(self) -> Clifford:
        """Return the conjugate of the Clifford gate."""
        return Clifford(CLIFFORD_CONJ[self.__index])

    @property
    def hsz(self) -> list[Clifford]:
        """Return a decomposition of the Clifford gate with the gates `H`, `S`, `Z`."""
        return [Clifford(i) for i in CLIFFORD_HSZ_DECOMPOSITION[self.__index]]

    @property
    def qasm3(self) -> tuple[str, ...]:
        """Return a decomposition of the Clifford gate as qasm3 gates."""
        return CLIFFORD_TO_QASM3[self.__index]

    def __matmul__(self, other: Clifford) -> Clifford:
        """Multiplication within the Clifford group (modulo unit factor)."""
        if isinstance(other, Clifford):
            return Clifford(CLIFFORD_MUL[self.__index][other.__index])
        return NotImplemented

    def measure(self, pauli: Pauli) -> Pauli:
        """Compute C† P C."""
        if pauli.symbol == IXYZ.I:
            return copy.deepcopy(pauli)
        table = CLIFFORD_MEASURE[self.__index]
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
            else:
                raise RuntimeError(f"{gate} should be either I, H, S or Z.")
        return Domains(s_domain, t_domain)

    @staticmethod
    def cliffords() -> Iterator[Clifford]:
        """Return an iterator over the Clifford gates."""
        for i in range(len(CLIFFORD)):
            yield Clifford(i)


Clifford.I = Clifford(0)
Clifford.X = Clifford(1)
Clifford.Y = Clifford(2)
Clifford.Z = Clifford(3)
Clifford.S = Clifford(4)
Clifford.H = Clifford(6)
