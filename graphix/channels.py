"""Quantum channels and noise models."""

from __future__ import annotations

import copy
import typing
from typing import TYPE_CHECKING, SupportsIndex, TypeVar

import numpy as np
import numpy.typing as npt

from graphix import linalg_validations as lv
from graphix.ops import Ops

if TYPE_CHECKING:
    from collections.abc import Iterable

_T = TypeVar("_T", bound=np.generic)


def _ilog2(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive.")
    return (n - 1).bit_length()


class KrausData:
    """Kraus operator data.

    Attributes
    ----------
    coef : complex
        Scalar prefactor of the operator.

    operator : npt.NDArray[np.complex128]
        Operator.
    """

    __coef: complex
    __operator: npt.NDArray[np.complex128]

    def __init__(self, coef: complex, operator: npt.NDArray[_T]) -> None:
        if not lv.is_square(operator):
            raise ValueError("Operator must be a square matrix.")
        if not lv.is_qubitop(operator):
            raise ValueError("Operator must be a qubit operator.")
        self.__coef = coef
        self.__operator = operator.astype(np.complex128, copy=True)
        self.__operator.flags.writeable = False

    @property
    def coef(self) -> complex:
        """Return the scalar prefactor."""
        return self.__coef

    @property
    def operator(self) -> npt.NDArray[np.complex128]:
        """Return the operator."""
        return self.__operator.view()

    @property
    def nqubit(self) -> int:
        """Validate the data."""
        size, _ = self.__operator.shape
        return _ilog2(size)


class KrausChannel:
    r"""Quantum channel class in the Kraus representation.

    Defined by Kraus operators :math:`K_i` with scalar prefactors :code:`coef`) :math:`c_i`,
    where the channel act on density matrix as :math:`\rho'  = \sum_i K_i^\dagger \rho K_i`.
    The data should satisfy :math:`\sum K_i^\dagger K_i = I`.
    """

    __nqubit: int
    __data: list[KrausData]

    @staticmethod
    def _nqubit(kraus_data: Iterable[KrausData]) -> int:
        # MEMO: kraus_data is not empty.
        it = iter(kraus_data)
        nqubit = next(it).nqubit

        if any(data.nqubit != nqubit for data in it):
            raise ValueError("All operators must have the same shape.")

        return nqubit

    def __init__(self, kraus_data: Iterable[KrausData]) -> None:
        """Initialize `KrausChannel` given a Kraus operator.

        Parameters
        ----------
        kraus_data : Iterable[KrausData]
            Iterable of Kraus operator data.

        Raises
        ------
        ValueError
            If kraus_data is empty.
        """
        kraus_data = list(copy.deepcopy(kdata) for kdata in kraus_data)

        if not kraus_data:
            raise ValueError("Cannot instantiate the channel with empty data.")

        self.__nqubit = self._nqubit(kraus_data)
        self.__data = kraus_data

        if len(self.__data) > 4**self.__nqubit:
            raise ValueError("len(kraus_data) cannot exceed 4**nqubit.")

        # Check that the channel is properly normalized, i.e., \sum_K_i^\dagger K_i = Identity.
        data = next(iter(self.__data))
        work = np.zeros_like(data.operator, dtype=np.complex128)
        for data in self.__data:
            m = data.coef * data.operator
            work += m.conj().T @ m
        if not np.allclose(work, np.eye(2**self.__nqubit)):
            raise ValueError("The specified channel is not normalized.")

    @typing.overload
    def __getitem__(self, index: SupportsIndex, /) -> KrausData: ...

    @typing.overload
    def __getitem__(self, index: slice, /) -> list[KrausData]: ...

    def __getitem__(self, index: SupportsIndex | slice, /) -> KrausData | list[KrausData]:
        """Return the Kraus operator at the given index."""
        return self.__data[index]

    def __len__(self) -> int:
        """Return the number of Kraus operators."""
        return len(self.__data)

    @property
    def nqubit(self) -> int:
        """Return the number of qubits."""
        return self.__nqubit


def dephasing_channel(prob: float) -> KrausChannel:
    r"""Single-qubit dephasing channel, :math:`(1-p) \rho + p Z  \rho Z`.

    Parameters
    ----------
    prob : float
        The probability associated to the channel

    Returns
    -------
    :class:`graphix.channels.KrausChannel` object
        containing the corresponding Kraus operators
    """
    return KrausChannel(
        [
            KrausData(np.sqrt(1 - prob), Ops.I),
            KrausData(np.sqrt(prob), Ops.Z),
        ]
    )


def depolarising_channel(prob: float) -> KrausChannel:
    r"""Single-qubit depolarizing channel.

    .. math::
        (1-p) \rho + \frac{p}{3} (X \rho X + Y \rho Y + Z \rho Z) = (1 - 4 \frac{p}{3}) \rho + 4 \frac{p}{3} id

    Parameters
    ----------
    prob : float
        The probability associated to the channel

    """
    return KrausChannel(
        [
            KrausData(np.sqrt(1 - prob), Ops.I),
            KrausData(np.sqrt(prob / 3.0), Ops.X),
            KrausData(np.sqrt(prob / 3.0), Ops.Y),
            KrausData(np.sqrt(prob / 3.0), Ops.Z),
        ]
    )


def pauli_channel(px: float, py: float, pz: float) -> KrausChannel:
    r"""Single-qubit Pauli channel.

    .. math::
        (1-p_X-p_Y-p_Z) \rho + p_X X \rho X + p_Y Y \rho Y + p_Z Z \rho Z)

    """
    if px + py + pz > 1:
        raise ValueError("The sum of probabilities must not exceed 1.")
    p_i = 1 - px - py - pz
    return KrausChannel(
        [
            KrausData(np.sqrt(1 - p_i), Ops.I),
            KrausData(np.sqrt(px / 3.0), Ops.X),
            KrausData(np.sqrt(py / 3.0), Ops.Y),
            KrausData(np.sqrt(pz / 3.0), Ops.Z),
        ]
    )


def two_qubit_depolarising_channel(prob: float) -> KrausChannel:
    r"""Two-qubit depolarising channel.

    .. math::
        \mathcal{E} (\rho) = (1-p) \rho + \frac{p}{15}  \sum_{P_i \in \{id, X, Y ,Z\}^{\otimes 2}/(id \otimes id)}P_i \rho P_i

    Parameters
    ----------
    prob : float
        The probability associated to the channel

    Returns
    -------
    :class:`graphix.channels.KrausChannel` object
        containing the corresponding Kraus operators
    """
    return KrausChannel(
        [
            KrausData(np.sqrt(1 - prob), np.kron(Ops.I, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.I)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.I, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.X, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.X)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Y, Ops.Z)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.Y)),
            KrausData(np.sqrt(prob / 15.0), np.kron(Ops.Z, Ops.X)),
        ]
    )


def two_qubit_depolarising_tensor_channel(prob: float) -> KrausChannel:
    r"""Two-qubit tensor channel of single-qubit depolarising channels with same probability.

    Kraus operators:

    .. math::
        \Big\{ \sqrt{(1-p)} id, \sqrt{(p/3)} X, \sqrt{(p/3)} Y , \sqrt{(p/3)} Z \Big\} \otimes \Big\{ \sqrt{(1-p)} id, \sqrt{(p/3)} X, \sqrt{(p/3)} Y , \sqrt{(p/3)} Z \Big\}

    Parameters
    ----------
    prob : float
        The probability associated to the channel

    Returns
    -------
    :class:`graphix.channels.KrausChannel` object
        containing the corresponding Kraus operators
    """
    return KrausChannel(
        [
            KrausData(1 - prob, np.kron(Ops.I, Ops.I)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.Y)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.Z)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.X, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.Y, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.Z, Ops.I)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.X)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.Y)),
            KrausData(np.sqrt(1 - prob) * np.sqrt(prob / 3.0), np.kron(Ops.I, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.Y)),
            KrausData(prob / 3.0, np.kron(Ops.X, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Y, Ops.Z)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.X)),
            KrausData(prob / 3.0, np.kron(Ops.Z, Ops.Y)),
        ]
    )
