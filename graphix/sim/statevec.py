"""MBQC state vector backend simulator based on 10.48550/arXiv.2506.08142."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat

import numba as nb
import numpy as np
import numpy.typing as npt
from typing_extensions import override

from graphix.parameter import ExpressionOrSupportsComplex
from graphix.sim.base_backend import DenseState, DenseStateBackend, DenseStateBackendKwargs, Matrix
from graphix.states import BasicStates, State

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal, Self, TypeAlias, TypeVar

    # Unpack introduced in Python 3.12
    from typing_extensions import Unpack

    from graphix.sim.data import Data

    _ENCODING = Literal["LSB", "MSB"]
    _ScalarT = TypeVar("_ScalarT", bound=np.generic[Any])

    from graphix.parameter import ExpressionOrSupportsComplex

    EvolveSingleJit: TypeAlias = Callable[
        [npt.NDArray[np.complex128], npt.NDArray[np.complex128], int, int], None
    ]  # type introduced in 3.12
    ExpectationSingleJit: TypeAlias = Callable[  # type introduced in 3.12
        [npt.NDArray[np.complex128], npt.NDArray[np.complex128], int, int], complex
    ]
    EntangleJit: TypeAlias = Callable[[npt.NDArray[np.complex128], int, int, int], None]  # type introduced in 3.12


NUM_QUBIT_PARALLEL = 15
"""This constant determines the number of qubits above which matrix operations are multi-threaded. For lower counts, the overhead does not compensate parallelization."""


class Statevec(DenseState):
    """Statevector object.

    Attributes
    ----------
    psi : numpy.ndarray of numpy.complex128
        Complex-valued 1-dimensional array representing the quantum statevector.
        Throughout the simulation ``psi`` has constant size ``2**max_qubits``. Only the first ``2**nqubit`` complex values have meaning.

    _max_qubits : int
        Maximum Hilbert space size allowed for internal computations. It determines the size of ``psi``. For circuit simulations, it corresponds to the number of qubits, while for pattern simulations it corresponds to the pattern's maximum space.

    _nqubit : int
        Number of active qubits at any given time.
    """

    psi: npt.NDArray[np.complex128]
    _max_qubits: int
    _nqubit: int

    def __init__(self, data: Data = BasicStates.PLUS, nqubit: int | None = None, max_qubits: int | None = None) -> None:
        """Initialize statevector objects.

        See :class:`graphix.sim.statevec.Statevec` for additional information.

        Parameters
        ----------
        data : Data, optional
            Input data to prepare the state. Can be a classical description or a numerical input, defaults to `graphix.states.BasicStates.PLUS`
        nqubit : int | None, optional
            Number of qubits to prepare. If ``None`` (default), it's inferred from ``data``.
        max_qubits : int | None, optional.
            Maximum Hilbert space size for array preallocation. If ``None`` (default), it's set equal to ``nqubit``.

        Raises
        ------
        ValueError
            If `max_qubits` is smaller than `nqubit` or the number of qubits inferred from ``data``.
        """
        if nqubit is not None and nqubit < 0:
            raise ValueError("`nqubit` must be a non-negative integer.")

        if max_qubits is not None and max_qubits < 0:
            raise ValueError("`max_qubits` must be a non-negative integer.")

        if isinstance(data, Statevec):
            if nqubit is not None and len(data.flatten()) != 1 << nqubit:
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the inferred number of qubit = {len(data.flatten())}."
                )
            self.psi = data.psi.copy()
            self._max_qubits = data.max_qubits
            self._nqubit = data.nqubit

            if max_qubits is not None:
                if max_qubits < data.max_qubits:
                    raise ValueError(
                        f"`max_qubits` can't be smaller than the capacity of input state: {max_qubits} < {data.max_qubits}"
                    )
                self.ensure_capacity(max_qubits)
            return

        # The type
        # list[states.State] | list[ExpressionOrSupportsComplex] | list[Iterable[ExpressionOrSupportsComplex]]
        # would be more precise, but given a value X of type Iterable[A] | Iterable[B],
        # mypy infers that list(X) has type list[A | B] instead of list[A] | list[B].
        input_list: list[State | ExpressionOrSupportsComplex | Iterable[ExpressionOrSupportsComplex]]
        if isinstance(data, State):
            if nqubit is None:
                nqubit = 1
            input_list = [data] * nqubit
        elif isinstance(data, Iterable):
            input_list = list(data)
        else:
            raise TypeError(f"Incorrect type for data: {type(data)}")

        if len(input_list) == 0:
            if nqubit is not None and nqubit != 0:
                raise ValueError("nqubit is not null but input state is empty.")
            nqubit = 0
            psi = np.array([1], dtype=np.complex128)

        elif isinstance(input_list[0], State):
            length = len(input_list)
            if nqubit is None:
                nqubit = length
            elif nqubit != length:
                raise ValueError(f"Mismatch between nqubit and length of input state: {nqubit} != {length}.")

            def state_to_statevector(
                s: State | ExpressionOrSupportsComplex | Iterable[ExpressionOrSupportsComplex],
            ) -> npt.NDArray[np.complex128]:
                if not isinstance(s, State):
                    raise TypeError("Data should be an homogeneous sequence of states.")
                return s.to_statevector()

            psi = functools.reduce(
                lambda m0, m1: np.kron(m0, m1).astype(np.complex128, copy=False),
                (state_to_statevector(s) for s in input_list),
            )

        # `SupportsFloat` is needed because `numpy.float64` is not an instance of `SupportsComplex`!
        elif isinstance(input_list[0], (SupportsComplex, SupportsFloat)):
            length = len(input_list)
            inferred_nqubit = length.bit_length() - 1
            if nqubit is None:
                if length & (length - 1):
                    raise ValueError(f"Length of input data is not a power of two: {length}")
                nqubit = inferred_nqubit
            elif nqubit != inferred_nqubit:
                raise ValueError(f"Mismatch between nqubit and inferred nqubit: {nqubit} != {inferred_nqubit}")
            psi = np.array(input_list, dtype=np.complex128)
            if not np.isclose(np.linalg.norm(psi), 1.0):
                raise ValueError("Input state is not normalized")

        else:
            raise TypeError(f"First element of data has type {type(input_list[0])} whereas Number or State is expected")

        if max_qubits is not None:
            if max_qubits < nqubit:
                raise ValueError(
                    f"`max_qubits` can't be smaller than the length of input state: {max_qubits} < {nqubit}"
                )
        else:
            max_qubits = nqubit

        self.psi = psi
        self._max_qubits = nqubit  # bootstrap for self.ensure_capacity
        self._nqubit = nqubit
        self.ensure_capacity(max_qubits)  # may extend both self.psi and self._max_qubits

    def __str__(self) -> str:
        """Return a string description."""
        sv = self.psi
        return f"Statevec object with statevector {sv} and length {len(sv)}."

    # Note that `@property` must appear before `@override` for pyright
    @property
    @override
    def nqubit(self) -> int:
        """Return the number of qubits."""
        return self._nqubit

    @property
    def max_qubits(self) -> int:
        """Return the preallocated number of qubits."""
        return self._max_qubits

    @property
    def size_valid_psi(self) -> int:
        """Return the number of meaningful elements in ``self.psi``."""
        return 1 << self.nqubit  # 2**self.nqubit

    def ensure_capacity(self, required_qubits: int) -> None:
        """Extend the state vector if the required qubit capacity exceeds the current one.

        Does nothing if ``required_qubits <= self.max_qubits``.

        Parameters
        ----------
        required_qubits : int
            Minimum number of qubits the state vector must support. If expansion
            is needed, ``self.psi`` is extended to size ``2**required_qubits``.
        """
        if required_qubits > self.max_qubits:
            offset = (1 << required_qubits) - len(self.psi)
            self.psi = np.concatenate([self.psi, np.empty(offset, dtype=self.psi.dtype)])
            self._max_qubits = required_qubits

    @override
    def flatten(self) -> Matrix:
        """Return flattened state.

        A view of only the first ``2**self.nqubit`` elements of ``self.psi`` is returned.
        """
        return self.psi[: self.size_valid_psi]

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        r"""Add nodes (qubits) to the state vector and initialize them in a specified state.

        Previously existing nodes remain unchanged.

        Parameters
        ----------
        nqubit : int
            The number of qubits to add to the state vector.

        data : Data, optional
            The state in which to initialize the newly added nodes.

            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of ``nodes``, and
              each node is initialized with its corresponding state.
            - A single-qubit state vector will be broadcast to all nodes.
            - A multi-qubit state vector of dimension :math:`2^n`, where :math:`n = \mathrm{len}(nodes)`, initializes the new nodes jointly.

        Notes
        -----
        This method can extend the size of ``self.psi`` for convenience, but this requires allocating a full new array.
        """
        self.ensure_capacity(required_qubits=self.nqubit + nqubit)
        sv_to_add = Statevec(nqubit=nqubit, data=data)
        self.tensor(sv_to_add)

    @override
    def entangle(self, qubits: tuple[int, int]) -> None:
        """Connect graph nodes.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        kernel = _entangle_jit_parallel if self.nqubit > NUM_QUBIT_PARALLEL else _entangle_jit
        kernel(self.psi, self.nqubit, *qubits)

    @override
    def evolve_single(self, op: Matrix, qubit: int) -> None:
        """Apply a single-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 matrix
        q : int
            qubit index
        """
        self._check_bounds(qubit)
        kernel = _evolve_single_jit_parallel if self.nqubit > NUM_QUBIT_PARALLEL else _evolve_single_jit
        # We cast to np.complex128 to match numba signature.
        kernel(self.psi, op.astype(np.complex128), self.nqubit, qubit)

    @override
    def expectation_single(self, op: Matrix, qubit: int) -> complex:
        """Return the expectation value of single-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 operator
        qubit : int
            target qubit index

        Returns
        -------
        complex : expectation value.
        """
        self._check_bounds(qubit)
        kernel = _expectation_single_jit_parallel if self.nqubit > NUM_QUBIT_PARALLEL else _expectation_single_jit
        # We cast to np.complex128 to match numba signature.
        return kernel(self.psi, op.astype(np.complex128), self.nqubit, qubit)

    @override
    def evolve(self, op: Matrix, qubits: Sequence[int]) -> None:
        """Apply a multi-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n matrix
        qubits : list of int
            target qubits' indices
        """
        nq = len(qubits)
        # treat op as a tensor with nq output + nq input legs
        op_t = op.reshape((2,) * (nq * 2)).astype(np.complex128, copy=False)
        psi_t = self.flatten().reshape((2,) * self.nqubit).astype(np.complex128, copy=False)

        psi_idx = np.array(range(self.nqubit))
        out_idx = np.array(range(self.nqubit, self.nqubit + nq))  # fresh labels

        op_idx = np.concatenate((out_idx, qubits))

        # result subscripts: same as psi but modified indices (qargs) replaced by out labels
        res_idx = psi_idx.copy()
        for i, s in enumerate(qubits):
            res_idx[s] = out_idx[i]

        self.psi[: self.size_valid_psi] = np.einsum(op_t, op_idx, psi_t, psi_idx, res_idx).reshape(1 << self.nqubit)

    def expectation_value(self, op: Matrix, qubits: Sequence[int]) -> complex:
        """Return the expectation value of multi-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n operator
        qubits : list of int
            target qubit indices

        Returns
        -------
        complex : expectation value
        """
        sv = deepcopy(self)
        sv.evolve(op, qubits)
        return complex(np.dot(self.flatten().conjugate(), sv.flatten()))

    @override
    def remove_qubit(self, qubit: int) -> None:
        r"""Remove a separable qubit from the system and assemble a statevector for remaining qubits.

        This results in the same result as partial trace, if the qubit *qarg* is separable from the rest.

        For a statevector :math:`\ket{\psi} = \sum c_i \ket{i}` with sum taken over
        :math:`i \in [ 0 \dots 00,\ 0\dots 01,\ \dots,\
        1 \dots 11 ]`, this method returns

        .. math::
            \begin{align}
                \ket{\psi}' =&
                    c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 00}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 00} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 01}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 01} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 10}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 10} \\
                    & + \dots \\
                    & + c_{1 \dots 1_{\mathrm{k-1}}0_{\mathrm{k}}1_{\mathrm{k+1}} \dots 11}
                    \ket{1 \dots 1_{\mathrm{k-1}}1_{\mathrm{k+1}} \dots 11},
           \end{align}

        (after normalization) for :math:`k =` qarg. If the :math:`k` th qubit is in :math:`\ket{1}` state,
        above will return zero amplitudes; in such a case the returned state will be the one above with
        :math:`0_{\mathrm{k}}` replaced with :math:`1_{\mathrm{k}}` .

        .. warning::
            This method assumes the qubit with index *qarg* to be separable from the rest,
            and is implemented as a significantly faster alternative for partial trace to
            be used after single-qubit measurements.
            Care needs to be taken when using this method.
            Checks for separability will be implemented soon as an option.

        Parameters
        ----------
        qubit : int
            qubit index
        """
        self._check_bounds(qubit)
        self._nqubit = _remove_qubit_jit(self.psi, self.nqubit, qubit, atol=1e-10)

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """Swap qubits.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        _swap_jit(self.psi, self.nqubit, *qubits)

    def tensor(self, other: Statevec) -> None:
        r"""Tensor product state with other qubits.

        Results in ``self`` :math:`\otimes` ``other``.

        Parameters
        ----------
        other : :class:`graphix.sim.statevec.Statevec`
            Statevector to be tensored with ``self``.
        """
        _tensor_jit(self.psi, other.psi, self.nqubit, other.nqubit)
        self._nqubit += other.nqubit

    def _check_bounds(self, qubit: int) -> None:
        """Check if qubit index is valid.

        This check is necessary because there is no bounds checking in Numba. See
        https://numba.pydata.org/numba-doc/dev/reference/pysemantics.html#bounds-checking
        """
        if not 0 <= qubit < self.nqubit:
            raise IndexError(f"Qubit index {qubit} out of range [0, {self.nqubit})")

    def fidelity(self, other: Statevec) -> float:
        r"""Calculate the fidelity against another statevector.

        The fidelity is defined as :math:`|\langle\psi_1|\psi_2\rangle|^2`.

        Parameters
        ----------
        other : :class:`Statevec`
            statevector to compare with

        Returns
        -------
        float
            Fidelity between the two statevectors.
        """
        inner = np.dot(self.flatten().conjugate(), other.flatten())
        return float(np.abs(inner) ** 2)

    def isclose(self, other: Statevec, *, rtol: float = 1e-09, atol: float = 0.0) -> bool:
        """Check if two quantum states are equal up to global phase.

        Two states are considered close if their fidelity is close to 1.

        Parameters
        ----------
        other : :class:`Statevec`
            statevector to compare with
        rtol : float
            relative tolerance for :func:`math.isclose`
        atol : float
            absolute tolerance for :func:`math.isclose`

        Returns
        -------
        bool
            ``True`` if the states are equal up to global phase.
        """
        return math.isclose(self.fidelity(other), 1, rel_tol=rtol, abs_tol=atol)

    def to_dict(
        self,
        encoding: _ENCODING = "MSB",
        *,
        rtol: float = 0.0,
        atol: float = 1e-8,
    ) -> dict[str, np.object_ | np.complex128]:
        r"""Convert the statevector to dictionary form.

        This dictionary representation uses a ket-like notation where the dictionary ``keys`` are qubit strings for the basis vectors and ``values`` are the corresponding complex amplitudes. Amplitudes below a certain threshold are filtered out.

        Parameters
        ----------
        encoding : Literal["LSB", "MSB"], default="MSB"
            Encoding for the basis kets. See notes for additional information.

        rtol : float, default=0.0
            Relative tolerance used when deciding whether a coefficient should be
            treated as zero. Values whose magnitude is within this relative tolerance
            of zero are omitted from the resulting dictionary.

        atol : float, default=1e-8
            Absolute tolerance used when deciding whether a coefficient should be
            treated as zero. Values whose magnitude is within this relative tolerance
            of zero are omitted from the resulting dictionary.

        Returns
        -------
        dict[str, complex]
            The statevector in dictionary form.

        Notes
        -----
        The encoding determines the bit ordering convention used when mapping basis states to dictionary
        keys. Consider a tensor product of three qubits:

        .. math::

        \lvert\psi\rangle = q_0 \otimes q_1 \otimes q_2.

        If ``encoding == "MSB"`` the first qubit is represented in the Most Significant Bit -> ``q0q1q2``. This is the default representation in Graphix.
        If ``encoding == "LSB"`` the first qubit is represented in the Least Significant Bit -> ``q2q1q0``. This is the default representation in other software packages such as Qiskit.

        Example
        -------
        >>> from graphix.states import BasicStates
        >>> from graphix.sim.statevec import Statevec
        >>> sv = Statevec(data=[BasicStates.ZERO, BasicStates.ONE])
        >>> sv.to_dict()
        {'01': np.complex128(1+0j)}
        >>> sv.to_dict(encoding="LSB")
        {'10': np.complex128(1+0j)}
        """
        return self._to_dict_map(lambda x: x, encoding, rtol=rtol, atol=atol)

    def to_prob_dict(
        self, encoding: _ENCODING = "MSB", *, rtol: float = 0.0, atol: float = 1e-8
    ) -> dict[str, np.object_ | np.float64]:
        r"""Convert the statevector to a probability distirbution in a dictionary form.

        This dictionary representation uses a ket-like notation where the dictionary ``keys`` are qubit strings for the basis vectors and ``values`` are the corresponding probabilities.
        Basis vector whose amplitude is below a certain threshold are filtered out.

        Parameters
        ----------
        encoding: Literal["LSB", "MSB"], default="MSB"
            Encoding for the basis kets. See :meth:`to_dict` for additional information.

        rtol : float, default=0.0
            Relative tolerance used when deciding whether a coefficient should be
            treated as zero. Values whose magnitude is within this relative tolerance
            of zero are omitted from the resulting dictionary.

        atol : float, default=1e-8
            Absolute tolerance used when deciding whether a coefficient should be
            treated as zero. Values whose magnitude is within this relative tolerance
            of zero are omitted from the resulting dictionary.

        Returns
        -------
        dict[str, float]
            The probability distribution associated to the statevector in dictionary form.

        See Also
        --------
        .. :meth:`to_dict`
        """
        return self._to_dict_map(lambda x: np.abs(x) ** 2, encoding, rtol=rtol, atol=atol)

    def _to_dict_map(
        self,
        f: Callable[[npt.NDArray[np.object_ | np.complex128]], npt.NDArray[_ScalarT]],
        encoding: _ENCODING = "MSB",
        *,
        rtol: float = 0.0,
        atol: float = 1e-8,
    ) -> dict[str, _ScalarT]:
        mask = np.logical_not(np.isclose(np.abs(self.flatten()), 0, rtol=rtol, atol=atol))
        i_vals = np.arange(1 << self.nqubit)[mask]
        amp_vals = f(self.flatten()[mask])

        return {_format_encoding(self.nqubit, i, encoding): amp for i, amp in zip(i_vals, amp_vals, strict=True)}


@dataclass(frozen=True)
class StatevectorBackend(DenseStateBackend[Statevec]):
    """MBQC state vector backend simulator based on 10.48550/arXiv.2506.08142."""

    state: Statevec = dataclasses.field(init=True, default_factory=lambda: Statevec(nqubit=0))

    @classmethod
    def with_capacity(
        cls, max_qubits: int, state: Statevec | None = None, **kwargs: Unpack[DenseStateBackendKwargs]
    ) -> Self:
        """Initialize the backend with the required capacity to perform the simulation.

        Parameters
        ----------
        max_qubits : int
            Number of qubits the state vector must support. For pattern simulations this corresponds to ``Pattern.max_space()``.
        """
        state_init = (
            Statevec(nqubit=0, max_qubits=max_qubits) if state is None else Statevec(state, max_qubits=max_qubits)
        )
        return cls(state_init, **kwargs)


@nb.njit("(c16[::1], c16[::1], int32, int32)")
def _tensor_jit(
    psi: npt.NDArray[np.complex128],
    psi_other: npt.NDArray[np.complex128],
    nqubit: int,
    nqubit_other: int,
) -> None:
    size_psi = 1 << nqubit
    size_other = 1 << nqubit_other
    # We update the elements of `psi` in-place.
    # This requires starting the update for the last element of the new psi, `size_psi * size_other - 1`
    k = size_psi * size_other - 1
    sp_m1 = size_psi - 1
    so_m1 = size_other - 1

    for i in range(size_psi):
        alpha_old = psi[sp_m1 - i]
        for j in range(size_other):
            psi[k] = alpha_old * psi_other[so_m1 - j]
            k -= 1


@nb.njit("(c16[::1], int32, int32, int32)")
def _swap_jit(psi: npt.NDArray[np.complex128], nqubit: int, q1: int, q2: int) -> None:

    if q1 == q2:
        return
    size_sv = 1 << nqubit
    mask_1 = 1 << nqubit - 1 - q1
    mask_2 = 1 << nqubit - 1 - q2
    mask = mask_1 | mask_2
    # `mask` is an integer number whose binary representation has 1s at positions `q1` and `q2` and 0s elsewhere.

    for i in range(size_sv):
        # i & mask_1 = 2^(nqubit - 1 - q_1) if the binary representation of `i` has a 1 at position `q1` and 0 otherwise.
        i_has_1_at_q1 = bool(i & mask_1)
        i_has_1_at_q2 = bool(i & mask_2)
        if i_has_1_at_q1 != i_has_1_at_q2:
            # `j` has the same binary representation as `i` except for bits `q1` and `q2` which are flipped.
            j = i ^ mask
            if j > i:  # Ensure we don't swap the same indices twice.
                psi[j], psi[i] = psi[i], psi[j]


def _evolve_single(psi: npt.NDArray[np.complex128], op: npt.NDArray[np.complex128], nqubit: int, q: int) -> None:
    r"""Apply a single-qubit operation.

    This function is inspired from 10.48550/arXiv.2506.08142.
    """
    nblocks = 1 << q
    size_block = 1 << nqubit - q  # 2**(nqubit - q)
    size_half_block = (
        size_block >> 1
    )  # Left-to-right tensor product encoding (first qubit corresponds to most significant bit). For right-to-left encoding use `size_half_block = 1 << i`

    for b in nb.prange(nblocks):
        # WARNING: setting `b0 += size_block` may result in a race condition if `parallel=True`
        b0 = size_block * b
        for offset in range(size_half_block):
            i1 = b0 | offset
            i2 = i1 | size_half_block
            psi1 = psi[i1]
            psi2 = psi[i2]
            psi[i1] = op[0, 0] * psi1 + op[0, 1] * psi2
            psi[i2] = op[1, 0] * psi1 + op[1, 1] * psi2


_evolve_single_jit: EvolveSingleJit = nb.njit("(c16[::1], c16[:, :], int32, int32)", parallel=False)(_evolve_single)
_evolve_single_jit_parallel: EvolveSingleJit = nb.njit("(c16[::1], c16[:, :], int32, int32)", parallel=True)(
    _evolve_single
)


def _expectation_single(
    psi: npt.NDArray[np.complex128], op: npt.NDArray[np.complex128], nqubit: int, q: int
) -> complex:
    nblocks = 1 << q
    size_block = 1 << nqubit - q
    size_half_block = (
        size_block >> 1
    )  # Left-to-right tensor product encoding (first qubit corresponds to most significant bit). For right-to-left encoding use `size_half_block = 1 << i`

    result = 0.0 + 0.0j

    for b in nb.prange(nblocks):
        # WARNING: setting `b0 += size_block` may result in a race condition if `parallel=True`
        b0 = b << nqubit - q
        for offset in range(size_half_block):
            i1 = b0 | offset
            i2 = i1 | size_half_block
            psi1 = psi[i1]
            psi2 = psi[i2]
            b1 = op[0, 0] * psi1 + op[0, 1] * psi2
            b2 = op[1, 0] * psi1 + op[1, 1] * psi2
            result += psi1.conjugate() * b1 + psi2.conjugate() * b2

    return result


_expectation_single_jit: ExpectationSingleJit = nb.njit("c16(c16[::1], c16[:, :], int32, int32)", parallel=False)(
    _expectation_single
)
_expectation_single_jit_parallel: ExpectationSingleJit = nb.njit(
    "c16(c16[::1], c16[:, :], int32, int32)", parallel=True
)(_expectation_single)


def _entangle(psi: npt.NDArray[np.complex128], nqubit: int, control: int, target: int) -> None:
    size_sv = 1 << nqubit
    mask_control = 1 << nqubit - 1 - control
    mask_target = 1 << nqubit - 1 - target
    mask = mask_control | mask_target
    # `mask` is an integer number whose binary representation has 1s at positions `control` and `target` and 0s elsewhere.

    for i in nb.prange(size_sv):
        if mask & i == mask:
            psi[i] = -psi[i]


_entangle_jit: EntangleJit = nb.njit("(c16[::1], int32, int32, int32)", parallel=False)(_entangle)
_entangle_jit_parallel: EntangleJit = nb.njit("(c16[::1], int32, int32, int32)", parallel=True)(_entangle)


@nb.njit("int32(c16[::1], int32, int32, f8)", parallel=False)
def _remove_qubit_jit(
    psi: npt.NDArray[np.complex128],
    nqubit: int,
    q: int,
    atol: float,
) -> int:
    new_nqubit = nqubit - 1

    n_blocks = 1 << q
    size_block = 1 << nqubit - q  # 2**(nqubits - q)
    size_half_block = size_block >> 1

    # Compute norm of branch 0
    norm2 = 0.0
    shift = 0
    b0 = shift
    for _ in range(n_blocks):
        # If parallelization, set `b0 = b * size_block + shift` with `b` the loop variable to avoid race condition.
        # Parallelization for norm computation is not worth, execution-time controlled by the update loop which can't be parallelized without cache.
        for j in range(size_half_block):
            a = psi[b0 | j]
            a_re = a.real
            a_im = a.imag
            norm2 += a_re * a_re + a_im * a_im
        b0 += size_block

    # If norm of branch 0 is 0, compute norm of branch 1 and set shift to branch 1
    if norm2 <= atol:
        norm2 = 0.0
        shift = size_half_block
        b0 = shift
        for _ in range(n_blocks):
            for j in range(size_half_block):
                a = psi[b0 | j]
                a_re = a.real
                a_im = a.imag
                norm2 += a_re * a_re + a_im * a_im
            b0 += size_block

    if norm2 <= atol:
        raise RuntimeError(f"Attempted to remove qubit {q} from 0-norm statevector.")

    b0 = shift
    k = 0
    inv_norm = 1.0 / math.sqrt(norm2)

    # Update `psi` with selected and normalized elements.
    for _ in range(n_blocks):
        for j in range(size_half_block):
            psi[k] = (
                psi[b0 | j] * inv_norm
            )  # b0 | j equivalent to b0 + j because the active bits of b0 and j don't overlap.
            k += 1
        b0 += size_block

    return new_nqubit


def _format_encoding(nqubit: int, i: int, encoding: _ENCODING) -> str:
    """Format the i-th basis vector as a ket. See :meth:`Statevec.to_dict` for additional details."""
    display_width = nqubit
    output = f"{i:0{display_width}b}"
    if encoding == "LSB":
        return output[::-1]
    return output
