"""Numba JIT-compiled MBQC statevector backend simulator based on Ref. [1].

[1] McGuffin, M. J., Robert J-M., and Ikeda K. "How to Write a Simulator for Quantum Circuits from Scratch: A Tutorial.", 2025 (arXiv:2506.08142).
"""

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

from graphix import states
from graphix.sim.base_backend import DenseState, DenseStateBackend, DenseStateBackendKwargs, Matrix
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal, Self, TypeVar

    # Unpack introduced in Python 3.12
    from typing_extensions import Unpack

    from graphix.parameter import ExpressionOrSupportsComplex
    from graphix.sim.data import Data

    _ENCODING = Literal["LSB", "MSB"]
    _ScalarT = TypeVar("_ScalarT", bound=np.generic[Any])


NUM_QUBIT_PARALLEL = 15
"""This constant determines the number of qubits above which matrix operations
are multi-threaded. For lower counts, the overhead does not compensate parallelization.
This number was determined empirically and it may be platform dependent."""


class Statevec(DenseState):
    """Statevector object.

    Attributes
    ----------
    _psi : npt.NDArray[np.complex128]
        Complex-valued 1-dimensional array representing the quantum statevector.
        Only the first ``2**nqubit`` complex values have meaning.

    _nqubit : int
        Number of active qubits at any given time.

    _max_qubits : int
        Maximum Hilbert space size allowed for internal computations. It determines
        the size of ``self._psi``. For circuit simulations, it corresponds to the number
        of qubits, while for pattern simulations it corresponds to the pattern's
        maximum space. The method :meth:`Statevec.ensure_capacity` allows to increase
        this number.

    Notes
    -----
    The internal representation of the quantum state is guaranteed to be
    normalized after initialization, and it is assumed to remain normalized
    thereafter.

    Using :meth:`evolve_single`, :meth:`expectation_single`, :meth:`evolve`,
    or :meth:`expectation_value` with non-unitary operators does not preserve
    the norm of the statevector and may lead to unexpected behavior.

    In pattern simulation, node measurements call :meth:`evolve_single` with
    a projector (a non-unitary operator). However, the measured qubit is
    immediately removed via :meth:`remove_qubit`, which restores the unit
    norm of the internal quantum state.
    See :meth:`graphix.sim.base_backend.DenseStateBackend.measure` for additional
    details.
    """

    _psi: npt.NDArray[np.complex128]
    _nqubit: int
    _max_qubits: int

    def __init__(self, data: Data = BasicStates.PLUS, nqubit: int | None = None, max_qubits: int | None = None) -> None:
        """Initialize a statevector object.

        `data` can be:
        - a single :class:`graphix.states.State` (classical description of a quantum state)
        - an iterable of :class:`graphix.states.State` objects
        - an iterable of scalars (a :math:`2^n` numerical statevector)
        - a single :class:`graphix.sim.statevec.Statevec`

        If ``nqubit`` is not provided, it is inferred from ``data``.
        If ``max_qubits`` is not provided, it is set to match the provided or inferred ``nqubit``.
        If only one :class:`graphix.states.State` is provided and ``nqubit`` is a valid integer, the statevector is initialized in the tensor product state.
        If a class:`graphix.sim.statevec.Statevec` is provided, a copy is returned.
        Consistency between provided ``nqubit``, ``max_qubits`` and ``data`` is checked.

        Parameters
        ----------
        data : Data, optional
            Input data to prepare the state. Can be a classical description or a numerical input, defaults to :class:`graphix.states.BasicStates.PLUS`.
        nqubit : int | None, optional
            Number of qubits to prepare. If ``None`` (default), it's inferred from ``data``.
        max_qubits : int | None, optional.
            Maximum Hilbert space size for array preallocation. If ``None`` (default), it's set equal to ``nqubit``.

        Raises
        ------
        ValueError
            If ``nqubit``, ``max_qubits`` or ``data`` are not consistent with each other.
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
            self._psi = data._psi.copy()
            self._max_qubits = data.max_qubits
            self._nqubit = data.nqubit

            if max_qubits is not None:
                if max_qubits < data.max_qubits:
                    raise ValueError(
                        f"`max_qubits` can't be smaller than the capacity of input state: {max_qubits} < {data.max_qubits}."
                    )
                self.ensure_capacity(max_qubits)
            return

        # The type
        # list[states.State] | list[ExpressionOrSupportsComplex] | list[Iterable[ExpressionOrSupportsComplex]]
        # would be more precise, but given a value X of type Iterable[A] | Iterable[B],
        # mypy infers that list(X) has type list[A | B] instead of list[A] | list[B].
        input_list: list[states.State | ExpressionOrSupportsComplex | Iterable[ExpressionOrSupportsComplex]]
        if isinstance(data, states.State):
            if nqubit is None:
                nqubit = 1
            input_list = [data] * nqubit
        elif isinstance(data, Iterable):
            input_list = list(data)
        else:
            raise TypeError(f"Incorrect type for data: {type(data)}.")

        if len(input_list) == 0:
            if nqubit is not None and nqubit != 0:
                raise ValueError("`nqubit` is not null but input state is empty.")
            nqubit = 0
            psi = np.array([1], dtype=np.complex128)

        elif isinstance(input_list[0], states.State):
            length = len(input_list)
            if nqubit is None:
                nqubit = length
            elif nqubit != length:
                raise ValueError(f"Mismatch between nqubit and length of input state: {nqubit} != {length}.")

            def state_to_statevector(
                s: states.State | ExpressionOrSupportsComplex | Iterable[ExpressionOrSupportsComplex],
            ) -> npt.NDArray[np.complex128]:
                if not isinstance(s, states.State):
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
                    raise ValueError(f"Length of input data is not a power of two: {length}.")
                nqubit = inferred_nqubit
            elif nqubit != inferred_nqubit:
                raise ValueError(f"Mismatch between nqubit and inferred nqubit: {nqubit} != {inferred_nqubit}.")
            psi = np.array(input_list, dtype=np.complex128)
            if not np.isclose(np.linalg.norm(psi), 1.0):
                raise ValueError("Input state is not normalized.")

        else:
            raise TypeError(
                f"First element of data has type {type(input_list[0])} whereas Number or State is expected."
            )

        if max_qubits is not None:
            if max_qubits < nqubit:
                raise ValueError(
                    f"`max_qubits` can't be smaller than the length of input state: {max_qubits} < {nqubit}."
                )
        else:
            max_qubits = nqubit

        self._psi = psi
        self._max_qubits = nqubit  # bootstrap for self.ensure_capacity
        self._nqubit = nqubit
        self.ensure_capacity(max_qubits)  # may extend both self._psi and self._max_qubits

    def __str__(self) -> str:
        """Return a string description."""
        sv = self.psi
        return f"Statevec object with statevector {sv} and length {len(sv)}."

    @property
    def psi(self) -> npt.NDArray[np.complex128]:
        """Return a view of the meaningful elements in ``self._psi``.

        These are the first ``2**self.nqubit`` elements.
        """
        size_valid_psi = 1 << self.nqubit  # 2**self.nqubit
        return self._psi[:size_valid_psi]

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

    def ensure_capacity(self, required_qubits: int) -> None:
        """Extend the state vector if the required qubit capacity exceeds the current one.

        Does nothing if ``required_qubits <= self.max_qubits``.

        Parameters
        ----------
        required_qubits : int
            Minimum number of qubits the state vector must support. If expansion
            is needed, ``self._psi`` is extended to size ``2**required_qubits``.
        """
        if required_qubits > self.max_qubits:
            offset = (1 << required_qubits) - len(self._psi)
            self._psi = np.concatenate([self._psi, np.empty(offset, dtype=self._psi.dtype)])
            self._max_qubits = required_qubits

    @override
    def flatten(self) -> Matrix:
        """Return flattened state.

        A view of only the first ``2**self.nqubit`` elements of ``self.psi`` is returned.
        """
        return self.psi

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        r"""Add nodes (qubits) to the state vector and initialize them in a specified state.

        Previously existing nodes remain unchanged.

        Parameters
        ----------
        nqubit : int
            The number of qubits to add to the state vector.

        data : Data
            The state in which to initialize the newly added nodes.

            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of ``nodes``, and
              each node is initialized with its corresponding state.
            - A single-qubit state vector will be broadcast to all nodes.
            - A multi-qubit state vector of dimension :math:`2^n`, where :math:`n = \mathrm{len}(nodes)`, initializes the new nodes jointly.

        Notes
        -----
        This method can extend the size of ``self._psi`` for convenience, but this requires allocating a full new array.
        """
        self.ensure_capacity(required_qubits=self.nqubit + nqubit)
        if nqubit == 1 and data is BasicStates.PLUS:
            # Simulating standard N commands falls in this branch.
            _add_default_node_jit(self._psi, self.nqubit)
            self._nqubit += 1
        else:
            sv_to_add = Statevec(nqubit=nqubit, data=data)
            self.tensor(sv_to_add)

    @override
    def entangle(self, qubits: tuple[int, int]) -> None:
        """Apply a CZ gate on two qubits.

        Parameters
        ----------
        qubits : tuple[int, int]
            (control, target) qubit indices.
        """
        # `_entangle_jit` is not unsafe if calle on out-of-bound indices but
        # we check them for robustness.
        for qubit in qubits:
            self._check_bounds(qubit)
        _entangle_jit(self._psi, self.nqubit, *qubits)

    @override
    def evolve_single(self, op: Matrix, qubit: int) -> None:
        """Apply a single-qubit operator.

        Parameters
        ----------
        op : npt.NDArray[np.complex128]
            Complex-valued matrix of shape :math:`(2, 2)` representing
            the operator to apply.
        qubit : int
            Target qubit index.
        """
        self._check_bounds(qubit)
        # Downcast from Matrix to np.complex128 to match numba signature.
        op_as_complex = _cast_op(op)
        _evolve_single_jit(self._psi, op_as_complex, self.nqubit, qubit)

    @override
    def expectation_single(self, op: Matrix, qubit: int) -> complex:
        """Return the expectation value of a single-qubit operator.

        Parameters
        ----------
        op : npt.NDArray[np.complex128]
            Complex-valued matrix of shape :math:`(2, 2)` representing
            the operator to measure.
        qubit : int
            Target qubit index.

        Returns
        -------
        complex
            Expectation value.

        Notes
        -----
        This method assumes that quantum state represented by ``self.psi`` is normalized. See the class docstring for details.
        """
        self._check_bounds(qubit)
        # Downcast from Matrix to np.complex128 to match numba signature.
        op_as_complex = _cast_op(op)
        return _expectation_single_jit(self._psi, op_as_complex, self.nqubit, qubit)

    @override
    def evolve(self, op: Matrix, qubits: Sequence[int]) -> None:
        r"""Apply a multi-qubit operator.

        Parameters
        ----------
        op : npt.NDArray[np.complex128]
            Complex-valued matrix of shape :math:`(2^n, 2^n)` representing
            the operator to apply.
        qubits : Sequence[int]
            Target qubit indices.

        Notes
        -----
        This method is a fallback for circuit simulation and it's not required
        for pattern simulation. It does not have an efficient JIT-compiled
        implementation.
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

        self._psi[: len(self.psi)] = np.einsum(op_t, op_idx, psi_t, psi_idx, res_idx).reshape(1 << self.nqubit)  # type: ignore[arg-type] # https://github.com/numpy/numpy/issues/31513

    def expectation_value(self, op: Matrix, qubits: Sequence[int]) -> complex:
        """Return the expectation value of a multi-qubit operator.

        Parameters
        ----------
        op : npt.NDArray[np.complex128]
            Complex-valued matrix of shape :math:`(2^n, 2^n)` representing
            the operator to measure.
        qubits : Sequence[int]
            Target qubit indices.

        Notes
        -----
        This method assumes that quantum state represented by ``self.psi`` is normalized.
        See the class docstring for details.
        """
        sv = deepcopy(self)
        sv.evolve(op, qubits)
        return complex(np.dot(self.flatten().conjugate(), sv.flatten()))

    @override
    def remove_qubit(self, qubit: int) -> None:
        r"""Remove a separable qubit from the system and assemble the statevector of the remaining qubits.

        This is equivalent to the partial trace if ``qubit`` corresponds to a separable qubit.

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

        (after normalization) for :math:`k =` ``qubit``. If the :math:`k` th qubit is in the :math:`\ket{1}` state, all the amplitudes above will be zero.
        In that case the returned state will be the one above with
        :math:`0_{\mathrm{k}}` replaced with :math:`1_{\mathrm{k}}`.

        .. warning::
            This method assumes the qubit ``qarg`` to be separable from the rest,
            and is implemented as a significantly faster alternative for partial trace to
            be used after single-qubit measurements.
            Separability is not checked.

        Parameters
        ----------
        qubit : int
            Target qubit index.

        Notes
        -----
        The implementation of this method does not support a parallelized kernel because data is read and written on the same array.
        """
        self._check_bounds(qubit)
        self._nqubit = _remove_qubit_jit(self._psi, self.nqubit, qubit, atol=1e-10)

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """Apply SWAP gate between two qubits.

        Parameters
        ----------
        qubits : tuple[int, int]
            (control, target) qubit indices.
        """
        _swap_jit(self._psi, self.nqubit, *qubits)

    def tensor(self, other: Statevec) -> None:
        r"""Tensor product state with other qubits.

        Results in ``self`` :math:`\otimes` ``other``.

        Parameters
        ----------
        other : :class:`graphix.sim.statevec.Statevec`
            Statevector to be tensored with ``self``.

        Notes
        -----
        This method is used internally by :meth:`add_nodes`.
        """
        _tensor_jit(self._psi, other.psi, self.nqubit, other.nqubit)
        self._nqubit += other.nqubit

    def _check_bounds(self, qubit: int) -> None:
        """Check if qubit index is valid.

        This check is necessary because there is no bounds checking in Numba. See
        https://numba.pydata.org/numba-doc/dev/reference/pysemantics.html#bounds-checking

        Parameters
        ----------
        qubit : int
            Target qubit index.

        Raises
        ------
        IndexError
        """
        if not 0 <= qubit < self.nqubit:
            raise IndexError(f"Qubit index {qubit} out of range [0, {self.nqubit} -1]")

    def fidelity(self, other: Statevec) -> float:
        r"""Calculate the fidelity against another statevector.

        The fidelity is defined as :math:`|\langle\psi_1|\psi_2\rangle|^2`.

        Parameters
        ----------
        other : :class:`graphix.sim.statevec.Statevec`
            Statevector to compare with.

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
        other : :class:`graphix.sim.statevec.Statevec`
            Statevector to compare with.
        rtol : float
            Relative tolerance for :func:`math.isclose`.
        atol : float
            Absolute tolerance for :func:`math.isclose`.

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


def _format_encoding(nqubit: int, i: int, encoding: _ENCODING) -> str:
    """Format the i-th basis vector as a ket.

    See :meth:`Statevec.to_dict` for additional details.
    """
    display_width = nqubit
    output = f"{i:0{display_width}b}"
    if encoding == "LSB":
        return output[::-1]
    return output


@dataclass(frozen=True)
class StatevectorBackend(DenseStateBackend[Statevec]):
    """Numba JIT-compiled MBQC statevector backend simulator based on Ref. [1].

    See Also
    --------
    graphix.sim.base_backend.DenseStateBackend
        Base class describing available parameters and shared behavior.

    Notes
    -----
    By default, the backend is initialized with a 0-dimensional statevector
    (a scalar ``1``) and ``max_qubits = 0``.

    The internal state representation can be expanded using
    ``StatevectorBackend.add_nodes``, but this is inefficient since it
    requires copying the full quantum state array.

    To preallocate memory for a fixed system size, use
    :meth:`StatevectorBackend.with_capacity`.

    References
    ----------
    [1] McGuffin, M. J., Robert J-M., and Ikeda K. "How to Write a Simulator for Quantum Circuits from Scratch: A Tutorial.", 2025 (arXiv:2506.08142).
    """

    state: Statevec = dataclasses.field(init=True, default_factory=lambda: Statevec(nqubit=0))

    def __post_init__(self) -> None:
        """Validate backend configuration.

        Raises
        ------
        ValueError
            If ``symbolic`` is ``True``, since the statevector backend
            does not support symbolic simulation.
        """
        if self.symbolic:
            raise ValueError(
                "Statevector backend does not support `symbolic` simulation. Consider using backend in `graphix-symbolic` plugin."
            )

    @classmethod
    def with_capacity(
        cls, max_qubits: int, state: Statevec | None = None, **kwargs: Unpack[DenseStateBackendKwargs]
    ) -> Self:
        """Initialize the backend with preallocated statevector capacity.

        Parameters
        ----------
        max_qubits : int
            Maximum number of qubits supported by the statevector. For pattern simulation this corresponds to ``Pattern.max_space()``.
        state: Statevec | None = None
            Initial backend state. If ``None``, the backend is initialized
            with a 0-dimensional statevector (scalar ``1``).
        **kwargs
            Options for :class:`graphix.sim.base_backend.DenseStateBackend`. See
            :class:`graphix.sim.base_backend.DenseStateBackendKwargs`.

        Returns
        -------
        Self
            Backend instance with capacity for up to ``max_qubits`` qubits.
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
    """Tensor in-place two state vectors.

    This function assumes that ``psi`` has enough capacity to contain the tensor product.
    """
    size_psi = 1 << nqubit
    size_other = 1 << nqubit_other
    # We update the elements of `psi` in-place.
    # This requires starting the update for the last element of the new psi, `size_psi * size_other - 1`
    k = size_psi * size_other - 1
    max_psi_idx = size_psi - 1
    max_other_idx = size_other - 1

    for i in range(size_psi):
        alpha_old = psi[max_psi_idx - i]
        for j in range(size_other):
            psi[k] = alpha_old * psi_other[max_other_idx - j]
            k -= 1


@nb.njit("(c16[::1], int32)")
def _add_default_node_jit(
    psi: npt.NDArray[np.complex128],
    nqubit: int,
) -> None:
    r"""Tensor in-place a one-qubit |+> state.

    This function follows the same logic as :func:`_tensor_jit` but is specialized to ``psi_other`` being a :math:`(1, 1)/\sqrt{2}` state.
    """
    read_idx = (1 << nqubit) - 1  # 2**nqubit - 1
    write_idx = (1 << nqubit + 1) - 1  # 2**(nqubit + 1) -1
    amp = 1 / np.sqrt(2)

    while read_idx >= 0:
        v = amp * psi[read_idx]

        psi[write_idx] = v
        psi[write_idx - 1] = v

        read_idx -= 1
        write_idx -= 2


@nb.njit("(c16[::1], int32, int32, int32)")
def _swap_jit(psi: npt.NDArray[np.complex128], nqubit: int, q1: int, q2: int) -> None:
    """Swap two qubits.

    This function is inspired from Ref. [1].
    """
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


@nb.njit("(c16[::1], c16[:,:], int32, int32, int32)")
def _compute_op_psi(
    psi: npt.NDArray[np.complex128], op: npt.NDArray[np.complex128], b0: int, offset: int, size_half_block: int
) -> None:
    i1 = b0 + offset
    i2 = i1 + size_half_block
    psi1 = psi[i1]
    psi2 = psi[i2]
    psi[i1] = op[0, 0] * psi1 + op[0, 1] * psi2
    psi[i2] = op[1, 0] * psi1 + op[1, 1] * psi2


@nb.njit("(c16[::1], c16[:,:], int32, int32)", parallel=True)
def _evolve_single_jit(psi: npt.NDArray[np.complex128], op: npt.NDArray[np.complex128], nqubit: int, q: int) -> None:
    r"""Apply a single-qubit operator.

    This function is inspired from Ref. [1].

    The kernel switches to a parallel implementation when the
    number of qubits exceeds ``NUM_QUBIT_PARALLEL`` (module constant).
    """
    nblocks = 1 << q
    size_block = 1 << nqubit - q  # 2**(nqubit - q)
    size_half_block = (
        size_block >> 1
    )  # Left-to-right tensor product encoding (first qubit corresponds to most significant bit). For right-to-left encoding use `size_half_block = 1 << i`

    if nqubit > NUM_QUBIT_PARALLEL:
        if nblocks > 1:
            for b in nb.prange(nblocks):
                # WARNING: setting `b0 += size_block` may result in a race condition if parallel loop.
                b0 = size_block * b
                for offset in range(size_half_block):
                    _compute_op_psi(psi, op, b0, offset, size_half_block)
        else:
            for offset in nb.prange(size_half_block):
                _compute_op_psi(psi, op, 0, offset, size_half_block)
    else:
        b0 = 0
        for _ in range(nblocks):
            for offset in range(size_half_block):
                _compute_op_psi(psi, op, b0, offset, size_half_block)
            b0 += size_block


@nb.njit("c16(c16[::1], c16[:,:], int32, int32, int32)")
def _compute_psic_op_psi(
    psi: npt.NDArray[np.complex128], op: npt.NDArray[np.complex128], b0: int, offset: int, size_half_block: int
) -> complex:
    i1 = b0 + offset
    i2 = i1 + size_half_block
    psi1 = psi[i1]
    psi2 = psi[i2]
    b1 = op[0, 0] * psi1 + op[0, 1] * psi2
    b2 = op[1, 0] * psi1 + op[1, 1] * psi2
    return psi1.conjugate() * b1 + psi2.conjugate() * b2  # type: ignore[no-any-return]


@nb.njit("c16(c16[::1], c16[:,:], int32, int32)", parallel=True)
def _expectation_single_jit(
    psi: npt.NDArray[np.complex128], op: npt.NDArray[np.complex128], nqubit: int, q: int
) -> complex:
    """Compute expectation value of single-qubit operator.

    This function applies ``op`` on ``psi`` in the same way as :func:`_evolve_single`.

    The kernel switches to a parallel implementation when the
    number of qubits exceeds ``NUM_QUBIT_PARALLEL`` (module constant).
    """
    nblocks = 1 << q
    size_block = 1 << nqubit - q
    size_half_block = (
        size_block >> 1
    )  # Left-to-right tensor product encoding (first qubit corresponds to most significant bit). For right-to-left encoding use `size_half_block = 1 << i`

    result = 0.0 + 0.0j

    if nqubit > NUM_QUBIT_PARALLEL:
        if nblocks > 1:
            for b in nb.prange(nblocks):
                # WARNING: setting `b0 += size_block` may result in a race condition if parallel loop.
                b0 = size_block * b
                for offset in range(size_half_block):
                    result += _compute_psic_op_psi(psi, op, b0, offset, size_half_block)
        else:
            for offset in nb.prange(size_half_block):
                result += _compute_psic_op_psi(psi, op, 0, offset, size_half_block)
    else:
        b0 = 0
        for _ in range(nblocks):
            for offset in range(size_half_block):
                result += _compute_psic_op_psi(psi, op, b0, offset, size_half_block)
            b0 += size_block

    return result


@nb.njit("(c16[::1], int32, int32, int32)", parallel=True)
def _entangle_jit(psi: npt.NDArray[np.complex128], nqubit: int, control: int, target: int) -> None:
    """Apply CZ gate on two qubits.

    This function is inspired from Ref. [1].

    The kernel switches to a parallel implementation when the
    number of qubits exceeds ``NUM_QUBIT_PARALLEL`` (module constant).
    """
    size_sv = 1 << nqubit
    mask_control = 1 << nqubit - 1 - control
    mask_target = 1 << nqubit - 1 - target
    mask = mask_control | mask_target
    # `mask` is an integer number whose binary representation has 1s at positions `control` and `target` and 0s elsewhere.

    if nqubit > NUM_QUBIT_PARALLEL:
        for i in nb.prange(size_sv):
            if mask & i == mask:
                psi[i] = -psi[i]
    else:
        for i in range(size_sv):
            if mask & i == mask:
                psi[i] = -psi[i]


@nb.njit("f8(c16[::1], int32, int32)")
def _compute_a2(
    psi: npt.NDArray[np.complex128],
    b0: int,
    j: int,
) -> float:
    a = psi[b0 + j]
    a_re = a.real
    a_im = a.imag
    return a_re * a_re + a_im * a_im  # type: ignore[no-any-return]


@nb.njit("f8(c16[::1], int32, int32, int32, int32, int32)", parallel=True)
def _compute_norm(
    psi: npt.NDArray[np.complex128],
    nqubit: int,
    n_blocks: int,
    size_block: int,
    size_half_block: int,
    shift: int,
) -> float:
    """Compute the norm of psi.

    The kernel switches to a parallel implementation when the
    number of qubits exceeds ``NUM_QUBIT_PARALLEL`` (module constant).
    """
    norm2 = 0.0
    if nqubit > NUM_QUBIT_PARALLEL:
        if n_blocks > 1:
            for b in nb.prange(n_blocks):
                b0 = b * size_block + shift
                for j in range(size_half_block):
                    norm2 += _compute_a2(psi, b0, j)
        else:
            for j in nb.prange(size_half_block):
                norm2 += _compute_a2(psi, shift, j)
    else:
        for b in range(n_blocks):
            b0 = b * size_block + shift
            for j in range(size_half_block):
                norm2 += _compute_a2(psi, b0, j)
    return norm2


@nb.njit("void(c16[::1], int32, int32, int32, int32, f8)")
def _scale_psi_kernel(
    psi: npt.NDArray[np.complex128],
    n_blocks: int,
    size_block: int,
    size_half_block: int,
    shift: int,
    inv_norm: float,
) -> None:
    """Update ``psi`` with selected and normalized elements.

    The implementation of this function does not support a parallelized
    kernel because data is read and written on the same array.
    """
    b0 = shift
    k = 0
    for _ in range(n_blocks):
        for j in range(size_half_block):
            psi[k] = psi[b0 + j] * inv_norm
            k += 1
        b0 += size_block


@nb.njit("void(c16[::1], int32, int32, int32, int32, f8)")
def _scale_psi(
    psi: npt.NDArray[np.complex128],
    n_blocks: int,
    size_block: int,
    size_half_block: int,
    shift: int,
    inv_norm: float,
) -> None:
    """Update ``psi`` with selected and normalized elements."""
    # If the inner loop in `_scale_psi_kernel` has too few elements
    # it introduces some overhead and it's best to unroll it.
    # Numba can do that only if ``size_half_block`` is a constant at
    # compile time, so we dispatch the relevant cases.
    # We observe speed improvements of up to 10%.
    if size_block == 2:
        _scale_psi_kernel(psi, n_blocks, 2, 1, shift, inv_norm)
    elif size_block == 4:
        _scale_psi_kernel(psi, n_blocks, 4, 2, shift, inv_norm)
    elif size_block == 8:
        _scale_psi_kernel(psi, n_blocks, 8, 4, shift, inv_norm)
    else:
        _scale_psi_kernel(psi, n_blocks, size_block, size_half_block, shift, inv_norm)


@nb.njit("int32(c16[::1], int32, int32, f8)")
def _remove_qubit_jit(
    psi: npt.NDArray[np.complex128],
    nqubit: int,
    q: int,
    atol: float,
) -> int:
    """Remove qubit.

    Argument ``atol`` controls the tolerance below which norm of statevector is 0.
    See :meth:`Statevec.remove_qubit` for additional details on the implementation.

    This implementation benefited from Jérôme Richard's advice.
    https://stackoverflow.com/questions/79948374/improving-efficiency-of-numba-jit-function
    """
    new_nqubit = nqubit - 1

    n_blocks = 1 << q
    size_block = 1 << nqubit - q  # 2**(nqubits - q)
    size_half_block = size_block >> 1

    # Compute norm of branch 0
    shift = 0
    norm2 = _compute_norm(psi, nqubit, n_blocks, size_block, size_half_block, shift)

    # If norm of branch 0 is 0, compute norm of branch 1 and set shift to branch 1
    if norm2 <= atol:
        shift = size_half_block
        norm2 = _compute_norm(psi, nqubit, n_blocks, size_block, size_half_block, shift)

    if norm2 <= atol:
        raise RuntimeError(f"Attempted to remove qubit {q} from 0-norm statevector.")

    inv_norm = 1.0 / math.sqrt(norm2)
    _scale_psi(psi, n_blocks, size_block, size_half_block, shift, inv_norm)

    return new_nqubit


def _cast_op(op: Matrix) -> npt.NDArray[np.complex128]:
    if op.dtype == np.object_:
        raise TypeError(
            "Statevector backend does not support symbolic operators. Consider using backend in `graphix-symbolic` plugin."
        )
    # By default, the numba signature c16[:,:] assumes a writeable array.
    # Arrays obtained from, e.g., `graphix.clifford.Clifford.H.matrix` or
    # `graphix.ops.Ops.X` are not writtable and therefore do not match the
    # numba signature.
    # Since ``op`` is a 2-by-2 matrix it's probably not worth it to dispatch
    # multiple jit kernels (with the appropriate numba signature)
    # depending on its WRITEABLE flag to avoid copying.
    # https://github.com/numba/numba/issues/4511#issuecomment-527350694
    # https://numba.pydata.org/numba-doc/0.17.0/reference/types.html#numba.types.Array
    return op.astype(np.complex128, copy=True)
