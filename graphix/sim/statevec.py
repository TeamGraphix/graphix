"""MBQC state vector backend."""

from __future__ import annotations

import copy
import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat

import numpy as np
import numpy.typing as npt

from graphix import parameter, states, utils
from graphix.parameter import Expression, ExpressionOrSupportsComplex
from graphix.sim.base_backend import Backend, State
from graphix.states import BasicStates

if TYPE_CHECKING:
    import collections
    from collections.abc import Mapping

    from graphix.parameter import ExpressionOrSupportsFloat, Parameter


class StatevectorBackend(Backend):
    """MBQC simulator with statevector method."""

    def __init__(self, **kwargs) -> None:
        """Construct a state vector backend."""
        super().__init__(Statevec(nqubit=0), **kwargs)


CZ_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, -1]]]],
    dtype=np.complex128,
)
CNOT_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]],
    dtype=np.complex128,
)
SWAP_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]],
    dtype=np.complex128,
)


class Statevec(State):
    """Statevector object."""

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
    ):
        """Initialize statevector objects.

        `data` can be:
        - a single :class:`graphix.states.State` (classical description of a quantum state)
        - an iterable of :class:`graphix.states.State` objects
        - an iterable of scalars (A 2**n numerical statevector)
        - a `graphix.statevec.Statevec` object

        If `nqubit` is not provided, the number of qubit is inferred from `data` and checked for consistency.
        If only one :class:`graphix.states.State` is provided and nqubit is a valid integer, initialize the statevector
        in the tensor product state.
        If both `nqubit` and `data` are provided, consistency of the dimensions is checked.
        If a `graphix.statevec.Statevec` is passed, returns a copy.


        :param data: input data to prepare the state. Can be a classical description or a numerical input, defaults to graphix.states.BasicStates.PLUS
        :type data: Data, optional
        :param nqubit: number of qubits to prepare, defaults to None
        :type nqubit: int, optional
        """
        if nqubit is not None and nqubit < 0:
            raise ValueError("nqubit must be a non-negative integer.")

        if isinstance(data, Statevec):
            # assert nqubit is None or len(state.flatten()) == 2**nqubit
            if nqubit is not None and len(data.flatten()) != 2**nqubit:
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the inferred number of qubit = {len(data.flatten())}."
                )
            self.psi = data.psi.copy()
            return

        if isinstance(data, states.State):
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

            self.psi = np.array(1, dtype=np.complex128)

        elif isinstance(input_list[0], states.State):
            utils.check_list_elements(input_list, states.State)
            if nqubit is None:
                nqubit = len(input_list)
            elif nqubit != len(input_list):
                raise ValueError("Mismatch between nqubit and length of input state.")
            list_of_sv = [s.get_statevector() for s in input_list]
            tmp_psi = functools.reduce(np.kron, list_of_sv)
            # reshape
            self.psi = tmp_psi.reshape((2,) * nqubit)
        # `SupportsFloat` is needed because `numpy.float64` is not an instance of `SupportsComplex`!
        elif isinstance(input_list[0], (Expression, SupportsComplex, SupportsFloat)):
            utils.check_list_elements(input_list, (Expression, SupportsComplex, SupportsFloat))
            if nqubit is None:
                length = len(input_list)
                if length & (length - 1):
                    raise ValueError("Length is not a power of two")
                nqubit = length.bit_length() - 1
            elif nqubit != len(input_list).bit_length() - 1:
                raise ValueError("Mismatch between nqubit and length of input state")
            psi = np.array(input_list)
            # check only if the matrix is not symbolic
            if psi.dtype != "O" and not np.allclose(np.sqrt(np.sum(np.abs(psi) ** 2)), 1):
                raise ValueError("Input state is not normalized")
            self.psi = psi.reshape((2,) * nqubit)
        else:
            raise TypeError(f"First element of data has type {type(input_list[0])} whereas Number or State is expected")

    def __str__(self) -> str:
        """Return a string description."""
        return f"Statevec object with statevector {self.psi} and length {self.dims()}."

    def add_nodes(self, nqubit, data) -> None:
        """Add nodes to the state vector."""
        sv_to_add = Statevec(nqubit=nqubit, data=data)
        self.tensor(sv_to_add)

    def evolve_single(self, op: npt.NDArray, i: int) -> None:
        """Apply a single-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 matrix
        i : int
            qubit index
        """
        psi = np.tensordot(op, self.psi, (1, i))
        self.psi = np.moveaxis(psi, 0, i)

    def evolve(self, op: np.ndarray, qargs: list[int]) -> None:
        """Apply a multi-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n matrix
        qargs : list of int
            target qubits' indices
        """
        op_dim = int(np.log2(len(op)))
        # TODO shape = (2,)* 2 * op_dim
        shape = [2 for _ in range(2 * op_dim)]
        op_tensor = op.reshape(shape)
        psi = np.tensordot(
            op_tensor,
            self.psi,
            (tuple(op_dim + i for i in range(len(qargs))), tuple(qargs)),
        )
        self.psi = np.moveaxis(psi, range(len(qargs)), qargs)

    def dims(self):
        """Return the dimensions."""
        return self.psi.shape

    def ptrace(self, qargs) -> None:
        """Perform partial trace of the selected qubits.

        .. warning::
            This method currently assumes qubits in qargs to be separable from the rest
            (checks not implemented for speed).
            Otherwise, the state returned will be forced to be pure which will result in incorrect output.
            Correct behaviour will be implemented as soon as the densitymatrix class, currently under development
            (PR #64), is merged.

        Parameters
        ----------
        qargs : list of int
            qubit indices to trace over
        """
        nqubit_after = len(self.psi.shape) - len(qargs)
        psi = self.psi
        rho = np.tensordot(psi, psi.conj(), axes=(qargs, qargs))  # density matrix
        rho = np.reshape(rho, (2**nqubit_after, 2**nqubit_after))
        evals, evecs = np.linalg.eig(rho)  # back to statevector
        # NOTE works since only one 1 in the eigenvalues corresponding to the state
        # TODO use np.eigh since rho is Hermitian?
        self.psi = np.reshape(evecs[:, np.argmax(evals)], (2,) * nqubit_after)

    def remove_qubit(self, qarg: int) -> None:
        r"""Remove a separable qubit from the system and assemble a statevector for remaining qubits.

        This results in the same result as partial trace, if the qubit `qarg` is separable from the rest.

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
            This method assumes the qubit with index `qarg` to be separable from the rest,
            and is implemented as a significantly faster alternative for partial trace to
            be used after single-qubit measurements.
            Care needs to be taken when using this method.
            Checks for separability will be implemented soon as an option.

        .. seealso::
            :meth:`graphix.sim.statevec.Statevec.ptrace` and warning therein.

        Parameters
        ----------
        qarg : int
            qubit index
        """
        norm = _get_statevec_norm(self.psi)
        if isinstance(norm, SupportsFloat):
            assert not np.isclose(norm, 0)
        psi = self.psi.take(indices=0, axis=qarg)
        norm = _get_statevec_norm(psi)
        self.psi = (
            psi
            if not isinstance(norm, SupportsFloat) or not np.isclose(norm, 0)
            else self.psi.take(indices=1, axis=qarg)
        )
        self.normalize()

    def entangle(self, edge: tuple[int, int]) -> None:
        """Connect graph nodes.

        Parameters
        ----------
        edge : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        psi = np.tensordot(CZ_TENSOR, self.psi, ((2, 3), edge))
        # sort back axes
        self.psi = np.moveaxis(psi, (0, 1), edge)

    def tensor(self, other: Statevec) -> None:
        r"""Tensor product state with other qubits.

        Results in self :math:`\otimes` other.

        Parameters
        ----------
        other : :class:`graphix.sim.statevec.Statevec`
            statevector to be tensored with self
        """
        psi_self = self.psi.flatten()
        psi_other = other.psi.flatten()

        total_num = len(self.dims()) + len(other.dims())
        self.psi = np.kron(psi_self, psi_other).reshape((2,) * total_num)

    def cnot(self, qubits):
        """Apply CNOT.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        psi = np.tensordot(CNOT_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(psi, (0, 1), qubits)

    def swap(self, qubits) -> None:
        """Swap qubits.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        psi = np.tensordot(SWAP_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(psi, (0, 1), qubits)

    def normalize(self) -> None:
        """Normalize the state in-place."""
        norm = _get_statevec_norm(self.psi)
        self.psi = self.psi / norm

    def flatten(self) -> npt.NDArray:
        """Return flattened statevector."""
        return self.psi.flatten()

    def expectation_single(self, op: np.NDArray, loc: int) -> complex:
        """Return the expectation value of single-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 operator
        loc : int
            target qubit index

        Returns
        -------
        complex : expectation value.
        """
        st1 = copy.copy(self)
        st1.normalize()
        st2 = copy.copy(st1)
        st1.evolve_single(op, loc)
        return np.dot(st2.psi.flatten().conjugate(), st1.psi.flatten())

    def expectation_value(self, op: np.NDArray, qargs: collections.abc.Iterable[int]) -> complex:
        """Return the expectation value of multi-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n operator
        qargs : list of int
            target qubit indices

        Returns
        -------
        complex : expectation value
        """
        st2 = copy.copy(self)
        st2.normalize()
        st1 = copy.copy(st2)
        st1.evolve(op, qargs)
        return np.dot(st2.psi.flatten().conjugate(), st1.psi.flatten())

    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Statevec:
        """Return a copy of the state vector where all occurrences of the given variable in measurement angles are substituted by the given value."""
        result = Statevec()
        result.psi = np.vectorize(lambda value: parameter.subs(value, variable, substitute))(self.psi)
        return result

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> Statevec:
        """Return a copy of the state vector where all occurrences of the given keys in measurement angles are substituted by the given values in parallel."""
        result = Statevec()
        result.psi = np.vectorize(lambda value: parameter.xreplace(value, assignment))(self.psi)
        return result


def _get_statevec_norm(psi):
    """Return norm of the state."""
    return np.sqrt(np.sum(psi.flatten().conj() * psi.flatten()))


if TYPE_CHECKING:
    from collections.abc import Iterable

    Data = states.State | Statevec | Iterable[states.State] | Iterable[ExpressionOrSupportsComplex]
else:
    from collections.abc import Iterable
    from typing import Union

    Data = Union[
        states.State,
        Statevec,
        Iterable[states.State],
        Iterable[ExpressionOrSupportsComplex],
    ]
