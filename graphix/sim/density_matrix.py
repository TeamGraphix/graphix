"""Density matrix simulator.

Simulate MBQC with density matrix representation.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat, cast

import numpy as np
from typing_extensions import override

from graphix import linalg_validations as lv
from graphix import parameter
from graphix.channels import KrausChannel
from graphix.parameter import Expression, ExpressionOrFloat, ExpressionOrSupportsComplex
from graphix.sim.base_backend import DenseState, FullStateBackend, Matrix, kron, matmul, outer, tensordot, vdot
from graphix.sim.statevec import CNOT_TENSOR, CZ_TENSOR, SWAP_TENSOR, Statevec
from graphix.states import BasicStates, State

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from graphix.parameter import ExpressionOrSupportsFloat, Parameter
    from graphix.sim.data import Data


class DensityMatrix(DenseState):
    """DensityMatrix object."""

    rho: Matrix

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
    ):
        """Initialize density matrix objects.

        The behaviour builds on the one of *graphix.statevec.Statevec*.
        `data` can be:
        - a single :class:`graphix.states.State` (classical description of a quantum state)
        - an iterable of :class:`graphix.states.State` objects
        - an iterable of iterable of scalars (A *2**n x 2**n* numerical density matrix)
        - a *graphix.statevec.DensityMatrix* object
        - a *graphix.statevec.Statevector* object

        If `nqubit` is not provided, the number of qubit is inferred from `data` and checked for consistency.
        If only one :class:`graphix.states.State` is provided and nqubit is a valid integer, initialize the statevector
        in the tensor product state.
        If both `nqubit` and `data` are provided, consistency of the dimensions is checked.
        If a *graphix.statevec.Statevec* or *graphix.statevec.DensityMatrix* is passed, returns a copy.


        :param data: input data to prepare the state. Can be a classical description or a numerical input, defaults to graphix.states.BasicStates.PLUS
        :type data: Data
        :param nqubit: number of qubits to prepare, defaults to *None*
        :type nqubit: int, optional
        """
        if nqubit is not None and nqubit < 0:
            raise ValueError("nqubit must be a non-negative integer.")

        def check_size_consistency(mat: Matrix) -> None:
            if nqubit is not None and mat.shape != (2**nqubit, 2**nqubit):
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the shape of the provided density matrix = {mat.shape}."
                )

        if isinstance(data, DensityMatrix):
            check_size_consistency(data.rho)
            # safe: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html
            self.rho = data.rho.copy()
            return
        if isinstance(data, Iterable):
            input_list = list(data)
            if len(input_list) != 0 and isinstance(input_list[0], Iterable):

                def get_row(
                    item: Iterable[ExpressionOrSupportsComplex] | State | Expression | SupportsComplex,
                ) -> list[ExpressionOrSupportsComplex]:
                    if isinstance(item, Iterable):
                        return list(item)
                    raise TypeError("Every row of a matrix should be iterable.")

                input_matrix: list[list[ExpressionOrSupportsComplex]] = list(map(get_row, input_list))
                if isinstance(input_matrix[0][0], (Expression, SupportsComplex, SupportsFloat)):
                    self.rho = np.array(input_matrix)
                    if not lv.is_qubitop(self.rho):
                        raise ValueError("Cannot interpret the provided density matrix as a qubit operator.")
                    check_size_consistency(self.rho)
                    if self.rho.dtype != "O":
                        if not lv.is_unit_trace(self.rho):
                            raise ValueError("Density matrix must have unit trace.")
                        if not lv.is_psd(self.rho):
                            raise ValueError("Density matrix must be positive semi-definite.")
                    return
        statevec = Statevec(data, nqubit)
        # NOTE this works since np.outer flattens the inputs!
        self.rho = outer(statevec.psi, statevec.psi.conj())

    @property
    def nqubit(self) -> int:
        """Return the number of qubits."""
        return self.rho.shape[0].bit_length() - 1

    def __str__(self) -> str:
        """Return a string description."""
        return f"DensityMatrix object, with density matrix {self.rho} and shape {self.dims()}."

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        r"""
        Add nodes (qubits) to the density matrix and initialize them in a specified state.

        Parameters
        ----------
        nqubit : int
            The number of qubits to add to the density matrix.

        data : Data, optional
            The state in which to initialize the newly added nodes.

            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of ``nodes``, and
              each node is initialized with its corresponding state.
            - A single-qubit state vector will be broadcast to all nodes.
            - A multi-qubit state vector of dimension :math:`2^n` initializes the new nodes jointly.
            - A density matrix must have shape :math:`2^n \times 2^n`,
              and is used to jointly initialize the new nodes.

        Notes
        -----
        Previously existing nodes remain unchanged.
        """
        dm_to_add = DensityMatrix(nqubit=nqubit, data=data)
        self.tensor(dm_to_add)

    @override
    def evolve_single(self, op: Matrix, i: int) -> None:
        """Single-qubit operation.

        Parameters
        ----------
            op : np.ndarray
                2*2 matrix.
            i : int
                Index of qubit to apply operator.
        """
        assert i >= 0
        assert i < self.nqubit
        if op.shape != (2, 2):
            raise ValueError("op must be 2*2 matrix.")

        rho_tensor = self.rho.reshape((2,) * self.nqubit * 2)
        rho_tensor = tensordot(tensordot(op, rho_tensor, axes=(1, i)), op.conj().T, axes=(i + self.nqubit, 0))
        rho_tensor = np.moveaxis(rho_tensor, (0, -1), (i, i + self.nqubit))
        self.rho = rho_tensor.reshape((2**self.nqubit, 2**self.nqubit))

    @override
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        """Multi-qubit operation.

        Args:
            op (np.array): 2^n*2^n matrix
            qargs (list of ints): target qubits' indexes
        """
        d = op.shape
        # check it is a matrix.
        if len(d) == 2:
            # check it is square
            if d[0] == d[1]:
                pass
            else:
                raise ValueError(f"The provided operator has shape {op.shape} and is not a square matrix.")
        else:
            raise ValueError(f"The provided data has incorrect shape {op.shape}.")

        nqb_op = np.log2(len(op))
        if not np.isclose(nqb_op, int(nqb_op)):
            raise ValueError("Incorrect operator dimension: not consistent with qubits.")
        nqb_op = int(nqb_op)

        if nqb_op != len(qargs):
            raise ValueError("The dimension of the operator doesn't match the number of targets.")

        if not all(0 <= i < self.nqubit for i in qargs):
            raise ValueError("Incorrect target indices.")
        if len(set(qargs)) != nqb_op:
            raise ValueError("A repeated target qubit index is not possible.")

        op_tensor = op.reshape((2,) * 2 * nqb_op)

        rho_tensor = self.rho.reshape((2,) * self.nqubit * 2)

        rho_tensor = tensordot(
            tensordot(op_tensor, rho_tensor, axes=(tuple(nqb_op + i for i in range(len(qargs))), tuple(qargs))),
            op.conj().T.reshape((2,) * 2 * nqb_op),
            axes=(tuple(i + self.nqubit for i in qargs), tuple(i for i in range(len(qargs)))),
        )
        rho_tensor = np.moveaxis(
            rho_tensor,
            list(range(len(qargs))) + [-i for i in range(1, len(qargs) + 1)],
            list(qargs) + [i + self.nqubit for i in reversed(list(qargs))],
        )
        self.rho = rho_tensor.reshape((2**self.nqubit, 2**self.nqubit))

    @override
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        """Return the expectation value of single-qubit operator.

        Args:
            op (np.array): 2*2 Hermite operator
            loc (int): Index of qubit on which to apply operator.

        Returns
        -------
            complex: expectation value (real for hermitian ops!).
        """
        if not (0 <= loc < self.nqubit):
            raise ValueError(f"Wrong target qubit {loc}. Must between 0 and {self.nqubit - 1}.")

        if op.shape != (2, 2):
            raise ValueError("op must be 2x2 matrix.")

        st1 = copy.copy(self)
        st1.normalize()

        nqubit = self.nqubit
        rho_tensor: Matrix = st1.rho.reshape((2,) * nqubit * 2)
        rho_tensor = tensordot(op, rho_tensor, axes=([1], [loc]))
        rho_tensor = np.moveaxis(rho_tensor, 0, loc)

        # complex() needed with mypy strict mode (no-any-return)
        return complex(np.trace(rho_tensor.reshape((2**nqubit, 2**nqubit))))

    def dims(self) -> tuple[int, ...]:
        """Return the dimensions of the density matrix."""
        return self.rho.shape

    def tensor(self, other: DensityMatrix) -> None:
        r"""Tensor product state with other density matrix.

        Results in self :math:`\otimes` other.

        Parameters
        ----------
            other : :class: `DensityMatrix` object
                DensityMatrix object to be tensored with self.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        self.rho = kron(self.rho, other.rho)

    def cnot(self, edge: tuple[int, int]) -> None:
        """Apply CNOT gate to density matrix.

        Parameters
        ----------
            edge : (int, int) or [int, int]
                Edge to apply CNOT gate.
        """
        self.evolve(CNOT_TENSOR.reshape(4, 4), edge)

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """Swap qubits.

        Parameters
        ----------
            qubits : (int, int)
                (control, target) qubits indices.
        """
        self.evolve(SWAP_TENSOR.reshape(4, 4), qubits)

    def entangle(self, edge: tuple[int, int]) -> None:
        """Connect graph nodes.

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubit indices.
        """
        self.evolve(CZ_TENSOR.reshape(4, 4), edge)

    def normalize(self) -> None:
        """Normalize density matrix."""
        if self.rho.dtype == np.object_:
            rho_o = cast("npt.NDArray[np.object_]", self.rho)
            rho_o /= np.trace(rho_o)
        else:
            rho_c = cast("npt.NDArray[np.complex128]", self.rho)
            rho_c /= np.trace(rho_c)

    @override
    def remove_qubit(self, qarg: int) -> None:
        """Remove a qubit."""
        self.ptrace(qarg)
        self.normalize()

    def ptrace(self, qargs: Collection[int] | int) -> None:
        """Partial trace.

        Parameters
        ----------
            qargs : list of ints or int
                Indices of qubit to trace out.
        """
        n = int(np.log2(self.rho.shape[0]))
        if isinstance(qargs, int):
            qargs = [qargs]
        assert isinstance(qargs, (list, tuple))
        qargs_num = len(qargs)
        nqubit_after = n - qargs_num
        assert n > 0
        assert all(qarg >= 0 and qarg < n for qarg in qargs)

        rho_res = self.rho.reshape((2,) * n * 2)
        # ket, bra indices to trace out
        trace_axes = list(qargs) + [n + qarg for qarg in qargs]
        op: Matrix = np.eye(2**qargs_num).reshape((2,) * qargs_num * 2).astype(np.complex128)
        rho_res = tensordot(op, rho_res, axes=(list(range(2 * qargs_num)), trace_axes))

        self.rho = rho_res.reshape((2**nqubit_after, 2**nqubit_after))

    def fidelity(self, statevec: Statevec) -> ExpressionOrFloat:
        """Calculate the fidelity against reference statevector.

        Parameters
        ----------
            statevec : numpy array
                statevector (flattened numpy array) to compare with
        """
        result = vdot(statevec.psi, matmul(self.rho, statevec.psi))
        if isinstance(result, Expression):
            return result
        assert math.isclose(result.imag, 0)
        return result.real

    def flatten(self) -> Matrix:
        """Return flattened density matrix."""
        return self.rho.flatten()

    @override
    def apply_channel(self, channel: KrausChannel, qargs: Sequence[int]) -> None:
        """Apply a channel to a density matrix.

        Parameters
        ----------
        :rho: density matrix.
        channel: :class:`graphix.channel.KrausChannel` object
            KrausChannel to be applied to the density matrix
        qargs: target qubit indices

        Returns
        -------
        nothing

        Raises
        ------
        ValueError
            If the final density matrix is not normalized after application of the channel.
            This shouldn't happen since :class:`graphix.channel.KrausChannel` objects are normalized by construction.
        ....
        """
        result_array = np.zeros((2**self.nqubit, 2**self.nqubit), dtype=np.complex128)

        if not isinstance(channel, KrausChannel):
            raise TypeError("Can't apply a channel that is not a Channel object.")

        for k_op in channel:
            dm = copy.copy(self)
            dm.evolve(k_op.operator, qargs)
            result_array += k_op.coef * np.conj(k_op.coef) * dm.rho
            # reinitialize to input density matrix

        if not np.allclose(result_array.trace(), 1.0):
            raise ValueError("The output density matrix is not normalized, check the channel definition.")

        self.rho = result_array

    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> DensityMatrix:
        """Return a copy of the density matrix where all occurrences of the given variable in measurement angles are substituted by the given value."""
        result = copy.copy(self)
        result.rho = np.vectorize(lambda value: parameter.subs(value, variable, substitute))(self.rho)
        return result

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> DensityMatrix:
        """Return a copy of the density matrix where all occurrences of the given keys in measurement angles are substituted by the given values in parallel."""
        result = copy.copy(self)
        result.rho = np.vectorize(lambda value: parameter.xreplace(value, assignment))(self.rho)
        return result


@dataclass(frozen=True)
class DensityMatrixBackend(FullStateBackend[DensityMatrix]):
    """MBQC simulator with density matrix method."""

    state: DensityMatrix = dataclasses.field(init=False, default_factory=lambda: DensityMatrix(nqubit=0))
