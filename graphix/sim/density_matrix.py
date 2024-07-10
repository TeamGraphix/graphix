"""Density matrix simulator.

Simulate MBQC with density matrix representation.
"""

from __future__ import annotations

import collections
import numbers
import sys
from copy import deepcopy

import numpy as np

import graphix.states
import graphix.types
from graphix.channels import KrausChannel
from graphix.linalg_validations import check_psd, check_square, check_unit_trace
from graphix.sim.base_backend import Backend, State
from graphix.sim.statevec import Statevec


class DensityMatrix(State):
    """DensityMatrix object."""

    def __init__(
        self,
        data: Data = graphix.states.BasicStates.PLUS,
        nqubit: graphix.types.PositiveOrNullInt | None = None,
    ):
        """Initialize density matrix objects. The behaviour builds on theo ne of `graphix.statevec.Statevec`.
        `data` can be:
        - a single :class:`graphix.states.State` (classical description of a quantum state)
        - an iterable of :class:`graphix.states.State` objects
        - an iterable of iterable of scalars (A 2**n x 2**n numerical density matrix)
        - a `graphix.statevec.DensityMatrix` object
        - a `graphix.statevec.Statevector` object

        If `nqubit` is not provided, the number of qubit is inferred from `data` and checked for consistency.
        If only one :class:`graphix.states.State` is provided and nqubit is a valid integer, initialize the statevector
        in the tensor product state.
        If both `nqubit` and `data` are provided, consistency of the dimensions is checked.
        If a `graphix.statevec.Statevec` or `graphix.statevec.DensityMatrix` is passed, returns a copy.


        :param data: input data to prepare the state. Can be a classical description or a numerical input, defaults to graphix.states.BasicStates.PLUS
        :type data: graphix.states.State | "DensityMatrix" | Statevec | collections.abc.Iterable[graphix.states.State] |collections.abc.Iterable[numbers.Number] | collections.abc.Iterable[collections.abc.Iterable[numbers.Number]], optional
        :param nqubit: number of qubits to prepare, defaults to None
        :type nqubit: int, optional
        """
        assert nqubit is None or isinstance(nqubit, numbers.Integral) and nqubit >= 0

        def check_size_consistency(mat):
            if nqubit is not None and mat.shape != (2**nqubit, 2**nqubit):
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the shape of the provided density matrix = {mat.shape}."
                )

        if isinstance(data, DensityMatrix):
            check_size_consistency(data)
            # safe: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html
            self.rho = data.rho.copy()
            return
        if isinstance(data, collections.abc.Iterable):
            input_list = list(data)
            if len(input_list) != 0:
                # needed since Object is iterable but not subscribable!
                try:
                    if isinstance(input_list[0], collections.abc.Iterable) and isinstance(
                        input_list[0][0], numbers.Number
                    ):
                        self.rho = np.array(input_list)
                        assert check_square(self.rho)
                        check_size_consistency(self.rho)
                        assert check_unit_trace(self.rho)
                        assert check_psd(self.rho)
                        return
                except TypeError:
                    pass
        statevec = Statevec(data, nqubit)
        # NOTE this works since np.outer flattens the inputs!
        self.rho = np.outer(statevec.psi, statevec.psi.conj())

    @property
    def Nqubit(self) -> int:
        return self.rho.shape[0].bit_length() - 1

    def copy(self) -> DensityMatrix:
        result = self.__new__(self.__class__)
        result.rho = self.rho
        return result

    def __repr__(self):
        return f"DensityMatrix object, with density matrix {self.rho} and shape{self.dims()}."

    def add_nodes(self, nqubit, data) -> None:
        dm_to_add = DensityMatrix(nqubit=nqubit, data=data)
        self.tensor(dm_to_add)

    def evolve_single(self, op, i) -> None:
        """Single-qubit operation.

        Parameters
        ----------
            op : np.ndarray
                2*2 matrix.
            i : int
                Index of qubit to apply operator.
        """
        assert i >= 0 and i < self.Nqubit
        if op.shape != (2, 2):
            raise ValueError("op must be 2*2 matrix.")

        rho_tensor = self.rho.reshape((2,) * self.Nqubit * 2)
        rho_tensor = np.tensordot(np.tensordot(op, rho_tensor, axes=(1, i)), op.conj().T, axes=(i + self.Nqubit, 0))
        rho_tensor = np.moveaxis(rho_tensor, (0, -1), (i, i + self.Nqubit))
        self.rho = rho_tensor.reshape((2**self.Nqubit, 2**self.Nqubit))

    def evolve(self, op, qargs) -> None:
        """Multi-qubit operation

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

        if not all(0 <= i < self.Nqubit for i in qargs):
            raise ValueError("Incorrect target indices.")
        if len(set(qargs)) != nqb_op:
            raise ValueError("A repeated target qubit index is not possible.")

        op_tensor = op.reshape((2,) * 2 * nqb_op)

        rho_tensor = self.rho.reshape((2,) * self.Nqubit * 2)

        rho_tensor = np.tensordot(
            np.tensordot(op_tensor, rho_tensor, axes=[tuple(nqb_op + i for i in range(len(qargs))), tuple(qargs)]),
            op.conj().T.reshape((2,) * 2 * nqb_op),
            axes=[tuple(i + self.Nqubit for i in qargs), tuple(i for i in range(len(qargs)))],
        )
        rho_tensor = np.moveaxis(
            rho_tensor,
            [i for i in range(len(qargs))] + [-i for i in range(1, len(qargs) + 1)],
            [i for i in qargs] + [i + self.Nqubit for i in reversed(list(qargs))],
        )
        self.rho = rho_tensor.reshape((2**self.Nqubit, 2**self.Nqubit))

    def expectation_single(self, op, i) -> complex:
        """Expectation value of single-qubit operator.

        Args:
            op (np.array): 2*2 Hermite operator
            loc (int): Index of qubit on which to apply operator.
        Returns:
            complex: expectation value (real for hermitian ops!).
        """

        if not (0 <= i < self.Nqubit):
            raise ValueError(f"Wrong target qubit {i}. Must between 0 and {self.Nqubit-1}.")

        if op.shape != (2, 2):
            raise ValueError("op must be 2x2 matrix.")

        st1 = self.copy()
        st1.normalize()

        rho_tensor = st1.rho.reshape((2,) * st1.Nqubit * 2)
        rho_tensor = np.tensordot(op, rho_tensor, axes=[1, i])
        rho_tensor = np.moveaxis(rho_tensor, 0, i)

        return np.trace(rho_tensor.reshape((2**self.Nqubit, 2**self.Nqubit)))

    def dims(self):
        return self.rho.shape

    def tensor(self, other) -> None:
        r"""Tensor product state with other density matrix.
        Results in self :math:`\otimes` other.

        Parameters
        ----------
            other : :class: `DensityMatrix` object
                DensityMatrix object to be tensored with self.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        self.rho = np.kron(self.rho, other.rho)

    def cnot(self, edge) -> None:
        """Apply CNOT gate to density matrix.

        Parameters
        ----------
            edge : (int, int) or [int, int]
                Edge to apply CNOT gate.
        """

        self.evolve(graphix.sim.statevec.CNOT_TENSOR.reshape(4, 4), edge)

    def swap(self, edge) -> None:
        """swap qubits

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubits indices.
        """

        self.evolve(graphix.sim.statevec.SWAP_TENSOR.reshape(4, 4), edge)

    def entangle(self, edge) -> None:
        """connect graph nodes

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubit indices.
        """

        self.evolve(graphix.sim.statevec.CZ_TENSOR.reshape(4, 4), edge)

    def normalize(self) -> None:
        """normalize density matrix"""
        self.rho = self.rho / np.trace(self.rho)

    def remove_qubit(self, loc) -> None:
        self.ptrace(loc)
        self.normalize()

    def ptrace(self, qargs) -> None:
        """partial trace

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
        assert all([qarg >= 0 and qarg < n for qarg in qargs])

        rho_res = self.rho.reshape((2,) * n * 2)
        # ket, bra indices to trace out
        trace_axes = list(qargs) + [n + qarg for qarg in qargs]
        rho_res = np.tensordot(
            np.eye(2**qargs_num).reshape((2,) * qargs_num * 2), rho_res, axes=(list(range(2 * qargs_num)), trace_axes)
        )

        self.rho = rho_res.reshape((2**nqubit_after, 2**nqubit_after))

    def fidelity(self, statevec):
        """calculate the fidelity against reference statevector.

        Parameters
        ----------
            statevec : numpy array
                statevector (flattened numpy array) to compare with
        """
        return np.abs(statevec.transpose().conj() @ self.rho @ statevec)

    def apply_channel(self, channel: KrausChannel, qargs) -> None:
        """Applies a channel to a density matrix.

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

        result_array = np.zeros((2**self.Nqubit, 2**self.Nqubit), dtype=np.complex128)
        tmp_dm = deepcopy(self)

        if not isinstance(channel, KrausChannel):
            raise TypeError("Can't apply a channel that is not a Channel object.")

        for k_op in channel.kraus_ops:
            dm = self.copy()
            dm.evolve(k_op["operator"], qargs)
            result_array += k_op["coef"] * np.conj(k_op["coef"]) * dm.rho
            # reinitialize to input density matrix

        if not np.allclose(result_array.trace(), 1.0):
            raise ValueError("The output density matrix is not normalized, check the channel definition.")

        self.rho = result_array


class DensityMatrixBackend(Backend):
    """MBQC simulator with density matrix method."""

    def __init__(self, pr_calc=True):
        super().__init__(DensityMatrix(nqubit=0), pr_calc=pr_calc)

    def apply_channel(self, channel: KrausChannel, qargs) -> DensityMatrixBackend:
        """backend version of apply_channel
        Parameters
        ----------
            qargs : list of ints. Target qubits
        """

        indices = [self.node_index[i] for i in qargs]
        new_dm = self.state.apply_channel(channel, indices)
        return self.with_changes(state=new_dm)


if sys.version_info >= (3, 10):
    from collections.abc import Iterable

    Data = (
        graphix.states.State
        | DensityMatrix
        | Statevec
        | Iterable[graphix.states.State]
        | Iterable[numbers.Number]
        | Iterable[Iterable[numbers.Number]]
    )
else:
    from typing import Iterable, Union

    Data = Union[
        graphix.states.State,
        DensityMatrix,
        Statevec,
        Iterable[graphix.states.State],
        Iterable[numbers.Number],
        Iterable[Iterable[numbers.Number]],
    ]
