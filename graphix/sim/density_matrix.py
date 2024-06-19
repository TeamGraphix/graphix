"""Density matrix simulator.

Simulate MBQC with density matrix representation.
"""

from __future__ import annotations

import collections
import functools
import numbers
import typing
from copy import deepcopy

import numpy as np
import pydantic

import graphix.sim.base_backend
import graphix.sim.statevec
import graphix.states
import graphix.types
from graphix.channels import KrausChannel
from graphix.clifford import CLIFFORD
from graphix.linalg_validations import check_hermitian, check_psd, check_square, check_unit_trace
from graphix.ops import Ops
from graphix.sim.base_backend import IndexedState, NodeIndex
from graphix.sim.statevec import Statevec

Data = typing.Union[
    graphix.states.State,
    "DensityMatrix",
    graphix.sim.statevec.Statevec,
    typing.Iterable[graphix.states.State],
    typing.Iterable[numbers.Number],
    typing.Iterable[typing.Iterable[numbers.Number]],
]


class DensityMatrix:
    """DensityMatrix object."""

    def __init__(
        self,
        data: Data = graphix.states.BasicStates.PLUS,
        nqubit: typing.Optional[graphix.types.PositiveInt] = None,
    ):
        """
        rewrite!
        Parameters
        ----------
            data : DensityMatrix, list, tuple, np.ndarray or None
                Density matrix of shape (2**nqubits, 2**nqubits).
            nqubit : int
                Number of qubits. Default is 1. If both `data` and `nqubit` are specified, consistency is checked.
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
            self.Nqubit = data.Nqubit
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

    @classmethod
    def __from_nparray(cls, rho) -> DensityMatrix:
        result = cls.__new__(cls)
        result.rho = rho
        return result

    def __repr__(self):
        return f"DensityMatrix object , with density matrix {self.rho} and shape{self.dims()}."

    def evolve_single(self, op, i) -> DensityMatrix:
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
        rho_tensor = np.tensordot(np.tensordot(op, rho_tensor, axes=[1, i]), op.conj().T, axes=[i + self.Nqubit, 0])
        rho_tensor = np.moveaxis(rho_tensor, (0, -1), (i, i + self.Nqubit))
        rho = rho_tensor.reshape((2**self.Nqubit, 2**self.Nqubit))
        return DensityMatrix.__from_nparray(rho)

    def evolve(self, op, qargs) -> DensityMatrix:
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
        rho = rho_tensor.reshape((2**self.Nqubit, 2**self.Nqubit))
        return DensityMatrix.__from_nparray(rho)

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

        st1 = self.normalize()

        rho_tensor = st1.rho.reshape((2,) * st1.Nqubit * 2)
        rho_tensor = np.tensordot(op, rho_tensor, axes=[1, i])
        rho_tensor = np.moveaxis(rho_tensor, 0, i)

        return np.trace(rho_tensor.reshape((2**self.Nqubit, 2**self.Nqubit)))

    def dims(self):
        return self.rho.shape

    def tensor(self, other) -> DensityMatrix:
        r"""Tensor product state with other density matrix.
        Results in self :math:`\otimes` other.

        Parameters
        ----------
            other : :class: `DensityMatrix` object
                DensityMatrix object to be tensored with self.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        rho = np.kron(self.rho, other.rho)
        return DensityMatrix.__from_nparray(rho)

    def cnot(self, edge) -> DensityMatrix:
        """Apply CNOT gate to density matrix.

        Parameters
        ----------
            edge : (int, int) or [int, int]
                Edge to apply CNOT gate.
        """

        return self.evolve(graphix.sim.statevec.CNOT_TENSOR.reshape(4, 4), edge)

    def swap(self, edge) -> DensityMatrix:
        """swap qubits

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubits indices.
        """

        return self.evolve(graphix.sim.statevec.SWAP_TENSOR.reshape(4, 4), edge)

    def entangle(self, edge) -> DensityMatrix:
        """connect graph nodes

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubit indices.
        """

        return self.evolve(graphix.sim.statevec.CZ_TENSOR.reshape(4, 4), edge)

    def normalize(self) -> DensityMatrix:
        """normalize density matrix"""
        rho = self.rho / np.trace(self.rho)
        return DensityMatrix.__from_nparray(rho)

    def ptrace(self, qargs) -> DensityMatrix:
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

        rho = rho_res.reshape((2**nqubit_after, 2**nqubit_after))
        return DensityMatrix.__from_nparray(rho)

    def fidelity(self, statevec):
        """calculate the fidelity against reference statevector.

        Parameters
        ----------
            statevec : numpy array
                statevector (flattened numpy array) to compare with
        """
        return np.abs(statevec.transpose().conj() @ self.rho @ statevec)

    def apply_channel(self, channel: KrausChannel, qargs):
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
            dm = self.evolve(k_op["operator"], qargs)
            result_array += k_op["coef"] * np.conj(k_op["coef"]) * dm.rho
            # reinitialize to input density matrix

        if not np.allclose(result_array.trace(), 1.0):
            raise ValueError("The output density matrix is not normalized, check the channel definition.")

        return DensityMatrix.__from_nparray(result_array)


class DensityMatrixBackend(graphix.sim.base_backend.Backend):
    """MBQC simulator with density matrix method."""

    def __init__(self, max_qubit_num=12, pr_calc=True, input_state: Data = graphix.states.BasicStates.PLUS):
        """
        Parameters
        ----------
            max_qubit_num : int
                optional argument specifying the maximum number of qubits
                to be stored in the statevector at a time.
            pr_calc : bool
                whether or not to compute the probability distribution before choosing the measurement result.
                if False, measurements yield results 0/1 with 50% probabilities each.
        """
        # check that pattern has output nodes configured
        self.max_qubit_num = max_qubit_num
        self.results = {}
        super().__init__(pr_calc)

    def initial_state(self) -> IndexedState:
        return IndexedState(DensityMatrix(nqubit=0), NodeIndex())

    def add_nodes(self, state, nodes, data: Data = graphix.states.BasicStates.PLUS) -> IndexedState:
        """add new qubit to the internal density matrix
        and asign the corresponding node number to list node_index.

        Parameters
        ----------
        nodes : list
            list of node indices
        qubit_to_add : DensityMatrix object
            qubit to be added to the graph states
        """
        n = len(nodes)
        dm_to_add = DensityMatrix(nqubit=n, data=data)
        new_dm = state.state.tensor(dm_to_add)
        new_node_index = state.node_index.extend(nodes)
        return IndexedState(new_dm, new_node_index)

    def entangle_nodes(self, state, edge):
        """Apply CZ gate to the two connected nodes.

        Parameters
        ----------
            edge : tuple (int, int)
                a pair of node indices
        """
        target = state.node_index[edge[0]]
        control = state.node_index[edge[1]]
        new_dm = state.state.entangle((target, control))
        return IndexedState(new_dm, state.node_index)

    def measure(self, state, node, measurement_description):
        """Perform measurement on the specified node and trase out the qubit.

        Parameters
        ----------
            cmd : list
                measurement command : ['M', node, plane, angle, s_domain, t_domain]
        """
        state, loc, result = self._perform_measure(
            state=state, node=node, measurement_description=measurement_description
        )
        # perform ptrace right after the measurement (destructive measurement).
        new_dm = state.state.normalize().ptrace(loc)
        return IndexedState(new_dm, state.node_index), result

    def correct_byproduct(self, state, results, cmd):
        """Byproduct correction
        correct for the X or Z byproduct operators,
        by applying the X or Z gate.
        """
        if np.mod(np.sum([results[j] for j in cmd[2]]), 2) == 1:
            loc = state.node_index[cmd[1]]
            if cmd[0] == "X":
                op = Ops.x
            elif cmd[0] == "Z":
                op = Ops.z
            return IndexedState(state.state.evolve_single(op, loc), state.node_index)
        return state

    def apply_clifford(self, state, cmd):
        """Apply single-qubit Clifford gate,
        specified by vop index specified in graphix.clifford.CLIFFORD
        """
        loc = node_index[cmd[1]]
        new_dm = state.state.evolve_single(CLIFFORD[cmd[2]], loc)
        return IndexedState(new_dm, state.node_index)

    def apply_channel(self, state, channel: KrausChannel, qargs):
        """backend version of apply_channel
        Parameters
        ----------
            qargs : list of ints. Target qubits
        """

        indices = [state.node_index[i] for i in qargs]
        new_dm = state.state.apply_channel(channel, indices)
        return IndexedState(new_dm, state.node_index)
