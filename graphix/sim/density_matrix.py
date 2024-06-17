"""Density matrix simulator.

Simulate MBQC with density matrix representation.
"""

from __future__ import annotations

import collections
import numbers
import typing
from copy import deepcopy

import numpy as np

import graphix.sim.base_backend
import graphix.states
import graphix.types
from graphix.channels import KrausChannel
from graphix.clifford import CLIFFORD
from graphix.linalg_validations import check_psd, check_square, check_unit_trace
from graphix.ops import Ops
from graphix.sim.statevec import CNOT_TENSOR, CZ_TENSOR, SWAP_TENSOR, Statevec


class DensityMatrix:
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
                        self.Nqubit = self.rho.shape[0].bit_length() - 1
                        return
                except TypeError:
                    pass
        statevec = Statevec(data, nqubit)
        # NOTE this works since np.outer flattens the inputs!
        self.rho = np.outer(statevec.psi, statevec.psi.conj())
        self.Nqubit = len(statevec.dims())

    def __repr__(self):
        return f"DensityMatrix object, with density matrix {self.rho} and shape {self.dims()}."

    def evolve_single(self, op, i):
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
        self.rho = rho_tensor.reshape((2**self.Nqubit, 2**self.Nqubit))

    def evolve(self, op, qargs):
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

    def expectation_single(self, op, i):
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

        st1 = deepcopy(self)
        st1.normalize()

        rho_tensor = st1.rho.reshape((2,) * st1.Nqubit * 2)
        rho_tensor = np.tensordot(op, rho_tensor, axes=[1, i])
        rho_tensor = np.moveaxis(rho_tensor, 0, i)
        st1.rho = rho_tensor.reshape((2**self.Nqubit, 2**self.Nqubit))

        return np.trace(st1.rho)

    def dims(self):
        return self.rho.shape

    def tensor(self, other):
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
        self.Nqubit += other.Nqubit

    def cnot(self, edge):
        """Apply CNOT gate to density matrix.

        Parameters
        ----------
            edge : (int, int) or [int, int]
                Edge to apply CNOT gate.
        """

        self.evolve(CNOT_TENSOR.reshape(4, 4), edge)

    def swap(self, edge):
        """swap qubits

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubits indices.
        """

        self.evolve(SWAP_TENSOR.reshape(4, 4), edge)

    def entangle(self, edge):
        """connect graph nodes

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubit indices.
        """

        self.evolve(CZ_TENSOR.reshape(4, 4), edge)

    def normalize(self):
        """normalize density matrix"""
        self.rho /= np.trace(self.rho)

    def ptrace(self, qargs):
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
        self.Nqubit = nqubit_after

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
            tmp_dm.evolve(k_op["operator"], qargs)
            result_array += k_op["coef"] * np.conj(k_op["coef"]) * tmp_dm.rho
            # reinitialize to input density matrix
            tmp_dm = deepcopy(self)

        # Performance?
        self.rho = deepcopy(result_array)

        if not np.allclose(self.rho.trace(), 1.0):
            raise ValueError("The output density matrix is not normalized, check the channel definition.")


class DensityMatrixBackend(graphix.sim.base_backend.Backend):
    """MBQC simulator with density matrix method."""

    def __init__(self, pattern, max_qubit_num=12, pr_calc=True, input_state: Data = graphix.states.BasicStates.PLUS):
        """
        Parameters
        ----------
            pattern : :class:`graphix.pattern.Pattern` object
                Pattern to be simulated.
            max_qubit_num : int
                optional argument specifying the maximum number of qubits
                to be stored in the statevector at a time.
            pr_calc : bool
                whether or not to compute the probability distribution before choosing the measurement result.
                if False, measurements yield results 0/1 with 50% probabilities each.
            input_state: same syntax as `graphix.statevec.DensityMatrix` constructor.
        """
        self.pattern = pattern
        if pattern._pauli_preprocessed and input_state != graphix.states.BasicStates.PLUS:
            raise ValueError("Pauli preprocessing is currently only available when inputs are initialized in |+> state")
        self.results = deepcopy(pattern.results)
        self.state = None
        self.node_index = []
        self.Nqubit = 0
        self.max_qubit_num = max_qubit_num
        if pattern.max_space() > max_qubit_num:
            raise ValueError("Pattern.max_space is larger than max_qubit_num. Increase max_qubit_num and try again.")
        super().__init__(pr_calc)

        # initialize input qubits to desired init_state
        self.add_nodes(pattern.input_nodes, input_state)

    def add_nodes(self, nodes, input_state: Data = graphix.states.BasicStates.PLUS):
        """add new qubit to the internal density matrix
        and asign the corresponding node number to list self.node_index.

        Parameters
        ----------
        nodes : list
            list of node indices
        qubit_to_add : DensityMatrix object
            qubit to be added to the graph states
        """
        if not self.state:
            self.state = DensityMatrix(nqubit=0)
        n = len(nodes)
        dm_to_add = DensityMatrix(nqubit=n, data=input_state)
        self.state.tensor(dm_to_add)
        self.node_index.extend(nodes)
        self.Nqubit += n

    def entangle_nodes(self, edge):
        """Apply CZ gate to the two connected nodes.

        Parameters
        ----------
            edge : tuple (int, int)
                a pair of node indices
        """
        target = self.node_index.index(edge[0])
        control = self.node_index.index(edge[1])
        self.state.entangle((target, control))

    def measure(self, cmd):
        """Perform measurement on the specified node and trase out the qubit.

        Parameters
        ----------
            cmd : list
                measurement command : ['M', node, plane, angle, s_domain, t_domain]
        """
        loc = self._perform_measure(cmd)
        self.state.normalize()
        # perform ptrace right after the measurement (destructive measurement).
        self.state.ptrace(loc)

    def correct_byproduct(self, cmd):
        """Byproduct correction
        correct for the X or Z byproduct operators,
        by applying the X or Z gate.
        """
        if np.mod(np.sum([self.results[j] for j in cmd[2]]), 2) == 1:
            loc = self.node_index.index(cmd[1])
            if cmd[0] == "X":
                op = Ops.x
            elif cmd[0] == "Z":
                op = Ops.z
            self.state.evolve_single(op, loc)

    def sort_qubits(self):
        """sort the qubit order in internal density matrix"""
        for i, ind in enumerate(self.pattern.output_nodes):
            if not self.node_index[i] == ind:
                move_from = self.node_index.index(ind)
                self.state.swap((i, move_from))
                self.node_index[i], self.node_index[move_from] = (
                    self.node_index[move_from],
                    self.node_index[i],
                )

    def apply_clifford(self, cmd):
        """backend version of apply_channel
        Parameters
        ----------
            qargs : list of ints. Target qubits
        """
        loc = self.node_index.index(cmd[1])
        self.state.evolve_single(CLIFFORD[cmd[2]], loc)

    def apply_channel(self, channel: KrausChannel, qargs):
        """backend version of apply_channel
        Parameters
        ----------
            qargs : list of ints. Target qubits
        """

        indices = [self.node_index.index(i) for i in qargs]
        self.state.apply_channel(channel, indices)

    def finalize(self):
        """To be run at the end of pattern simulation."""
        self.sort_qubits()
        self.state.normalize()


## Python <3.10:
## TypeError: unsupported operand type(s) for |: 'ABCMeta' and 'type'
## TypeError: 'ABCMeta' object is not subscriptable
# Data = (
#    graphix.states.State
#    | DensityMatrix
#    | Statevec
#    | collections.abc.Iterable[graphix.states.State]
#    | collections.abc.Iterable[numbers.Number]
#    | collections.abc.Iterable[collections.abc.Iterable[numbers.Number]]
# )
Data = typing.Union[
    graphix.states.State,
    DensityMatrix,
    Statevec,
    typing.Iterable[graphix.states.State],
    typing.Iterable[numbers.Number],
    typing.Iterable[typing.Iterable[numbers.Number]],
]
