"""Density matrix simulator.

Simulate MBQC with density matrix representation.
"""

from copy import deepcopy

import numpy as np

from graphix.linalg_validations import check_square, check_hermitian, check_unit_trace
from graphix.channels import KrausChannel
from graphix.ops import Ops
from graphix.clifford import CLIFFORD
from graphix.sim.statevec import CNOT_TENSOR, CZ_TENSOR, SWAP_TENSOR, meas_op, Statevec
import graphix.sim.base_backend
import graphix.states

import typing
from typing_extensions import Annotated
from annotated_types import Ge
import functools


PositiveInt = Annotated[int, Ge(0)]  # includes 0


class DensityMatrix:
    """DensityMatrix object."""

    def __init__(
        self,
        nqubit: typing.Optional[PositiveInt] = None,
        state: typing.Union[
            graphix.states.State,
            "DensityMatrix",
            Statevec,
            typing.Iterable[graphix.states.State],
            typing.Iterable[complex],
        ] = graphix.states.BasicStates.PLUS,
    ):

        """
        rewrite!
        Parameters
        ----------
            data : DensityMatrix, list, tuple, np.ndarray or None
                Density matrix of shape (2**nqubits, 2**nqubits).
            nqubit : int
                Number of qubits. Default is 1. If both `data` and `nqubit` are specified, `nqubit` is ignored.
        """
        if nqubit == 0:
            self.rho = np.array(1, dtype=np.complex128)

        elif isinstance(state, graphix.states.State):

            if nqubit is None:
                raise ValueError("Incorrect value for nqubit.")

            # or directly get_dm from get_statevec? Can make it inherit from abc?
            vec = state.get_statevector()
            dm = np.outer(vec, vec.conj())
            # these vecs are normalized
            if nqubit == 1:  # or None
                self.rho = dm

            # build tensor product |state>^{\otimes nqubit}
            # can only be >1 int.
            else:
                # build tensor product
                # comma in tuple is for disambiguation with paranthesed expression
                self.rho = functools.reduce(np.kron, (dm,) * nqubit)
                # no reshape

        # nqubit is None : on prend la longeur de l'iterable
        elif isinstance(state, typing.Iterable):
            # iterateur
            it = iter(state)
            head = next(it)
            # type constraint in head doesn't progpagate to all elts
            if isinstance(head, graphix.states.State):
                # assert isinstance(head, typing.Iterator[graphix.states.State])
                if nqubit is None:
                    # liste persistante state pour eviter la transience
                    states = [head] + list(it)
                    nqubit = len(states)
                    # self.Nqubit = nqubit
                # else take nqubit elts
                else:  # ignore for now
                    states = [head] + [next(it) for _ in range(nqubit - 1)]

                    # self.Nqubit = nqubit

                list_of_sv = [s.get_statevector() for s in states]
                list_of_dm = [np.outer(sv, sv.conj()) for sv in list_of_sv]
                self.rho = functools.reduce(np.kron, list_of_dm)
                # no reshape
                # self.psi = tmp_psi.reshape((2,) * nqubit)

            else:  # now 2**n by 2**n matrices also iterable?
                if nqubit is None:
                    # BUG what to do with that?
                    states = [head] + list(it)
                    # need a shape so just np.ndarray?
                    inferred_shape = state.shape

                    # this is done in check_square for matrix do it on shape
                    if len(inferred_shape) != 2:
                        raise ValueError(f"The object has {len(inferred_shape)} axes but must have 2 to be a matrix.")
                    if inferred_shape[0] != inferred_shape[1]:
                        raise ValueError(f"Matrix must be square but has different dimensions {inferred_shape}.")
                    inferred_size = inferred_shape[0]
                    if inferred_size & (inferred_size - 1) != 0:
                        raise ValueError(f"Matrix size must be a power of two but is {inferred_size}.")

                    nqubit = inferred_size.bit_length() - 1

                else:  # ignore for now
                    # BUG what to do with that?
                    states = state[: 2**nqubit, : 2**nqubit].copy()
                    # [head] + [next(it) for _ in range(2**nqubit - 1)]

                # psi = np.array(states)

                dm = np.array([np.outer(s, s.conj()) for s in states])

                # hermicity and trace checked later. Or do it from statevec?
                # from pure states, should be ok by default so just when input data?
                assert check_hermitian(self.rho)
                assert check_unit_trace(self.rho)

            # TODO now: statevec and densitymatrix
        # if already a valid statevec transform it to DensityMatrix.
        if isinstance(state, Statevec):
            if nqubit is not None or len(state.flatten()) != 2**nqubit:
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the size of the provided statevector = {len(state.flatten())}."
                )
            vect = state.psi.copy()
            self.rho = np.outer(vect, vect.conj())

        # if DensityMatrix, just copy it
        elif isinstance(state, DensityMatrix):
            if nqubit is not None or state.dims() != (2**nqubit, 2**nqubit):
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the shape of the provided density matrix = {state.dims()}."
                )

            self.rho = state.rho.copy()

    def __repr__(self):
        return f"DensityMatrix object , with density matrix {self.rho} and shape{self.dims()}."

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

    def __init__(self, pattern, max_qubit_num=12, pr_calc=True):
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
        """
        # check that pattern has output nodes configured
        # assert len(pattern.output_nodes) > 0
        self.pattern = pattern
        self.results = deepcopy(pattern.results)
        self.state = None
        self.node_index = []
        self.Nqubit = 0
        self.max_qubit_num = max_qubit_num
        if pattern.max_space() > max_qubit_num:
            raise ValueError("Pattern.max_space is larger than max_qubit_num. Increase max_qubit_num and try again.")
        super().__init__(pr_calc)

    def add_nodes(self, nodes, qubit_to_add=None):
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
        if qubit_to_add is None:
            dm_to_add = DensityMatrix(nqubit=n)
        else:
            assert isinstance(qubit_to_add, DensityMatrix)
            assert qubit_to_add.nqubit == 1
            dm_to_add = qubit_to_add
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
