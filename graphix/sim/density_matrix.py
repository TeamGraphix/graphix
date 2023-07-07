"""Density matrix simulator.

Simulate MBQC with density matrix representation.
"""

from copy import deepcopy
import numpy as np
from graphix.ops import Ops
from graphix.sim.statevec import meas_op, CNOT_TENSOR, SWAP_TENSOR, CZ_TENSOR


class DensityMatrix:
    """DensityMatrix object."""

    def __init__(self, data=None, plus_state=True, nqubit=1):
        """
        Parameters
        ----------
            data : DensityMatrix, list, tuple, np.ndarray or None
                Density matrix.
            nqubit : int
                Number of qubits. Default is 1. If both `data` and `nqubit` are specified, `nqubit` is ignored.
        """
        if data is None:
            assert nqubit >= 0
            self.Nqubit = nqubit
            if plus_state:
                self.rho = np.ones(2 ** (2 * nqubit)).reshape(2**nqubit, 2**nqubit) / 2**nqubit
            else:
                self.rho = np.zeros(2 ** (2 * nqubit)).reshape(2**nqubit, 2**nqubit)
                self.rho[0, 0] = 1
        else:
            if isinstance(data, DensityMatrix):
                data = data.rho
            elif isinstance(data, (list, tuple)):
                data = np.asarray(data, dtype=complex)
            elif isinstance(data, np.ndarray):
                pass
            else:
                raise TypeError("data must be DensityMatrix, list, tuple, or np.ndarray.")
            nqubit = np.log2(len(data.flatten())) / 2
            assert np.isclose(nqubit, int(nqubit))
            nqubit = int(nqubit)
            self.Nqubit = nqubit

            # conversion done above
            self.rho = data.reshape((2**nqubit, 2**nqubit))
            np.testing.assert_almost_equal(self.rho.trace(), 1, err_msg='The data provided does not have unit trace!')

    def __repr__(self):
        return f"DensityMatrix, data={self.rho}, shape={self.dims()}"

    def evolve_single(self, op, i):
        """Single-qudit operation.

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
        i, j = edge
        n = int(np.log2(self.rho.shape[0]))
        assert i >= 0 and j >= 0
        assert n > i and n > j
        assert i != j

        rho_tensor = self.rho.reshape((2,) * n * 2)
        rho_tensor = np.tensordot(
            np.tensordot(CNOT_TENSOR, rho_tensor, axes=[(2, 3), edge]),
            CNOT_TENSOR.conj().T,
            axes=[(edge[1] + n, edge[0] + n), [0, 1]],
        )
        rho_tensor = np.moveaxis(rho_tensor, (0, 1, -1, -2), (edge[0], edge[1], edge[0] + n, edge[1] + n))
        self.rho = rho_tensor.reshape((2**n, 2**n))

    def swap(self, edge):
        """swap qubits

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubits indices.
        """
        i, j = edge
        n = int(np.log2(self.rho.shape[0]))
        assert i >= 0 and j >= 0
        assert n > i and n > j
        assert i != j

        rho_tensor = self.rho.reshape((2,) * n * 2)
        rho_tensor = np.tensordot(
            np.tensordot(SWAP_TENSOR, rho_tensor, axes=[(2, 3), edge]),
            SWAP_TENSOR.conj().T,
            axes=[(edge[1] + n, edge[0] + n), [0, 1]],
        )
        rho_tensor = np.moveaxis(rho_tensor, (0, 1, -1, -2), (edge[0], edge[1], edge[0] + n, edge[1] + n))
        self.rho = rho_tensor.reshape((2**n, 2**n))

    def entangle(self, edge):
        """connect graph nodes

        Parameters
        ----------
            edge : (int, int) or [int, int]
                (control, target) qubit indices.
        """
        i, j = edge
        n = int(np.log2(self.rho.shape[0]))
        assert i >= 0 and j >= 0
        assert n > i and n > j
        assert i != j

        rho_tensor = self.rho.reshape((2,) * n * 2)
        rho_tensor = np.tensordot(
            np.tensordot(CZ_TENSOR, rho_tensor, axes=[(2, 3), edge]),
            CZ_TENSOR.conj().T,
            axes=[(edge[1] + n, edge[0] + n), [0, 1]],
        )
        rho_tensor = np.moveaxis(rho_tensor, (0, 1, -1, -2), (edge[0], edge[1], edge[0] + n, edge[1] + n))
        self.rho = rho_tensor.reshape((2**n, 2**n))

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

        self.rho = rho_res.reshape((2**nqubit_after, 2**nqubit_after)) / np.linalg.norm(rho_res)
        self.Nqubit = nqubit_after

    def fidelity(self, statevec):
        """calculate the fidelity against reference statevector.

        Parameters
        ----------
            statevec : numpy array
                statevector (flattened numpy array) to compare with
        """
        return np.abs(statevec.conj() @ self.rho @ statevec)


class DensityMatrixBackend:
    """MBQC simulator with density matrix method."""

    def __init__(self, pattern, max_qubit_num=12):
        """
        Parameters
        ----------
            pattern : :class:`graphix.pattern.Pattern` object
                Pattern to be simulated.
            max_qubit_num : int
                optional argument specifying the maximum number of qubits
                to be stored in the statevector at a time.
        """
        # check that pattern has output nodes configured
        assert len(pattern.output_nodes) > 0
        self.pattern = pattern
        self.results = deepcopy(pattern.results)
        self.state = None
        self.node_index = []
        self.Nqubit = 0
        self.max_qubit_num = max_qubit_num
        if pattern.max_space() > max_qubit_num:
            raise ValueError("Pattern.max_space is larger than max_qubit_num. Increase max_qubit_num and try again.")

    def dephase(self, p=0):
        """Apply dephasing channel to all nodes. Phase is flipped with probability p.

        :math:`(1-p) \rho + p Z \rho Z`

        Parameters
        ----------
            p : float
                dephase probability
        """
        n = int(np.log2(self.state.rho.shape[0]))
        Z = np.array([[1, 0], [0, -1]])
        rho_tensor = self.state.rho.reshape((2,) * n * 2)
        for node in range(n):
            dephase_part = np.tensordot(np.tensordot(Z, rho_tensor, axes=(1, node)), Z, axes=(node + n, 0))
            dephase_part = np.moveaxis(dephase_part, (0, -1), (node, node + n))
            rho_tensor = (1 - p) * rho_tensor + p * dephase_part
        self.state.rho = rho_tensor.reshape((2**n, 2**n))

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
        result = np.random.choice([0, 1])
        self.results[cmd[1]] = result

        s_signal = np.sum([self.results[j] for j in cmd[4]])
        t_signal = np.sum([self.results[j] for j in cmd[5]])
        angle = cmd[3] * np.pi * (-1) ** s_signal + np.pi * t_signal
        if len(cmd) == 7:
            m_op = meas_op(angle, vop=cmd[6], plane=cmd[2], choice=result)
        else:
            m_op = meas_op(angle, plane=cmd[2], choice=result)
        loc = self.node_index.index(cmd[1])
        self.state.evolve_single(m_op, loc)
        self.state.normalize()
        # perform ptrace right after measurement as in real devices
        self.state.ptrace(loc)
        self.node_index.remove(cmd[1])

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

    def finalize(self):
        """To be run at the end of pattern simulation."""
        self.sort_qubits()
        self.state.normalize()
