from copy import deepcopy

import numpy as np

from graphix.clifford import CLIFFORD, CLIFFORD_CONJ, CLIFFORD_MUL
from graphix.ops import Ops


class StatevectorBackend:
    """MBQC simulator with statevector method."""

    def __init__(self, pattern, max_qubit_num=20):
        """
        Parameters
        -----------
        pattern : :class:`graphix.pattern.Pattern` object
            MBQC pattern to be simulated.
        backend : str, 'statevector'
            optional argument for simulation.
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
        self.to_trace = []
        self.to_trace_loc = []
        self.max_qubit_num = max_qubit_num
        if pattern.max_space() > max_qubit_num:
            raise ValueError("Pattern.max_space is larger than max_qubit_num. Increase max_qubit_num and try again")

    def qubit_dim(self):
        """Returns the qubit number in the internal statevector

        Returns
        -------
        n_qubit : int
        """
        return len(self.state.dims())

    def add_nodes(self, nodes):
        """add new qubit to internal statevector
        and assign the corresponding node number
        to list self.node_index.

        Parameters
        ----------
        nodes : list of node indices
        """
        if not self.state:
            self.state = Statevec(nqubit=0)
        n = len(nodes)
        sv_to_add = Statevec(nqubit=n)
        self.state.tensor(sv_to_add)
        self.node_index.extend(nodes)
        self.Nqubit += n

    def entangle_nodes(self, edge):
        """Apply CZ gate to two connected nodes

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        target = self.node_index.index(edge[0])
        control = self.node_index.index(edge[1])
        self.state.entangle((target, control))

    def measure(self, cmd):
        """Perform measurement of a node in the internal statevector and trace out the qubit

        Parameters
        ----------
        cmd : list
            measurement command : ['M', node, plane angle, s_domain, t_domain]
        """
        # choose the measurement result randomly
        result = np.random.choice([0, 1])
        self.results[cmd[1]] = result

        # extract signals for adaptive angle
        s_signal = np.sum([self.results[j] for j in cmd[4]])
        t_signal = np.sum([self.results[j] for j in cmd[5]])
        angle = cmd[3] * np.pi
        if len(cmd) == 7:
            vop = cmd[6]
        else:
            vop = 0
        if int(s_signal % 2) == 1:
            vop = CLIFFORD_MUL[1, vop]
        if int(t_signal % 2) == 1:
            vop = CLIFFORD_MUL[3, vop]
        m_op = meas_op(angle, vop=vop, plane=cmd[2], choice=result)
        loc = self.node_index.index(cmd[1])
        self.state.evolve_single(m_op, loc)

        self.state.remove_qubit(loc)
        self.node_index.remove(cmd[1])
        self.Nqubit -= 1

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

    def apply_clifford(self, cmd):
        """Apply single-qubit Clifford gate,
        specified by vop index specified in graphix.clifford.CLIFFORD
        """
        loc = self.node_index.index(cmd[1])
        self.state.evolve_single(CLIFFORD[cmd[2]], loc)

    def finalize(self):
        """to be run at the end of pattern simulation."""
        self.sort_qubits()
        self.state.normalize()

    def sort_qubits(self):
        """sort the qubit order in internal statevector"""
        for i, ind in enumerate(self.pattern.output_nodes):
            if not self.node_index[i] == ind:
                move_from = self.node_index.index(ind)
                self.state.swap((i, move_from))
                self.node_index[i], self.node_index[move_from] = (
                    self.node_index[move_from],
                    self.node_index[i],
                )


def meas_op(angle, vop=0, plane="XY", choice=0):
    """Returns the projection operator for given measurement angle and local Clifford op (VOP).

    .. seealso:: :mod:`graphix.clifford`

    Parameters
    ----------
    angle : float
        original measurement angle in radian
    vop : int
        index of local Clifford (vop), see graphq.clifford.CLIFFORD
    plane : 'XY', 'YZ' or 'ZX'
        measurement plane on which angle shall be defined
    choice : 0 or 1
        choice of measurement outcome. measured eigenvalue would be (-1)**choice.

    Returns
    -------
    op : numpy array
        projection operator

    """
    assert vop in np.arange(24)
    assert choice in [0, 1]
    assert plane in ["XY", "YZ", "XZ"]
    if plane == "XY":
        vec = (np.cos(angle), np.sin(angle), 0)
    elif plane == "YZ":
        vec = (0, np.cos(angle), np.sin(angle))
    elif plane == "XZ":
        vec = (np.cos(angle), 0, np.sin(angle))
    op_mat = np.eye(2, dtype=np.complex128) / 2
    for i in range(3):
        op_mat += (-1) ** (choice) * vec[i] * CLIFFORD[i + 1] / 2
    op_mat = CLIFFORD[CLIFFORD_CONJ[vop]] @ op_mat @ CLIFFORD[vop]
    return op_mat


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


class Statevec:
    """Simple statevector simulator"""

    def __init__(self, nqubit=1, plus_states=True):
        """Initialize statevector

        Parameters
        ----------
        nqubit : int, optional:
            number of qubits. Defaults to 1.
        plus_states : bool, optional
            whether or not to start all qubits in + state or 0 state. Defaults to +
        """
        if plus_states:
            self.psi = np.ones((2,) * nqubit) / 2 ** (nqubit / 2)
        else:
            self.psi = np.zeros((2,) * nqubit)
            self.psi[(0,) * nqubit] = 1

    def __repr__(self):
        return f"Statevec, data={self.psi}, shape={self.dims()}"

    def evolve_single(self, op, i):
        """Single-qubit operation

        Parameters
        ----------
        op : numpy.ndarray
            2*2 matrix
        i : int
            qubit index
        """
        self.psi = np.tensordot(op, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)

    def evolve(self, op, qargs):
        """Multi-qubit operation

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n matrix
        qargs : list of int
            target qubits' indices
        """
        op_dim = int(np.log2(len(op)))
        shape = [2 for _ in range(2 * op_dim)]
        op_tensor = op.reshape(shape)
        self.psi = np.tensordot(
            op_tensor,
            self.psi,
            (tuple(op_dim + i for i in range(len(qargs))), tuple(qargs)),
        )
        self.psi = np.moveaxis(self.psi, [i for i in range(len(qargs))], qargs)

    def dims(self):
        return self.psi.shape

    def ptrace(self, qargs):
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
        self.psi = np.reshape(evecs[:, np.argmax(evals)], (2,) * nqubit_after)

    def remove_qubit(self, qarg):
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
        assert not np.isclose(_get_statevec_norm(self.psi), 0)
        psi = self.psi.take(indices=0, axis=qarg)
        self.psi = psi if not np.isclose(_get_statevec_norm(psi), 0) else self.psi.take(indices=1, axis=qarg)
        self.normalize()

    def entangle(self, edge):
        """connect graph nodes

        Parameters
        ----------
        edge : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = np.tensordot(CZ_TENSOR, self.psi, ((2, 3), edge))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), edge)

    def tensor(self, other):
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

    def CNOT(self, qubits):
        """apply CNOT

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = np.tensordot(CNOT_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), qubits)

    def swap(self, qubits):
        """swap qubits

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = np.tensordot(SWAP_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), qubits)

    def normalize(self):
        """normalize the state"""
        norm = _get_statevec_norm(self.psi)
        self.psi = self.psi / norm

    def flatten(self):
        """returns flattened statevector"""
        return self.psi.flatten()

    def expectation_single(self, op, loc):
        """Expectation value of single-qubit operator.

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
        st1 = deepcopy(self)
        st1.normalize()
        st2 = st1.deepcopy(st1)
        st1.evolve_single(op, loc)
        return np.dot(st2.psi.flatten().conjugate(), st1.psi.flatten())

    def expectation_value(self, op, qargs):
        """Expectation value of multi-qubit operator.

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
        st1 = deepcopy(self)
        st1.normalize()
        st2 = deepcopy(st1)
        st1.evolve(op, qargs)
        return np.dot(st2.psi.flatten().conjugate(), st1.psi.flatten())


def _get_statevec_norm(psi):
    """returns norm of the state"""
    return np.sqrt(np.sum(psi.flatten().conj() * psi.flatten()))
