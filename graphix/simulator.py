"""MBQC simulator

Simulates MBQC by executing the pattern.

"""

import numpy as np
import qiskit.quantum_info as qi
from graphix.ops import Ops
from graphix.clifford import CLIFFORD_MEASURE, CLIFFORD
from copy import deepcopy

class PatternSimulator():
    """MBQC simulator

    Executes the measurement pattern.
    """

    def __init__(self, pattern, backend='statevector', max_qubit_num=12):
        """
        Parameteres
        -----------
        pattern: :class:`graphix.pattern.Pattern` object
            MBQC pattern to be simulated.
        backend: str, 'statevector'
            optional argument for simulation.
        max_qubit_num : int
            optional argument specifying the maximum number of qubits in statevector at a time.
        """
        # check that pattern has output nodes configured
        assert len(pattern.output_nodes) > 0
        self.backend = backend
        self.pattern = pattern
        self.results = deepcopy(pattern.results)
        self.sv = qi.Statevector([])
        self.node_index = []
        self.max_qubit_num = max_qubit_num
        if pattern.max_space() > max_qubit_num:
            raise ValueError('Pattern.max_space is larger than max_qubit_num. Increase max_qubit_num and try again')

    def qubit_dim(self):
        """Returns the qubit number in the internal statevector
        Returns
        -------
        n_qubit : int
        """
        return len(self.sv.dims())

    def normalize_state(self):
        """Normalize the internal statevector
        """
        self.sv = self.sv/self.sv.trace()**0.5

    def initialize_statevector(self):
        """Initialize the internal statevector
        """
        self.sv = qi.Statevector([])

    def add_nodes(self, nodes):
        """add new qubit to internal statevector
        and assign the corresponding node number
        to list self.node_index.

        Parameters
        ---------
        nodes : list of node indices
        """
        if not self.sv:
            self.sv = qi.Statevector([1])
        n = len(nodes)
        sv_to_add = qi.Statevector([1 for i in range(2**n)])
        sv_to_add = sv_to_add/sv_to_add.trace()**0.5
        self.sv = self.sv.expand(sv_to_add)
        self.node_index.extend(nodes)

    def entangle_nodes(self, edge):
        """ Apply CZ gate to two connected nodes

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        target = self.node_index.index(edge[0])
        control = self.node_index.index(edge[1])
        self.sv = self.sv.evolve(Ops.cz, [control, target])

    def measure(self, cmd, to_trace, to_trace_loc):
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
        angle = cmd[3] * np.pi * (-1)**s_signal + np.pi * t_signal
        if len(cmd) == 7:
            vop = cmd[6]
        else:
            vop = 0
        meas_op = self.meas_op(angle, vop, plane=cmd[2], choice=result)
        loc = self.node_index.index(cmd[1])
        self.sv = self.sv.evolve(meas_op, [loc])

        to_trace.append(cmd[1])
        to_trace_loc.append(loc)

        return to_trace, to_trace_loc

    def correct_byproduct(self, cmd):
        """Byproduct correction
        correct for the X or Z byproduct operators,
        by applying the X or Z gate.
        """
        if np.mod(np.sum([self.results[j] for j in cmd[2]]), 2) == 1:
            loc = self.node_index.index(cmd[1])
            if cmd[0] == 'X':
                op = Ops.x
            elif cmd[0] == 'Z':
                op = Ops.z
            self.sv = self.sv.evolve(op, [loc])

    def apply_clifford(self, cmd):
        """Apply single-qubit Clifford gate,
        specified by vop index specified in graphix.clifford.CLIFFORD
        """
        loc = self.node_index.index(cmd[1])
        self.sv = self.sv.evolve(qi.Operator(CLIFFORD[cmd[2]]), [loc])

    def run(self):
        """Perform the simulation.

        Returns
        -------
        state : qiskit.quantum_info.Statevector
            output state of the MBQC.
        """
        self.initialize_statevector()
        to_trace = []
        to_trace_loc = []
        Nqubit = 0
        threshold = self.max_qubit_num
        for cmd in self.pattern.seq:
            if cmd[0] == 'N':
                self.add_nodes([cmd[1]])
                Nqubit += 1
            elif cmd[0] == 'E':
                self.entangle_nodes(cmd[1])
            elif cmd[0] == 'M':
                to_trace, to_trace_loc = self.measure(cmd, to_trace, to_trace_loc)
            elif cmd[0] == 'X':
                self.correct_byproduct(cmd)
            elif cmd[0] == 'Z':
                self.correct_byproduct(cmd)
            elif cmd[0] == 'C':
                self.apply_clifford(cmd)
            else:
                raise ValueError("invalid commands")

            # trace out
            if (Nqubit == threshold) or (self.pattern.seq[-1] == cmd):
                self.normalize_state()
                self.sv = qi.partial_trace(self.sv, to_trace_loc).to_statevector()
                for node in to_trace:
                    self.node_index.remove(node)
                Nqubit -= len(to_trace)
                to_trace = []
                to_trace_loc = []
        self.sort_qubits()
        return self.sv

    def sort_qubits(self):
        """sort the qubit order in internal statevector"""
        for i, ind in enumerate(self.pattern.output_nodes):
            if not self.node_index[i] == ind:
                move_from = self.node_index.index(ind)
                self.sv = self.sv.evolve(Ops.swap, [i, move_from])
                self.node_index[i], self.node_index[move_from] = \
                    self.node_index[move_from], self.node_index[i]

    @staticmethod
    def meas_op(angle, vop, plane='XY', choice=0):
        """Returns the projection operator for given measurement angle and local Clifford op (VOP).

        Parameters
        ----------
        angle: float
            original measurement angle in radian
        vop : int
            index of local Clifford (vop), see graphq.clifford.CLIFFORD
        plane : 'XY', 'YZ' or 'ZX'
            measurement plane on which angle shall be defined
        choice : 0 or 1
            choice of measurement outcome. measured eigenvalue would be (-1)**choice.

        Returns
        -------
        op : qi.Operator
            projection operator

        """
        assert vop in np.arange(24)
        assert choice in [0, 1]
        assert plane in ['XY', 'YZ', 'XZ']
        if plane == 'XY':
            vec = (np.cos(angle), np.sin(angle), 0)
        elif plane == 'YZ':
            vec = (0, np.cos(angle), np.sin(angle))
        elif plane == 'XZ':
            vec = (np.cos(angle), 0, np.sin(angle))
        op_mat = np.eye(2, dtype=np.complex128) / 2
        for i in range(3):
            op_mat += (-1)**(choice + CLIFFORD_MEASURE[vop][i][1]) \
                * vec[CLIFFORD_MEASURE[vop][i][0]] * CLIFFORD[i + 1] / 2
        return qi.Operator(op_mat)
