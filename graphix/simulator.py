"""MBQC simulator

Simulates MBQC by running Pauli measurements on stabilizer simulator
and other measurements on qiskit.qi.

"""

import numpy as np
import qiskit.quantum_info as qi
from graphix.graphsim import GraphState
from graphix.ops import Ops
from graphix.clifford import CLIFFORD_MEASURE, CLIFFORD


class Simulator():
    """MBQC simulator

    Perform Pauli measurements of the transpiled measurement pattern using graphsim.
    Non-Pauli measurements are directly simulated by applying projection operators to
    the graph state, using qiskit.quantum_info module.

    Attributes:
    -----------
    graph : nx.Graph
        graph state to be simulated.
    angles : dict
        measurement angles or 'output' label for output qubits
    domains : list of dicts
        domains of dependence ('s' and 't') for adaptive measurements.
    byprocuts : list of dicts
        domains of dependence for byproducts (X, Z) that applies to output qubits
    input_qubits : list
        vertex labels for input qubits.
    unmeasured : list
        unmeasured qubits after the graph state simulation
    output_qubits : list
        vertex labels for output qubits
    measurement_results : dict
        measurement results at each qubit, either 0 or 1
    vop : dict
        local Clifford (vertex operator, vop) on each qubit.
    node_pos : dict
        position of nodes for nx.draw() call.
    """

    def __init__(self, circuit):
        """
        Parameteres:
        --------
        circuit: graphq.transpiler.Circuit
        """
        circuit.sort_outputs()
        self.circ = circuit

        # specify graph states
        self.nodes = circuit.Nnode
        self.edges = circuit.edges
        self.vop = {j: 0 for j in range(self.nodes)}
        self.node_pos = circuit.pos

        # measurement pattern
        self.measurement_order = circuit.measurement_order
        self.byproductx = circuit.byproductx
        self.byproductz = circuit.byproductz
        self.domains = circuit.domains
        self.angles = circuit.angles
        self.measurement_results = {}
        self.output_nodes = circuit.out
        self._pauli_measurements_performed = False

    def pauli_nodes(self):
        """Find nodes that can be measured with graph algorithm

        Returns
        --------
        to_measure: list
            list of nodes that can be measured by graph transformation
        """
        to_measure = []  # Pauli measured and do not depend on non-Pauli meas
        cannot_measure = []  # all other nodes
        for i in self.circ.measurement_order:
            bpx = np.any(np.isin(self.circ.byproductx.get(i), cannot_measure))
            bpz = np.any(np.isin(self.circ.byproductz.get(i), cannot_measure))
            domain = self.circ.domains.get(i)
            if domain:
                dms0 = np.any(np.isin(self.circ.domains[i][0], cannot_measure))
                dms1 = np.any(np.isin(self.circ.domains[i][1], cannot_measure))
            else:
                dms0, dms1 = False, False
                cannot_measure.append(i)
            if np.mod(self.angles[i], 1) == 0:  # \pm x Pauli measurement
                if bpz or dms1:
                    cannot_measure.append(i)
                else:
                    to_measure.append(i)
            elif np.mod(self.angles[i], 1) == 0.5:  # \pm y Pauli measurement
                if bpx or bpz or dms1 or dms0:
                    cannot_measure.append(i)
                else:
                    to_measure.append(i)
            else:
                cannot_measure.append(i)

        return to_measure, cannot_measure

    def measure_pauli(self):
        """
        Perform Pauli measurements by graph transformations.
        uses graphq.graphsim module.

        Updates the internal graph and meausrement pattern.
        """
        assert not self._pauli_measurements_performed
        graph_state = GraphState(nodes=np.arange(self.nodes), edges=list(self.edges))
        to_measure, non_pauli_nodes = self.pauli_nodes()
        results = {}

        for i in to_measure:
            # treatment of byproduct during graph transformation
            s_signal, t_signal = self.extract_signal(i, results)
            if self.angles[i] in [-0.5, 0.5]:
                angle = self.angles[i] * (-1)**(s_signal + t_signal)
            elif self.angles[i] in [0, -1]:
                angle = self.angles[i] + np.mod(t_signal, 2)

            if angle == 0:  # +x measurement
                results[i] = graph_state.measure_x(i, choice=0)
            elif angle in [-1, 1]:  # -x measurement
                results[i] = 1 - graph_state.measure_x(i, choice=1)
            elif angle == 0.5:  # +y measurement
                results[i] = graph_state.measure_y(i, choice=0)
            elif angle == -0.5:  # -y measurement
                results[i] = 1 - graph_state.measure_y(i, choice=1)

        # relabel nodes from 0 to len(graph_state.nodes)
        vops = graph_state.get_vops()
        unmeasured = list(graph_state.nodes)
        nqubit = len(graph_state.nodes)
        mapping = [unmeasured[j] for j in range(nqubit)]
        out = [mapping.index(j) for j in self.output_nodes]
        new_edge = [(mapping.index(i), mapping.index(j)) for i, j in iter(graph_state.edges)]
        new_pos = {j: self.node_pos[mapping[j]] for j in range(nqubit)}
        new_vops = {j: vops[mapping[j]] for j in range(nqubit)}
        byproductx = dict()
        byproductz = dict()
        domains = dict()
        new_angles = dict()
        # relabel byproduct domains and apply byproduct effects from Pauli measurements
        for i in range(nqubit):
            s_signal, t_signal = 0, 0
            if self.byproductx.get(mapping[i]) is not None:
                bpx = []
                for j in self.byproductx[mapping[i]]:
                    if j in mapping:
                        bpx.append(mapping.index(j))
                    else:
                        s_signal += results[j]
                if bpx:
                    byproductx[i] = bpx
            else:
                byproductx[i] = []

            if self.byproductz.get(mapping[i]) is not None:
                bpz = []
                for j in self.byproductz[mapping[i]]:
                    if j in mapping:
                        bpz.append(mapping.index(j))
                    else:
                        t_signal += results[j]
                if bpz:
                    byproductz[i] = bpz
            else:
                byproductz[i] = []

            if self.domains.get(mapping[i]) is not None:
                dmt, dms = [], []
                for j in self.domains[mapping[i]][0]:
                    if j in mapping:
                        dmt.append(mapping.index(j))
                    else:
                        s_signal += results[j]
                for j in self.domains[mapping[i]][1]:
                    if j in mapping:
                        dms.append(mapping.index(j))
                    else:
                        t_signal += results[j]
                domains[i] = [dmt, dms]
            else:
                domains[i] = [[], []]

            if self.angles.get(mapping[i]) is not None:
                new_angles[i] = self.angles[mapping[i]] * (-1)**s_signal + t_signal

        self.nodes = nqubit
        self.edges = new_edge
        self.measurement_order = [mapping.index(j) for j in non_pauli_nodes]
        self.byproductx = byproductx
        self.byproductz = byproductz
        self.domains = domains
        self.angles = new_angles
        self.vop = new_vops
        self.node_pos = new_pos
        self.output_nodes = out
        self._pauli_measurements_performed = True
        return graph_state

    def extract_signal(self, node, results):
        """Signal extraction for adaptive measurement and byproduct of measured qubits

        returns the adaptive signal to change the measurement angle.
        s flips polarity of measurement angle and t add pi to the measurement.

        Returns:
        --------
        s_signal : int
        t_signal : int
        """
        bpx, bpz = 0, 0
        if not np.mod(self.angles[node], 1) == 0:  # NOT \pm x Pauli measurement
            if node in self.byproductx.keys():
                if self.byproductx[node]:
                    bpx = np.sum([results[j] for j in self.byproductx[node]])
        if node in self.byproductz.keys():
            if self.byproductz[node]:
                bpz = np.sum([results[j] for j in self.byproductz[node]])

        signal_s, signal_t = 0, 0
        if not np.mod(self.angles[node], 1) == 0:  # NOT \pm x Pauli measurement
            if self.domains[node][0]:
                signal_s = np.sum([results[j] for j in self.domains[node][0]])
        if self.domains[node][1]:
            signal_t = np.sum([results[j] for j in self.domains[node][1]])
        return signal_s + bpx, signal_t + bpz

    def simulate_mbqc(self):
        """MBQC simulation

        Using qiskit.qi, create the graph state and simulate measurements by
        applying projection operator.

        Returns:
        --------
        gstate: qiskit.quantum_info.State
            output state after measurement and partial trace over measured qubits

        TODO: memory-efficient simulation
        TODO: arbitrary input state
        """
        gstate = qi.Statevector(np.ones(2**self.nodes) / np.sqrt(2**self.nodes))
        for i, j in self.edges:
            gstate = gstate.evolve(Ops.cz, [i, j])

        to_trace = []
        results = {}
        for i in self.measurement_order:
            if i in self.output_nodes:
                pass
            else:
                result = np.random.choice([0, 1])
                results[i] = result

                s_signal, t_signal = self.extract_signal(i, results)
                angle = self.angles[i] * np.pi * (-1)**s_signal + np.pi * t_signal

                meas_op = self.meas_op(angle, self.vop[i], choice=result)
                gstate = gstate.evolve(meas_op, [i])
                to_trace.append(i)

        gstate = qi.Statevector(  # normalize
            gstate.data / np.sqrt(np.dot(gstate.data.conjugate(), gstate.data)))

        for i in self.output_nodes:
            if i in self.byproductx.keys():
                if np.mod(np.sum([results[j] for j in self.byproductx[i]]), 2):
                    gstate = gstate.evolve(Ops.x, [i])
            if i in self.byproductz.keys():
                if np.mod(np.sum([results[j] for j in self.byproductz[i]]), 2):
                    gstate = gstate.evolve(Ops.z, [i])

        # trace out meaured nodes
        gstate = qi.partial_trace(gstate, to_trace).to_statevector()
        # apply VOP to output vertices
        for i, j in enumerate(self.output_nodes):
            gstate = gstate.evolve(qi.Operator(CLIFFORD[self.vop[j]]), [i])

        return gstate

    @staticmethod
    def meas_op(angle, vop, choice=0):
        """Returns the projection operator for given measurement angle and local Clifford op (VOP).

        Parameters
        ----------
        angle: float
            original measurement angle (xy-plane) in radian
        vop : int
            index of local Clifford (vop), see graphq.clifford.CLIFFORD
        choice : 0 or 1
            choice of measurement outcome. measured eigenvalue would be (-1)**choice.

        Returns
        -------
        op : qi.Operator
            projection operator

        """
        assert vop in np.arange(24)
        assert choice in [0, 1]
        vec = (np.cos(angle), np.sin(angle), 0)
        op_mat = np.eye(2, dtype=np.complex128) / 2
        for i in range(3):
            op_mat += (-1)**(choice + CLIFFORD_MEASURE[vop][i][1]) \
                * vec[CLIFFORD_MEASURE[vop][i][0]] * CLIFFORD[i + 1] / 2
        return qi.Operator(op_mat)
