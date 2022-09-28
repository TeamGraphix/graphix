import unittest
from graphix.graphsim import GraphState
import qiskit.quantum_info as qi
from graphix.ops import Ops, States
import numpy as np


def get_state(g):
    gstate = qi.Statevector(np.ones(2**len(g.nodes)) / np.sqrt(2**len(g.nodes)))
    node_list = list(g.nodes)
    nqubit = len(g.nodes)
    imapping = {node_list[i]: i for i in range(nqubit)}
    mapping = [node_list[i] for i in range(nqubit)]
    for i, j in g.edges:
        gstate = gstate.evolve(Ops.cz, [imapping[i], imapping[j]])
    for i in range(nqubit):
        if g.nodes[mapping[i]]['sign']:
            gstate = gstate.evolve(Ops.z, [i])
    for i in range(nqubit):
        if g.nodes[mapping[i]]['loop']:
            gstate = gstate.evolve(Ops.s, [i])
    for i in range(nqubit):
        if g.nodes[mapping[i]]['hollow']:
            gstate = gstate.evolve(Ops.h, [i])
    return gstate


class TestGraphSim(unittest.TestCase):

    def test_fig2(self):
        """Example of three single-qubit measurements
        presented in Fig.2 of M. Elliot et al (2010)
        """
        nqubit = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges)
        gstate = get_state(g)
        g.measure_x(0)
        gstate = gstate.evolve(States.xplus_state.to_operator(), [0])
        gstate = qi.Statevector(gstate.data / np.sqrt(np.dot(gstate.data.conjugate(), gstate.data)))
        gstate = qi.partial_trace(gstate, [0]).to_statevector()
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate2.data)), 1)

        g.measure_y(1, choice=0)
        gstate = gstate.evolve(States.yplus_state.to_operator(), [0])
        gstate = qi.Statevector(gstate.data / np.sqrt(np.dot(gstate.data.conjugate(), gstate.data)))
        gstate = qi.partial_trace(gstate, [0]).to_statevector()
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate2.data)), 1)

        g.measure_z(3)
        gstate = gstate.evolve(States.zplus_state.to_operator(), [1])
        gstate = qi.Statevector(gstate.data / np.sqrt(np.dot(gstate.data.conjugate(), gstate.data)))
        gstate = qi.partial_trace(gstate, [1]).to_statevector()
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate2.data)), 1)

    def test_E2(self):
        nqubit = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges)
        g.h(3)
        gstate = get_state(g)

        g.equivalent_graph_E2(3, 4)
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate2.data)), 1)

        g.equivalent_graph_E2(4, 0)
        gstate3 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate3.data)), 1)

        g.equivalent_graph_E2(4, 5)
        gstate4 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate4.data)), 1)

        g.equivalent_graph_E2(0, 3)
        gstate5 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate5.data)), 1)

        g.equivalent_graph_E2(0, 3)
        gstate6 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate6.data)), 1)

    def test_E1(self):
        nqubit = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges)
        g.nodes[3]['loop'] = True
        gstate = get_state(g)
        g.equivalent_graph_E1(3)

        gstate2 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate2.data)), 1)
        g.z(4)
        gstate = get_state(g)
        g.equivalent_graph_E1(4)
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate2.data)), 1)
        g.equivalent_graph_E1(4)
        gstate3 = get_state(g)
        np.testing.assert_almost_equal(
            np.abs(np.dot(gstate.data.conjugate(), gstate3.data)), 1)


if __name__ == '__main__':
    unittest.main()
