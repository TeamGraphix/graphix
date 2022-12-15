import unittest
from graphix.graphsim import GraphState
from graphix.ops import Ops
from graphix.sim.statevec import Statevec, meas_op
import numpy as np


def get_state(g):
    node_list = list(g.nodes)
    nqubit = len(g.nodes)
    gstate = Statevec(nqubit=nqubit)
    imapping = {node_list[i]: i for i in range(nqubit)}
    mapping = [node_list[i] for i in range(nqubit)]
    for i, j in g.edges:
        gstate.entangle((imapping[i], imapping[j]))
    for i in range(nqubit):
        if g.nodes[mapping[i]]["sign"]:
            gstate.evolve_single(Ops.z, i)
    for i in range(nqubit):
        if g.nodes[mapping[i]]["loop"]:
            gstate.evolve_single(Ops.s, i)
    for i in range(nqubit):
        if g.nodes[mapping[i]]["hollow"]:
            gstate.evolve_single(Ops.h, i)
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
        gstate.evolve_single(meas_op(0), [0])  # x meas
        gstate.normalize()
        gstate.ptrace([0])
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())), 1)

        g.measure_y(1, choice=0)
        gstate.evolve_single(meas_op(0.5 * np.pi), [0])  # y meas
        gstate.normalize()
        gstate.ptrace([0])
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())), 1)

        g.measure_z(3)
        gstate.evolve_single(meas_op(0.5 * np.pi, plane="YZ"), 1)  # z meas
        gstate.normalize()
        gstate.ptrace([1])
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())), 1)

    def test_E2(self):
        nqubit = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges)
        g.h(3)
        gstate = get_state(g)

        g.equivalent_graph_E2(3, 4)
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())), 1)

        g.equivalent_graph_E2(4, 0)
        gstate3 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate3.flatten())), 1)

        g.equivalent_graph_E2(4, 5)
        gstate4 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate4.flatten())), 1)

        g.equivalent_graph_E2(0, 3)
        gstate5 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate5.flatten())), 1)

        g.equivalent_graph_E2(0, 3)
        gstate6 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate6.flatten())), 1)

    def test_E1(self):
        nqubit = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges)
        g.nodes[3]["loop"] = True
        gstate = get_state(g)
        g.equivalent_graph_E1(3)

        gstate2 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())), 1)
        g.z(4)
        gstate = get_state(g)
        g.equivalent_graph_E1(4)
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())), 1)
        g.equivalent_graph_E1(4)
        gstate3 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate3.flatten())), 1)


if __name__ == "__main__":
    unittest.main()
