import sys
import unittest

import numpy as np
from networkx import Graph
from networkx.utils import graphs_equal
from parameterized import parameterized_class

try:
    from rustworkx import PyGraph
except ModuleNotFoundError:
    pass

from graphix.graphsim.graphstate import GraphState
from graphix.graphsim.utils import convert_rustworkx_to_networkx, is_graphs_equal
from graphix.ops import Ops
from graphix.sim.statevec import Statevec, meas_op


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


@parameterized_class([{"use_rustworkx": False}, {"use_rustworkx": True}])
class TestGraphSim(unittest.TestCase):
    def setUp(self):
        if sys.modules.get("rustworkx") is None and self.use_rustworkx is True:
            self.skipTest("rustworkx not installed")

    def test_fig2(self):
        """Example of three single-qubit measurements
        presented in Fig.2 of M. Elliot et al (2010)
        """
        nqubit = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges, use_rustworkx=self.use_rustworkx)
        gstate = get_state(g)
        g.measure_x(0)
        gstate.evolve_single(meas_op(0), [0])  # x meas
        gstate.normalize()
        gstate.remove_qubit(0)
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())), 1)

        g.measure_y(1, choice=0)
        gstate.evolve_single(meas_op(0.5 * np.pi), [0])  # y meas
        gstate.normalize()
        gstate.remove_qubit(0)
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())), 1)

        g.measure_z(3)
        gstate.evolve_single(meas_op(0.5 * np.pi, plane="YZ"), 1)  # z meas
        gstate.normalize()
        gstate.remove_qubit(1)
        gstate2 = get_state(g)
        np.testing.assert_almost_equal(np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())), 1)

    def test_E2(self):
        nqubit = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges, use_rustworkx=self.use_rustworkx)
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
        g = GraphState(nodes=np.arange(nqubit), edges=edges, use_rustworkx=self.use_rustworkx)
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

    def test_local_complement(self):
        nqubit = 6
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        exp_edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 0)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges, use_rustworkx=self.use_rustworkx)
        g.local_complement(1)
        exp_g = GraphState(nodes=np.arange(nqubit), edges=exp_edges)
        self.assertTrue(is_graphs_equal(g, exp_g))


@unittest.skipIf(sys.modules.get("rustworkx") is None, "rustworkx not installed")
class TestGraphSimUtils(unittest.TestCase):
    def test_is_graphs_equal_nx_nx(self):
        nnode = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g1 = GraphState(nodes=range(nnode), edges=edges, use_rustworkx=True)
        g2 = GraphState(nodes=range(nnode), edges=edges, use_rustworkx=True)
        self.assertTrue(is_graphs_equal(g1, g2))

    def test_is_graphs_equal_nx_rx(self):
        nnode = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g1 = GraphState(nodes=range(nnode), edges=edges, use_rustworkx=True)
        g2 = GraphState(nodes=range(nnode), edges=edges, use_rustworkx=True)
        self.assertTrue(is_graphs_equal(g1, g2))

    def test_is_graphs_equal_rx_nx(self):
        nnode = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g1 = GraphState(nodes=range(nnode), edges=edges, use_rustworkx=True)
        g2 = GraphState(nodes=range(nnode), edges=edges, use_rustworkx=True)
        self.assertTrue(is_graphs_equal(g1, g2))

    def test_is_graphs_equal_rx_rx(self):
        nnode = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g1 = GraphState(nodes=range(nnode), edges=edges, use_rustworkx=True)
        g2 = GraphState(nodes=range(nnode), edges=edges, use_rustworkx=True)
        self.assertTrue(is_graphs_equal(g1, g2))

    def test_convert_rustworkx_to_networkx(self):
        nnode = 6
        data = {"dummy": 1}
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g_nx = Graph()
        g_nx.add_nodes_from([(n, d) for n, d in zip(range(nnode), [data] * nnode)])
        g_nx.add_edges_from(edges)
        g_rx = PyGraph()
        g_rx.add_nodes_from([(n, d) for n, d in zip(range(nnode), [data] * nnode)])
        g_rx.add_edges_from_no_data(edges)
        g_rx = convert_rustworkx_to_networkx(g_rx)
        self.assertTrue(graphs_equal(g_nx, g_rx))

    def test_convert_rustworkx_to_networkx_throw_error(self):
        nnode = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g_rx = PyGraph()
        g_rx.add_nodes_from(range(nnode))
        g_rx.add_edges_from_no_data(edges)
        with self.assertRaises(TypeError):
            g_rx = convert_rustworkx_to_networkx(g_rx)


if __name__ == "__main__":
    unittest.main()
