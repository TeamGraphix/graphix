from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest

from graphix.clifford import Clifford
from graphix.fundamentals import Plane
from graphix.graphsim import GraphState
from graphix.ops import Ops
from graphix.sim.statevec import Statevec


def get_state(g: GraphState) -> Statevec:
    node_list = list(g.nodes)
    nqubit = len(g.nodes)
    gstate = Statevec(nqubit=nqubit)
    imapping = {node_list[i]: i for i in range(nqubit)}
    mapping = [node_list[i] for i in range(nqubit)]
    for i, j in g.edges:
        gstate.entangle((imapping[i], imapping[j]))
    for i in range(nqubit):
        if g.nodes[mapping[i]]["sign"]:
            gstate.evolve_single(Ops.Z, i)
    for i in range(nqubit):
        if g.nodes[mapping[i]]["loop"]:
            gstate.evolve_single(Ops.S, i)
    for i in range(nqubit):
        if g.nodes[mapping[i]]["hollow"]:
            gstate.evolve_single(Ops.H, i)
    return gstate


def meas_op(
    angle: float, vop: Clifford = Clifford.I, plane: Plane = Plane.XY, choice: int = 0
) -> npt.NDArray[np.complex128]:
    """Return the projection operator for given measurement angle and local Clifford op (VOP).

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
    assert choice in {0, 1}
    if plane == Plane.XY:
        vec = (np.cos(angle), np.sin(angle), 0)
    elif plane == Plane.YZ:
        vec = (0, np.cos(angle), np.sin(angle))
    elif plane == Plane.XZ:
        vec = (np.cos(angle), 0, np.sin(angle))
    op_mat = np.eye(2, dtype=np.complex128) / 2
    for i in range(3):
        op_mat += (-1) ** (choice) * vec[i] * Clifford(i + 1).matrix / 2
    return (vop.conj.matrix @ op_mat @ vop.matrix).astype(np.complex128, copy=False)


class TestGraphSim:
    def test_fig2(self) -> None:
        """Three single-qubit measurements presented in Fig.2 of M. Elliot et al (2010)."""
        nqubit = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges)
        gstate = get_state(g)
        g.measure_x(0)
        gstate.evolve_single(meas_op(0), 0)  # x meas
        gstate.normalize()
        gstate.remove_qubit(0)
        gstate2 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())) == pytest.approx(1)

        g.measure_y(1, choice=0)
        gstate.evolve_single(meas_op(0.5 * np.pi), 0)  # y meas
        gstate.normalize()
        gstate.remove_qubit(0)
        gstate2 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())) == pytest.approx(1)

        g.measure_z(3)
        gstate.evolve_single(meas_op(0.5 * np.pi, plane=Plane.YZ), 1)  # z meas
        gstate.normalize()
        gstate.remove_qubit(1)
        gstate2 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())) == pytest.approx(1)

    def test_e2(self) -> None:
        nqubit = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges)
        g.h(3)
        gstate = get_state(g)

        g.equivalent_graph_e2(3, 4)
        gstate2 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())) == pytest.approx(1)

        g.equivalent_graph_e2(4, 0)
        gstate3 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate3.flatten())) == pytest.approx(1)

        g.equivalent_graph_e2(4, 5)
        gstate4 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate4.flatten())) == pytest.approx(1)

        g.equivalent_graph_e2(0, 3)
        gstate5 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate5.flatten())) == pytest.approx(1)

        g.equivalent_graph_e2(0, 3)
        gstate6 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate6.flatten())) == pytest.approx(1)

    def test_e1(self) -> None:
        nqubit = 6
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges)
        g.nodes[3]["loop"] = True
        gstate = get_state(g)
        g.equivalent_graph_e1(3)

        gstate2 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())) == pytest.approx(1)
        g.z(4)
        gstate = get_state(g)
        g.equivalent_graph_e1(4)
        gstate2 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate2.flatten())) == pytest.approx(1)
        g.equivalent_graph_e1(4)
        gstate3 = get_state(g)
        assert np.abs(np.dot(gstate.flatten().conjugate(), gstate3.flatten())) == pytest.approx(1)

    def test_local_complement(self) -> None:
        nqubit = 6
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        exp_edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 0)]
        g = GraphState(nodes=np.arange(nqubit), edges=edges)
        g.local_complement(1)
        exp_g = GraphState(nodes=np.arange(nqubit), edges=exp_edges)
        assert nx.utils.graphs_equal(g, exp_g)  # type:ignore[no-untyped-call]
