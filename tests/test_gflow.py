from __future__ import annotations

import dataclasses
from collections import Counter
from typing import TYPE_CHECKING

import networkx as nx
import pytest

from graphix import gflow
from graphix.fundamentals import Plane
from graphix.gflow import PauliPlane
from graphix.random_objects import rand_circuit

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclasses.dataclass
class GraphForTest:
    graph: nx.Graph[int]
    inputs: set[int]
    outputs: set[int]
    meas_planes: dict[int, Plane]
    meas_pplanes: dict[int, PauliPlane]
    flow_exist: bool | None
    gflow_exist: bool
    pauliflow_exist: bool


def generate_g1() -> GraphForTest:
    # no measurement
    # 1
    # |
    # 2
    return GraphForTest(
        nx.Graph([(1, 2)]),
        {1, 2},
        {1, 2},
        meas_planes={},
        meas_pplanes={},
        flow_exist=True,
        gflow_exist=True,
        pauliflow_exist=True,
    )


def generate_g2() -> GraphForTest:
    # line graph with flow and gflow
    # 1 - 2 - 3 - 4 - 5
    return GraphForTest(
        nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5)]),
        {1},
        {5},
        meas_planes={1: Plane.XY, 2: Plane.XY, 3: Plane.XY, 4: Plane.XY},
        meas_pplanes={1: PauliPlane.XY, 2: PauliPlane.XY, 3: PauliPlane.XY, 4: PauliPlane.XY},
        flow_exist=True,
        gflow_exist=True,
        pauliflow_exist=True,
    )


def generate_g3() -> GraphForTest:
    # graph with flow and gflow
    # 1 - 3 - 5
    #     |
    # 2 - 4 - 6
    return GraphForTest(
        nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
        {1, 2},
        {5, 6},
        meas_planes={1: Plane.XY, 2: Plane.XY, 3: Plane.XY, 4: Plane.XY},
        meas_pplanes={1: PauliPlane.XY, 2: PauliPlane.XY, 3: PauliPlane.XY, 4: PauliPlane.XY},
        flow_exist=True,
        gflow_exist=True,
        pauliflow_exist=True,
    )


def generate_g4() -> GraphForTest:
    # graph with gflow but flow
    #   ______
    #  /      |
    # 1 - 4   |
    #    /    |
    #   /     |
    #  /      |
    # 2 - 5   |
    #  \ /    |
    #   X    /
    #  / \  /
    # 3 - 6
    return GraphForTest(
        nx.Graph([(1, 4), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)]),
        {1, 2, 3},
        {4, 5, 6},
        meas_planes={1: Plane.XY, 2: Plane.XY, 3: Plane.XY},
        meas_pplanes={1: PauliPlane.XY, 2: PauliPlane.XY, 3: PauliPlane.XY},
        flow_exist=False,
        gflow_exist=True,
        pauliflow_exist=True,
    )


def generate_g5() -> GraphForTest:
    # graph with extended gflow but flow
    #   0 - 1
    #  /|   |
    # 4 |   |
    #  \|   |
    #   2 - 5 - 3
    return GraphForTest(
        nx.Graph([(0, 1), (0, 2), (0, 4), (1, 5), (2, 4), (2, 5), (3, 5)]),
        {0, 1},
        {4, 5},
        meas_planes={0: Plane.XY, 1: Plane.XY, 2: Plane.XZ, 3: Plane.YZ},
        meas_pplanes={0: PauliPlane.XY, 1: PauliPlane.XY, 2: PauliPlane.XZ, 3: PauliPlane.YZ},
        flow_exist=None,  # flow not defined
        gflow_exist=True,
        pauliflow_exist=True,
    )


def generate_g6() -> GraphForTest:
    # graph with no flow and no gflow
    # 1 - 3
    #  \ /
    #   X
    #  / \
    # 2 - 4
    return GraphForTest(
        nx.Graph([(1, 3), (1, 4), (2, 3), (2, 4)]),
        {1, 2},
        {3, 4},
        meas_planes={1: Plane.XY, 2: Plane.XY},
        meas_pplanes={1: PauliPlane.XY, 2: PauliPlane.XY},
        flow_exist=False,
        gflow_exist=False,
        pauliflow_exist=False,
    )


def generate_g7() -> GraphForTest:
    # graph with no flow or gflow but pauliflow, No.1
    #     3
    #     |
    #     2
    #     |
    # 0 - 1 - 4
    return GraphForTest(
        nx.Graph([(0, 1), (1, 2), (1, 4), (2, 3)]),
        {0},
        {4},
        meas_planes={0: Plane.XY, 1: Plane.XY, 2: Plane.XY, 3: Plane.XY},
        meas_pplanes={0: PauliPlane.XY, 1: PauliPlane.X, 2: PauliPlane.XY, 3: PauliPlane.X},
        flow_exist=False,
        gflow_exist=False,
        pauliflow_exist=True,
    )


def generate_g8() -> GraphForTest:
    # graph with no flow or gflow but pauliflow, No.2
    # 1   2   3
    # | /     |
    # 0 - - - 4
    return GraphForTest(
        nx.Graph([(0, 1), (0, 2), (0, 4), (3, 4)]),
        {0},
        {4},
        meas_planes={0: Plane.YZ, 1: Plane.XZ, 2: Plane.XY, 3: Plane.YZ},
        meas_pplanes={0: PauliPlane.Z, 1: PauliPlane.Z, 2: PauliPlane.Y, 3: PauliPlane.Y},
        flow_exist=None,  # flow not defined
        gflow_exist=False,
        pauliflow_exist=True,
    )


def generate_g9() -> GraphForTest:
    # graph with no flow or gflow but pauliflow, No.3
    # 0 - 1 -- 3
    #    \|   /|
    #     |\ / |
    #     | /\ |
    #     2 -- 4
    return GraphForTest(
        nx.Graph([(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]),
        {0},
        {3, 4},
        meas_planes={0: Plane.YZ, 1: Plane.XZ, 2: Plane.XY},
        meas_pplanes={0: PauliPlane.Z, 1: PauliPlane.XZ, 2: PauliPlane.Y},
        flow_exist=None,  # flow not defined
        gflow_exist=False,
        pauliflow_exist=True,
    )


def generate_gft() -> list[GraphForTest]:
    return [
        generate_g1(),
        generate_g2(),
        generate_g3(),
        generate_g4(),
        generate_g5(),
        generate_g6(),
        generate_g7(),
        generate_g8(),
        generate_g9(),
    ]


def randgraph(rng: Generator, n: int, nio: int) -> tuple[nx.Graph[int], set[int], set[int]]:
    """Generate a random graph and input/output sets."""
    g: nx.Graph[int] = nx.fast_gnp_random_graph(n, 0.3, seed=42)
    v = list(g.nodes)
    rng.shuffle(v)
    ni = rng.integers(0, nio + 1)
    no = rng.integers(0, nio + 1)
    iset = set(v[:ni])
    oset = set(v[ni : ni + no])
    return g, iset, oset


def randgraph_p(rng: Generator, n: int, nio: int) -> tuple[nx.Graph[int], set[int], set[int], dict[int, Plane]]:
    g, iset, oset = randgraph(rng, n, nio)
    vset = set(g.nodes)
    sel = list(Plane)
    planes = {k: sel[rng.choice(len(sel))] for k in vset if k not in oset}
    return g, iset, oset, planes


def randgraph_pp(rng: Generator, n: int, nio: int) -> tuple[nx.Graph[int], set[int], set[int], dict[int, PauliPlane]]:
    g, iset, oset = randgraph(rng, n, nio)
    vset = set(g.nodes)
    sel = list(PauliPlane)
    pplanes: dict[int, PauliPlane] = {}
    while not any(pp in {PauliPlane.X, PauliPlane.Y, PauliPlane.Z} for pp in pplanes.values()):
        pplanes = {k: sel[rng.choice(len(sel))] for k in vset if k not in oset}
    return g, iset, oset, pplanes


class TestGflow:
    @pytest.mark.parametrize("gft", generate_gft())
    def test_flow_default(self, gft: GraphForTest) -> None:
        cmp = gflow.find_flow(gft.graph, gft.inputs, gft.outputs)
        ref = gflow.find_flow(gft.graph, gft.inputs, gft.outputs, dict.fromkeys(gft.meas_planes, Plane.XY))
        assert cmp == ref

    @pytest.mark.parametrize("gft", generate_gft())
    def test_flow(self, gft: GraphForTest) -> None:
        if gft.flow_exist is None:
            pytest.skip("Contains non-XY measurements.")
        res = gflow.find_flow(gft.graph, gft.inputs, gft.outputs)
        assert gft.flow_exist == (res is not None)
        if res is not None:
            assert gflow.verify_flow(res, gft.graph, gft.inputs, gft.outputs)

    @pytest.mark.parametrize("gft", generate_gft())
    def test_gflow_default(self, gft: GraphForTest) -> None:
        cmp = gflow.find_gflow(gft.graph, gft.inputs, gft.outputs)
        ref = gflow.find_gflow(gft.graph, gft.inputs, gft.outputs, dict.fromkeys(gft.meas_planes, Plane.XY))
        assert cmp == ref

    @pytest.mark.parametrize("gft", generate_gft())
    def test_gflow(self, gft: GraphForTest) -> None:
        res = gflow.find_gflow(gft.graph, gft.inputs, gft.outputs, gft.meas_planes)
        assert gft.gflow_exist == (res is not None)
        if res is not None:
            assert gflow.verify_gflow(res, gft.graph, gft.inputs, gft.outputs, gft.meas_planes)

    @pytest.mark.parametrize("gft", generate_gft())
    def test_pflow_default(self, gft: GraphForTest) -> None:
        cmp = gflow.find_pauliflow(gft.graph, gft.inputs, gft.outputs)
        ref = gflow.find_pauliflow(gft.graph, gft.inputs, gft.outputs, dict.fromkeys(gft.meas_pplanes, PauliPlane.XY))
        assert cmp == ref

    @pytest.mark.parametrize("gft", generate_gft())
    def test_pflow(self, gft: GraphForTest) -> None:
        res = gflow.find_pauliflow(gft.graph, gft.inputs, gft.outputs, gft.meas_pplanes)
        assert gft.pauliflow_exist == (res is not None)
        if res is not None:
            assert gflow.verify_pauliflow(res, gft.graph, gft.inputs, gft.outputs, gft.meas_pplanes)

    def test_randcirc_flow(self, fx_rng: Generator) -> None:
        # test for large graph
        # graph transpiled from circuit always has a flow
        circ = rand_circuit(50, 50, fx_rng)
        pattern = circ.transpile().pattern
        _, edges = pattern.get_graph()
        graph = nx.Graph(edges)
        input_ = set(pattern.input_nodes)
        output = set(pattern.output_nodes)
        meas_planes = pattern.get_meas_plane()
        assert all(p == Plane.XY for p in meas_planes.values())
        res = gflow.find_flow(graph, input_, output)
        assert res is not None
        assert gflow.verify_flow(res, graph, input_, output)

    def test_randcirc_gflow(self, fx_rng: Generator) -> None:
        # test for large graph
        # pauli-node measured graph always has gflow
        circ = rand_circuit(30, 30, fx_rng)
        pattern = circ.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        _, edges = pattern.get_graph()
        graph = nx.Graph(edges)
        output = set(pattern.output_nodes)
        meas_planes = pattern.get_meas_plane()
        res = gflow.find_gflow(graph, set(), output, meas_planes)
        assert res is not None
        assert gflow.verify_gflow(res, graph, set(), output, meas_planes)

    def test_randcirc_pflow(self, fx_rng: Generator) -> None:
        circ = rand_circuit(20, 20, fx_rng)
        pattern = circ.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        _, edges = pattern.get_graph()
        graph = nx.Graph(edges)
        output = set(pattern.output_nodes)
        meas_planes = pattern.get_meas_plane()
        meas_angles = pattern.get_angles()
        meas_pplanes = {k: PauliPlane.from_plane(v, meas_angles[k]) for k, v in meas_planes.items()}
        res = gflow.find_pauliflow(graph, set(), output, meas_pplanes)
        assert res is not None
        assert gflow.verify_pauliflow(res, graph, set(), output, meas_pplanes)

    def test_randgraph_flow(self, fx_rng: Generator) -> None:
        cnt: Counter[bool] = Counter()
        while any(v < 5 for v in cnt.values()):
            g, iset, oset = randgraph(fx_rng, 15, 4)
            res = gflow.find_flow(g, iset, oset)
            if res is not None:
                cnt[True] += 1
                assert gflow.verify_flow(res, g, iset, oset)
            else:
                cnt[False] += 1

    def test_randgraph_gflow(self, fx_rng: Generator) -> None:
        cnt: Counter[bool] = Counter()
        while any(v < 5 for v in cnt.values()):
            g, iset, oset, planes = randgraph_p(fx_rng, 15, 4)
            res = gflow.find_gflow(g, iset, oset, planes)
            if res is not None:
                cnt[True] += 1
                assert gflow.verify_gflow(res, g, iset, oset, planes)
            else:
                cnt[False] += 1

    def test_randgraph_pflow(self, fx_rng: Generator) -> None:
        cnt: Counter[bool] = Counter()
        while any(v < 5 for v in cnt.values()):
            g, iset, oset, pplanes = randgraph_pp(fx_rng, 15, 4)
            res = gflow.find_pauliflow(g, iset, oset, pplanes)
            if res is not None:
                cnt[True] += 1
                assert gflow.verify_pauliflow(res, g, iset, oset, pplanes)
            else:
                cnt[False] += 1

    def test_odd_neighbor(self) -> None:
        g = nx.complete_graph(4)
        assert gflow.odd_neighbor(g, {0}) == {1, 2, 3}
        assert gflow.odd_neighbor(g, {0, 1}) == {0, 1}
        assert gflow.odd_neighbor(g, {0, 1, 2}) == {3}
        assert gflow.odd_neighbor(g, {0, 1, 2, 3}) == {0, 1, 2, 3}

    def test_group_layers(self) -> None:
        l = {0: 3, 1: 0, 2: 1, 3: 2, 4: 2}
        assert gflow.group_layers(l) == (3, {0: {1}, 1: {2}, 2: {3, 4}, 3: {0}})
