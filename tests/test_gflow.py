from __future__ import annotations

import dataclasses
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
    meas_planes: dict[int, Plane] = dataclasses.field(kw_only=True)
    meas_pplanes: dict[int, PauliPlane] = dataclasses.field(kw_only=True)
    flow_exist: bool | None = dataclasses.field(kw_only=True)
    gflow_exist: bool = dataclasses.field(kw_only=True)
    pauliflow_exist: bool = dataclasses.field(kw_only=True)


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
        generate_g4(),
        generate_g7(),
        generate_g8(),
        generate_g9(),
    ]


class TestGflow:
    @pytest.mark.parametrize("gft", generate_gft())
    def test_flow(self, gft: GraphForTest) -> None:
        if gft.flow_exist is None:
            pytest.skip("Contains non-XY measurements.")
        res = gflow.find_flow(gft.graph, gft.inputs, gft.outputs)
        assert gft.flow_exist == (res is not None)
        if res is not None:
            assert gflow.verify_flow(res, gft.graph, gft.inputs, gft.outputs)

    @pytest.mark.parametrize("gft", generate_gft())
    def test_gflow(self, gft: GraphForTest) -> None:
        res = gflow.find_gflow(gft.graph, gft.inputs, gft.outputs, gft.meas_planes)
        assert gft.gflow_exist == (res is not None)
        if res is not None:
            assert gflow.verify_gflow(res, gft.graph, gft.inputs, gft.outputs, gft.meas_planes)

    @pytest.mark.parametrize("gft", generate_gft())
    def test_pflow(self, gft: GraphForTest) -> None:
        res = gflow.find_pauliflow(gft.graph, gft.inputs, gft.outputs, gft.meas_pplanes)
        assert gft.pauliflow_exist == (res is not None)
        if res is not None:
            assert gflow.verify_pauliflow(res, gft.graph, gft.inputs, gft.outputs, gft.meas_pplanes)

    def test_with_rand_circ(self, fx_rng: Generator) -> None:
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

    def test_rand_circ_gflow(self, fx_rng: Generator) -> None:
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
