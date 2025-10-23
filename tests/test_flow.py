from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import pytest

from graphix.flow.flow import CausalFlow, GFlow
from graphix.fundamentals import AbstractPlanarMeasurement, Plane
from graphix.measurements import Measurement
from graphix.opengraph_ import OpenGraph

if TYPE_CHECKING:
    from collections.abc import Mapping


def generate_causal_flow_0() -> CausalFlow[Plane]:
    """Generate causal flow on linear open graph.

    Open graph structure:

        [0]-1-2-(3)

    Causal flow:
        c(0) = 1, c(1) = 2, c(2) = 3
        {3} > {2} > {1} > {0}
    """
    og = OpenGraph(
                        graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
                        input_nodes=[0],
                        output_nodes=[3],
                        measurements=dict.fromkeys(range(3), Plane.XY),
                    )
    return CausalFlow(
                    og=og,
                    correction_function={0: {1}, 1: {2}, 2: {3}},
                    partial_order_layers=[{3}, {2}, {1}, {0}],
                )


def generate_causal_flow_1() -> CausalFlow[Measurement]:
    """Generate causal flow on H-shaped open graph.

    Open graph structure:

        [0]-2-(4)
            |
        [1]-3-(5)

    Causal flow:
        c(0) = 2, c(1) = 3, c(2) = 4, c(3) = 5
        {4, 5} > {2, 3} > {0, 1}
    """
    og = OpenGraph(
                    graph=nx.Graph([(0, 2), (2, 3), (1, 3), (2, 4), (3, 5)]),
                    input_nodes=[0, 1],
                    output_nodes=[4, 5],
                    measurements=dict.fromkeys(range(4), Measurement(angle=0, plane=Plane.XY)))
    return CausalFlow(
                    og=og,
                    correction_function={0: {2}, 1: {3}, 2: {4}, 3: {5}},
                    partial_order_layers=[{4, 5}, {2, 3}, {0, 1}],
                )


def generate_gflow_0() -> GFlow[Measurement]:
    """Generate gflow on H-shaped open graph.

    Open graph structure:

        [0]-2-(4)
            |
        [1]-3-(5)

    GFlow:
        g(0) = {2, 5}, g(1) = {3, 4}, g(2) = {4}, g(3) = {5}
        {4, 5} > {0, 1, 2, 3}
    """
    og = OpenGraph(
                    graph=nx.Graph([(0, 2), (2, 3), (1, 3), (2, 4), (3, 5)]),
                    input_nodes=[0, 1],
                    output_nodes=[4, 5],
                    measurements=dict.fromkeys(range(4), Measurement(angle=0, plane=Plane.XY)))
    return GFlow(
                    og=og,
                    correction_function={0: {2, 5}, 1: {3, 4}, 2: {4}, 3: {5}},
                    partial_order_layers=[{4, 5}, {0, 1, 2, 3}],
                )


def generate_gflow_1() -> GFlow[Plane]:
    r"""Generate gflow on open graph without causal flow.

    Open graph structure:

        1
         \
         (4)-[0]-(3)
         /
        2

    GFlow:
        g(0) = {3}, g(1) = {1}, g(2) = {2, 3, 4}
        {3, 4} > {1} > {0, 2}
    """
    og = OpenGraph(
                    graph=nx.Graph([(0, 3), (0, 4), (1, 4), (2, 4)]),
                    input_nodes=[0],
                    output_nodes=[3, 4],
                    measurements={0: Plane.XY, 1: Plane.YZ, 2: Plane.XZ},
                )
    return GFlow(
                    og=og,
                    correction_function={0: {3}, 1: {1}, 2: {2, 3, 4}},
                    partial_order_layers=[{3, 4}, {1}, {0, 2}],
                )


def generate_gflow_2() -> GFlow[Plane]:
    r"""Generate gflow on open graph without causal flow.

    Open graph structure:

        [0]-(3)
           X
        [1]-(4)
           X
        [2]-(5)

    GFlow:
        g(0) = {4, 5}, g(1) = {3, 4, 5}, g(2) = {3, 4}
        {3, 4, 5} > {0, 1, 2}
    """
    og = OpenGraph(
                    graph=nx.Graph([(0, 3), (0, 4), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5)]),
                    input_nodes=[0, 1, 2],
                    output_nodes=[3, 4, 5],
                    measurements=dict.fromkeys(range(3), Plane.XY),
    )
    return GFlow(
                    og=og,
                    correction_function={0: {4, 5}, 1: {3, 4, 5}, 2: {3, 4}},
                    partial_order_layers=[{3, 4}, {1}, {0, 2}],
                )


def prepare_test_causal_flow() -> list[CausalFlow[AbstractPlanarMeasurement]]:
    return [generate_causal_flow_0(), generate_causal_flow_1()]


def prepare_test_gflow() -> list[GFlow[AbstractPlanarMeasurement]]:
    return [generate_gflow_0(), generate_gflow_1(), generate_gflow_2()]

   #         # Open graph without causal flow or gflow.
    #         FlowTestCase(
    #             og=OpenGraph(
    #                 graph=nx.Graph([(0, 2), (1, 2), (2, 3), (2, 4)]),
    #                 input_nodes=[0, 1],
    #                 output_nodes=[3, 4],
    #                 measurements=dict.fromkeys(range(3), Measurement(angle=Placeholder("Angle"), plane=Plane.XY)),
    #             ),
    #             has_causal_flow=False,
    #             has_gflow=False,
    #         ),
    #     )
    # )


class TestFlow:
    @pytest.mark.parametrize("test_case", prepare_test_causal_flow())
    def test_causal_flow(self, test_case: CausalFlow[AbstractPlanarMeasurement]) -> None:
        flow = test_case.og.find_causal_flow()

        assert flow is not None
        assert flow.correction_function == test_case.correction_function
        assert flow.partial_order_layers == test_case.partial_order_layers

    # @pytest.mark.parametrize("test_case", prepare_test_gflow())
    # def test_gflow(self, test_case: GFlow[AbstractPlanarMeasurement]) -> None:
    #     flow = test_case.og.find_gflow()

    #     assert flow is not None
    #     assert flow.correction_function == test_case.correction_function
    #     assert flow.partial_order_layers == test_case.partial_order_layers


class XZCorrectionsTestCase(NamedTuple):
    flow: CausalFlow[AbstractPlanarMeasurement] | GFlow[AbstractPlanarMeasurement]
    x_corr: Mapping[int, set[int]]
    z_corr: Mapping[int, set[int]]

# TODO: add pattern, add dag


def prepare_test_xzcorrections() -> list[XZCorrectionsTestCase]:
    test_cases: list[XZCorrectionsTestCase] = []

    test_cases.extend(
        (
            XZCorrectionsTestCase(
                flow=generate_causal_flow_0(),
                x_corr={1: {0}, 2: {1}, 3: {2}},
                z_corr={2: {0}, 3: {1}},
            ),
            XZCorrectionsTestCase(
                flow=generate_causal_flow_1(),
                x_corr={2: {0}, 3: {1}, 4: {2}, 5: {3}},
                z_corr={3: {0}, 4: {0}, 2: {1}, 5: {1}},
            ),
            # Same open graph as before but now we consider a gflow which has lower depth than the causal flow.
            XZCorrectionsTestCase(
                flow=generate_gflow_0(),
                x_corr={2: {0}, 5: {0, 3}, 3: {1}, 4: {1, 2}},
                z_corr={4: {0}, 5: {1}},
            ),
            XZCorrectionsTestCase(
                flow=generate_gflow_1(),
                x_corr={3: {0, 2}, 4: {2}},
                z_corr={1: {2}, 4: {1, 2}},
            ),
            XZCorrectionsTestCase(
                flow=generate_gflow_2(),
                x_corr={3: {1, 2}, 4: {0, 1, 2}, 5: {0, 1}},
                z_corr={},
            ),
        )
    )

    return test_cases


class TestXZCorrections:
    @pytest.mark.parametrize("test_case", prepare_test_xzcorrections())
    def test_causal_flow(self, test_case: XZCorrectionsTestCase) -> None:
        flow = test_case.flow
        corrections = flow.to_corrections()
        assert corrections.z_corrections == test_case.z_corr
        assert corrections.x_corrections == test_case.x_corr
