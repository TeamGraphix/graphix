from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import pytest

from graphix.fundamentals import AbstractPlanarMeasurement, Plane
from graphix.measurements import Measurement
from graphix.opengraph_ import OpenGraph
from graphix.parameter import Placeholder

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class FlowTestCase(NamedTuple):
    og: OpenGraph[AbstractPlanarMeasurement]
    has_causal_flow: bool
    has_gflow: bool
    cf: Mapping[int, set[int]] | None = None
    c_layers: Sequence[set[int]] | None = None
    c_x_corr: Mapping[int, set[int]] | None = None
    c_z_corr: Mapping[int, set[int]] | None = None
    gf: Mapping[int, set[int]] | None = None
    g_layers: Sequence[set[int]] | None = None
    g_x_corr: Mapping[int, set[int]] | None = None
    g_z_corr: Mapping[int, set[int]] | None = None


def prepare_test_flow() -> list[FlowTestCase]:
    test_cases: list[FlowTestCase] = []

    test_cases.extend(
        (
            # Linear open graph with causal flow.
            FlowTestCase(
                og=OpenGraph(
                    graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
                    input_nodes=[0],
                    output_nodes=[3],
                    measurements=dict.fromkeys(range(3), Plane.XY),  # Plane labels
                ),
                has_causal_flow=True,
                has_gflow=True,
                cf={0: {1}, 1: {2}, 2: {3}},
                c_layers=[{3}, {2}, {1}, {0}],
                c_x_corr={1: {0}, 2: {1}, 3: {2}},
                c_z_corr={2: {0}, 3: {1}},
                gf={0: {1}, 1: {2}, 2: {3}},
                g_layers=[{3}, {2}, {1}, {0}],
                g_x_corr={1: {0}, 2: {1}, 3: {2}},
                g_z_corr={2: {0}, 3: {1}},
            ),
            # H-shaped open graph with causal flow.
            FlowTestCase(
                og=OpenGraph(
                    graph=nx.Graph([(0, 2), (2, 3), (1, 3), (2, 4), (3, 5)]),
                    input_nodes=[0, 1],
                    output_nodes=[4, 5],
                    measurements=dict.fromkeys(range(3), Measurement(angle=0, plane=Plane.XY)),  # Measurement labels
                ),
                has_causal_flow=True,
                cf={0: {2}, 1: {3}, 2: {4}, 3: {5}},
                c_layers=[{4, 5}, {2, 3}, {0, 1}],
                c_x_corr={2: {0}, 3: {1}, 4: {2}, 5: {3}},
                c_z_corr={3: {0}, 4: {0}, 2: {1}, 5: {1}},
            ),
            # Open graph without causal flow but gflow.
            FlowTestCase(
                og=OpenGraph(
                    graph=nx.Graph([(0, 3), (0, 4), (1, 4), (2, 4)]),
                    input_nodes=[0],
                    output_nodes=[3, 4],
                    measurements={0: Plane.XY, 1: Plane.YZ, 2: Plane.XZ}
                ),
                has_causal_flow=False,
            ),
            # Open graph without causal flow but gflow.
            FlowTestCase(
                og=OpenGraph(
                    graph=nx.Graph([(0, 3), (0, 4), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5)]),
                    input_nodes=[0, 1, 2],
                    output_nodes=[3, 4, 5],
                    measurements=dict.fromkeys(range(3), Plane.XY)
                ),
                has_causal_flow=False,
            ),
            # Open graph without causal flow or gflow.
            FlowTestCase(
                og=OpenGraph(
                    graph=nx.Graph([(0, 2), (1, 2), (2, 3), (2, 4)]),
                    input_nodes=[0, 1],
                    output_nodes=[3, 4],
                    measurements=dict.fromkeys(range(3), Measurement(angle=Placeholder('Angle'), plane=Plane.XY)),
                ),
                has_causal_flow=False,
            ),
        )
    )

    return test_cases


class TestFlow:

    @pytest.mark.parametrize("test_case", prepare_test_flow())
    def test_causal_flow(self, test_case: FlowTestCase) -> None:
        og = test_case.og
        flow = og.find_causal_flow()

        if test_case.has_causal_flow:
            assert flow is not None
            assert flow.correction_function == test_case.cf
            assert flow.partial_order_layers == test_case.c_layers

            corrections = flow.to_corrections()
            assert corrections.z_corrections == test_case.c_z_corr
            assert corrections.x_corrections == test_case.c_x_corr
        else:
            assert flow is None
