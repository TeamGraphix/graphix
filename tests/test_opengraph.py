"""Unit tests for the class `:class: graphix.opengraph.OpenGraph`.

This module tests the full conversion Open Graph -> Flow -> XZ-corrections -> Pattern for all three classes of flow.
Output correctness is verified by checking if the resulting pattern is deterministic (when the flow exists).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
import pytest

from graphix.command import E
from graphix.fundamentals import Axis, Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph, OpenGraphError, _M_co
from graphix.pattern import Pattern
from graphix.random_objects import rand_circuit
from graphix.states import PlanarState

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.random import Generator

    from graphix.fundamentals import AbstractMeasurement


class OpenGraphFlowTestCase(NamedTuple):
    og: OpenGraph[Measurement]
    has_cflow: bool
    has_gflow: bool
    has_pflow: bool


OPEN_GRAPH_FLOW_TEST_CASES: list[OpenGraphFlowTestCase] = []


def register_open_graph_flow_test_case(
    test_case: Callable[[], OpenGraphFlowTestCase],
) -> Callable[[], OpenGraphFlowTestCase]:
    OPEN_GRAPH_FLOW_TEST_CASES.append(test_case())
    return test_case


@register_open_graph_flow_test_case
def _og_0() -> OpenGraphFlowTestCase:
    """Generate open graph.

    Structure:

    [(0)]-[(1)]
    """
    meas: dict[int, Measurement] = {}
    og = OpenGraph(
        graph=nx.Graph([(0, 1)]),
        input_nodes=[0, 1],
        output_nodes=[0, 1],
        measurements=meas,
    )
    return OpenGraphFlowTestCase(og, has_cflow=True, has_gflow=True, has_pflow=True)


@register_open_graph_flow_test_case
def _og_1() -> OpenGraphFlowTestCase:
    """Generate open graph.

    Structure:

    [0]-1-20-30-4-(5)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 20), (20, 30), (30, 4), (4, 5)]),
        input_nodes=[0],
        output_nodes=[5],
        measurements={
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.2, Plane.XY),
            20: Measurement(0.3, Plane.XY),
            30: Measurement(0.4, Plane.XY),
            4: Measurement(0.5, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=True, has_gflow=True, has_pflow=True)


@register_open_graph_flow_test_case
def _og_2() -> OpenGraphFlowTestCase:
    """Generate open graph.

    Structure:

    [1]-3-(5)
        |
    [2]-4-(6)
    """
    og = OpenGraph(
        graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
        input_nodes=[1, 2],
        output_nodes=[5, 6],
        measurements={
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.2, Plane.XY),
            3: Measurement(0.3, Plane.XY),
            4: Measurement(0.4, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=True, has_gflow=True, has_pflow=True)


@register_open_graph_flow_test_case
def _og_3() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [1]-(4)
      \ /
       X____
      /     |
    [2]-(5) |
      \ /   |
       X    |
      / \  /
    [3]-(6)
    """
    og = OpenGraph(
        graph=nx.Graph([(1, 4), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)]),
        input_nodes=[1, 2, 3],
        output_nodes=[4, 5, 6],
        measurements={
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.2, Plane.XY),
            3: Measurement(0.3, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=True, has_pflow=True)


@register_open_graph_flow_test_case
def _og_4() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

       [0]-[1]
       /|   |
     (4)|   |
       \|   |
        2--(5)-3
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 2), (0, 4), (1, 5), (2, 4), (2, 5), (3, 5)]),
        input_nodes=[0, 1],
        output_nodes=[4, 5],
        measurements={
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.2, Plane.XZ),
            3: Measurement(0.3, Plane.YZ),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=True, has_pflow=True)


@register_open_graph_flow_test_case
def _og_5() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [1]-(3)
      \ /
       X
      / \
    [2]-(4)
    """
    og = OpenGraph(
        graph=nx.Graph([(1, 3), (1, 4), (2, 3), (2, 4)]),
        input_nodes=[1, 2],
        output_nodes=[3, 4],
        measurements={
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.1, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


@register_open_graph_flow_test_case
def _og_6() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]
     |
     1-2-3
     |
    (4)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (1, 4), (2, 3)]),
        input_nodes=[0],
        output_nodes=[4],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0, Plane.XY),  # X
            2: Measurement(0.1, Plane.XY),  # XY
            3: Measurement(0, Plane.XY),  # X
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=True)


@register_open_graph_flow_test_case
def _og_7() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-1-(2)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2)]),
        input_nodes=[0],
        output_nodes=[2],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.5, Plane.YZ),  # Y
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=True)


@register_open_graph_flow_test_case
def _og_8() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-2-4-(6)
        | |
    [1]-3-5-(7)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)]),
        input_nodes=[1, 0],
        output_nodes=[6, 7],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XZ),  # XZ
            2: Measurement(0.5, Plane.XZ),  # X
            3: Measurement(0.5, Plane.YZ),  # Y
            4: Measurement(0.5, Plane.YZ),  # Y
            5: Measurement(0.1, Plane.YZ),  # YZ
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


@register_open_graph_flow_test_case
def _og_9() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-2-4-(6)
        | |
    [1]-3-5-(7)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)]),
        input_nodes=[0, 1],
        output_nodes=[6, 7],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XY),  # XY
            2: Measurement(0.0, Plane.XY),  # X
            3: Measurement(0.1, Plane.XY),  # XY
            4: Measurement(0.0, Plane.XY),  # X
            5: Measurement(0.5, Plane.XY),  # Y
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=True, has_gflow=True, has_pflow=True)


@register_open_graph_flow_test_case
def _og_10() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

          3-(5)
         /| \
    [0]-2-4-(6)
        | | /|
        | 1  |
        |____|

    Notes
    -----
    Example from Fig. 1 in Mitosek and Backens, 2024 (arXiv:2410.23439).
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (2, 4), (3, 4), (4, 6), (1, 4), (1, 6), (2, 3), (3, 5), (2, 6), (3, 6)]),
        input_nodes=[0],
        output_nodes=[5, 6],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XZ),  # XZ
            2: Measurement(0.5, Plane.YZ),  # Y
            3: Measurement(0.1, Plane.XY),  # XY
            4: Measurement(0, Plane.XZ),  # Z
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=True)


@register_open_graph_flow_test_case
def _og_11() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-2--3--4-7-(8)
        |  |  |
       (6)[1](5)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 6), (3, 4), (4, 7), (4, 5), (7, 8)]),
        input_nodes=[0, 1],
        output_nodes=[5, 6, 8],
        measurements={
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.0, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.5, Plane.XY),
            7: Measurement(0, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=True, has_gflow=True, has_pflow=True)


@register_open_graph_flow_test_case
def _og_12() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-2-(3)-(4)
        |
      [(1)]
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 2), (2, 3), (3, 4)]),
        input_nodes=[0, 1],
        output_nodes=[1, 3, 4],
        measurements={0: Measurement(0.1, Plane.XY), 2: Measurement(0.5, Plane.YZ)},
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=True)


@register_open_graph_flow_test_case
def _og_13() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

          0-[1]
          |  |
    5-(2)-3--4-(7)
          |
         (6)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7)]),
        input_nodes=[1],
        output_nodes=[6, 2, 7],
        measurements={
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.YZ),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=True, has_pflow=True)


@register_open_graph_flow_test_case
def _og_14() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    0-(1)  (4)-6
    |  |
    2-(3)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (4, 6)]),
        input_nodes=[],
        output_nodes=[1, 3, 4],
        measurements={0: Measurement(0.5, Plane.XZ), 2: Measurement(0, Plane.YZ), 6: Measurement(0.2, Plane.XY)},
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=True, has_pflow=True)


@register_open_graph_flow_test_case
def _og_15() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [1]--0
     |   |
     4---3-(2)
     |   |  |
    (7)-(6)-5
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7), (5, 6), (6, 7)]),
        input_nodes=[1],
        output_nodes=[6, 2, 7],
        measurements={
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


@register_open_graph_flow_test_case
def _og_16() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-(1)  (4)-6
     |   |
     2--(3)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (4, 6)]),
        input_nodes=[0],
        output_nodes=[1, 3, 4],
        measurements={0: Measurement(0.1, Plane.XZ), 2: Measurement(0, Plane.YZ), 6: Measurement(0.2, Plane.XY)},
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


@register_open_graph_flow_test_case
def _og_17() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

          0-[1]     [8]
          |  |
    5-(2)-3--4-(7)
          |
         (6)

    Notes
    -----
    Graph is constructed by adding a disconnected input to OG 13.
    """
    graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7)])
    graph.add_node(8)
    og = OpenGraph(
        graph=graph,
        input_nodes=[1, 8],
        output_nodes=[6, 2, 7],
        measurements={
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.YZ),
            8: Measurement(0.1, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


@register_open_graph_flow_test_case
def _og_18() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-2--3--4-7-(8)
        |  |  |
       (6)[1](5)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 6), (3, 4), (4, 7), (4, 5), (7, 8)]),
        input_nodes=[0, 1],
        output_nodes=[5, 6, 8],
        measurements={
            0: Measurement(0, Plane.XY),
            1: Measurement(0, Plane.XY),
            2: Measurement(0, Plane.XZ),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.5, Plane.XY),
            7: Measurement(0, Plane.YZ),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


@register_open_graph_flow_test_case
def _og_19() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    0-2--3--4-7-(8)
      |  |  |
     (6) 1 (5)

    Notes
    -----
    Even though any node is measured in the XY plane, OG has Pauli flow because none of them are inputs.
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 6), (3, 4), (4, 7), (4, 5), (7, 8)]),
        input_nodes=[],
        output_nodes=[5, 6, 8],
        measurements={
            0: Measurement(0, Plane.XZ),
            1: Measurement(0, Plane.XZ),
            2: Measurement(0, Plane.XZ),
            3: Measurement(0, Plane.XZ),
            4: Measurement(0, Plane.XZ),
            7: Measurement(0, Plane.XZ),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=True)


class OpenGraphComposeTestCase(NamedTuple):
    og1: OpenGraph[AbstractMeasurement]
    og2: OpenGraph[AbstractMeasurement]
    og_ref: OpenGraph[AbstractMeasurement]
    mapping: dict[int, int]
    comparison_method: Callable[..., bool] = (
        OpenGraph.__eq__
    )  # Replace by `OpenGraph.isclose` if `OpenGraph` is of type `Measurement`.


# Parallel composition
def _compose_0() -> OpenGraphComposeTestCase:
    """Generate composition test.

    Graph 1
    [1] -- (2)

    Graph 2 = Graph 1

    Mapping: 1 -> 100, 2 -> 200

    Expected graph
     [1]  --  (2)

    [100] -- (200)
    """
    g: nx.Graph[int] = nx.Graph([(1, 2)])
    inputs = [1]
    outputs = [2]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og1 = OpenGraph(g, inputs, outputs, meas)
    og2 = OpenGraph(g, inputs, outputs, meas)
    og_ref = OpenGraph(
        nx.Graph([(1, 2), (100, 200)]),
        input_nodes=[1, 100],
        output_nodes=[2, 200],
        measurements={1: Measurement(0, Plane.XY), 100: Measurement(0, Plane.XY)},
    )

    mapping = {1: 100, 2: 200}

    return OpenGraphComposeTestCase(og1, og2, og_ref, mapping, OpenGraph.isclose)


# Series composition
def _compose_1() -> OpenGraphComposeTestCase:
    """Generate composition test.

    Graph 1
    [0] -- 17 -- (23)
           |
    [3] -- 4  -- (13)

    Graph 2
    [6] -- 17 -- (1)
     |     |
    [7] -- 4  -- (2)

    Mapping: 6 -> 23, 7 -> 13, 1 -> 100, 2 -> 200, 17 -> 90

    Expected graph
    [0] -- 17 -- 23 -- 90 -- (100)
           |     |     |
    [3] -- 4  -- 13 -- 201 -- (200)
    """
    g: nx.Graph[int] = nx.Graph([(0, 17), (17, 23), (17, 4), (3, 4), (4, 13)])
    inputs = [0, 3]
    outputs = [13, 23]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og1 = OpenGraph(g, inputs, outputs, meas)

    g = nx.Graph([(6, 7), (6, 17), (17, 1), (7, 4), (17, 4), (4, 2)])
    inputs = [6, 7]
    outputs = [1, 2]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og2 = OpenGraph(g, inputs, outputs, meas)

    mapping = {6: 23, 7: 13, 1: 100, 2: 200, 17: 90}

    g = nx.Graph(
        [(0, 17), (17, 23), (17, 4), (3, 4), (4, 13), (23, 13), (23, 90), (13, 201), (90, 201), (90, 100), (201, 200)]
    )
    inputs = [0, 3]
    outputs = [100, 200]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_ref = OpenGraph(g, inputs, outputs, meas)

    return OpenGraphComposeTestCase(og1, og2, og_ref, mapping, OpenGraph.isclose)


# Full overlap
def _compose_2() -> OpenGraphComposeTestCase:
    """Generate composition test.

    Graph 1
    [0] -- 17 -- (23)
           |
    [3] -- 4  -- (13)

    Graph 2 = Graph 1

    Mapping: 0 -> 0, 3 -> 3, 17 -> 17, 4 -> 4, 23 -> 23, 13 -> 13

    Expected graph = Graph 1
    """
    g: nx.Graph[int]
    g = nx.Graph([(0, 17), (17, 23), (17, 4), (3, 4), (4, 13)])
    inputs = [0, 3]
    outputs = [13, 23]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og1 = OpenGraph(g, inputs, outputs, meas)
    og2 = OpenGraph(g, inputs, outputs, meas)
    og_ref = OpenGraph(g, inputs, outputs, meas)

    mapping = {i: i for i in g.nodes}

    return OpenGraphComposeTestCase(og1, og2, og_ref, mapping, OpenGraph.isclose)


# Overlap inputs/outputs
def _compose_3() -> OpenGraphComposeTestCase:
    """Generate composition test.

    Graph 1
    ([17]) -- (3)
      |
     [18]

    Graph 2
    [1] -- 2 -- (3)

    Mapping: 1 -> 17, 3 -> 300

    Expected graph
    (300) -- 301 -- [17] -- (3)
                     |
                    [18]
    """
    g: nx.Graph[int] = nx.Graph([(18, 17), (17, 3)])
    inputs = [17, 18]
    outputs = [3, 17]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og1 = OpenGraph(g, inputs, outputs, meas)

    g = nx.Graph([(1, 2), (2, 3)])
    inputs = [1]
    outputs = [3]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og2 = OpenGraph(g, inputs, outputs, meas)

    mapping = {1: 17, 3: 300}

    g = nx.Graph([(18, 17), (17, 3), (17, 301), (301, 300)])
    inputs = [17, 18]  # the input character of node 17 is kept because node 1 (in G2) is an input.
    outputs = [3, 300]  # the output character of node 17 is lost because node 1 (in G2) is not an output
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_ref = OpenGraph(g, inputs, outputs, meas)

    return OpenGraphComposeTestCase(og1, og2, og_ref, mapping, OpenGraph.isclose)


# Inverse series composition
def _compose_4() -> OpenGraphComposeTestCase:
    """Generate composition test.

    Graph 1
    [1] -- (2)
     |
    [3]

    Graph 2
    [3] -- (4)

    Mapping: 4 -> 1, 3 -> 300

    Expected graph
    [300] -- 1 -- (2)
             |
            [3]
    """
    g: nx.Graph[int] = nx.Graph([(1, 2), (1, 3)])
    inputs = [1, 3]
    outputs = [2]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og1 = OpenGraph(g, inputs, outputs, meas)

    g = nx.Graph([(3, 4)])
    inputs = [3]
    outputs = [4]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og2 = OpenGraph(g, inputs, outputs, meas)

    mapping = {4: 1, 3: 300}

    g = nx.Graph([(1, 2), (1, 3), (1, 300)])
    inputs = [3, 300]
    outputs = [2]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_ref = OpenGraph(g, inputs, outputs, meas)

    return OpenGraphComposeTestCase(og1, og2, og_ref, mapping, OpenGraph.isclose)


def _compose_5() -> OpenGraphComposeTestCase:
    """Generate composition test.

    Graph 1
    [1] -- (2)

    Graph 2 = Graph 1

    Mapping: 1 -> 2

    Expected graph
    [1] -- 2 -- (3)

    """
    g: nx.Graph[int] = nx.Graph([(1, 2)])
    inputs = [1]
    outputs = [2]
    meas = dict.fromkeys(g.nodes - set(outputs), Plane.XY)
    og1 = OpenGraph(g, inputs, outputs, meas)
    og2 = OpenGraph(g, inputs, outputs, meas)
    og_ref = OpenGraph(
        nx.Graph([(1, 2), (2, 3)]), input_nodes=[1], output_nodes=[3], measurements={1: Plane.XY, 2: Plane.XY}
    )

    mapping = {1: 2}

    return OpenGraphComposeTestCase(og1, og2, og_ref, mapping)


def prepare_test_og_compose() -> list[OpenGraphComposeTestCase]:
    n_og_samples = 6
    test_cases: list[OpenGraphComposeTestCase] = [globals()[f"_compose_{i}"]() for i in range(n_og_samples)]

    return test_cases


def check_determinism(pattern: Pattern, fx_rng: Generator, n_shots: int = 3) -> bool:
    """Verify if the input pattern is deterministic."""
    for plane in {Plane.XY, Plane.XZ, Plane.YZ}:
        alpha = 2 * np.pi * fx_rng.random()
        state_ref = pattern.simulate_pattern(input_state=PlanarState(plane, alpha))

        for _ in range(n_shots):
            state = pattern.simulate_pattern(input_state=PlanarState(plane, alpha))
            result = np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten()))

            if result:
                continue
            return False

    return True


class TestOpenGraph:
    def test_odd_neighbors(self) -> None:
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (1, 3), (1, 2), (2, 3), (1, 4)])
        og = OpenGraph(graph=graph, input_nodes=[0], output_nodes=[1, 2, 3, 4], measurements={0: Plane.XY})

        assert og.odd_neighbors([0]) == {1, 2}
        assert og.odd_neighbors([0, 1]) == {0, 1, 3, 4}
        assert og.odd_neighbors([1, 2, 3]) == {4}
        assert og.odd_neighbors([]) == set()

    def test_neighbors(self) -> None:
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (1, 3), (1, 2), (2, 3), (1, 4)])
        og = OpenGraph(graph=graph, input_nodes=[0], output_nodes=[1, 2, 3, 4], measurements={0: Plane.XY})

        assert og.neighbors([4]) == {1}
        assert og.neighbors([0, 1]) == {0, 1, 2, 3, 4}
        assert og.neighbors([1, 2, 3]) == {0, 1, 2, 3, 4}
        assert og.neighbors([]) == set()

    @pytest.mark.parametrize("test_case", OPEN_GRAPH_FLOW_TEST_CASES)
    def test_cflow(self, test_case: OpenGraphFlowTestCase, fx_rng: Generator) -> None:
        og = test_case.og

        if test_case.has_cflow:
            pattern = og.extract_causal_flow().to_corrections().to_pattern()
            assert check_determinism(pattern, fx_rng)
        else:
            with pytest.raises(OpenGraphError, match=r"The open graph does not have a causal flow."):
                og.extract_causal_flow()

    @pytest.mark.parametrize("test_case", OPEN_GRAPH_FLOW_TEST_CASES)
    def test_gflow(self, test_case: OpenGraphFlowTestCase, fx_rng: Generator) -> None:
        og = test_case.og

        if test_case.has_gflow:
            pattern = og.extract_gflow().to_corrections().to_pattern()
            assert check_determinism(pattern, fx_rng)
        else:
            with pytest.raises(OpenGraphError, match=r"The open graph does not have a gflow."):
                og.extract_gflow()

    @pytest.mark.parametrize("test_case", OPEN_GRAPH_FLOW_TEST_CASES)
    def test_pflow(self, test_case: OpenGraphFlowTestCase, fx_rng: Generator) -> None:
        og = test_case.og

        if test_case.has_pflow:
            pattern = og.extract_pauli_flow().to_corrections().to_pattern()
            assert check_determinism(pattern, fx_rng)
        else:
            with pytest.raises(OpenGraphError, match=r"The open graph does not have a Pauli flow."):
                og.extract_pauli_flow()

    def test_double_entanglement(self) -> None:
        pattern = Pattern(input_nodes=[0, 1], cmds=[E((0, 1)), E((0, 1))])
        pattern2 = pattern.extract_opengraph().to_pattern()
        state = pattern.simulate_pattern()
        state2 = pattern2.simulate_pattern()
        assert np.abs(np.dot(state.flatten().conjugate(), state2.flatten())) == pytest.approx(1)

    def test_from_to_pattern(self, fx_rng: Generator) -> None:
        n_qubits = 2
        depth = 2
        circuit = rand_circuit(n_qubits, depth, fx_rng)
        pattern_ref = circuit.transpile().pattern
        pattern = pattern_ref.extract_opengraph().to_pattern()

        for plane in {Plane.XY, Plane.XZ, Plane.YZ}:
            alpha = 2 * np.pi * fx_rng.random()
            state_ref = pattern_ref.simulate_pattern(input_state=PlanarState(plane, alpha))
            state = pattern.simulate_pattern(input_state=PlanarState(plane, alpha))
            assert np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

<<<<<<< HEAD
    @pytest.mark.parametrize("test_case", prepare_test_og_compose())
    def test_compose(self, test_case: OpenGraphComposeTestCase) -> None:
        og1, og2, og_ref, mapping, compare = test_case

        og, mapping_complete = og1.compose(og2, mapping)
=======
    @pytest.mark.parametrize(
        "test_case",
        [
            (
                OpenGraph(
                    graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
                    input_nodes=[0],
                    output_nodes=[3],
                    measurements=dict.fromkeys(range(3), Plane.XY),
                ),
                OpenGraph(
                    graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
                    input_nodes=[0],
                    output_nodes=[3],
                    measurements=dict.fromkeys(range(3), Axis.X),
                ),
            ),
            (
                OpenGraph(
                    graph=nx.Graph([(0, 1)]),
                    input_nodes=[0],
                    output_nodes=[],
                    measurements=dict.fromkeys(range(2), Plane.XY),
                ),
                OpenGraph(
                    graph=nx.Graph([(0, 1)]),
                    input_nodes=[0, 1],
                    output_nodes=[],
                    measurements=dict.fromkeys(range(2), Plane.XY),
                ),
            ),
            (
                OpenGraph(
                    graph=nx.Graph([(0, 1)]),
                    input_nodes=[0],
                    output_nodes=[1],
                    measurements={0: Measurement(0.6, Plane.XY)},
                ),
                OpenGraph(
                    graph=nx.Graph([(0, 2)]),
                    input_nodes=[0],
                    output_nodes=[2],
                    measurements={0: Measurement(0.6, Plane.XY)},
                ),
            ),
        ],
    )
    def test_eq(self, test_case: Sequence[tuple[OpenGraph[_M_co], OpenGraph[_M_co]]]) -> None:
        og_1, og_2 = test_case
        assert og_1 == og_1  # noqa: PLR0124
        assert og_1 != og_2
        assert og_2 == og_2  # noqa: PLR0124

    def test_isclose(self) -> None:
        og_1 = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
            input_nodes=[0],
            output_nodes=[3],
            measurements=dict.fromkeys(range(3), Measurement(0.1, Plane.XY)),
        )
        og_2 = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
            input_nodes=[0],
            output_nodes=[3],
            measurements=dict.fromkeys(range(3), Measurement(0.15, Plane.XY)),
        )
        assert og_1.isclose(og_2, abs_tol=0.1)
        assert not og_1.isclose(og_2)


# TODO: rewrite as parametric tests
>>>>>>> rf_og_compare

        assert compare(og, og_ref)
        assert mapping.keys() <= mapping_complete.keys()
        assert set(mapping.values()) <= set(mapping_complete.values())

    def test_compose_exception(self) -> None:
        g: nx.Graph[int] = nx.Graph([(0, 1)])
        inputs = [0]
        outputs = [1]
        mapping = {0: 0, 1: 1}

        og1 = OpenGraph(g, inputs, outputs, measurements={0: Measurement(0, Plane.XY)})
        og2 = OpenGraph(g, inputs, outputs, measurements={0: Measurement(0.5, Plane.XY)})

        with pytest.raises(
            OpenGraphError,
            match=re.escape(
                "Attempted to merge nodes with different measurements: (0, Measurement(angle=0.5, plane=Plane.XY)) -> (0, Measurement(angle=0, plane=Plane.XY))."
            ),
        ):
            og1.compose(og2, mapping)

        og3 = OpenGraph(g, inputs, outputs, measurements={0: Plane.XY})
        og4 = OpenGraph(g, inputs, outputs, measurements={0: Plane.XZ})

        with pytest.raises(
            OpenGraphError,
            match=re.escape("Attempted to merge nodes with different measurements: (0, Plane.XZ) -> (0, Plane.XY)."),
        ):
            og3.compose(og4, mapping)
