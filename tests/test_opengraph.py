from __future__ import annotations

import networkx as nx

from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph


# Tests whether an open graph can be converted to and from a pattern and be
# successfully reconstructed.
def test_open_graph_to_pattern() -> None:
    g: nx.Graph[int]
    g = nx.Graph([(0, 1), (1, 2)])
    inputs = [0]
    outputs = [2]
    meas = {0: Measurement(0, Plane.XY), 1: Measurement(0, Plane.XY)}
    og = OpenGraph(g, meas, inputs, outputs)

    pattern = og.to_pattern()
    og_reconstructed = OpenGraph.from_pattern(pattern)

    assert og.isclose(og_reconstructed)

    # 0 -- 1 -- 2
    #      |
    # 3 -- 4 -- 5
    g = nx.Graph([(0, 1), (1, 2), (1, 4), (3, 4), (4, 5)])
    inputs = [0, 3]
    outputs = [2, 5]
    meas = {
        0: Measurement(0, Plane.XY),
        1: Measurement(1.0, Plane.XY),
        3: Measurement(0.5, Plane.YZ),
        4: Measurement(1.0, Plane.XY),
    }

    og = OpenGraph(g, meas, inputs, outputs)

    pattern = og.to_pattern()
    og_reconstructed = OpenGraph.from_pattern(pattern)

    assert og.isclose(og_reconstructed)


# Tests composition of two graphs


# Parallel composition
def test_compose_1() -> None:
    # Graph 1
    # [1] -- (2)
    #
    # Graph 2 = Graph 1
    #
    # Mapping: 1 -> 100, 2 -> 200
    #
    # Expected graph
    #  [1]  --  (2)
    #
    # [100] -- (200)

    g: nx.Graph[int]
    g = nx.Graph([(1, 2)])
    inputs = [1]
    outputs = [2]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_1 = OpenGraph(g, meas, inputs, outputs)

    mapping = {1: 100, 2: 200}

    og, mapping_complete = og_1.compose(og_1, mapping)

    expected_graph: nx.Graph[int]
    expected_graph = nx.Graph([(1, 2), (100, 200)])
    assert nx.is_isomorphic(og.inside, expected_graph)
    assert og.inputs == [1, 100]
    assert og.outputs == [2, 200]

    outputs_c = {i for i in og.inside.nodes if i not in og.outputs}
    assert og.measurements.keys() == outputs_c
    assert mapping.keys() <= mapping_complete.keys()
    assert set(mapping.values()) <= set(mapping_complete.values())


# Series composition
def test_compose_2() -> None:
    # Graph 1
    # [0] -- 17 -- (23)
    #        |
    # [3] -- 4  -- (13)
    #
    # Graph 2
    # [6] -- 17 -- (1)
    #  |     |
    # [7] -- 4  -- (2)
    #
    # Mapping: 6 -> 23, 7 -> 13, 1 -> 100, 2 -> 200
    #
    # Expected graph
    # [0] -- 17 -- 23 -- o -- (100)
    #        |     |     |
    # [3] -- 4  -- 13 -- o -- (200)

    g: nx.Graph[int]
    g = nx.Graph([(0, 17), (17, 23), (17, 4), (3, 4), (4, 13)])
    inputs = [0, 3]
    outputs = [13, 23]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_1 = OpenGraph(g, meas, inputs, outputs)

    g = nx.Graph([(6, 7), (6, 17), (17, 1), (7, 4), (17, 4), (4, 2)])
    inputs = [6, 7]
    outputs = [1, 2]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_2 = OpenGraph(g, meas, inputs, outputs)

    mapping = {6: 23, 7: 13, 1: 100, 2: 200}

    og, mapping_complete = og_1.compose(og_2, mapping)

    expected_graph: nx.Graph[int]
    expected_graph = nx.Graph(
        [(0, 17), (17, 23), (17, 4), (3, 4), (4, 13), (23, 13), (23, 1), (13, 2), (1, 2), (1, 100), (2, 200)]
    )
    assert nx.is_isomorphic(og.inside, expected_graph)
    assert og.inputs == [0, 3]
    assert og.outputs == [100, 200]

    outputs_c = {i for i in og.inside.nodes if i not in og.outputs}
    assert og.measurements.keys() == outputs_c
    assert mapping.keys() <= mapping_complete.keys()
    assert set(mapping.values()) <= set(mapping_complete.values())


# Full overlap
def test_compose_3() -> None:
    # Graph 1
    # [0] -- 17 -- (23)
    #        |
    # [3] -- 4  -- (13)
    #
    # Graph 2 = Graph 1
    #
    # Mapping: 0 -> 0, 3 -> 3, 17 -> 17, 4 -> 4, 23 -> 23, 13 -> 13
    #
    # Expected graph = Graph 1

    g: nx.Graph[int]
    g = nx.Graph([(0, 17), (17, 23), (17, 4), (3, 4), (4, 13)])
    inputs = [0, 3]
    outputs = [13, 23]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_1 = OpenGraph(g, meas, inputs, outputs)

    mapping = {i: i for i in g.nodes}

    og, mapping_complete = og_1.compose(og_1, mapping)

    assert og.isclose(og_1)
    assert mapping.keys() <= mapping_complete.keys()
    assert set(mapping.values()) <= set(mapping_complete.values())


# Overlap inputs/outputs
def test_compose_4() -> None:
    # Graph 1
    # ([17]) -- (3)
    #   |
    #  [18]
    #
    # Graph 2
    # [1] -- 2 -- (3)
    #
    # Mapping: 1 -> 17, 3 -> 300
    #
    # Expected graph
    # (300) -- 2 -- [17] -- (3)
    #                |
    #               [18]

    g: nx.Graph[int]
    g = nx.Graph([(18, 17), (17, 3)])
    inputs = [17, 18]
    outputs = [3, 17]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_1 = OpenGraph(g, meas, inputs, outputs)

    g = nx.Graph([(1, 2), (2, 3)])
    inputs = [1]
    outputs = [3]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_2 = OpenGraph(g, meas, inputs, outputs)

    mapping = {1: 17, 3: 300}

    og, mapping_complete = og_1.compose(og_2, mapping)

    expected_graph: nx.Graph[int]
    expected_graph = nx.Graph([(18, 17), (17, 3), (17, 2), (2, 300)])
    assert nx.is_isomorphic(og.inside, expected_graph)
    assert og.inputs == [17, 18]  # the input character of node 17 is kept because node 1 (in G2) is an input
    assert og.outputs == [3, 300]  # the output character of node 17 is lost because node 1 (in G2) is not an output

    outputs_c = {i for i in og.inside.nodes if i not in og.outputs}
    assert og.measurements.keys() == outputs_c
    assert mapping.keys() <= mapping_complete.keys()
    assert set(mapping.values()) <= set(mapping_complete.values())


# Inverse series composition
def test_compose_5() -> None:
    # Graph 1
    # [1] -- (2)
    #  |
    # [3]
    #
    # Graph 2
    # [3] -- (4)
    #
    # Mapping: 4 -> 1, 3 -> 300
    #
    # Expected graph
    # [300] -- 1 -- (2)
    #          |
    #         [3]

    g: nx.Graph[int]
    g = nx.Graph([(1, 2), (1, 3)])
    inputs = [1, 3]
    outputs = [2]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_1 = OpenGraph(g, meas, inputs, outputs)

    g = nx.Graph([(3, 4)])
    inputs = [3]
    outputs = [4]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_2 = OpenGraph(g, meas, inputs, outputs)

    mapping = {4: 1, 3: 300}

    og, mapping_complete = og_1.compose(og_2, mapping)

    expected_graph: nx.Graph[int]
    expected_graph = nx.Graph([(1, 2), (1, 3), (1, 300)])
    assert nx.is_isomorphic(og.inside, expected_graph)
    assert og.inputs == [3, 300]
    assert og.outputs == [2]

    outputs_c = {i for i in og.inside.nodes if i not in og.outputs}
    assert og.measurements.keys() == outputs_c
    assert mapping.keys() <= mapping_complete.keys()
    assert set(mapping.values()) <= set(mapping_complete.values())
