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
    meas = {0: Measurement(Plane.XY), 1: Measurement(Plane.XY)}
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
        0: Measurement(Plane.XY),
        1: Measurement(Plane.XY, 1),
        3: Measurement(Plane.YZ, 1),
        4: Measurement(Plane.XY, 1),
    }

    og = OpenGraph(g, meas, inputs, outputs)

    pattern = og.to_pattern()
    og_reconstructed = OpenGraph.from_pattern(pattern)

    assert og.isclose(og_reconstructed)
