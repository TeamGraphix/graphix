"""Test empty graph fix."""

from __future__ import annotations

from typing import Any

import networkx as nx

from graphix.opengraph import OpenGraph


def test_empty_graph_well_formed() -> None:

    og: OpenGraph[Any] = OpenGraph(graph=nx.Graph(), input_nodes=[], output_nodes=[], measurements={})
    pf = og.extract_causal_flow()
    pf.check_well_formed()
    assert True
