from __future__ import annotations

import networkx as nx

from graphix.opengraph import OpenGraph


def test_empty_graph_well_formed() -> None:
    """Test that empty graphs pass the well-formedness check.

    This is a regression test for issue #531.
    """
    og: OpenGraph = OpenGraph(graph=nx.Graph(), input_nodes=[], output_nodes=[], measurements={})
    pf = og.extract_causal_flow()
    pf.check_well_formed()
    assert True
