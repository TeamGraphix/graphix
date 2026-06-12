from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from graphix.opengraph import OpenGraph

if TYPE_CHECKING:
    from graphix.measurement import Measurement


def test_empty_graph_well_formed() -> None:
    """Test that empty graphs pass the well-formedness check.

    This is a regression test for issue #531.
    """
    og: OpenGraph[Measurement] = OpenGraph(graph=nx.Graph(), input_nodes=[], output_nodes=[], measurements={})
    pf = og.extract_causal_flow()
    pf.check_well_formed()
    assert True
