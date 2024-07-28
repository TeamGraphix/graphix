from __future__ import annotations

import networkx as nx
from networkx.utils import graphs_equal

from .basegraphstate import BaseGraphState
from .nxgraphstate import NXGraphState


def try_to_networkx(g: BaseGraphState) -> nx.Graph:
    if isinstance(g, NXGraphState):
        return g.graph

    from graphix.graphsim import rxgraphstate
    from graphix.graphsim.rxgraphstate import RXGraphState

    if not isinstance(g, RXGraphState):
        msg = f"Unknown graph type {type(g).__name__}"
        raise TypeError(msg)

    return rxgraphstate.convert_rustworkx_to_networkx(g.graph)


def is_graphs_equal(graph1: BaseGraphState, graph2: BaseGraphState) -> bool:
    """Check if graphs are equal.

    Parameters
    ----------
    graph1, graph2 : GraphState

    Returns
    -------
    bool
        True if graphs are equal, False otherwise.
    """
    g1 = try_to_networkx(graph1)
    g2 = try_to_networkx(graph2)
    return graphs_equal(g1, g2)
