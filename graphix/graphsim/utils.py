from networkx import Graph
from networkx.utils import graphs_equal

from .graphstate import RUSTWORKX_INSTALLED

if RUSTWORKX_INSTALLED:
    from rustworkx import PyGraph
else:
    PyGraph = None

from .basegraphstate import BaseGraphState
from .nx_graphstate import NetworkxGraphState
from .rx_graphstate import RustworkxGraphState


def convert_rustworkx_to_networkx(graph: PyGraph) -> Graph:
    """Convert a rustworkx PyGraph to a networkx graph."""
    if not isinstance(graph, PyGraph):
        raise TypeError("graph must be a rustworkx PyGraph")
    node_list = list(graph.nodes())
    edge_list = list(graph.edge_list())
    g = Graph()
    for node in node_list:
        g.add_node(node[0])
        for k, v in node[1].items():
            g.nodes[node[0]][k] = v
    for uidx, vidx in edge_list:
        g.add_edge(node_list[uidx][0], node_list[vidx][0])
    return g


def is_graph_equal(graph1: BaseGraphState, graph2: BaseGraphState) -> bool:
    """Check if graphs are equal.

    Parameters
    ----------
    graph1, graph2 : GraphState

    Returns
    -------
    bool
        True if graphs are equal, False otherwise.
    """
    if isinstance(graph1, RustworkxGraphState):
        graph1 = convert_rustworkx_to_networkx(graph1.graph)
    elif isinstance(graph1, NetworkxGraphState):
        graph1 = graph1.graph
    else:
        raise TypeError(f"Unknown graph type {type(graph1)}")
    if isinstance(graph2, RustworkxGraphState):
        graph2 = convert_rustworkx_to_networkx(graph2.graph)
    elif isinstance(graph2, NetworkxGraphState):
        graph2 = graph2.graph
    else:
        raise TypeError(f"Unknown graph type {type(graph2)}")

    return graphs_equal(graph1, graph2)
