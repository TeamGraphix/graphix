from networkx import Graph
from networkx.utils import graphs_equal

from .graphstate import RUSTWORKX_INSTALLED

if RUSTWORKX_INSTALLED:
    from rustworkx import PyGraph
else:
    PyGraph = None

from .basegraphstate import BaseGraphState
from .nxgraphstate import NXGraphState
from .rxgraphstate import RXGraphState


def convert_rustworkx_to_networkx(graph: PyGraph) -> Graph:
    """Convert a rustworkx PyGraph to a networkx graph.

    .. caution::
        The node in the rustworkx graph must be a tuple of the form (node_num, node_data),
        where node_num is an integer and node_data is a dictionary of node data.
    """
    if not isinstance(graph, PyGraph):
        raise TypeError("graph must be a rustworkx PyGraph")
    node_list = graph.nodes()
    if not all(
        isinstance(node, tuple) and len(node) == 2 and (int(node[0]) == node[0]) and isinstance(node[1], dict)
        for node in node_list
    ):
        raise TypeError("All the nodes in the graph must be tuple[int, dict]")
    edge_list = list(graph.edge_list())
    g = Graph()
    for node in node_list:
        g.add_node(node[0])
        for k, v in node[1].items():
            g.nodes[node[0]][k] = v
    for uidx, vidx in edge_list:
        g.add_edge(node_list[uidx][0], node_list[vidx][0])
    return g


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
    if isinstance(graph1, RXGraphState):
        graph1 = convert_rustworkx_to_networkx(graph1.graph)
    elif isinstance(graph1, NXGraphState):
        graph1 = graph1.graph
    else:
        raise TypeError(f"Unknown graph type {type(graph1)}")
    if isinstance(graph2, RXGraphState):
        graph2 = convert_rustworkx_to_networkx(graph2.graph)
    elif isinstance(graph2, NXGraphState):
        graph2 = graph2.graph
    else:
        raise TypeError(f"Unknown graph type {type(graph2)}")

    return graphs_equal(graph1, graph2)
