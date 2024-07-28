from __future__ import annotations

import networkx as nx

from .basegraphstate import BaseGraphState
from .rxgraphviews import EdgeList, NodeList

try:
    import rustworkx as rx
    from rustworkx import PyGraph
except ModuleNotFoundError as e:
    msg = "Cannot find rustworkx (optional dependency)."
    raise RuntimeError(msg) from e


class RXGraphState(BaseGraphState):
    """Graph state simulator implemented with rustworkx.
    See :class:`~graphix.graphsim.basegraphstate.BaseGraphState` for more details.
    """

    def __init__(
        self,
        nodes: list[int] | None = None,
        edges: list[tuple[int, int]] | None = None,
        vops: dict[int, int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        nodes : list[int]
            A container of nodes (list, dict, etc)
        edges : list[tuple[int, int]]
            list of tuples (i,j) for pairs to be entangled.
        vops : dict[int, int]
            dict of local Clifford gates with keys for node indices and
            values for Clifford index (see graphix.clifford.CLIFFORD)
        """
        super().__init__()
        self._graph = rx.PyGraph()
        self._nodes = NodeList()
        self._edges = EdgeList()
        if nodes is not None:
            self.add_nodes_from(nodes)
        if edges is not None:
            self.add_edges_from(edges)
        if vops is not None:
            self.apply_vops(vops)

    @property
    def nodes(self) -> NodeList:
        return self._nodes

    @property
    def edges(self) -> EdgeList:
        return self._edges

    @property
    def graph(self) -> rx.PyGraph:
        return self._graph

    def degree(self) -> Iterator[tuple[int, int]]:
        ret = []
        for n in self.nodes:
            nidx = self.nodes.get_node_index(n)
            degree = self._graph.degree(nidx)
            ret.append((n, degree))
        return iter(ret)

    def neighbors(self, node) -> Iterator:
        nidx = self.nodes.get_node_index(node)
        return iter(self._graph.neighbors(nidx))

    def subgraph(self, nodes: list) -> rx.PyGraph:
        nidx = [self.nodes.get_node_index(n) for n in nodes]
        return self._graph.subgraph(nidx)

    def number_of_edges(self, u: int | None = None, v: int | None = None) -> int:
        if u is None and v is None:
            return len(self.edges)
        elif u is None or v is None:
            raise ValueError("u and v must be specified together")
        uidx = self.nodes.get_node_index(u)
        vidx = self.nodes.get_node_index(v)
        return len(self._graph.get_all_edge_data(uidx, vidx))

    def adjacency(self) -> Iterator:
        ret = []
        for n in self.nodes:
            nidx = self.nodes.get_node_index(n)
            adjacency_dict = self._graph.adj(nidx)
            new_adjacency_dict = {}
            for nidx in adjacency_dict.keys():
                new_adjacency_dict[self.nodes.get_node_index(nidx)] = {}  # replace None with {}
            ret.append((n, new_adjacency_dict))
        return iter(ret)

    def remove_node(self, node: int) -> None:
        nidx = self.nodes.get_node_index(node)
        self._graph.remove_node(nidx)
        self.nodes.remove_node(node)
        self.edges.remove_edges_by_node(node)

    def remove_nodes_from(self, nodes: list[int]) -> None:
        for n in nodes:
            self.remove_node(n)

    def remove_edge(self, u: int, v: int) -> None:
        uidx = self.nodes.get_node_index(u)
        vidx = self.nodes.get_node_index(v)
        self._graph.remove_edge(uidx, vidx)
        self.edges.remove_edge((u, v))

    def remove_edges_from(self, edges: list[tuple[int, int]]) -> None:
        for e in edges:
            self.remove_edge(e[0], e[1])

    def add_nodes_from(self, nodes: list[int]) -> None:
        node_indices = self._graph.add_nodes_from([(n, {"loop": False, "sign": False, "hollow": False}) for n in nodes])
        for nidx in node_indices:
            self.nodes.add_node(self._graph[nidx][0], self._graph[nidx][1], nidx)

    def add_edges_from(self, edges) -> None:
        for u, v in edges:
            # adding edges may add new nodes
            if u not in self.nodes:
                nidx = self._graph.add_node((u, {"loop": False, "sign": False, "hollow": False}))
                self.nodes.add_node(self._graph[nidx][0], self._graph[nidx][1], nidx)
            if v not in self.nodes:
                nidx = self._graph.add_node((v, {"loop": False, "sign": False, "hollow": False}))
                self.nodes.add_node(self._graph[nidx][0], self._graph[nidx][1], nidx)
            uidx = self.nodes.get_node_index(u)
            vidx = self.nodes.get_node_index(v)
            eidx = self._graph.add_edge(uidx, vidx, None)
            self.edges.add_edge((self._graph[uidx][0], self._graph[vidx][0]), None, eidx)

    def local_complement(self, node) -> None:
        g = self.subgraph(list(self.neighbors(node)))
        g_new = rx.complement(g)
        g_edge_list = []
        for uidx, vidx in g.edge_list():
            u = g.get_node_data(uidx)[0]
            v = g.get_node_data(vidx)[0]
            g_edge_list.append((u, v))
        g_new_eidx_list = []
        for uidx, vidx in g_new.edge_list():
            u = g_new.get_node_data(uidx)[0]
            v = g_new.get_node_data(vidx)[0]
            g_new_eidx_list.append((u, v))
        self.remove_edges_from(g_edge_list)
        self.add_edges_from(g_new_eidx_list)

    def get_isolates(self) -> list[int]:
        # return list(rx.isolates(self.graph))  # will work with rustworkx>=0.14.0
        return [nnum for nnum, deg in self.degree() if deg == 0]


def convert_rustworkx_to_networkx(graph: PyGraph) -> nx.Graph:
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
    g = nx.Graph()
    for node in node_list:
        g.add_node(node[0])
        for k, v in node[1].items():
            g.nodes[node[0]][k] = v
    for uidx, vidx in edge_list:
        g.add_edge(node_list[uidx][0], node_list[vidx][0])
    return g
