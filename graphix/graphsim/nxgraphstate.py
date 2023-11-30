from __future__ import annotations

import networkx as nx
from networkx.classes.reportviews import EdgeView, NodeView

from .basegraphstate import BaseGraphState


class NXGraphState(BaseGraphState):
    """Graph state simulator implemented with networkx.
    See :class:`~graphix.graphsim.basegraphstate.BaseGraphState` for more details.
    """

    def __init__(self, nodes: list[int] = None, edges: list[tuple[int, int]] = None, vops: dict[int, int] = None):
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
        self._graph = nx.Graph()
        if nodes is not None:
            self.add_nodes_from(nodes)
        if edges is not None:
            self.add_edges_from(edges)
        if vops is not None:
            self.apply_vops(vops)

    @property
    def nodes(self) -> NodeView:
        return self._graph.nodes

    @property
    def edges(self) -> EdgeView:
        return self._graph.edges

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def degree(self) -> iter[tuple[int, int]]:
        return iter(self._graph.degree())

    def add_nodes_from(self, nodes):
        self._graph.add_nodes_from(nodes, loop=False, sign=False, hollow=False)

    def add_edges_from(self, edges):
        self._graph.add_edges_from(edges)
        # adding edges may add new nodes
        for u, v in edges:
            if u not in self._graph.nodes:
                self._graph.nodes[u]["loop"] = False
                self._graph.nodes[u]["sign"] = False
                self._graph.nodes[u]["hollow"] = False
            if v not in self._graph.nodes:
                self._graph.nodes[v]["loop"] = False
                self._graph.nodes[v]["sign"] = False
                self._graph.nodes[v]["hollow"] = False

    def number_of_edges(self, u: int | None = None, v: int | None = None) -> int:
        if u is None and v is None:
            return len(self.edges)
        elif u is None or v is None:
            raise ValueError("u and v must be specified together")
        return self._graph.number_of_edges(u, v)

    def neighbors(self, node) -> iter:
        return self._graph.neighbors(node)

    def subgraph(self, nodes: list) -> nx.Graph:
        return self._graph.subgraph(nodes)

    def remove_node(self, node: int) -> None:
        self._graph.remove_node(node)

    def remove_nodes_from(self, nodes: list[int]) -> None:
        self._graph.remove_nodes_from(nodes)

    def remove_edge(self, u: int, v: int) -> None:
        self._graph.remove_edge(u, v)

    def remove_edges_from(self, edges: list[tuple[int, int]]) -> None:
        self._graph.remove_edges_from(edges)

    def adjacency(self) -> iter:
        return self._graph.adjacency()

    def local_complement(self, node):
        g = self.subgraph(list(self.neighbors(node)))
        g_new = nx.complement(g)
        self.remove_edges_from(g.edges)
        self.add_edges_from(g_new.edges)

    def get_isolates(self) -> list[int]:
        return list(nx.isolates(self.graph))
