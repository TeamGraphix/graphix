"""Graph state simulator implemented with networkx."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from graphix.graphsim.basegraphstate import BaseGraphState

if TYPE_CHECKING:
    from collections.abc import Iterator

    from networkx.classes.reportviews import EdgeView, NodeView


class NXGraphState(BaseGraphState):
    """Graph state simulator implemented with networkx.

    See :class:`~graphix.graphsim.basegraphstate.BaseGraphState` for more details.
    """

    def __init__(
        self,
        nodes: list[int] | None = None,
        edges: list[tuple[int, int]] | None = None,
        vops: dict[int, int] | None = None,
    ):
        """Instantiate a graph simulator.

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
        """Return the set of nodes."""
        return self._graph.nodes

    @property
    def edges(self) -> EdgeView:
        """Return the set of edges."""
        return self._graph.edges

    @property
    def graph(self) -> nx.Graph:
        """Return the graph itself."""
        return self._graph

    def degree(self) -> Iterator[tuple[int, int]]:
        """Return an iterator for (node, degree) tuples, where degree is the number of edges adjacent to the node."""
        return iter(self._graph.degree())

    def add_nodes_from(self, nodes) -> None:
        """Add nodes and initialize node properties.

        See :meth:`BaseGraphState.add_nodes_from`.
        """
        self._graph.add_nodes_from(nodes, loop=False, sign=False, hollow=False)

    def add_edges_from(self, edges) -> None:
        """Add edges and initialize node properties of newly added nodes.

        See :meth:`BaseGraphState.add_edges_from`.
        """
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
        """Return the number of edges between two nodes.

        See :meth:`BaseGraphState.number_of_edges`.
        """
        if u is None and v is None:
            return len(self.edges)
        elif u is None or v is None:
            raise ValueError("u and v must be specified together")
        return self._graph.number_of_edges(u, v)

    def neighbors(self, node) -> Iterator:
        """Return an iterator over all neighbors of node n.

        See :meth:`BaseGraphState.neighbors`.
        """
        return self._graph.neighbors(node)

    def subgraph(self, nodes: list) -> nx.Graph:
        """Return a subgraph of the graph.

        See :meth:`BaseGraphState.subgraph`.
        """
        return self._graph.subgraph(nodes)

    def remove_node(self, node: int) -> None:
        """Remove a node from the graph.

        See :meth:`BaseGraphState.remove_node`.
        """
        self._graph.remove_node(node)

    def remove_nodes_from(self, nodes: list[int]) -> None:
        """Remove all nodes specified in the list.

        See :meth:`BaseGraphState.remove_nodes_from`.
        """
        self._graph.remove_nodes_from(nodes)

    def remove_edge(self, u: int, v: int) -> None:
        """Remove an edge from the graph.

        See :meth:`BaseGraphState.remove_edge`.
        """
        self._graph.remove_edge(u, v)

    def remove_edges_from(self, edges: list[tuple[int, int]]) -> None:
        """Remove all edges specified in the list.

        See :meth:`BaseGraphState.remove_edges_from`.
        """
        self._graph.remove_edges_from(edges)

    def adjacency(self) -> Iterator:
        """Return an iterator over (node, adjacency dict) tuples for all nodes.

        See :meth:`BaseGraphState.adjacency`.
        """
        return self._graph.adjacency()

    def local_complement(self, node):
        """Perform local complementation of a graph.

        See :meth:`BaseGraphState.local_complement`.
        """
        g = self.subgraph(list(self.neighbors(node)))
        g_new = nx.complement(g)
        self.remove_edges_from(g.edges)
        self.add_edges_from(g_new.edges)

    def get_isolates(self) -> list[int]:
        """Return a list of isolated nodes (nodes with no edges).

        See :meth:`BaseGraphState.get_isolates`.
        """
        return list(nx.isolates(self.graph))
