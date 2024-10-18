"""Graph state simulator implemented with rustworkx."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix.graphsim.basegraphstate import RUSTWORKX_INSTALLED, BaseGraphState
from graphix.graphsim.rxgraphviews import EdgeList, NodeList

if TYPE_CHECKING:
    from collections.abc import Iterator

if RUSTWORKX_INSTALLED:
    import rustworkx as rx
else:
    rx = None


class RXGraphState(BaseGraphState):
    """Graph state simulator implemented with rustworkx.

    See :class:`~graphix.graphsim.basegraphstate.BaseGraphState` for more details.
    """

    def __init__(
        self,
        nodes: list[int] | None = None,
        edges: list[tuple[int, int]] | None = None,
        vops: dict[int, int] | None = None,
    ):
        """Initialize graph state simulator.

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
        """Return the set of nodes."""
        return self._nodes

    @property
    def edges(self) -> EdgeList:
        """Return the set of edges."""
        return self._edges

    @property
    def graph(self) -> rx.PyGraph:
        """Return the graph itself."""
        return self._graph

    def degree(self) -> Iterator[tuple[int, int]]:
        """Return an iterator for (node, degree) tuples, where degree is the number of edges adjacent to the node."""
        ret = []
        for n in self.nodes:
            nidx = self.nodes.get_node_index(n)
            degree = self._graph.degree(nidx)
            ret.append((n, degree))
        return iter(ret)

    def neighbors(self, node) -> Iterator[int]:
        """Return an iterator over all neighbors of node n.

        See :meth:`BaseGraphState.neighbors`.
        """
        nidx = self.nodes.get_node_index(node)
        return (self.nodes.idx_to_num[idx] for idx in self._graph.neighbors(nidx))

    def subgraph(self, nodes: list) -> rx.PyGraph:
        """Return a subgraph of the graph.

        See :meth:`BaseGraphState.subgraph`.
        """
        nidx = [self.nodes.get_node_index(n) for n in nodes]
        return self._graph.subgraph(nidx)

    def number_of_edges(self, u: int | None = None, v: int | None = None) -> int:
        """Return the number of edges between two nodes.

        See :meth:`BaseGraphState.number_of_edges`.
        """
        if u is None and v is None:
            return len(self.edges)
        elif u is None or v is None:
            raise ValueError("u and v must be specified together")
        uidx = self.nodes.get_node_index(u)
        vidx = self.nodes.get_node_index(v)
        return len(self._graph.get_all_edge_data(uidx, vidx))

    def adjacency(self) -> Iterator:
        """Return an iterator over (node, adjacency dict) tuples for all nodes.

        See :meth:`BaseGraphState.adjacency`.
        """
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
        """Remove a node from the graph.

        See :meth:`BaseGraphState.remove_node`.
        """
        nidx = self.nodes.get_node_index(node)
        self._graph.remove_node(nidx)
        self.nodes.remove_node(node)
        self.edges.remove_edges_by_node(node)

    def remove_nodes_from(self, nodes: list[int]) -> None:
        """Remove all nodes specified in the list.

        See :meth:`BaseGraphState.remove_nodes_from`.
        """
        for n in nodes:
            self.remove_node(n)

    def remove_edge(self, u: int, v: int) -> None:
        """Remove an edge from the graph.

        See :meth:`BaseGraphState.remove_edge`.
        """
        uidx = self.nodes.get_node_index(u)
        vidx = self.nodes.get_node_index(v)
        self._graph.remove_edge(uidx, vidx)
        self.edges.remove_edge((u, v))

    def remove_edges_from(self, edges: list[tuple[int, int]]) -> None:
        """Remove all edges specified in the list.

        See :meth:`BaseGraphState.remove_edges_from`.
        """
        for e in edges:
            self.remove_edge(e[0], e[1])

    def add_nodes_from(self, nodes: list[int]):
        """Add nodes and initialize node properties.

        See :meth:`BaseGraphState.add_nodes_from`.
        """
        node_indices = self._graph.add_nodes_from([(n, {"loop": False, "sign": False, "hollow": False}) for n in nodes])
        for nidx in node_indices:
            self.nodes.add_node(self._graph[nidx][0], self._graph[nidx][1], nidx)

    def add_edges_from(self, edges):
        """Add edges and initialize node properties of newly added nodes.

        See :meth:`BaseGraphState.add_edges_from`.
        """
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

    def local_complement(self, node):
        """Perform local complementation of a graph.

        See :meth:`BaseGraphState.local_complement`.
        """
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
        """Return a list of isolated nodes (nodes with no edges).

        See :meth:`BaseGraphState.get_isolates`.
        """
        # return list(rx.isolates(self.graph))  # will work with rustworkx>=0.14.0
        return [nnum for nnum, deg in self.degree() if deg == 0]
