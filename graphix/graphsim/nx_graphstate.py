from __future__ import annotations

import networkx as nx

from graphix.clifford import CLIFFORD_HSZ_DECOMPOSITION, CLIFFORD_MUL

from .basegraphstate import BaseGraphState


class NetworkxGraphState(BaseGraphState):
    """Graph state simulator implemented with networkx.

    Performs Pauli measurements on graph states.
    Inherits methods and attributes from networkx.Graph.

    ref: M. Elliot, B. Eastin & C. Caves, JPhysA 43, 025301 (2010)
    and PRA 77, 042307 (2008)

    Each node has attributes:
        :`hollow`: True if node is hollow
        :`sign`: True if node has negative sign
        :`loop`: True if node has loop
    """

    def __init__(self, nodes=None, edges=None, vops=None):
        """
        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, etc)
        edges : list
            list of tuples (i,j) for pairs to be entangled.
        vops : dict
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
    def nodes(self):
        return self._graph.nodes

    @property
    def edges(self):
        return self._graph.edges

    @property
    def graph(self):
        return self._graph

    def degree(self):
        return self._graph.degree()

    def add_nodes_from(self, nodes):
        """Add nodes and initialize node properties.

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, etc)
        """
        self._graph.add_nodes_from(nodes, loop=False, sign=False, hollow=False)

    def add_edges_from(self, edges):
        """Add edges and initialize node properties of newly added nodes.

        Parameters
        ----------
        edges : iterable container
            must be given as list of 2-tuples (u, v)
        """
        self._graph.add_edges_from(edges)
        # adding edges may add new nodes
        for i in self._graph.nodes:
            if "loop" not in self._graph.nodes[i]:
                self._graph.nodes[i]["loop"] = False  # True for having loop
                self._graph.nodes[i]["sign"] = False  # True for minus
                self._graph.nodes[i]["hollow"] = False  # True for hollow node

    def number_of_edges(self, u: int | None = None, v: int | None = None) -> int:
        """Returns the number of edges between two nodes.

        Parameters
        ----------
        u : int, optional
            A node in the graph
        v : int, optional
            A node in the graph

        Returns
        ----------
        int
            The number of edges in the graph. If u and v are specified,
            return the number of edges between those nodes.
        """
        if u is None and v is None:
            return len(self.edges)
        elif u is None or v is None:
            raise ValueError("u and v must be specified together")
        return self._graph.number_of_edges(u, v)

    def neighbors(self, node) -> iter:
        """Returns an iterator over all neighbors of node n.

        Parameters
        ----------
        node : int
            A node in the graph

        Returns
        ----------
        iter
            An iterator over all neighbors of node n.
        """
        return self._graph.neighbors(node)

    def subgraph(self, nodes: list) -> nx.Graph:
        """Returns a subgraph of the graph.

        Parameters
        ----------
        nodes : list
            A list of node indices to generate the subgraph from.

        Returns
        ----------
        GraphObject
            A subgraph of the graph.
        """
        return self._graph.subgraph(nodes)

    def remove_node(self, node: int) -> None:
        """Remove a node from the graph.

        Parameters
        ----------
        node : int
            A node in the graph

        Returns
        ----------
        None
        """
        self._graph.remove_node(node)

    def remove_nodes_from(self, nodes: list[int]) -> None:
        """Remove all nodes specified in the list.

        Parameters
        ----------
        nodes : list
            A list of nodes to remove from the graph.

        Returns
        ----------
        None
        """
        self._graph.remove_nodes_from(nodes)

    def remove_edge(self, u: int, v: int) -> None:
        """Remove an edge from the graph.

        Parameters
        ----------
        u : int
            A node in the graph
        v : int
            A node in the graph

        Returns
        ----------
        None
        """
        self._graph.remove_edge(u, v)

    def remove_edges_from(self, edges: list[tuple[int, int]]) -> None:
        """Remove all edges specified in the list.

        Parameters
        ----------
        edges : list of tuples
            A list of edges to remove from the graph.

        Returns
        ----------
        None
        """
        self._graph.remove_edges_from(edges)

    def adjacency(self) -> iter:
        """Returns an iterator over (node, adjacency dict) tuples for all nodes.

        Returns
        ----------
        iter
            An iterator over (node, adjacency dictionary) for all nodes in the graph.
        """
        return self._graph.adjacency()

    def local_complement(self, node):
        """Perform local complementation of a graph

        Parameters
        ----------
        node : int
            chosen node for the local complementation
        """
        g = self.subgraph(list(self.neighbors(node)))
        g_new = nx.complement(g)
        self.remove_edges_from(g.edges)
        self.add_edges_from(g_new.edges)
