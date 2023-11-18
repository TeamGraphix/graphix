from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import networkx as nx

RUSTWORKX_INSTALLED = False
try:
    import rustworkx as rx

    RUSTWORKX_INSTALLED = True
except ImportError:
    rx = None

GraphObject: Union[nx.Graph, rx.PyGraph]


class BaseGraphState(ABC):
    """Base class for graph state simulator."""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def nodes(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def edges(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def graph(self):
        raise NotImplementedError

    @abstractmethod
    def degree(self):
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def subgraph(self, nodes: list) -> GraphObject:
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def adjacency(self) -> iter:
        """Returns an iterator over (node, adjacency dict) tuples for all nodes.

        Returns
        ----------
        iter
            An iterator over (node, adjacency dictionary) for all nodes in the graph.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def apply_vops(self, vops: dict):
        """Apply local Clifford operators to the graph state from a dictionary

        Parameters
        ----------
            vops : dict
                dict containing node indices as keys and
                local Clifford indices as values (see graphix.clifford.CLIFFORD)
        """
        raise NotImplementedError

    @abstractmethod
    def add_nodes_from(self, nodes) -> None:
        """Add nodes and initialize node properties.

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, etc)

        Returns
        ----------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def add_edges_from(self, edges) -> None:
        """Add edges and initialize node properties of newly added nodes.

        Parameters
        ----------
        edges : iterable container
            must be given as list of 2-tuples (u, v)

        Returns
        ----------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def get_vops(self) -> dict:
        """Apply local Clifford operators to the graph state from a dictionary

        Parameters
        ----------
            vops : dict
                dict containing node indices as keys and
                local Clifford indices as values (see graphix.clifford.CLIFFORD)
        """
        raise NotImplementedError

    @abstractmethod
    def flip_fill(self, node):
        """Flips the fill (local H) of a node.

        Parameters
        ----------
        node : int
            graph node to flip the fill
        """
        raise NotImplementedError

    @abstractmethod
    def flip_sign(self, node):
        """Flips the sign (local Z) of a node.
        Note that application of Z gate is different from `flip_sign`
        if there exist an edge from the node.

        Parameters
        ----------
        node : int
            graph node to flip the sign
        """
        raise NotImplementedError

    @abstractmethod
    def advance(self, node):
        """Flips the loop (local S) of a node.
        If the loop already exist, sign is also flipped,
        reflecting the relation SS=Z.
        Note that application of S gate is different from `advance`
        if there exist an edge from the node.

        Parameters
        ----------
        node : int
            graph node to advance the loop.
        """
        raise NotImplementedError

    @abstractmethod
    def h(self, node):
        """Apply H gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply H gate
        """
        raise NotImplementedError

    @abstractmethod
    def s(self, node):
        """Apply S gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply S gate
        """
        raise NotImplementedError

    @abstractmethod
    def z(self, node):
        """Apply Z gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply Z gate
        """
        raise NotImplementedError

    @abstractmethod
    def equivalent_graph_E1(self, node):
        """Tranform a graph state to a different graph state
        representing the same stabilizer state.
        This rule applies only to a node with loop.

        Parameters
        ----------
        node1 : int
            A graph node with a loop to apply rule E1
        """
        raise NotImplementedError

    @abstractmethod
    def equivalent_graph_E2(self, node1, node2):
        """Tranform a graph state to a different graph state
        representing the same stabilizer state.
        This rule applies only to two connected nodes without loop.

        Parameters
        ----------
        node1, node2 : int
            connected graph nodes to apply rule E2
        """
        raise NotImplementedError

    @abstractmethod
    def local_complement(self, node):
        """Perform local complementation of a graph

        Parameters
        ----------
        node : int
            chosen node for the local complementation
        """
        raise NotImplementedError

    @abstractmethod
    def equivalent_fill_node(self, node):
        """Fill the chosen node by graph transformation rules E1 and E2,
        If the selected node is hollow and isolated, it cannot be filled
        and warning is thrown.

        Parameters
        ----------
        node : int
            node to fill.

        Returns
        ----------
        result : int
            if the selected node is hollow and isolated, `result` is 1.
            if filled and isolated, 2.
            otherwise it is 0.
        """
        raise NotImplementedError

    @abstractmethod
    def measure_x(self, node, choice=0):
        """perform measurement in X basis
        According to original paper, we realise X measurement by
        applying H gate to the measured node before Z measurement.

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice
        """
        raise NotImplementedError

    @abstractmethod
    def measure_y(self, node, choice=0):
        """perform measurement in Y basis
        According to original paper, we realise Y measurement by
        applying S,Z and H gate to the measured node before Z measurement.

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice
        """
        raise NotImplementedError

    @abstractmethod
    def measure_z(self, node, choice=0):
        """perform measurement in Z basis
        To realize the simple Z measurement on undecorated graph state,
        we first fill the measured node (remove local H gate)

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice
        """
        raise NotImplementedError

    @abstractmethod
    def draw(self, fill_color="C0", **kwargs):
        """Draw decorated graph state.
        Negative nodes are indicated by negative sign of node labels.

        Parameters
        ----------
        fill_color : str, optional
            fill color of nodes
        kwargs : keyword arguments, optional
            additional arguments to supply networkx.draw().
        """
        raise NotImplementedError

    @abstractmethod
    def to_statevector(self):
        raise NotImplementedError
