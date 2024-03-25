from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import networkx as nx
import networkx.classes.reportviews as nx_reportviews

from graphix.clifford import CLIFFORD_HSZ_DECOMPOSITION, CLIFFORD_MUL
from graphix.ops import Ops
from graphix.sim.statevec import Statevec

from .rxgraphviews import EdgeList, NodeList

RUSTWORKX_INSTALLED = False
try:
    import rustworkx as rx
    from rustworkx import PyGraph

    RUSTWORKX_INSTALLED = True
except ModuleNotFoundError:
    rx = None
    PyGraph = None

NodesObject = Union[nx_reportviews.NodeView, NodeList]
EdgesObject = Union[nx_reportviews.EdgeView, EdgeList]
GraphObject = Union[nx.Graph, PyGraph]


class BaseGraphState(ABC):
    """Base class for graph state simulator.

    Performs Pauli measurements on graph states.
    You can choose between networkx and rustworkx as the backend.
    The default is rustworkx if installed, otherwise networkx.

    ref: M. Elliot, B. Eastin & C. Caves, JPhysA 43, 025301 (2010)
    and PRA 77, 042307 (2008)

    Each node has attributes:
        :`hollow`: True if node is hollow (has local H operator)
        :`sign`: True if node has negative sign (local Z operator)
        :`loop`: True if node has loop (local S operator)
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def nodes(self) -> NodesObject:
        raise NotImplementedError

    @property
    @abstractmethod
    def edges(self) -> EdgesObject:
        raise NotImplementedError

    @property
    @abstractmethod
    def graph(self) -> GraphObject:
        raise NotImplementedError

    @abstractmethod
    def degree(self) -> iter[tuple[int, int]]:
        """Returns an iterator for (node, degree) tuples,
        where degree is the number of edges adjacent to the node
        """
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, node: int) -> iter:
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

    def apply_vops(self, vops: dict) -> None:
        """Apply local Clifford operators to the graph state from a dictionary

        Parameters
        ----------
        vops : dict
            dict containing node indices as keys and
            local Clifford indices as values (see graphix.clifford.CLIFFORD)

        Returns
        ----------
        None
        """
        for node, vop in vops.items():
            for lc in reversed(CLIFFORD_HSZ_DECOMPOSITION[vop]):
                if lc == 3:
                    self.z(node)
                elif lc == 6:
                    self.h(node)
                elif lc == 4:
                    self.s(node)

    @abstractmethod
    def add_nodes_from(self, nodes: list[int]) -> None:
        """Add nodes and initialize node properties.

        Parameters
        ----------
        nodes : list[int]
            A list of nodes.

        Returns
        ----------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def add_edges_from(self, edges: list[tuple[int, int]]) -> None:
        """Add edges and initialize node properties of newly added nodes.

        Parameters
        ----------
        edges : list[tuple[int, int]]
            must be given as list of 2-tuples (u, v)

        Returns
        ----------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def get_isolates(self) -> list[int]:
        """Returns a list of isolated nodes (nodes with no edges).

        Returns
        ----------
        list[int]
            A list of isolated nodes.
        """
        raise NotImplementedError

    def get_vops(self) -> dict:
        """Apply local Clifford operators to the graph state from a dictionary

        Parameters
        ----------
            vops : dict
                dict containing node indices as keys and
                local Clifford indices as values (see graphix.clifford.CLIFFORD)
        """
        vops = {}
        for i in self.nodes:
            vop = 0
            if self.nodes[i]["sign"]:
                vop = CLIFFORD_MUL[3, vop]
            if self.nodes[i]["loop"]:
                vop = CLIFFORD_MUL[4, vop]
            if self.nodes[i]["hollow"]:
                vop = CLIFFORD_MUL[6, vop]
            vops[i] = vop
        return vops

    def flip_fill(self, node: int) -> None:
        """Flips the fill (local H) of a node.

        Parameters
        ----------
        node : int
            graph node to flip the fill

        Returns
        ----------
        None
        """
        self.nodes[node]["hollow"] = not self.nodes[node]["hollow"]

    def flip_sign(self, node) -> None:
        """Flips the sign (local Z) of a node.
        Note that application of Z gate is different from `flip_sign`
        if there exist an edge from the node.

        Parameters
        ----------
        node : int
            graph node to flip the sign

        Returns
        ----------
        None
        """
        self.nodes[node]["sign"] = not self.nodes[node]["sign"]

    def advance(self, node: int) -> None:
        """Flips the loop (local S) of a node.
        If the loop already exist, sign is also flipped,
        reflecting the relation SS=Z.
        Note that application of S gate is different from `advance`
        if there exist an edge from the node.

        Parameters
        ----------
        node : int
            graph node to advance the loop.

        Returns
        ----------
        None
        """
        if self.nodes[node]["loop"]:
            self.nodes[node]["loop"] = False
            self.flip_sign(node)
        else:
            self.nodes[node]["loop"] = True

    def h(self, node: int) -> None:
        """Apply H gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply H gate

        Returns
        ----------
        None
        """
        self.flip_fill(node)

    def s(self, node: int) -> None:
        """Apply S gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply S gate

        Returns
        ----------
        None
        """
        if self.nodes[node]["hollow"]:
            if self.nodes[node]["loop"]:
                self.flip_fill(node)
                self.nodes[node]["loop"] = False
                self.local_complement(node)
                for i in self.neighbors(node):
                    self.advance(i)
            else:
                self.local_complement(node)
                for i in self.neighbors(node):
                    self.advance(i)
                if self.nodes[node]["sign"]:
                    for i in self.neighbors(node):
                        self.flip_sign(i)
        else:  # solid
            self.advance(node)

    def z(self, node: int) -> None:
        """Apply Z gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply Z gate

        Returns
        ----------
        None
        """
        if self.nodes[node]["hollow"]:
            for i in self.neighbors(node):
                self.flip_sign(i)
            if self.nodes[node]["loop"]:
                self.flip_sign(node)
        else:  # solid
            self.flip_sign(node)

    def equivalent_graph_E1(self, node: int) -> None:
        """Tranform a graph state to a different graph state
        representing the same stabilizer state.
        This rule applies only to a node with loop.

        Parameters
        ----------
        node1 : int
            A graph node with a loop to apply rule E1

        Returns
        ----------
        None
        """
        if not self.nodes[node]["loop"]:
            raise ValueError("node must have loop")
        self.flip_fill(node)
        self.local_complement(node)
        for i in self.neighbors(node):
            self.advance(i)
        self.flip_sign(node)
        if self.nodes[node]["sign"]:
            for i in self.neighbors(node):
                self.flip_sign(i)

    def equivalent_graph_E2(self, node1: int, node2: int) -> None:
        """Tranform a graph state to a different graph state
        representing the same stabilizer state.
        This rule applies only to two connected nodes without loop.

        Parameters
        ----------
        node1, node2 : int
            connected graph nodes to apply rule E2

        Returns
        ----------
        None
        """
        if (node1, node2) not in self.edges and (node2, node1) not in self.edges:
            raise ValueError("nodes must be connected by an edge")
        if self.nodes[node1]["loop"] or self.nodes[node2]["loop"]:
            raise ValueError("nodes must not have loop")
        sg1 = self.nodes[node1]["sign"]
        sg2 = self.nodes[node2]["sign"]
        self.flip_fill(node1)
        self.flip_fill(node2)
        # local complement along edge between node1, node2
        self.local_complement(node1)
        self.local_complement(node2)
        self.local_complement(node1)
        for i in iter(set(self.neighbors(node1)) & set(self.neighbors(node2))):
            self.flip_sign(i)
        if sg1:
            self.flip_sign(node1)
            for i in self.neighbors(node1):
                self.flip_sign(i)
        if sg2:
            self.flip_sign(node2)
            for i in self.neighbors(node2):
                self.flip_sign(i)

    @abstractmethod
    def local_complement(self, node: int) -> None:
        """Perform local complementation of a graph

        Parameters
        ----------
        node : int
            chosen node for the local complementation

        Returns
        ----------
        None
        """
        raise NotImplementedError

    def equivalent_fill_node(self, node: int) -> int:
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
        if self.nodes[node]["hollow"]:
            if self.nodes[node]["loop"]:
                self.equivalent_graph_E1(node)
                return 0
            else:  # node = hollow and loopless
                if len(list(self.neighbors(node))) == 0:
                    return 1
                for i in self.neighbors(node):
                    if not self.nodes[i]["loop"]:
                        self.equivalent_graph_E2(node, i)
                        return 0
                # if all neighbor has loop, pick one and apply E1, then E1 to the node.
                i = next(self.neighbors(node))
                self.equivalent_graph_E1(i)  # this gives loop to node.
                self.equivalent_graph_E1(node)
                return 0
        else:
            if len(list(self.neighbors(node))) == 0:
                return 2
            else:
                return 0

    def measure_x(self, node: int, choice: int = 0) -> int:
        """perform measurement in X basis
        According to original paper, we realise X measurement by
        applying H gate to the measured node before Z measurement.

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice

        Returns
        ----------
        result : int
            measurement outcome. 0 or 1.
        """
        if choice not in [0, 1]:
            raise ValueError("choice must be 0 or 1")
        # check if isolated
        if len(list(self.neighbors(node))) == 0:
            if self.nodes[node]["hollow"] or self.nodes[node]["loop"]:
                choice_ = choice
            elif self.nodes[node]["sign"]:  # isolated and state is |->
                choice_ = 1
            else:  # isolated and state is |+>
                choice_ = 0
            self.remove_node(node)
            return choice_
        else:
            self.h(node)
            return self.measure_z(node, choice=choice)

    def measure_y(self, node: int, choice: int = 0) -> int:
        """perform measurement in Y basis
        According to original paper, we realise Y measurement by
        applying S,Z and H gate to the measured node before Z measurement.

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice

        Returns
        ----------
        result : int
            measurement outcome. 0 or 1.
        """
        if choice not in [0, 1]:
            raise ValueError("choice must be 0 or 1")
        self.s(node)
        self.z(node)
        self.h(node)
        return self.measure_z(node, choice=choice)

    def measure_z(self, node: int, choice: int = 0) -> int:
        """perform measurement in Z basis
        To realize the simple Z measurement on undecorated graph state,
        we first fill the measured node (remove local H gate)

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice

        Returns
        ----------
        result : int
            measurement outcome. 0 or 1.
        """
        if choice not in [0, 1]:
            raise ValueError("choice must be 0 or 1")
        isolated = self.equivalent_fill_node(node)
        if choice:
            for i in self.neighbors(node):
                self.flip_sign(i)
        if not isolated:
            result = choice
        else:
            result = int(self.nodes[node]["sign"])
        self.remove_node(node)
        return result

    def draw(self, fill_color: str = "C0", **kwargs):
        """Draw decorated graph state.
        Negative nodes are indicated by negative sign of node labels.

        Parameters
        ----------
        fill_color : str, optional
            fill color of nodes
        kwargs : keyword arguments, optional
            additional arguments to supply networkx.draw().
        """
        nqubit = len(self.nodes)
        nodes = list(self.nodes)
        edges = list(self.edges)
        labels = {i: i for i in iter(self.nodes)}
        colors = [fill_color for _ in range(nqubit)]
        for i in range(nqubit):
            if self.nodes[nodes[i]]["loop"]:
                edges.append((nodes[i], nodes[i]))
            if self.nodes[nodes[i]]["hollow"]:
                colors[i] = "white"
            if self.nodes[nodes[i]]["sign"]:
                labels[nodes[i]] = -1 * labels[nodes[i]]
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        nx.draw(g, labels=labels, node_color=colors, edgecolors="k", **kwargs)

    def to_statevector(self) -> Statevec:
        node_list = list(self.nodes)
        nqubit = len(self.nodes)
        gstate = Statevec(nqubit=nqubit)
        # map graph node indices into 0 - (nqubit-1) for qubit indexing in statevec
        imapping = {node_list[i]: i for i in range(nqubit)}
        mapping = [node_list[i] for i in range(nqubit)]
        for i, j in self.edges:
            gstate.entangle((imapping[i], imapping[j]))
        for i in range(nqubit):
            if self.nodes[mapping[i]]["sign"]:
                gstate.evolve_single(Ops.z, i)
        for i in range(nqubit):
            if self.nodes[mapping[i]]["loop"]:
                gstate.evolve_single(Ops.s, i)
        for i in range(nqubit):
            if self.nodes[mapping[i]]["hollow"]:
                gstate.evolve_single(Ops.h, i)
        return gstate
