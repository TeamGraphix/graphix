"""Graph state simulator implemented with networkx."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx

from graphix.clifford import Clifford
from graphix.ops import Ops
from graphix.sim.statevec import Statevec

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class NXGraphState(nx.Graph[int]):
    """Graph state simulator implemented with networkx.

    Performs Pauli measurements on graph states.

    ref: M. Elliot, B. Eastin & C. Caves, JPhysA 43, 025301 (2010)
    and PRA 77, 042307 (2008)

    Each node has attributes:
        :`hollow`: True if node is hollow (has local H operator)
        :`sign`: True if node has negative sign (local Z operator)
        :`loop`: True if node has loop (local S operator)
    """

    def __init__(
        self,
        nodes: Iterable[int] | None = None,
        edges: Iterable[tuple[int, int]] | None = None,
        vops: Mapping[int, int] | None = None,
    ):
        """Instantiate a graph simulator.

        Parameters
        ----------
        nodes : Iterable[int]
            A container of nodes (list, dict, etc)
        edges : Iterable[tuple[int, int]]
            list of tuples (i,j) for pairs to be entangled.
        vops : Mapping[int, int]
            dict of local Clifford gates with keys for node indices and
            values for Clifford index (see graphix.clifford.CLIFFORD)
        """
        super().__init__()
        if nodes is not None:
            self.add_nodes_from(nodes)
        if edges is not None:
            self.add_edges_from(edges)
        if vops is not None:
            self.apply_vops(vops)

    def local_complement(self, node: int) -> None:
        """Perform local complementation of a graph.

        See :meth:`BaseGraphState.local_complement`.
        """
        g = self.subgraph(self.neighbors(node))
        g_new = nx.complement(g)
        self.remove_edges_from(g.edges)
        self.add_edges_from(g_new.edges)

    def apply_vops(self, vops: Mapping[int, int]) -> None:
        """Apply local Clifford operators to the graph state from a dictionary.

        Parameters
        ----------
        vops : Mapping[int, int]
            dict containing node indices as keys and
            local Clifford indices as values (see graphix.clifford.CLIFFORD)

        Returns
        -------
        None
        """
        for node, vop in vops.items():
            for lc in reversed(Clifford(vop).hsz):
                if lc == Clifford.Z:
                    self.z(node)
                elif lc == Clifford.H:
                    self.h(node)
                elif lc == Clifford.S:
                    self.s(node)

    def get_vops(self) -> dict[int, int]:
        """Apply local Clifford operators to the graph state from a dictionary.

        Parameters
        ----------
            vops : dict
                dict containing node indices as keys and
                local Clifford indices as values (see graphix.clifford.CLIFFORD)
        """
        vops = {}
        for i in self.nodes:
            vop: int = 0
            if self.nodes[i]["sign"]:
                vop = (Clifford.Z @ Clifford(vop)).value
            if self.nodes[i]["loop"]:
                vop = (Clifford.S @ Clifford(vop)).value
            if self.nodes[i]["hollow"]:
                vop = (Clifford.H @ Clifford(vop)).value
            vops[i] = vop
        return vops

    def flip_fill(self, node: int) -> None:
        """Flips the fill (local H) of a node.

        Parameters
        ----------
        node : int
            graph node to flip the fill

        Returns
        -------
        None
        """
        self.nodes[node]["hollow"] = not self.nodes[node]["hollow"]

    def flip_sign(self, node: int) -> None:
        """Flip the sign (local Z) of a node.

        Note that application of Z gate is different from `flip_sign`
        if there exist an edge from the node.

        Parameters
        ----------
        node : int
            graph node to flip the sign

        Returns
        -------
        None
        """
        self.nodes[node]["sign"] = not self.nodes[node]["sign"]

    def advance(self, node: int) -> None:
        """Flip the loop (local S) of a node.

        If the loop already exist, sign is also flipped,
        reflecting the relation SS=Z.
        Note that application of S gate is different from `advance`
        if there exist an edge from the node.

        Parameters
        ----------
        node : int
            graph node to advance the loop.

        Returns
        -------
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
        -------
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
        -------
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
        -------
        None
        """
        if self.nodes[node]["hollow"]:
            for i in self.neighbors(node):
                self.flip_sign(i)
            if self.nodes[node]["loop"]:
                self.flip_sign(node)
        else:  # solid
            self.flip_sign(node)

    def equivalent_graph_e1(self, node: int) -> None:
        """Tranform a graph state to a different graph state representing the same stabilizer state.

        This rule applies only to a node with loop.

        Parameters
        ----------
        node1 : int
            A graph node with a loop to apply rule E1

        Returns
        -------
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

    def equivalent_graph_e2(self, node1: int, node2: int) -> None:
        """Tranform a graph state to a different graph state representing the same stabilizer state.

        This rule applies only to two connected nodes without loop.

        Parameters
        ----------
        node1, node2 : int
            connected graph nodes to apply rule E2

        Returns
        -------
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

    def equivalent_fill_node(self, node: int) -> int:
        """Fill the chosen node by graph transformation rules E1 and E2.

        If the selected node is hollow and isolated, it cannot be filled
        and warning is thrown.

        Parameters
        ----------
        node : int
            node to fill.

        Returns
        -------
        result : int
            if the selected node is hollow and isolated, `result` is 1.
            if filled and isolated, 2.
            otherwise it is 0.
        """
        if self.nodes[node]["hollow"]:
            if self.nodes[node]["loop"]:
                self.equivalent_graph_e1(node)
                return 0
            # node = hollow and loopless
            if len(list(self.neighbors(node))) == 0:
                return 1
            for i in self.neighbors(node):
                if not self.nodes[i]["loop"]:
                    self.equivalent_graph_e2(node, i)
                    return 0
            # if all neighbor has loop, pick one and apply E1, then E1 to the node.
            i = next(self.neighbors(node))
            self.equivalent_graph_e1(i)  # this gives loop to node.
            self.equivalent_graph_e1(node)
            return 0
        if len(list(self.neighbors(node))) == 0:
            return 2
        return 0

    def measure_x(self, node: int, choice: int = 0) -> int:
        """Perform measurement in X basis.

        According to original paper, we realise X measurement by
        applying H gate to the measured node before Z measurement.

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice

        Returns
        -------
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
        self.h(node)
        return self.measure_z(node, choice=choice)

    def measure_y(self, node: int, choice: int = 0) -> int:
        """Perform measurement in Y basis.

        According to original paper, we realise Y measurement by
        applying S,Z and H gate to the measured node before Z measurement.

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice

        Returns
        -------
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
        """Perform measurement in Z basis.

        To realize the simple Z measurement on undecorated graph state,
        we first fill the measured node (remove local H gate)

        Parameters
        ----------
        node : int
            qubit index to be measured
        choice : int, 0 or 1
            choice of measurement outcome. observe (-1)^choice

        Returns
        -------
        result : int
            measurement outcome. 0 or 1.
        """
        if choice not in [0, 1]:
            raise ValueError("choice must be 0 or 1")
        isolated = self.equivalent_fill_node(node)
        if choice:
            for i in self.neighbors(node):
                self.flip_sign(i)
        result = choice if not isolated else int(self.nodes[node]["sign"])
        self.remove_node(node)
        return result

    def draw(self, fill_color: str = "C0", **kwargs: dict[str, Any]) -> None:
        """Draw decorated graph state.

        Negative nodes are indicated by negative sign of node labels.

        Parameters
        ----------
        fill_color : str
            optional, fill color of nodes
        kwargs :
            optional, additional arguments to supply networkx.draw().
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
        g: nx.Graph[int] = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        nx.draw(g, labels=labels, node_color=colors, edgecolors="k", **kwargs)

    def to_statevector(self) -> Statevec:
        """Convert the graph state into a state vector."""
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
                gstate.evolve_single(Ops.Z, i)
        for i in range(nqubit):
            if self.nodes[mapping[i]]["loop"]:
                gstate.evolve_single(Ops.S, i)
        for i in range(nqubit):
            if self.nodes[mapping[i]]["hollow"]:
                gstate.evolve_single(Ops.H, i)
        return gstate
