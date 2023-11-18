from __future__ import annotations

import networkx as nx

from graphix.clifford import CLIFFORD_HSZ_DECOMPOSITION, CLIFFORD_MUL
from graphix.ops import Ops
from graphix.sim.statevec import Statevec

from .basegraphstate import BaseGraphState
from .graphstate import RUSTWORKX_INSTALLED

if RUSTWORKX_INSTALLED:
    import rustworkx as rx
else:
    rx = None


class NodeList:
    """Node list class for RustworkxGraphState
    In rustworkx, node data is stored in a tuple (node_num, node_data),
    and adding/removing nodes by node_num is not supported.
    This class defines a node list with node_num as key.
    """

    def __init__(self, node_nums: list[int] = [], node_datas: list[dict] = [], node_indices: list[int] = []):
        if not (len(node_nums) == len(node_datas) and len(node_nums) == len(node_indices)):
            raise ValueError("node_nums, node_datas and node_indices must have the same length")
        self.nodes = set(node_nums)
        self.num_to_data = {nnum: node_datas[nidx] for nidx, nnum in zip(node_indices, node_nums)}
        self.num_to_idx = {nnum: nidx for nidx, nnum in zip(node_indices, node_nums)}

    def __getitem__(self, nnum: int):
        return self.num_to_data[nnum]

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __repr__(self) -> str:
        return "NodeList" + str(list(self.nodes))

    def get_node_index(self, nnum: int):
        return self.num_to_idx[nnum]

    def add_node(self, nnum: int, ndata: dict, nidx: int):
        if nnum in self.num_to_data:
            raise ValueError(f"Node {nnum} already exists")
        self.nodes.add(nnum)
        self.num_to_data[nnum] = ndata
        self.num_to_idx[nnum] = nidx

    def add_nodes_from(self, node_nums: list[int], node_datas: list[dict], node_indices: list[int]):
        if not (len(node_nums) == len(node_datas) and len(node_nums) == len(node_indices)):
            raise ValueError("node_nums, node_datas and node_indices must have the same length")
        for nnum in node_nums:
            if nnum in self.num_to_data:
                raise ValueError(f"Node {nnum} already exists")
        for nnum, ndata, nidx in zip(node_nums, node_datas, node_indices):
            self.add_node(nnum, ndata, nidx)

    def remove_node(self, nnum: int):
        if nnum not in self.num_to_data:
            raise ValueError(f"Node {nnum} does not exist")
        self.nodes.remove(nnum)
        del self.num_to_data[nnum]
        del self.num_to_idx[nnum]

    def remove_nodes_from(self, node_nums: list[int]):
        for nnum in node_nums:
            if nnum not in self.num_to_data:
                raise ValueError(f"Node {nnum} does not exist")
        for nnum in node_nums:
            self.remove_node(nnum)


class EdgeList:
    """Edge list class for RustworkxGraphState
    In rustworkx, edge data is stored in a tuple (parent, child, edge_data),
    and adding/removing edges by (parent, child) is not supported.
    This class defines a edge list with (parent, child) as key.
    """

    def __init__(
        self, edge_nums: list[tuple[int, int]] = [], edge_datas: list[dict] = [], edge_indices: list[int] = []
    ):
        if not (len(edge_nums) == len(edge_datas) and len(edge_nums) == len(edge_indices)):
            raise ValueError("edge_nums, edge_datas and edge_indices must have the same length")
        self.edges = set(edge_nums)
        self.num_to_data = {enum: edge_datas[eidx] for eidx, enum in zip(edge_indices, edge_nums)}
        self.num_to_idx = {enum: eidx for eidx, enum in zip(edge_indices, edge_nums)}

    def __getitem__(self, enum: tuple[int, int]):
        return self.num_to_data[enum]

    def __len__(self):
        return len(self.edges)

    def __iter__(self):
        return iter(self.edges)

    def __repr__(self) -> str:
        return "EdgeList" + str(list(self.edges))

    def get_edge_index(self, enum: tuple[int, int]):
        return self.num_to_idx[enum]

    def add_edge(self, enum: tuple[int, int], edata: dict, eidx: int):
        if enum in self.num_to_data:
            raise ValueError(f"Edge {enum} already exists")
        self.edges.add(enum)
        self.num_to_data[enum] = edata
        self.num_to_idx[enum] = eidx

    def add_edges_from(self, edge_nums: list[tuple[int, int]], edge_datas: list[dict], edge_indices: list[int]):
        if not (len(edge_nums) == len(edge_datas) and len(edge_nums) == len(edge_indices)):
            raise ValueError("edge_nums, edge_datas and edge_indices must have the same length")
        for enum in edge_nums:
            if enum in self.num_to_data:
                raise ValueError(f"Edge {enum} already exists")
        for enum, edata, eidx in zip(edge_nums, edge_datas, edge_indices):
            self.add_edge(enum, edata, eidx)

    def remove_edge(self, enum: tuple[int, int]):
        if enum not in self.num_to_data:
            raise ValueError(f"Edge {enum} does not exist")
        self.edges.remove(enum)
        del self.num_to_data[enum]
        del self.num_to_idx[enum]

    def remove_edges_from(self, edge_nums: list[tuple[int, int]]):
        for enum in edge_nums:
            if enum not in self.num_to_data:
                raise ValueError(f"Edge {enum} does not exist")
        for enum in edge_nums:
            self.remove_edge(enum)


class RustworkxGraphState(BaseGraphState):
    """Graph state simulator implemented with rustworkx"""

    def __init__(self, nodes=None, edges=None, vops=None):
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
    def graph(self):
        return self._graph

    def degree(self):
        ret = []
        for n in self._nodes:
            nidx = self._nodes.get_node_index(n)
            degree = self._graph.degree(nidx)
            ret.append((n, degree))
        return iter(ret)

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
        nidx = self._nodes.get_node_index(node)
        return iter(self._graph.neighbors(nidx))

    def subgraph(self, nodes: list) -> rx.PyGraph:
        """Returns a subgraph of the graph.

        Parameters
        ----------
        nodes : list
            A list of node indices to generate the subgraph from.

        Returns
        ----------
        PyGraph
            A subgraph of the graph.
        """
        nidx = [self._nodes.get_node_index(n) for n in nodes]
        return self._graph.subgraph(nidx)

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
            return len(self._edges)
        elif u is None or v is None:
            raise ValueError("u and v must be specified together")
        uidx = self._nodes.get_node_index(u)
        vidx = self._nodes.get_node_index(v)
        return len(self._graph.get_all_edge_data(uidx, vidx))

    def adjacency(self) -> iter:
        """Returns an iterator over (node, adjacency dict) tuples for all nodes.

        Returns
        ----------
        iter
            An iterator over (node, adjacency dictionary) for all nodes in the graph.
        """
        ret = []
        for n in self._nodes:
            nidx = self._nodes.get_node_index(n)
            adjacencies = self._graph.adj(nidx)
            ret.append((n, adjacencies))
        return iter(ret)

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
        nidx = self._nodes.get_node_index(node)
        self._graph.remove_node(nidx)
        self._nodes.remove_node(node)

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
        self._nodes.remove_nodes_from(nodes)
        for n in nodes:
            nidx = self._nodes.get_node_index(n)
            self._graph.remove_node(nidx)

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
        uidx = self._nodes.get_node_index(u)
        vidx = self._nodes.get_node_index(v)
        self._graph.remove_edge(uidx, vidx)
        self._edges.remove_edge((u, v))

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
        self._edges.remove_edges_from(edges)
        for u, v in edges:
            uidx = self._nodes.get_node_index(u)
            vidx = self._nodes.get_node_index(v)
            self._graph.remove_edge(uidx, vidx)

    def apply_vops(self, vops: dict):
        """Apply local Clifford operators to the graph state from a dictionary

        Parameters
        ----------
            vops : dict
                dict containing node indices as keys and
                local Clifford indices as values (see graphix.clifford.CLIFFORD)
        """
        for node, vop in vops.items():
            for lc in reversed(CLIFFORD_HSZ_DECOMPOSITION[vop]):
                if lc == 3:
                    self.z(node)
                elif lc == 6:
                    self.h(node)
                elif lc == 4:
                    self.s(node)

    def add_nodes_from(self, nodes: list[int]):
        """Add nodes and initialize node properties.

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, etc)
        """
        node_indices = self._graph.add_nodes_from([(n, {"loop": False, "sign": False, "hollow": False}) for n in nodes])
        self._nodes.add_nodes_from(nodes, [{"loop": False, "sign": False, "hollow": False}] * len(nodes), node_indices)

    def add_edges_from(self, edges):
        """Add edges and initialize node properties of newly added nodes.

        Parameters
        ----------
        edges : iterable container
            must be given as list of 2-tuples (u, v)
        """
        for u, v in edges:
            # adding edges may add new nodes
            if u not in self._nodes:
                eidx = self._graph.add_node((u, {"loop": False, "sign": False, "hollow": False}))
                self._nodes.add_node(u, {"loop": False, "sign": False, "hollow": False}, eidx)
            if v not in self._nodes:
                eidx = self._graph.add_node((v, {"loop": False, "sign": False, "hollow": False}))
                self._nodes.add_node(v, {"loop": False, "sign": False, "hollow": False}, eidx)
            uidx = self._nodes.get_node_index(u)
            vidx = self._nodes.get_node_index(v)
            eidx = self._graph.add_edge(uidx, vidx, None)
            self._edges.add_edge((u, v), None, eidx)

    def get_vops(self):
        """Returns a dict containing clifford labels for each nodes.
        labels 0 - 23 specify one of 24 single-qubit Clifford gates.
        see graphq.clifford for the definition of all unitaries.

        Returns
        ----------
        vops : dict
            clifford gate indices as defined in `graphq.clifford`.


        .. seealso:: :mod:`graphix.clifford`
        """
        vops = {}
        for nidx in self._graph.node_indexes():
            vop = 0
            if self._graph[nidx][1]["sign"]:
                vop = CLIFFORD_MUL[3, vop]
            if self._graph[nidx][1]["loop"]:
                vop = CLIFFORD_MUL[4, vop]
            if self._graph[nidx][1]["hollow"]:
                vop = CLIFFORD_MUL[6, vop]
            vops[self._graph[nidx][0]] = vop
        return vops

    def flip_fill(self, node):
        """Flips the fill (local H) of a node.

        Parameters
        ----------
        node : int
            graph node to flip the fill
        """
        nidx = self._nodes.get_node_index(node)
        self._nodes[node]["hollow"] = not self._nodes[node]["hollow"]
        self._graph[nidx][1]["hollow"] = not self._graph[nidx][1]["hollow"]

    def flip_sign(self, node):
        """Flips the sign (local Z) of a node.
        Note that application of Z gate is different from `flip_sign`
        if there exist an edge from the node.

        Parameters
        ----------
        node : int
            graph node to flip the sign
        """
        nidx = self._nodes.get_node_index(node)
        self._nodes[node]["sign"] = not self._nodes[node]["sign"]
        self._graph[nidx][1]["sign"] = not self._graph[nidx][1]["sign"]

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
        nidx = self._nodes.get_node_index(node)
        if self._graph[nidx][1]["loop"]:
            self._graph[nidx][1]["loop"] = False
            self._nodes[node]["loop"] = False
            self.flip_sign(node)
        else:
            self._graph[nidx][1]["loop"] = True
            self._nodes[node]["loop"] = True

    def h(self, node):
        """Apply H gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply H gate
        """
        self.flip_fill(node)

    def s(self, node):
        """Apply S gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply S gate
        """
        nidx = self._nodes.get_node_index(node)
        if self._graph[nidx][1]["hollow"]:
            if self._graph[nidx][1]["loop"]:
                self.flip_fill(node)
                self._graph[nidx][1]["loop"] = False
                self._nodes[node]["loop"] = False
                self.local_complement(node)
                for i in self.neighbors(node):
                    self.advance(i)
            else:
                self.local_complement(node)
                for i in self.neighbors(node):
                    self.advance(i)
                if self._graph[nidx][1]["sign"]:
                    for i in self.neighbors(node):
                        self.flip_sign(i)
        else:  # solid
            self.advance(node)

    def z(self, node):
        """Apply Z gate to a qubit (node).

        Parameters
        ----------
        node : int
            graph node to apply Z gate
        """
        nidx = self._nodes.get_node_index(node)
        if self._graph[nidx][1]["hollow"]:
            for i in self.neighbors(node):
                self.flip_sign(i)
            if self._graph[nidx][1]["loop"]:
                self.flip_sign(node)
        else:  # solid
            self.flip_sign(node)

    def equivalent_graph_E1(self, node):
        """Tranform a graph state to a different graph state
        representing the same stabilizer state.
        This rule applies only to a node with loop.

        Parameters
        ----------
        node1 : int
            A graph node with a loop to apply rule E1
        """
        nidx = self._nodes.get_node_index(node)
        if not self._graph[nidx][1]["loop"]:
            raise ValueError("node must have loop")
        self.flip_fill(node)
        self.local_complement(node)
        for i in self.neighbors(node):
            self.advance(i)
        self.flip_sign(node)
        if self._graph[nidx][1]["sign"]:
            for i in self.neighbors(node):
                self.flip_sign(i)

    def equivalent_graph_E2(self, node1, node2):
        """Tranform a graph state to a different graph state
        representing the same stabilizer state.
        This rule applies only to two connected nodes without loop.

        Parameters
        ----------
        node1, node2 : int
            connected graph nodes to apply rule E2
        """
        if (node1, node2) not in list(self._edges) and (node2, node1) not in list(self._edges):
            raise ValueError("nodes must be connected by an edge")
        nidx1 = self._nodes.get_node_index(node1)
        nidx2 = self._nodes.get_node_index(node2)
        if self._graph[nidx1][1]["loop"] or self._graph[nidx2][1]["loop"]:
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

    def local_complement(self, node):
        """Perform local complementation of a graph

        Parameters
        ----------
        node : int
            chosen node for the local complementation
        """
        g = self.subgraph(list(self.neighbors(node))).copy()
        g_new = rx.complement(g).copy()
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
        self.s(node)
        self.z(node)
        self.h(node)
        return self.measure_z(node, choice=choice)

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
        nqubit = len(self.nodes)
        nodes = list(self.nodes)
        edges = list(self.edges)
        labels = {i: i for i in iter(self.nodes)}
        colors = [fill_color for i in range(nqubit)]
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

    def to_statevector(self):
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
