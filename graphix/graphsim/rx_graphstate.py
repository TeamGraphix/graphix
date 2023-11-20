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
        for n in self.nodes:
            nidx = self.nodes.get_node_index(n)
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
        nidx = self.nodes.get_node_index(node)
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
        nidx = [self.nodes.get_node_index(n) for n in nodes]
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
            return len(self.edges)
        elif u is None or v is None:
            raise ValueError("u and v must be specified together")
        uidx = self.nodes.get_node_index(u)
        vidx = self.nodes.get_node_index(v)
        return len(self._graph.get_all_edge_data(uidx, vidx))

    def adjacency(self) -> iter:
        """Returns an iterator over (node, adjacency dict) tuples for all nodes.

        Returns
        ----------
        iter
            An iterator over (node, adjacency dictionary) for all nodes in the graph.
        """
        ret = []
        for n in self.nodes:
            nidx = self.nodes.get_node_index(n)
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
        nidx = self.nodes.get_node_index(node)
        self._graph.remove_node(nidx)
        self.nodes.remove_node(node)
        edge_list = list(self.edges)
        for e in edge_list:
            if node in e:
                self.edges.remove_edge(e)

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
        for n in nodes:
            self.remove_node(n)

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
        uidx = self.nodes.get_node_index(u)
        vidx = self.nodes.get_node_index(v)
        self._graph.remove_edge(uidx, vidx)
        self.edges.remove_edge((u, v))

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
        for e in edges:
            self.remove_edge(e[0], e[1])

    def add_nodes_from(self, nodes: list[int]):
        """Add nodes and initialize node properties.

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, etc)
        """
        node_indices = self._graph.add_nodes_from([(n, {"loop": False, "sign": False, "hollow": False}) for n in nodes])
        for nidx in node_indices:
            self.nodes.add_node(self._graph[nidx][0], self._graph[nidx][1], nidx)

    def add_edges_from(self, edges):
        """Add edges and initialize node properties of newly added nodes.

        Parameters
        ----------
        edges : iterable container
            must be given as list of 2-tuples (u, v)
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
            self._edges.add_edge((self._graph[uidx][0], self._graph[vidx][0]), None, eidx)

    def local_complement(self, node):
        """Perform local complementation of a graph

        Parameters
        ----------
        node : int
            chosen node for the local complementation
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
