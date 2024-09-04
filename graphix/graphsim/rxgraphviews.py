"""Node list class for RXGraphState."""

from __future__ import annotations

from typing import Any, Iterator


class NodeList:
    """Node list class for RXGraphState.

    In rustworkx, node data is stored in a tuple (node_num, node_data),
    and adding/removing nodes by node_num is not supported.
    This class defines a node list with node_num as key.
    """

    def __init__(
        self,
        node_nums: list[int] | None = None,
        node_datas: list[dict] | None = None,
        node_indices: list[int] | None = None,
    ):
        """Initialize a node list."""
        if node_indices is None:
            node_indices = []
        if node_datas is None:
            node_datas = []
        if node_nums is None:
            node_nums = []
        if not (len(node_nums) == len(node_datas) and len(node_nums) == len(node_indices)):
            raise ValueError("node_nums, node_datas and node_indices must have the same length")
        self.nodes = set(node_nums)
        self.num_to_data = {nnum: node_datas[nidx] for nidx, nnum in zip(node_indices, node_nums)}
        self.num_to_idx = {nnum: nidx for nidx, nnum in zip(node_indices, node_nums)}

    def __contains__(self, nnum: int) -> bool:
        """Return `True` if the node `nnum` belongs to the list, `False` otherwise."""
        return nnum in self.nodes

    def __getitem__(self, nnum: int) -> Any:
        """Return the data associated to node `nnum`."""
        return self.num_to_data[nnum]

    def __len__(self) -> int:
        """Return the number of nodes."""
        return len(self.nodes)

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over nodes."""
        return iter(self.nodes)

    # TODO: This is not an evaluable __repr__. Define __str__ instead?
    def __repr__(self) -> str:
        """Return a string representation for the node list."""
        return "NodeList" + str(list(self.nodes))

    def get_node_index(self, nnum: int) -> int:
        """Return the index of the node `nnum`."""
        return self.num_to_idx[nnum]

    def add_node(self, nnum: int, ndata: dict, nidx: int) -> None:
        """Add a node to the list."""
        if nnum in self.num_to_data:
            raise ValueError(f"Node {nnum} already exists")
        self.nodes.add(nnum)
        self.num_to_data[nnum] = ndata
        self.num_to_idx[nnum] = nidx

    def add_nodes_from(self, node_nums: list[int], node_datas: list[dict], node_indices: list[int]) -> None:
        """Add nodes to the list."""
        if not (len(node_nums) == len(node_datas) and len(node_nums) == len(node_indices)):
            raise ValueError("node_nums, node_datas and node_indices must have the same length")
        for nnum, ndata, nidx in zip(node_nums, node_datas, node_indices):
            if nnum in self.nodes:
                continue
            self.add_node(nnum, ndata, nidx)

    def remove_node(self, nnum: int) -> None:
        """Remove a node from the list."""
        if nnum not in self.num_to_data:
            raise ValueError(f"Node {nnum} does not exist")
        self.nodes.remove(nnum)
        del self.num_to_data[nnum]
        del self.num_to_idx[nnum]

    def remove_nodes_from(self, node_nums: list[int]) -> None:
        """Remove nodes from the list."""
        for nnum in node_nums:
            if nnum not in self.nodes:
                continue
            self.remove_node(nnum)


class EdgeList:
    """Edge list class for RXGraphState.

    In rustworkx, edge data is stored in a tuple (parent, child, edge_data),
    and adding/removing edges by (parent, child) is not supported.
    This class defines a edge list with (parent, child) as key.
    """

    def __init__(
        self,
        edge_nums: list[tuple[int, int]] | None = None,
        edge_datas: list[dict] | None = None,
        edge_indices: list[int] | None = None,
    ):
        """Initialize an edge list."""
        if edge_indices is None:
            edge_indices = []
        if edge_datas is None:
            edge_datas = []
        if edge_nums is None:
            edge_nums = []
        if not (len(edge_nums) == len(edge_datas) and len(edge_nums) == len(edge_indices)):
            raise ValueError("edge_nums, edge_datas and edge_indices must have the same length")
        self.edges = set(edge_nums)
        self.num_to_data = {enum: edge_datas[eidx] for eidx, enum in zip(edge_indices, edge_nums)}
        self.num_to_idx = {enum: eidx for eidx, enum in zip(edge_indices, edge_nums)}
        self.nnum_to_edges = {}
        for enum in edge_nums:
            if enum[0] not in self.nnum_to_edges:
                self.nnum_to_edges[enum[0]] = set()
            if enum[1] not in self.nnum_to_edges:
                self.nnum_to_edges[enum[1]] = set()
            self.nnum_to_edges[enum[0]].add(enum)
            self.nnum_to_edges[enum[1]].add(enum)

    def __contains__(self, enum: tuple[int, int]) -> bool:
        """Return `True` if the edge `enum` belongs to the list, `False` otherwise."""
        return enum in self.edges

    def __getitem__(self, enum: tuple[int, int]) -> Any:
        """Return the data associated to edge `enum`."""
        return self.num_to_data[enum]

    def __len__(self):
        """Return the number of edges."""
        return len(self.edges)

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over edges."""
        return iter(self.edges)

    # TODO: This is not an evaluable __repr__. Define __str__ instead?
    def __repr__(self) -> str:
        """Return a string representation for the edge list."""
        return "EdgeList" + str(list(self.edges))

    def get_edge_index(self, enum: tuple[int, int]) -> int:
        """Return the index of the edge `enum`."""
        return self.num_to_idx[enum]

    def add_edge(self, enum: tuple[int, int], edata: dict, eidx: int) -> None:
        """Add an edge to the list."""
        if enum in self.num_to_data:
            raise ValueError(f"Edge {enum} already exists")
        self.edges.add(enum)
        self.num_to_data[enum] = edata
        self.num_to_idx[enum] = eidx
        if enum[0] not in self.nnum_to_edges:
            self.nnum_to_edges[enum[0]] = set()
        if enum[1] not in self.nnum_to_edges:
            self.nnum_to_edges[enum[1]] = set()
        self.nnum_to_edges[enum[0]].add(enum)
        self.nnum_to_edges[enum[1]].add(enum)

    def add_edges_from(self, edge_nums: list[tuple[int, int]], edge_datas: list[dict], edge_indices: list[int]) -> None:
        """Add edges to the list."""
        if not (len(edge_nums) == len(edge_datas) and len(edge_nums) == len(edge_indices)):
            raise ValueError("edge_nums, edge_datas and edge_indices must have the same length")
        for enum, edata, eidx in zip(edge_nums, edge_datas, edge_indices):
            if enum in self.edges:
                continue
            self.add_edge(enum, edata, eidx)

    def remove_edge(self, enum: tuple[int, int]) -> None:
        """Remove an edge from the list."""
        if enum not in self.num_to_data:
            raise ValueError(f"Edge {enum} does not exist")
        self.edges.remove(enum)
        del self.num_to_data[enum]
        del self.num_to_idx[enum]
        if enum[0] not in self.nnum_to_edges:
            self.nnum_to_edges[enum[0]] = set()
        if enum[1] not in self.nnum_to_edges:
            self.nnum_to_edges[enum[1]] = set()
        self.nnum_to_edges[enum[0]].remove(enum)
        self.nnum_to_edges[enum[1]].remove(enum)

    def remove_edges_from(self, edge_nums: list[tuple[int, int]]) -> None:
        """Remove edges from the list."""
        for enum in edge_nums:
            if enum not in self.edges:
                continue
            self.remove_edge(enum)

    def remove_edges_by_node(self, nnum: int):
        """Remove all edges connected to the node `nnum`."""
        if nnum in self.nnum_to_edges:
            for enum in self.nnum_to_edges[nnum]:
                self.edges.remove(enum)
                del self.num_to_data[enum]
                del self.num_to_idx[enum]
                if enum[0] == nnum:
                    self.nnum_to_edges[enum[1]].remove(enum)
                else:
                    self.nnum_to_edges[enum[0]].remove(enum)
