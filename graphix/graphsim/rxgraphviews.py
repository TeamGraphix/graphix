from __future__ import annotations


class NodeList:
    """Node list class for RXGraphState
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

    def __contains__(self, nnum: int):
        return nnum in self.nodes

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
        for nnum, ndata, nidx in zip(node_nums, node_datas, node_indices):
            if nnum in self.nodes:
                continue
            self.add_node(nnum, ndata, nidx)

    def remove_node(self, nnum: int):
        if nnum not in self.num_to_data:
            raise ValueError(f"Node {nnum} does not exist")
        self.nodes.remove(nnum)
        del self.num_to_data[nnum]
        del self.num_to_idx[nnum]

    def remove_nodes_from(self, node_nums: list[int]):
        for nnum in node_nums:
            if nnum not in self.nodes:
                continue
            self.remove_node(nnum)


class EdgeList:
    """Edge list class for RXGraphState
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
        self.nnum_to_edges = {}
        for enum in edge_nums:
            if enum[0] not in self.nnum_to_edges:
                self.nnum_to_edges[enum[0]] = set()
            if enum[1] not in self.nnum_to_edges:
                self.nnum_to_edges[enum[1]] = set()
            self.nnum_to_edges[enum[0]].add(enum)
            self.nnum_to_edges[enum[1]].add(enum)

    def __contains__(self, enum: tuple[int, int]):
        return enum in self.edges

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
        if enum[0] not in self.nnum_to_edges:
            self.nnum_to_edges[enum[0]] = set()
        if enum[1] not in self.nnum_to_edges:
            self.nnum_to_edges[enum[1]] = set()
        self.nnum_to_edges[enum[0]].add(enum)
        self.nnum_to_edges[enum[1]].add(enum)

    def add_edges_from(self, edge_nums: list[tuple[int, int]], edge_datas: list[dict], edge_indices: list[int]):
        if not (len(edge_nums) == len(edge_datas) and len(edge_nums) == len(edge_indices)):
            raise ValueError("edge_nums, edge_datas and edge_indices must have the same length")
        for enum, edata, eidx in zip(edge_nums, edge_datas, edge_indices):
            if enum in self.edges:
                continue
            self.add_edge(enum, edata, eidx)

    def remove_edge(self, enum: tuple[int, int]):
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

    def remove_edges_from(self, edge_nums: list[tuple[int, int]]):
        for enum in edge_nums:
            if enum not in self.edges:
                continue
            self.remove_edge(enum)

    def remove_edges_by_node(self, nnum: int):
        if nnum in self.nnum_to_edges:
            for enum in self.nnum_to_edges[nnum]:
                self.edges.remove(enum)
                del self.num_to_data[enum]
                del self.num_to_idx[enum]
                if enum[0] == nnum:
                    self.nnum_to_edges[enum[1]].remove(enum)
                else:
                    self.nnum_to_edges[enum[0]].remove(enum)
