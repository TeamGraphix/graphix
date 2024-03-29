from __future__ import annotations

import uuid
from itertools import combinations

import networkx as nx
import numpy as np

from graphix.gflow import find_flow, find_gflow
from graphix.pattern import Pattern

COLOR_MAP = {
    "XY": "chartreuse",
    "YZ": "red",
    "XZ": "gold",
    "hadamard node": "yellow",
    None: "lightgray",
    "input": "slategrey",
    "harmard edge": "deepskyblue",
    "non-hadamard": "k",
}


class MGraph(nx.MultiGraph):
    """Measurement graph

    Attributes
    ----------
    input: list
        input nodes
    output: list
        output nodes
    flow: dict
        (g)flow of the graph
    layers: list
        layers of the graph in terms of (g)flow
    """

    def __init__(self, inputs=[], outputs=[], **kwargs):
        super().__init__(**kwargs)
        """Measurement graph
        """
        self.flow = dict()
        self.layers = dict()
        self.node_num = 0
        self.set_input_nodes(inputs)
        self.output_nodes = outputs

        self.node2id = dict()
        self.edge2id = dict()

    def add_node(self, node: int, plane=None, angle=None, output=False, **kwargs):
        """add a node to the graph

        Parameters
        ----------
        node: int
            node to add
        plane: str, optional
            measurement plane, by default "XY"
        angle: int, optional
            measurement angle, by default 0
        """
        super().add_node(node, plane=plane, angle=angle, output=output, **kwargs)

        self.node_num += 1

    def add_hadamard(self, node):
        super().add_node(node, plane="hadamard node", angle=None, output=False)

        self.node_num += 1

    def add_edge(self, u: int, v: int, hadamard: bool = False):
        """add an edge to the graph

        Parameters
        ----------
        u: int
            first node
        v: int
            second node
        hadamard: bool, optional
            hadamard edge or not, by default False
        """
        super().add_edge(u, v, hadamard=hadamard)

    def find_odd_neighbors(self, nodes: set[int]):
        OddN = set()
        for node in nodes:
            OddN ^= set(self.neighbors(node))
        return OddN

    def assign_measurement_info(self, node: int, plane: str, angle: int):
        """asign measurement info to a node

        Parameters
        ----------
        node: int
            node to asign measurement info
        plane: str
            measurement plane
        angle: int
            measurement angle
        """
        self.nodes[node]["plane"] = plane
        self.nodes[node]["angle"] = angle

        self.nodes[node]["output"] = False

    def set_input_nodes(self, nodes: set):
        """add input nodes to graph

        Parameters
        ----------
        nodes: list
            list of input nodes
        """
        self.input_nodes = nodes
        for node in nodes:
            self.add_node(node, output=True)

    def get_meas_planes(self):
        """get measurement planes of the graph

        Returns
        -------
        meas_planes: dict
            dictionary of measurement planes
        """
        meas_planes = dict()
        for node in self.nodes:
            if self.nodes[node]["output"]:
                continue
            meas_planes[node] = self.nodes[node]["plane"]

        return meas_planes

    def get_pattern(self):
        """returns the pattern of the graph

        Returns
        -------
        Pattern: graphix.pattern.Pattern
            pattern of the graph
        """
        pattern = Pattern(input_nodes=self.input_nodes)
        for node in self.nodes:
            pattern.add(["N", node])

        for edge in self.edges:
            pattern.add(["E", edge])

        x_signals, z_signals = self.collect_signals()

        depth = len(self.layers)
        for k in range(depth - 1, 0, -1):
            layer = self.layers[k]
            for node in layer:
                pattern.add(
                    [
                        "M",
                        node,
                        self.nodes[node]["plane"],
                        self.nodes[node]["angle"],
                        x_signals[node],
                        z_signals[node],
                    ]
                )

        for node in self.output_nodes:
            if len(x_signals[node]) > 0:
                pattern.add(["X", node, x_signals[node]])
            if len(z_signals[node]) > 0:
                pattern.add(["Z", node, z_signals[node]])

        pattern.output_nodes = self.output_nodes

        return pattern

    def collect_signals(self):
        """collects the signals of the graph from the (g)flow

        Returns
        -------
        x_signals: dict
            dictionary of x signals
        z_signals: dict
            dictionary of z signals
        """
        x_signals = {node: set() for node in self.nodes}
        z_signals = {node: set() for node in self.nodes}

        for node in self.nodes - set(self.output_nodes):
            for node_fg in self.flow[node]:
                x_signals[node_fg] |= {node}

            odd_neighbors = self.odd_neighbors(self.flow[node])
            for node_fg in odd_neighbors:
                if node_fg not in self.output_nodes:
                    if node == node_fg:
                        continue
                z_signals[node_fg] ^= {node}

        return x_signals, z_signals

    def get_colors(self):
        """get color for displaying measurement graph

        Returns
        -------
        tuple[list[str], list[str]]: color map for nodes and edges
        """
        node_color = [COLOR_MAP[node["plane"]] for node in self.nodes.values()]
        edge_color = ["deepskyblue" if e["hadamard"] else COLOR_MAP["non-hadamard"] for e in self.edges.values()]
        return node_color, edge_color

    #############################
    # Elementary Graph Operations
    #############################

    def fusion(self, u: int, v: int, new_index: int, copy: bool = False):
        """Apply fusion to two nodes

        Parameters
        ----------
        u: int
            First node
        v: int
            Second node
        new_index: int
            New node index
        copy: bool, optional
            Copy the graph, by default False
        """
        if self.nodes[u]["plane"] != self.nodes[v]["plane"]:
            raise ValueError("Color mismatch")
        else:
            new_plane = self.nodes[u]["plane"]
        num_edges = self.number_of_edges(u, v)

        # check if there is a hadamard edge
        for k in range(num_edges):
            edge_info = self.edges[u, v, k]["hadamard"]
            if edge_info:
                raise ValueError("Hadamard edge")

        graph = self.copy() if copy else self
        u_neighbors = set(self.neighbors(u))
        v_neighbors = set(self.neighbors(v))

        new_angle = self.nodes[u]["angle"] + self.nodes[v]["angle"]
        graph.add_node(new_index, new_plane, new_angle)
        for node in u_neighbors:
            num_edges = graph.number_of_edges(u, node)
            for k in range(num_edges):
                graph.add_edge(new_index, node, hadamard=self.edges[u, node, k]["hadamard"])
        for node in v_neighbors:
            num_edges = graph.number_of_edges(v, node)
            for k in range(num_edges):
                graph.add_edge(new_index, node, hadamard=self.edges[v, node, k]["hadamard"])

        graph.remove_nodes_from([u, v])

        return graph

    def hermite_conj(self, u: int, copy: bool = False):
        """Apply hermite conjugation to a node

        Parameters
        ----------
        u: int
            Node to apply hermite conjugation
        copy: bool, optional
            Copy the graph, by default False
        """
        graph = self.copy() if copy else self
        graph.nodes[u]["plane"] = self.convert_plane_by_h(self.nodes[u]["plane"])
        for node in self.neighbors(u):
            num_edges = graph.number_of_edges(u, node)
            for k in range(num_edges):
                graph.edges[u, node, k]["hadamard"] = not graph.edges[u, node, k]["hadamard"]

        return graph

    def identity(self, u: int, copy: bool = False):
        """Apply identity transform on the node

        Parameters
        ----------
        u: int
            Node with "XY" measurement plane and 0 angle
        copy: bool, optional
            Copy the graph, by default False
        """
        graph = self.copy() if copy else self
        neighbors = set(graph.neighbors(u))
        if graph.nodes[u]["plane"] == "XY" and graph.nodes[u]["angle"] == 0 and len(neighbors) == 2:
            # ensure edges are not Hadamard and multiple
            single_non_Hadamard = True
            for neighbor in neighbors:
                single_non_Hadamard &= graph.edges[u, neighbor, 0]["hadamard"]
                single_non_Hadamard &= graph.number_of_edges(u, neighbor) == 1
            graph.remove_node(u)
            graph.add_edge(*neighbors, hadamard=False)
        return graph

    def hadamard_cancel(self, u: int, v: int, new_index: int, copy: bool = False):
        """Convert two neighboring Hadamard nodes into a single XY-spilder with 0 angle

        Parameters
        ----------
        u: int
            first Hadamard node
        v: int
            second Hadamard node
        new_index: int
            new node index
        copy: bool
            Copy the graph, by default False
        """
        graph = self.copy() if copy else self
        assert graph.nodes[u]["plane"] == "hadamard node"
        assert graph.nodes[v]["plane"] == "hadamard node"
        assert graph.number_of_edges(u, v) == 1
        assert graph.edges[u, v, 0]["hadamard"] == False

        u_neighbors = set(graph.neighbors(u))
        v_neighbors = set(graph.neighbors(v))

        graph.add_node(new_index, "XY", 0, output=False)

        for u_neighbor in u_neighbors - {v}:
            num_edges = graph.number_of_edges(u, u_neighbor)
            for k in range(num_edges):
                graph.add_edge(
                    new_index,
                    u_neighbor,
                    hadamard=graph.edges[u, u_neighbor, k]["hadamard"],
                )

        for v_neighbor in v_neighbors - {u}:
            num_edges = graph.number_of_edges(v, v_neighbor)
            for k in range(num_edges):
                graph.add_edge(
                    new_index,
                    v_neighbor,
                    hadamard=graph.edges[v, v_neighbor, k]["hadamard"],
                )

        graph.remove_node(u)
        graph.remove_node(v)

        return graph

    def pi_transport(self, u: int, pi: int, new_indices: dict[int, int], copy: bool = False):
        """Apply pi transport to two nodes

        Parameters
        ----------
        u: int
            Target node
        pi: int
            pi node
        new_indices: dict[int, int]
            new index for new pi nodes
        copy: bool, optional
            Copy the graph, by default False
        """
        graph = self.copy() if copy else self
        assert graph.nodes[pi]["angle"] == np.pi, "Node v is not a pi node"
        assert graph.nodes[pi]["plane"] == "YZ", "Node v is not X spider"
        assert graph.nodes[u]["plane"] == "XY", "Node u should be XY spider"
        assert graph.number_of_edges(u, pi) == 1, "The edge between u and v should be single"
        assert graph.edges[u, pi, 0]["hadamard"] == False, "The edge between u and v should not be Hadamard"

        u_neighbors = set(graph.neighbors(u))
        pi_neighbors = set(graph.neighbors(pi))

        for u_neighbor in u_neighbors - {pi}:
            graph.add_node(new_indices[u_neighbor], plane="YZ", angle=np.pi, output=False)

            graph.add_edge(u, new_indices[u_neighbor], hadamard=False)
            num_edges = graph.number_of_edges(u, u_neighbor)
            for k in range(num_edges):
                graph.add_edge(
                    u_neighbor,
                    new_indices[u_neighbor],
                    hadamard=graph.edges[u, u_neighbor, k]["hadamard"],
                )

            graph.remove_edge(u, u_neighbor)

        for pi_neighbor in pi_neighbors - {u}:
            num_edges = graph.number_of_edges(pi, pi_neighbor)

            for k in range(num_edges):
                graph.add_edge(u, pi_neighbor, hadamard=graph.edges[pi, pi_neighbor, k]["hadamard"])

        graph.nodes[u]["angle"] = -graph.nodes[u]["angle"]

        graph.remove_node(pi)

        return graph

    def anchor_divide(self, u: int, anchor: int, new_indices: dict[int, int], copy: bool = False):
        """Apply anchor division rule

        u: int
            Node to apply anchor rule
        anchor: int
            Anchor node
        new_indices: dict[int, int]
            new index for new anchor
        copy: bool, optional
            Copy the graph, by default False
        """
        assert self.nodes[anchor]["plane"] == "YZ", f"Node {anchor} is not a X spider"
        assert self.nodes[anchor]["angle"] == 0, f"Node {anchor} is not a anchor node"
        assert self.nodes[u]["plane"] == "XY", f"Node {u} should be XY spider"

        graph = self.copy() if copy else self

        graph.remove_node(anchor)

        u_neighbors = set(graph.neighbors(u))

        for u_neighbor in u_neighbors:
            graph.add_node(new_indices[u_neighbor], "YZ", 0)
            num_edges = graph.number_of_edges(u, u_neighbor)
            for k in range(num_edges):
                graph.add_edge(
                    u_neighbor,
                    new_indices[u_neighbor],
                    hadamard=graph.edges[u, u_neighbor, k],
                )

        graph.remove_node(u)

        return graph

    def bipartite(self, u: int, v: int, new_indices: dict[int, int], copy: bool = False):
        """Apply bipartite rule

        Parameters
        ----------
        u: int
            YZ spider
        v: int
            XY spider
        new_indices: dict[int, int]
            new node indicies.
            It must include 4 nodes because in this implementation we remove u and v.
            dict should be {"u1": ind1, "u2": ind2, "v1": ind3, "v2": ind4} in the current implementation.
            TODO: improve index generation
        copy: bool, optional
            Copy the graph, by default False
        """
        graph = self.copy() if copy else self
        assert graph.nodes[u]["plane"] == "YZ"
        assert graph.nodes[u]["angle"] == 0
        assert graph.nodes[v]["plane"] == "XY"
        assert graph.nodes[v]["angle"] == 0
        assert graph.number_of_edges(u, v) == 1
        assert graph.edges[u, v, 0]["hadamard"] == False

        assert len(set(graph.neighbors(u))) == 3
        assert len(set(graph.neighbors(v))) == 3

        u_neighbors = set(graph.neighbors(u)) - {v}
        v_neighbors = set(graph.neighbors(v)) - {u}

        graph.add_node(new_indices["u1"], "XY", 0)
        graph.add_node(new_indices["u2"], "XY", 0)
        graph.add_node(new_indices["v1"], "YZ", 0)
        graph.add_node(new_indices["v2"], "YZ", 0)

        graph.add_edge(new_indices["u1"], new_indices["v1"], hadamard=False)
        graph.add_edge(new_indices["u1"], new_indices["v2"], hadamard=False)
        graph.add_edge(new_indices["u2"], new_indices["v1"], hadamard=False)
        graph.add_edge(new_indices["u2"], new_indices["v2"], hadamard=False)

        u_neighbor1 = u_neighbors.pop()
        graph.add_edge(
            new_indices["u1"],
            u_neighbor1,
            hadamard=graph.edges[u, u_neighbor1, 0]["hadamard"],
        )

        u_neighbor2 = u_neighbors.pop()
        graph.add_edge(
            new_indices["u2"],
            u_neighbor2,
            hadamard=graph.edges[u, u_neighbor2, 0]["hadamard"],
        )

        v_neighbor1 = v_neighbors.pop()
        graph.add_edge(
            new_indices["v1"],
            v_neighbor1,
            hadamard=graph.edges[v, v_neighbor1, 0]["hadamard"],
        )

        v_neighbor2 = v_neighbors.pop()
        graph.add_edge(
            new_indices["v2"],
            v_neighbor2,
            hadamard=graph.edges[v, v_neighbor2, 0]["hadamard"],
        )

        graph.remove_node(u)
        graph.remove_node(v)

        return graph

    def convert_plane_by_h(self, plane: str) -> str:
        if plane == "XY":
            return "YZ"
        elif plane == "YZ":
            return "XY"
        elif plane == "XZ":
            return "XZ"

    #############################

    def local_complementation(self, target: int):
        """Apply local complementation to a node

        Parameters
        ----------
        target: int
            Node to apply local complementation
        """
        assert self.has_node(target), "Node not in graph"
        neighbors_target = list(self.neighbors(target))
        neighbors_complete_edges = set((u, v, 0) for u, v in combinations(neighbors_target, 2))
        remove_edges = set(self.edges) & neighbors_complete_edges
        add_edges = neighbors_complete_edges - remove_edges
        self.remove_edges_from(remove_edges)
        for edge in add_edges:
            self.add_edge(edge[0], edge[1], hadamard=True)

        # modify measurement planes
        if self.nodes[target]["plane"] == "XY":
            self.nodes[target]["plane"] = "XZ"
        elif self.nodes[target]["plane"] == "XZ":
            self.nodes[target]["plane"] = "XY"
        elif self.nodes[target]["plane"] == "YZ":
            pass

        non_output = set(self.nodes) - set(self.output_nodes) - {target}
        for node in non_output:
            if self.nodes[target]["plane"] == "XZ":
                self.nodes[target]["plane"] = "YZ"
            elif self.nodes[target]["plane"] == "YZ":
                self.nodes[target]["plane"] = "XZ"
            elif self.nodes[target]["plane"] == "XY":
                pass

        # modify flow
        if len(self.flow) != 0:
            if self.nodes[target]["plane"] == "XY" or "XZ":
                self.flow[target] = self.flow[target] ^ {target}
            elif self.nodes[target]["plane"] == "YZ":
                pass

            for node in non_output:
                odd_neighbors_node = self.find_odd_neighbors(self.flow[node])
                if target in odd_neighbors_node:
                    self.flow[node] = self.flow[node] ^ {target}
                    if self.nodes[target]["plane"] != None:
                        self.flow[node] = self.flow[node] ^ self.flow[target]
                else:
                    pass

    def update_flow(self):
        """Update the flow of the graph"""
        fg, l_k = find_flow(
            self,
            input=set(self.input_nodes),
            output=set(self.output_nodes),
            meas_planes=self.get_meas_planes(),
        )
        if fg == None:
            fg, l_k = find_gflow(
                self,
                input=set(self.input_nodes),
                output=set(self.output_nodes),
                meas_planes=self.get_meas_planes(),
            )

        if fg == None:
            raise ValueError("No flow found")

        self.flow = fg
        for node, k in l_k.items():
            if k not in self.layers.keys():
                self.layers[k] = {node}
            else:
                self.layers[k] = self.layers[k] | {node}

    def odd_neighbors(self, nodes: set):
        """Find odd neighbors of a node

        Parameters
        ----------
        node: set
            Nodes to find odd neighbors

        Returns
        -------
        odd_neighbors: set
            Set of odd neighbors
        """
        odd_neighbors = set()
        for node in nodes:
            odd_neighbors ^= set(self.neighbors(node))
        return odd_neighbors

    def pivot(self, u: int, v: int):
        """Apply pivot to two nodes

        Parameters
        ----------
        u: int
            First node
        v: int
            Second node
        """
        u_neighbors = set(self.neighbors(u))
        v_neighbors = set(self.neighbors(v))
        uv_all_neighbors = u_neighbors | v_neighbors

        uv_neighbors = u_neighbors & v_neighbors
        u_vnot_neighbors = uv_all_neighbors - v_neighbors
        unot_v_neighbors = uv_all_neighbors - u_neighbors

        complete_edges_uv_uvnot = {(i, j) for i in uv_neighbors for j in u_vnot_neighbors}
        complete_edges_uv_unotv = {(i, j) for i in uv_neighbors for j in unot_v_neighbors}
        complete_edges_uvnot_unotv = {(i, j) for i in u_vnot_neighbors for j in unot_v_neighbors}

        E = set(self.edges)
        E = E ^ complete_edges_uv_uvnot
        E = E ^ complete_edges_uv_unotv
        E = E ^ complete_edges_uvnot_unotv

        self.update(edges=E)
        self = nx.relabel_nodes(self, {u: v, v: u})

        # modify measurement planes
        for a in {u, v}:
            if self.meas_planes[a] == "XY":
                self.meas_planes[a] = "YZ"
            elif self.meas_planes[a] == "XZ":
                pass
            elif self.meas_planes[a] == "YZ":
                self.meas_planes[a] = "XY"

        # flow?

    # def to_pyzx(self):
    # graph = zx.Graph()
    # index_map = dict()
    # for node in self.nodes - set(self.output_nodes):
    #     index = graph.add_vertex(type=zx.VertexType.Z, phase=node.angle)
    #     index_map[node] = index
    # for node in set(self.output_nodes):
    #     index = graph.add_vertex(type=zx.VertexType.BOUNDARY)
    #     index_map[node] = index
    # for edge in self.edges:
    #     graph.add_edge(
    #         graph.edge(index_map[edge[0]], index_map[edge[1]]),
    #         type=zx.EdgeType.HADAMARD,
    #     )

    # return graph

    def simulate_mbqc(self, **kwargs):
        """Simulate the graph using MBQC

        Returns
        -------
        simulator: graphix.simulator.Simulator
            Simulator object
        """
        pass