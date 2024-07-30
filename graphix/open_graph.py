"""Provides a class for open graphs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pyzx as zx


@dataclass
class Measurement:
    """An MBQC measurement.

    :param angle: the angle of the measurement. Should be between [0, 2pi)
    :param plane: the measurement plane: 'XY', 'XZ', 'YZ'
    """

    angle: float
    plane: str

    def __eq__(self, other):
        """Compares if two measurements are equal

        Example
        -------
        >>> from graphix.open_graph import Measurement
        >>> Measurement(0.0, "XY") == Measurement(0.0, "XY")
        True
        >>> Measurement(0.0, "XY") == Measurement(0.0, "YZ")
        False
        >>> Measurement(0.1, "XY") == Measurement(0.0, "XY")
        False
        """
        return np.allclose(self.angle, other.angle) and self.plane == other.plane

    def is_z_measurement(self) -> bool:
        """Indicates whether it is a Z measurement

        Example
        -------
        >>> from graphix.open_graph import Measurement
        >>> Measurement(0.0, "XY").is_z_measurement()
        True
        >>> Measurement(0.0, "YZ").is_z_measurement()
        False
        >>> Measurement(0.1, "XY").is_z_measurement()
        False
        """
        return np.allclose(self.angle, 0.0) and self.plane == "XY"


@dataclass
class Fusion:
    """A fusion between two nodes

    :param node1: ID of one of the nodes in the fusion
    :param node2: ID of the other node in the fusion
    :param fusion_type: The type of fusion. Currently either: "X", "Y"

    Example
    -------
    >>> from graphix.open_graph import Fusion
    >>> Fusion(0, 1, "X") == Fusion(0, 1, "X")
    True
    >>> Fusion(0, 1, "X") == Fusion(0, 1, "Y")
    False
    >>> Fusion(0, 1, "X") == Fusion(1, 0, "X")
    True
    >>> Fusion(0, 1, "X") == Fusion(0, 2, "X")
    False
    """

    node1: int
    node2: int
    fusion_type: str

    def __eq__(self, other) -> bool:
        if self.fusion_type != other.fusion_type:
            return False

        if self.node1 == other.node1 and self.node2 == other.node2:
            return True

        if self.node1 == other.node2 and self.node2 == other.node1:
            return True

        return False

    def contains(self, node_id: int) -> bool:
        """Indicates whether the node is part of the fusion"""
        return node_id in (self.node1, self.node2)


class OpenGraph:
    """Open graph contains the graph, measurement, and input and output
    nodes. This is the graph we wish to implement deterministically

    :param inside: the underlying graph state
    :param measurements: a dictionary whose key is the ID of a node and the
        value is the measurement at that node
    :param inputs: a set of IDs of the nodes that are inputs to the graph
    :param outputs: a set of IDs of the nodes that are outputs of the graph

    Example
    -------
    >>> import networkx as nx
    >>> from graphix.open_graph import OpenGraph, Measurement
    >>>
    >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    >>>
    >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    >>> inputs = [0]
    >>> outputs = [2]
    >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
    """

    # All the measurement, inputs and outputs information is stored in this
    # graph. This is to simplify certain operations such as composition of
    # graphs and checking equality.
    # These attributes shouldn't be accessed directly by the user, instead,
    # they should use the inputs(), outputs(), and measurements() properties.
    #
    # The following attributes are stored in the nodes:
    #
    # Non-output nodes:
    # * "measurement" - The Measurement object for the node
    #
    # Input nodes:
    # * "is_input"    - The value is always True
    # * "input_order" - A zero-indexed integer used to preserve the ordering of
    #                   the inputs
    #
    # Output nodes:
    # * "is_output"    - The value is always True
    # * "output_order" - A zero-indexed integer used to preserve the ordering of
    #                    the outputs
    inside: nx.Graph

    def __eq__(self, other):
        """Checks the two open graphs are equal

        This doesn't check they are equal up to an isomorphism"""

        g1 = self.perform_z_deletions()
        g2 = other.perform_z_deletions()

        return (
            g1.inputs == g2.inputs
            and g1.outputs == g2.outputs
            and nx.utils.graphs_equal(g1.inside, g2.inside)
            and g1.measurements == g2.measurements
        )

    def __init__(
        self,
        inside: nx.Graph,
        measurements: dict[int, Measurement],
        inputs: list[int],
        outputs: list[int],
    ):
        """Constructs a new OpenGraph instance

        The inputs() and outputs() methods will preserve the order that was
        original given in to this methods.
        """
        self.inside = inside

        if any(node in outputs for node in measurements):
            raise ValueError("output node can not be measured")

        for node_id, measurement in measurements.items():
            self.inside.nodes[node_id]["measurement"] = measurement

        for i, node_id in enumerate(inputs):
            self.inside.nodes[node_id]["is_input"] = True
            self.inside.nodes[node_id]["input_order"] = i

        for i, node_id in enumerate(outputs):
            self.inside.nodes[node_id]["is_output"] = True
            self.inside.nodes[node_id]["output_order"] = i

    def to_pyzx_graph(self) -> zx.graph.base.BaseGraph:
        """Return a PyZX graph corresponding to the the open graph.

        Example
        -------
        >>> import networkx as nx
        >>> g = nx.Graph([(0, 1), (1, 2)])
        >>> inputs = [0]
        >>> outputs = [2]
        >>> measurements = {0: Measurement(0, "XY"), 1: Measurement(1, "ZY")}
        >>> og = OpenGraph(g, measurements, inputs, outputs)
        >>> reconstructed_pyzx_graph = og.to_pyzx_graph()
        """
        g = zx.Graph()

        # Add vertices into the graph and set their type
        def add_vertices(n: int, ty: zx.VertexType) -> list[zx.VertexType]:
            verts = g.add_vertices(n)
            for vert in verts:
                g.set_type(vert, ty)

            return verts

        # Add input boundary nodes
        in_verts = add_vertices(len(self.inputs), zx.VertexType.BOUNDARY)
        g.set_inputs(in_verts)

        # Add nodes for internal Z spiders - not including the phase gadgets
        body_verts = add_vertices(len(self.inside), zx.VertexType.Z)

        # Add nodes for the phase gadgets. In OpenGraph we don't store the
        # effect as a seperate node, it is instead just stored in the
        # "measurement" attribute of the node it measures.
        x_meas = [i for i, m in self.measurements.items() if m.plane == "YZ"]
        x_meas_verts = add_vertices(len(x_meas), zx.VertexType.Z)

        out_verts = add_vertices(len(self.outputs), zx.VertexType.BOUNDARY)
        g.set_outputs(out_verts)

        # Maps a node's ID in the Open Graph to it's corresponding node ID in
        # the PyZX graph and vice versa.
        map_to_og = dict(zip(body_verts, self.inside.nodes()))
        map_to_pyzx = {v: i for i, v in map_to_og.items()}

        # Open Graph's don't have boundary nodes, so we need to connect the
        # input and output Z spiders to their corresponding boundary nodes in
        # pyzx.
        for pyzx_index, og_index in zip(in_verts, self.inputs):
            g.add_edge((pyzx_index, map_to_pyzx[og_index]))
        for pyzx_index, og_index in zip(out_verts, self.outputs):
            g.add_edge((pyzx_index, map_to_pyzx[og_index]))

        og_edges = self.inside.edges()
        pyzx_edges = [(map_to_pyzx[a], map_to_pyzx[b]) for a, b in og_edges]
        g.add_edges(pyzx_edges, zx.EdgeType.HADAMARD)

        # Add the edges between the Z spiders in the graph body
        for og_index, meas in self.measurements.items():
            # If it's an X measured node, then we handle it in the next loop
            if meas.plane == "XY":
                g.set_phase(map_to_pyzx[og_index], meas.angle)

        # Connect the X measured vertices
        for og_index, pyzx_index in zip(x_meas, x_meas_verts):
            g.add_edge((map_to_pyzx[og_index], pyzx_index), zx.EdgeType.HADAMARD)
            g.set_phase(pyzx_index, self.measurements[og_index].angle)

        return g

    @classmethod
    def from_pyzx_graph(cls, g: zx.graph.base.BaseGraph) -> OpenGraph:
        """Constructs an Optyx Open Graph from a PyZX graph.

        This method may add additional nodes to the graph so that it adheres
        with the definition of an OpenGraph. For instance, if the final node on
        a qubit is measured, it will add two nodes behind it so that no output
        nodes are measured to satisfy the requirements of an open graph.

        Example
        -------
        >>> import pyzx as zx
        >>> from graphix.open_graph import OpenGraph
        >>> circ = zx.qasm("qreg q[2]; h q[1]; cx q[0], q[1]; h q[1];")
        >>> g = circ.to_graph()
        >>> og = OpenGraph.from_pyzx_graph(g)
        """
        zx.simplify.to_graph_like(g)
        zx.simplify.full_reduce(g)

        measurements = {}
        inputs = g.inputs()
        outputs = g.outputs()

        g_nx = nx.Graph(g.edges())

        # We need to do this since the full reduce simplification can
        # leave either hadamard or plain wires on the inputs and outputs
        for inp in g.inputs():
            nbrs = list(g.neighbors(inp))
            et = g.edge_type((nbrs[0], inp))

            if et == zx.EdgeType.SIMPLE:
                g_nx.remove_node(inp)
                inputs = [i if i != inp else nbrs[0] for i in inputs]

        for out in g.outputs():
            nbrs = list(g.neighbors(out))
            et = g.edge_type((nbrs[0], out))

            if et == zx.EdgeType.SIMPLE:
                g_nx.remove_node(out)
                outputs = [o if o != out else nbrs[0] for o in outputs]

        # Turn all phase gadgets into measurements
        # Since we did a full reduce, any node that isn't an input or output
        # node and has only one neighbour is definitely a phase gadget.
        nodes = list(g_nx.nodes())
        for v in nodes:
            if v in inputs or v in outputs:
                continue

            nbrs = list(g.neighbors(v))
            if len(nbrs) == 1:
                measurements[nbrs[0]] = Measurement(g.phase(v), "YZ")
                g_nx.remove_node(v)

        next_id = max(g_nx.nodes) + 1

        # Since outputs can't be measured, we need to add an extra two nodes
        # in to counter it
        for out in outputs:
            if g.phase(out) == 0:
                continue

            g_nx.add_edges_from([(out, next_id), (next_id, next_id + 1)])
            measurements[next_id] = Measurement(0, "XY")

            outputs = [o if o != out else next_id + 1 for o in outputs]
            next_id += 2

        # Add the phase to all XY measured nodes
        for v in g_nx.nodes:
            if v in outputs or v in measurements:
                continue

            measurements[v] = Measurement(g.phase(v), "XY")

        return cls(g_nx, measurements, inputs, outputs)

    @property
    def inputs(self) -> list[int]:
        """Returns the inputs of the graph sorted by their position.

        Example
        ------
        >>> import networkx as nx
        >>> from graphix.open_graph import OpenGraph, Measurement
        >>>
        >>> g = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
        >>> inputs = [0]
        >>> outputs = [2]
        >>>
        >>> og = OpenGraph(g, measurements, inputs, outputs)
        >>> og.inputs == inputs
        True
        """
        inputs = [i for i in self.inside.nodes(data=True) if "is_input" in i[1]]
        sorted_inputs = sorted(inputs, key=lambda x: x[1]["input_order"])
        input_node_ids = [i[0] for i in sorted_inputs]
        return input_node_ids

    @property
    def outputs(self) -> list[int]:
        """Returns the outputs of the graph sorted by their position.

        Example
        ------
        >>> import networkx as nx
        >>> from graphix.open_graph import OpenGraph, Measurement
        >>>
        >>> g = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
        >>> inputs = [0]
        >>> outputs = [2]
        >>>
        >>> og = OpenGraph(g, measurements, inputs, outputs)
        >>> og.outputs == outputs
        True
        """
        outputs = [i for i in self.inside.nodes(data=True) if "is_output" in i[1]]
        sorted_outputs = sorted(outputs, key=lambda x: x[1]["output_order"])
        output_node_ids = [i[0] for i in sorted_outputs]
        return output_node_ids

    @property
    def measurements(self) -> dict[int, Measurement]:
        """Returns a dictionary which maps each node to its measurement. Output
        nodes are not measured and hence are not included in the dictionary.

        Example
        ------
        >>> import networkx as nx
        >>> from graphix.open_graph import OpenGraph, Measurement
        >>>
        >>> g = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
        >>> inputs = [0]
        >>> outputs = [2]
        >>>
        >>> og = OpenGraph(g, measurements, inputs, outputs)
        >>> og.measurements == {
        ...     0: Measurement(0.0, "XY"),
        ...     1: Measurement(0.5, "XY"),
        ... }
        True
        """
        return {
            n[0]: n[1]["measurement"]
            for n in self.inside.nodes(data=True)
            if "measurement" in n[1]
        }

    def perform_z_deletions_in_place(self):
        """Removes the Z-deleted nodes from the graph in place"""
        z_measured_nodes = [
            node
            for node in self.inside.nodes
            if "measurement" in self.inside.nodes[node]
            and self.inside.nodes[node]["measurement"].is_z_measurement()
        ]

        for node in z_measured_nodes:
            self.inside.remove_node(node)

    def perform_z_deletions(self) -> OpenGraph:
        """Removes the Z-deleted nodes from the graph"""
        g = deepcopy(self)
        g.perform_z_deletions_in_place()
        return g
