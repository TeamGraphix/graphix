"""Provides a class for open graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from graphix.generator import generate_from_graph
from graphix.pauli import Plane

if TYPE_CHECKING:
    from graphix.pattern import Pattern


@dataclass
class Measurement:
    """An MBQC measurement.

    :param angle: the angle of the measurement. Should be between [0, 2)
    :param plane: the measurement plane
    """

    angle: float
    plane: Plane

    def __eq__(self, other: object) -> bool:
        """Compares if two measurements are equal

        Example
        -------
        >>> from graphix.opengraph import Measurement
        >>> from graphix.pauli import Plane
        >>> Measurement(0.0, Plane.XY) == Measurement(0.0, Plane.XY)
        True
        >>> Measurement(0.0, Plane.XY) == Measurement(0.0, Plane.YZ)
        False
        >>> Measurement(0.1, Plane.XY) == Measurement(0.0, Plane.XY)
        False
        """
        if not isinstance(other, Measurement):
            return NotImplemented

        return np.allclose(self.angle, other.angle) and self.plane == other.plane

    def is_z_measurement(self) -> bool:
        """Indicates whether it is a Z measurement

        Example
        -------
        >>> from graphix.opengraph import Measurement
        >>> Measurement(0.0, Plane.XY).is_z_measurement()
        True
        >>> Measurement(0.0, Plane.YZ).is_z_measurement()
        False
        >>> Measurement(0.1, Plane.XY).is_z_measurement()
        False
        """
        return np.allclose(self.angle, 0.0) and self.plane == Plane.XY


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
    >>> from graphix.opengraph import OpenGraph, Measurement
    >>>
    >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    >>>
    >>> measurements = {i: Measurement(0.5 * i, Plane.XY) for i in range(2)}
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
    _inside: nx.Graph

    def __eq__(self, other):
        """Checks the two open graphs are equal

        This doesn't check they are equal up to an isomorphism"""

        return (
            self.inputs == other.inputs
            and self.outputs == other.outputs
            and nx.utils.graphs_equal(self._inside, other._inside)
            and self.measurements == other.measurements
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
        self._inside = inside

        if any(node in outputs for node in measurements):
            raise ValueError("output node can not be measured")

        for node_id, measurement in measurements.items():
            self._inside.nodes[node_id]["measurement"] = measurement

        for i, node_id in enumerate(inputs):
            self._inside.nodes[node_id]["is_input"] = True
            self._inside.nodes[node_id]["input_order"] = i

        for i, node_id in enumerate(outputs):
            self._inside.nodes[node_id]["is_output"] = True
            self._inside.nodes[node_id]["output_order"] = i

    @property
    def inputs(self) -> list[int]:
        """Returns the inputs of the graph sorted by their position.

        Example
        ------
        >>> import networkx as nx
        >>> from graphix.opengraph import OpenGraph, Measurement
        >>>
        >>> g = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = {i: Measurement(0.5 * i, Plane.XY) for i in range(2)}
        >>> inputs = [0]
        >>> outputs = [2]
        >>>
        >>> og = OpenGraph(g, measurements, inputs, outputs)
        >>> og.inputs == inputs
        True
        """
        inputs = [i for i in self._inside.nodes(data=True) if "is_input" in i[1]]
        sorted_inputs = sorted(inputs, key=lambda x: x[1]["input_order"])

        # Returns only the input ids
        return [i[0] for i in sorted_inputs]

    @property
    def outputs(self) -> list[int]:
        """Returns the outputs of the graph sorted by their position.

        Example
        ------
        >>> import networkx as nx
        >>> from graphix.opengraph import OpenGraph, Measurement
        >>>
        >>> g = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = {i: Measurement(0.5 * i, Plane.XY) for i in range(2)}
        >>> inputs = [0]
        >>> outputs = [2]
        >>>
        >>> og = OpenGraph(g, measurements, inputs, outputs)
        >>> og.outputs == outputs
        True
        """
        outputs = [i for i in self._inside.nodes(data=True) if "is_output" in i[1]]
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
        >>> from graphix.opengraph import OpenGraph, Measurement
        >>>
        >>> g = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = {i: Measurement(0.5 * i, Plane.XY) for i in range(2)}
        >>> inputs = [0]
        >>> outputs = [2]
        >>>
        >>> og = OpenGraph(g, measurements, inputs, outputs)
        >>> og.measurements == {
        ...     0: Measurement(0.0, Plane.XY),
        ...     1: Measurement(0.5, Plane.XY),
        ... }
        True
        """
        return {n[0]: n[1]["measurement"] for n in self._inside.nodes(data=True) if "measurement" in n[1]}

    @classmethod
    def from_pattern(cls, pattern: Pattern) -> OpenGraph:
        """Initialises an `OpenGraph` object based on the resource-state graph
        associated with the measurement pattern. """
        g = nx.Graph()
        nodes, edges = pattern.get_graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        inputs = pattern.input_nodes
        outputs = pattern.output_nodes

        meas_planes = pattern.get_meas_plane()
        meas_angles = pattern.get_angles()
        meas = {node: Measurement(meas_angles[node], meas_planes[node]) for node in meas_angles}

        return cls(g, meas, inputs, outputs)

    def to_pattern(self) -> Pattern:
        """Converts the `OpenGraph` into a `Pattern`.

        Will raise an exception if the open graph does not have flow, gflow, or
        Pauli flow.
        The pattern will be generated using maximally-delayed flow.
        """

        g = self._inside.copy()
        inputs = self.inputs
        outputs = self.outputs
        meas = self.measurements

        angles = {node: m.angle for node, m in meas.items()}
        planes = {node: m.plane for node, m in meas.items()}

        return generate_from_graph(g, angles, inputs, outputs, planes)
