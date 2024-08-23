"""Provides a class for open graphs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from graphix.generator import generate_from_graph

if TYPE_CHECKING:
    from graphix.pattern import Pattern
    from graphix.pauli import Plane


@dataclass(frozen=True)
class Measurement:
    """An MBQC measurement.

    :param angle: the angle of the measurement. Should be between [0, 2)
    :param plane: the measurement plane
    """

    angle: float
    plane: Plane

    def isclose(self, other: Measurement, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        """Compares if two measurements have the same plane and their angles
        are close.

        Example
        -------
        >>> from graphix.opengraph import Measurement
        >>> from graphix.pauli import Plane
        >>> Measurement(0.0, Plane.XY).isclose(Measurement(0.0, Plane.XY))
        True
        >>> Measurement(0.0, Plane.XY).isclose(Measurement(0.0, Plane.YZ))
        False
        >>> Measurement(0.1, Plane.XY).isclose(Measurement(0.0, Plane.XY))
        False
        """
        return math.isclose(self.angle, other.angle, rel_tol=rel_tol, abs_tol=abs_tol) and self.plane == other.plane


@dataclass(frozen=True)
class OpenGraph:
    """Open graph contains the graph, measurement, and input and output
    nodes. This is the graph we wish to implement deterministically

    :param inside: the underlying graph state
    :param measurements: a dictionary whose key is the ID of a node and the
        value is the measurement at that node
    :param inputs: an ordered list of node IDs that are inputs to the graph
    :param outputs: an ordered list of node IDs that are outputs of the graph

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

    inside: nx.Graph
    measurements: dict[int, Measurement]
    inputs: list[int]  # Inputs are ordered
    outputs: list[int]  # Outputs are ordered

    def __post_init__(self) -> None:
        if not all(node in self.inside.nodes for node in self.measurements):
            raise ValueError("All measured nodes must be part of the graph's nodes.")
        if not all(node in self.inside.nodes for node in self.inputs):
            raise ValueError("All input nodes must be part of the graph's nodes.")
        if not all(node in self.inside.nodes for node in self.outputs):
            raise ValueError("All output nodes must be part of the graph's nodes.")
        if any(node in self.outputs for node in self.measurements):
            raise ValueError("Output node cannot be measured.")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Input nodes contain duplicates.")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Output nodes contain duplicates.")

    def isclose(self, other: OpenGraph, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        """Compared two open graphs implement approximately the same unitary
        operator by ensuring the structure of the graphs are the same and all
        measurement angles are sufficiently close.

        This doesn't check they are equal up to an isomorphism"""

        if not nx.utils.graphs_equal(self.inside, other.inside):
            return False

        if self.inputs != other.inputs or self.outputs != other.outputs:
            return False

        if set(self.measurements.keys()) != set(other.measurements.keys()):
            return False

        return all(m.isclose(other.measurements[node]) for node, m in self.measurements.items())

    @classmethod
    def from_pattern(cls, pattern: Pattern) -> OpenGraph:
        """Initialises an `OpenGraph` object based on the resource-state graph
        associated with the measurement pattern."""
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

        g = self.inside.copy()
        inputs = self.inputs
        outputs = self.outputs
        meas = self.measurements

        angles = {node: m.angle for node, m in meas.items()}
        planes = {node: m.plane for node, m in meas.items()}

        return generate_from_graph(g, angles, inputs, outputs, planes)
