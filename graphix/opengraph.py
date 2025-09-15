"""Provides a class for open graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

import graphix.generator
from graphix.measurements import Measurement

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from graphix.pattern import Pattern


@dataclass(frozen=True)
class OpenGraph:
    """Open graph contains the graph, measurement, and input and output nodes.

    This is the graph we wish to implement deterministically.

    :param inside: the underlying :class:`networkx.Graph` state
    :param measurements: a dictionary whose key is the ID of a node and the
        value is the measurement at that node
    :param inputs: an ordered list of node IDs that are inputs to the graph
    :param outputs: an ordered list of node IDs that are outputs of the graph

    Example
    -------
    >>> import networkx as nx
    >>> from graphix.fundamentals import Plane
    >>> from graphix.opengraph import OpenGraph, Measurement
    >>>
    >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    >>>
    >>> measurements = {i: Measurement(0.5 * i, Plane.XY) for i in range(2)}
    >>> inputs = [0]
    >>> outputs = [2]
    >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
    """

    inside: nx.Graph[int]
    measurements: dict[int, Measurement]
    inputs: list[int]  # Inputs are ordered
    outputs: list[int]  # Outputs are ordered

    def __post_init__(self) -> None:
        """Validate the open graph."""
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
        """Return `True` if two open graphs implement approximately the same unitary operator.

        Ensures the structure of the graphs are the same and all
        measurement angles are sufficiently close.

        This doesn't check they are equal up to an isomorphism.

        """
        if not nx.utils.graphs_equal(self.inside, other.inside):
            return False

        if self.inputs != other.inputs or self.outputs != other.outputs:
            return False

        if set(self.measurements.keys()) != set(other.measurements.keys()):
            return False

        return all(
            m.isclose(other.measurements[node], rel_tol=rel_tol, abs_tol=abs_tol)
            for node, m in self.measurements.items()
        )

    @staticmethod
    def from_pattern(pattern: Pattern) -> OpenGraph:
        """Initialise an `OpenGraph` object based on the resource-state graph associated with the measurement pattern."""
        g: nx.Graph[int] = nx.Graph()
        nodes, edges = pattern.get_graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        inputs = pattern.input_nodes
        outputs = pattern.output_nodes

        meas_planes = pattern.get_meas_plane()
        meas_angles = pattern.get_angles()
        meas = {node: Measurement(meas_angles[node], meas_planes[node]) for node in meas_angles}

        return OpenGraph(g, meas, inputs, outputs)

    def to_pattern(self) -> Pattern:
        """Convert the `OpenGraph` into a `Pattern`.

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

        return graphix.generator.generate_from_graph(g, angles, inputs, outputs, planes)

    def compose(self, other: OpenGraph, mapping: Mapping[int, int]) -> tuple[OpenGraph, dict[int, int]]:
        r"""Compose two open graphs by merging subsets of nodes from `self` and `other`, and relabeling the nodes of `other` that were not merged.

        Parameters
        ----------
        other : OpenGraph
            Open graph to be composed with `self`.
        mapping: dict[int, int]
            Partial relabelling of the nodes in `other`, with `keys` and `values` denoting the old and new node labels, respectively.

        Returns
        -------
        og: OpenGraph
            composed open graph
        mapping_complete: dict[int, int]
            Complete relabelling of the nodes in `other`, with `keys` and `values` denoting the old and new node label, respectively.

        Notes
        -----
        Let's denote :math:`\{G(V_1, E_1), I_1, O_1\}` the open graph `self`, :math:`\{G(V_2, E_2), I_2, O_2\}` the open graph `other`, :math:`\{G(V, E), I, O\}` the resulting open graph `og` and `{v:u}` an element of `mapping`.

        We define :math:`V, U` the set of nodes in `mapping.keys()` and `mapping.values()`, and :math:`M = U \cap V_1` the set of merged nodes.

        The open graph composition requires that
        - :math:`V \subseteq V_2`.
        - If both `v` and `u` are measured, the corresponding measurements must have the same plane and angle.
         The returned open graph follows this convention:
        - :math:`I = (I_1 \cup I_2) \setminus M \cup (I_1 \cap I_2 \cap M)`,
        - :math:`O = (O_1 \cup O_2) \setminus M \cup (O_1 \cap O_2 \cap M)`,
        - If only one node of the pair `{v:u}` is measured, this measure is assigned to :math:`u \in V` in the resulting open graph.
        - Input (and, respectively, output) nodes in the returned open graph have the order of the open graph `self` followed by those of the open graph `other`. Merged nodes are removed, except when they are input (or output) nodes in both open graphs, in which case, they appear in the order they originally had in the graph `self`.
        """
        if not (mapping.keys() <= other.inside.nodes):
            raise ValueError("Keys of mapping must be correspond to nodes of other.")
        if len(mapping) != len(set(mapping.values())):
            raise ValueError("Values in mapping contain duplicates.")
        for v, u in mapping.items():
            if (
                (vm := other.measurements.get(v)) is not None
                and (um := self.measurements.get(u)) is not None
                and not vm.isclose(um)
            ):
                raise ValueError(f"Attempted to merge nodes {v}:{u} but have different measurements")

        shift = max(*self.inside.nodes, *mapping.values()) + 1

        mapping_sequential = {
            node: i for i, node in enumerate(sorted(other.inside.nodes - mapping.keys()), start=shift)
        }  # assigns new labels to nodes in other not specified in mapping

        mapping_complete = {**mapping, **mapping_sequential}

        g2_shifted = nx.relabel_nodes(other.inside, mapping_complete)
        g = nx.compose(self.inside, g2_shifted)

        merged = set(mapping_complete.values()) & self.inside.nodes

        def merge_ports(p1: Iterable[int], p2: Iterable[int]) -> list[int]:
            p2_mapped = [mapping_complete[node] for node in p2]
            p2_set = set(p2_mapped)
            part1 = [node for node in p1 if node not in merged or node in p2_set]
            part2 = [node for node in p2_mapped if node not in merged]
            return part1 + part2

        inputs = merge_ports(self.inputs, other.inputs)
        outputs = merge_ports(self.outputs, other.outputs)

        measurements_shifted = {mapping_complete[i]: meas for i, meas in other.measurements.items()}
        measurements = {**self.measurements, **measurements_shifted}

        return OpenGraph(g, measurements, inputs, outputs), mapping_complete
