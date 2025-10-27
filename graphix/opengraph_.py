"""Provides a class for open graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from graphix.flow._find_cflow import find_cflow
from graphix.flow._find_pflow import AlgebraicOpenGraph, PlanarAlgebraicOpenGraph, compute_correction_matrix
from graphix.flow.flow import CausalFlow, GFlow, PauliFlow
from graphix.fundamentals import AbstractMeasurement, AbstractPlanarMeasurement
from graphix.measurements import Measurement

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    import networkx as nx

    from graphix.pattern import Pattern

# TODO
# I think we should treat Plane and Axes on the same footing (are likewise for Measurement and PauliMeasurement)
# Otherwise, shall we define Plane.XY-only open graphs.
# Maybe move these definitions to graphix.fundamentals and graphix.measurements ?

_M_co = TypeVar("_M_co", bound=AbstractMeasurement, covariant=True)
_PM_co = TypeVar("_PM_co", bound=AbstractPlanarMeasurement, covariant=True)


@dataclass(frozen=True)
class OpenGraph(Generic[_M_co]):
    """Open graph contains the graph, measurement, and input and output nodes.

    This is the graph we wish to implement deterministically.

    :param graph: the underlying :class:`networkx.Graph` state
    :param measurements: a dictionary whose key is the ID of a node and the
        value is the measurement at that node
    :param input_nodes: an ordered list of node IDs that are inputs to the graph
    :param output_nodes: an ordered list of node IDs that are outputs of the graph

    Example
    -------
    >>> import networkx as nx
    >>> from graphix.fundamentals import Plane
    >>> from graphix.opengraph import OpenGraph, Measurement
    >>>
    >>> graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    >>>
    >>> measurements = {i: Measurement(0.5 * i, Plane.XY) for i in range(2)}
    >>> input_nodes = [0]
    >>> output_nodes = [2]
    >>> og = OpenGraph(graph, measurements, input_nodes, output_nodes)
    """

    graph: nx.Graph[int]
    measurements: Mapping[int, _M_co]  # TODO: Rename `measurement_labels` ?
    input_nodes: list[int]  # Inputs are ordered
    output_nodes: list[int]  # Outputs are ordered

    def __post_init__(self) -> None:
        """Validate the open graph."""
        if not set(self.measurements).issubset(self.graph.nodes):
            raise ValueError("All measured nodes must be part of the graph's nodes.")
        if not set(self.input_nodes).issubset(self.graph.nodes):
            raise ValueError("All input nodes must be part of the graph's nodes.")
        if not set(self.output_nodes).issubset(self.graph.nodes):
            raise ValueError("All output nodes must be part of the graph's nodes.")
        if set(self.output_nodes) & self.measurements.keys():
            raise ValueError("Output node cannot be measured.")
        if len(set(self.input_nodes)) != len(self.input_nodes):
            raise ValueError("Input nodes contain duplicates.")
        if len(set(self.output_nodes)) != len(self.output_nodes):
            raise ValueError("Output nodes contain duplicates.")

    # def isclose(self, other: OpenGraph, rel_Mol: float = 1e-09, abs_Mol: float = 0.0) -> bool:
    #     """Return `True` if two open graphs implement approximately the same unitary operator.

    #     Ensures the structure of the graphs are the same and all
    #     measurement angles are sufficiently close.

    #     This doesn't check they are equal up to an isomorphism.

    #     """
    #     if not nx.utils.graphs_equal(self.graph, other.graph):
    #         return False

    #     if self.input_nodes != other.input_nodes or self.output_nodes != other.output_nodes:
    #         return False

    #     if set(self.measurements.keys()) != set(other.measurements.keys()):
    #         return False

    #     return all(
    #         m.isclose(other.measurements[node], rel_Mol=rel_Mol, abs_Mol=abs_Mol)
    #         for node, m in self.measurements.items()
    #     )

    @staticmethod
    def from_pattern(pattern: Pattern) -> OpenGraph[Measurement]:
        """Initialise an `OpenGraph` object based on the resource-state graph associated with the measurement pattern."""
        graph = pattern.extract_graph()

        input_nodes = pattern.input_nodes
        output_nodes = pattern.output_nodes

        meas_planes = pattern.get_meas_plane()
        meas_angles = pattern.get_angles()
        measurements: Mapping[int, Measurement] = {
            node: Measurement(meas_angles[node], meas_planes[node]) for node in meas_angles
        }

        return OpenGraph(graph, measurements, input_nodes, output_nodes)

    # def to_pattern(self) -> Pattern:
    #     """Convert the `OpenGraph` into a `Pattern`.

    #     Will raise an exception if the open graph does not have flow, gflow, or
    #     Pauli flow.
    #     The pattern will be generated using maximally-delayed flow.
    #     """
    #     g = self.graph.copy()
    #     input_nodes = self.input_nodes
    #     output_nodes = self.output_nodes
    #     meas = self.measurements

    #     angles = {node: m.angle for node, m in meas.items()}
    #     planes = {node: m.plane for node, m in meas.items()}

    #     return graphix.generator.generate_from_graph(g, angles, input_nodes, output_nodes, planes)

    # def compose(self, other: OpenGraph[_M], mapping: Mapping[int, int]) -> tuple[OpenGraph[_M], dict[int, int]]:
    #     r"""Compose two open graphs by merging subsets of nodes from `self` and `other`, and relabeling the nodes of `other` that were not merged.

    #     Parameters
    #     ----------
    #     other : OpenGraph
    #         Open graph to be composed with `self`.
    #     mapping: dict[int, int]
    #         Partial relabelling of the nodes in `other`, with `keys` and `values` denoting the old and new node labels, respectively.

    #     Returns
    #     -------
    #     og: OpenGraph
    #         composed open graph
    #     mapping_complete: dict[int, int]
    #         Complete relabelling of the nodes in `other`, with `keys` and `values` denoting the old and new node label, respectively.

    #     Notes
    #     -----
    #     Let's denote :math:`\{G(V_1, E_1), I_1, O_1\}` the open graph `self`, :math:`\{G(V_2, E_2), I_2, O_2\}` the open graph `other`, :math:`\{G(V, E), I, O\}` the resulting open graph `og` and `{v:u}` an element of `mapping`.

    #     We define :math:`V, U` the set of nodes in `mapping.keys()` and `mapping.values()`, and :math:`M = U \cap V_1` the set of merged nodes.

    #     The open graph composition requires that
    #     - :math:`V \subseteq V_2`.
    #     - If both `v` and `u` are measured, the corresponding measurements must have the same plane and angle.
    #      The returned open graph follows this convention:
    #     - :math:`I = (I_1 \cup I_2) \setminus M \cup (I_1 \cap I_2 \cap M)`,
    #     - :math:`O = (O_1 \cup O_2) \setminus M \cup (O_1 \cap O_2 \cap M)`,
    #     - If only one node of the pair `{v:u}` is measured, this measure is assigned to :math:`u \in V` in the resulting open graph.
    #     - Input (and, respectively, output) nodes in the returned open graph have the order of the open graph `self` followed by those of the open graph `other`. Merged nodes are removed, except when they are input (or output) nodes in both open graphs, in which case, they appear in the order they originally had in the graph `self`.
    #     """
    #     if not (mapping.keys() <= other.graph.nodes):
    #         raise ValueError("Keys of mapping must be correspond to nodes of other.")
    #     if len(mapping) != len(set(mapping.values())):
    #         raise ValueError("Values in mapping contain duplicates.")
    #     for v, u in mapping.items():
    #         if (
    #             (vm := other.measurements.get(v)) is not None
    #             and (um := self.measurements.get(u)) is not None
    #             and not vm.isclose(um)  # TODO: How do we ensure that planes, axis, etc. are the same ?
    #         ):
    #             raise ValueError(f"Attempted to merge nodes {v}:{u} but have different measurements")

    #     shift = max(*self.graph.nodes, *mapping.values()) + 1

    #     mapping_sequential = {
    #         node: i for i, node in enumerate(sorted(other.graph.nodes - mapping.keys()), start=shift)
    #     }  # assigns new labels to nodes in other not specified in mapping

    #     mapping_complete = {**mapping, **mapping_sequential}

    #     g2_shifted = nx.relabel_nodes(other.graph, mapping_complete)
    #     g = nx.compose(self.graph, g2_shifted)

    #     merged = set(mapping_complete.values()) & self.graph.nodes

    #     def merge_ports(p1: Iterable[int], p2: Iterable[int]) -> list[int]:
    #         p2_mapped = [mapping_complete[node] for node in p2]
    #         p2_set = set(p2_mapped)
    #         part1 = [node for node in p1 if node not in merged or node in p2_set]
    #         part2 = [node for node in p2_mapped if node not in merged]
    #         return part1 + part2

    #     input_nodes = merge_ports(self.input_nodes, other.input_nodes)
    #     output_nodes = merge_ports(self.output_nodes, other.output_nodes)

    #     measurements_shifted = {mapping_complete[i]: meas for i, meas in other.measurements.items()}
    #     measurements = {**self.measurements, **measurements_shifted}

    #     return OpenGraph(g, measurements, input_nodes, output_nodes), mapping_complete

    # TODO: check if nodes in input belong to open graph ?
    def neighbors(self, nodes: Collection[int]) -> set[int]:
        """Return the set containing the neighborhood of a set of nodes.

        Parameters
        ----------
        nodes : Collection[int]
            Set of nodes whose neighborhood is to be found

        Returns
        -------
        neighbors_set : set[int]
            Neighborhood of set `nodes`.
        """
        neighbors_set: set[int] = set()
        for node in nodes:
            neighbors_set |= set(self.graph.neighbors(node))
        return neighbors_set

    def odd_neighbors(self, nodes: Collection[int]) -> set[int]:
        """Return the set containing the odd neighborhood of a set of nodes.

        Parameters
        ----------
        nodes : Collection[int]
            Set of nodes whose odd neighborhood is to be found

        Returns
        -------
        odd_neighbors_set : set[int]
            Odd neighborhood of set `nodes`.
        """
        odd_neighbors_set: set[int] = set()
        for node in nodes:
            odd_neighbors_set ^= self.neighbors([node])
        return odd_neighbors_set

    def find_causal_flow(self: OpenGraph[_PM_co]) -> CausalFlow | None:
        return find_cflow(self)

    def find_gflow(self: OpenGraph[_PM_co]) -> GFlow | None:
        aog = PlanarAlgebraicOpenGraph(self)
        correction_matrix = compute_correction_matrix(aog)
        if correction_matrix is None:
            return None
        return GFlow.from_correction_matrix(
            correction_matrix
        )  # The constructor can return `None` if the correction matrix is not compatible with any partial order on the open graph.

    def find_pauli_flow(self: OpenGraph[_M_co]) -> PauliFlow | None:
        aog = AlgebraicOpenGraph(self)
        correction_matrix = compute_correction_matrix(aog)
        if correction_matrix is None:
            return None
        return PauliFlow.from_correction_matrix(
            correction_matrix
        )  # The constructor can return `None` if the correction matrix is not compatible with any partial order on the open graph.
