"""Module for flow classes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Generic, Self, override

import networkx as nx
import numpy as np

from graphix._linalg import MatGF2
from graphix.command import E, M, N, X, Z
from graphix.flow._find_gpflow import CorrectionMatrix, _M_co, _PM_co, compute_partial_order_layers
from graphix.pattern import Pattern

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet
    from typing import Self

    from graphix.measurements import Measurement
    from graphix.opengraph_ import OpenGraph

TotalOrder = Sequence[int]


@dataclass(frozen=True)
class XZCorrections(Generic[_M_co]):
    og: OpenGraph[_M_co]
    x_corrections: Mapping[int, AbstractSet[int]]  # {domain: nodes}
    z_corrections: Mapping[int, AbstractSet[int]]  # {domain: nodes}
    partial_order_layers: Sequence[AbstractSet[int]]
    # Often XZ-corrections are extracted from a flow whose partial order can be used to construct a pattern from the corrections. We store it to avoid recalculating it twice.

    @staticmethod
    def from_measured_nodes_mapping(
        og: OpenGraph[_M_co],
        x_corrections: Mapping[int, AbstractSet[int]] | None = None,
        z_corrections: Mapping[int, AbstractSet[int]] | None = None,
    ) -> XZCorrections[_M_co]:
        """Create an `XZCorrections` instance from the XZ-corrections mappings.

        Parameters
        ----------
        og : OpenGraph[_M_co]
            Open graph with respect to which the corrections are defined.
        x_corrections : Mapping[int, AbstractSet[int]] | None
            Mapping of X-corrections: in each (`key`, `value`) pair, `key` is a measured node, and `value` is the set of nodes on which an X-correction must be applied depending on the measurement result of `key`.
        z_corrections : Mapping[int, AbstractSet[int]] | None
            Mapping of X-corrections: in each (`key`, `value`) pair, `key` is a measured node, and `value` is the set of nodes on which an X-correction must be applied depending on the measurement result of `key`.

        Returns
        -------
        XZCorrections[_M_co]

        Notes
        -----
        This method computes the partial order induced by the XZ-corrections.
        """
        x_corrections = x_corrections or {}
        z_corrections = z_corrections or {}

        nodes_set = set(og.graph.nodes)
        outputs_set = set(og.output_nodes)
        non_outputs_set = nodes_set - outputs_set

        if not set(x_corrections).issubset(non_outputs_set):
            raise ValueError("Keys of input X-corrections contain non-measured nodes.")
        if not set(z_corrections).issubset(non_outputs_set):
            raise ValueError("Keys of input Z-corrections contain non-measured nodes.")

        dag = _corrections_to_dag(x_corrections, z_corrections)
        partial_order_layers = _dag_to_partial_order_layers(dag)

        # The first element in the output of `_dag_to_partial_order_layers(dag)` may or may not contain a subset of the output nodes.
        # The first element in `XZCorrections.partial_order_layers` should contain all output nodes.

        shift = 1 if partial_order_layers[0].issubset(outputs_set) else 0
        partial_order_layers = [outputs_set, *partial_order_layers[shift:]]

        ordered_nodes = {node for layer in partial_order_layers for node in layer}
        if not ordered_nodes.issubset(nodes_set):
            raise ValueError("Values of input mapping contain labels which are not nodes of the input open graph.")

        # We append to the last layer (first measured nodes) all the non-output nodes not involved in the corrections
        if unordered_nodes := nodes_set - ordered_nodes:
            partial_order_layers.append(unordered_nodes)

        return XZCorrections(og, x_corrections, z_corrections, partial_order_layers)

    def to_pattern(
        self: XZCorrections[Measurement],
        total_measurement_order: TotalOrder | None = None,
    ) -> Pattern:
        if total_measurement_order is None:
            total_measurement_order = self.generate_total_measurement_order()
        elif not self.is_compatible(total_measurement_order):
            raise ValueError(
                "The input total measurement order is not compatible with the partial order induced by the correction sets."
            )

        pattern = Pattern(input_nodes=self.og.input_nodes)
        non_input_nodes = set(self.og.graph.nodes) - set(self.og.input_nodes)

        for i in non_input_nodes:
            pattern.add(N(node=i))
        for e in self.og.graph.edges:
            pattern.add(E(nodes=e))

        for measured_node in total_measurement_order:
            measurement = self.og.measurements[measured_node]
            pattern.add(M(node=measured_node, plane=measurement.plane, angle=measurement.angle))

            for corrected_node in self.z_corrections.get(measured_node, []):
                pattern.add(Z(node=corrected_node, domain={measured_node}))

            for corrected_node in self.x_corrections.get(measured_node, []):
                pattern.add(X(node=corrected_node, domain={measured_node}))

        pattern.reorder_output_nodes(self.og.output_nodes)
        return pattern

    def generate_total_measurement_order(self) -> TotalOrder:
        """Generate a sequence of all the non-output nodes in the open graph in an arbitrary order compatible with the intrinsic partial order of the XZ-corrections.

        Returns
        -------
        TotalOrder
        """
        total_order = [node for layer in reversed(self.partial_order_layers[1:]) for node in layer]

        assert set(total_order) == set(self.og.graph.nodes) - set(self.og.output_nodes)
        return total_order

    def extract_dag(self) -> nx.DiGraph[int]:
        """Extract the directed graph induced by the corrections.

        Returns
        -------
        nx.DiGraph[int]
            Directed graph in which an edge `i -> j` represents a correction applied to qubit `j`, conditioned on the measurement outcome of qubit `i`.

        Notes
        -----
        - Not all nodes of the underlying open graph are nodes of the returned directed graph, but only those involved in a correction, either as corrected qubits or belonging to a correction domain.
        - Despite the name, the output of this method is not guranteed to be a directed acyclical graph (i.e., a directed graph without any loops). This is only the case if the `XZCorrections` object is well formed, which is verified by the method :func:`XZCorrections.is_wellformed`.
        """
        return _corrections_to_dag(self.x_corrections, self.z_corrections)

    def is_compatible(self, total_measurement_order: TotalOrder) -> bool:
        """Verify if a given total measurement order is compatible with the intrisic partial order of the XZ-corrections.

        Parameters
        ----------
        total_measurement_order: TotalOrder
            An ordered sequence of all the non-output nodes in the open graph.

        Returns
        -------
        bool
            `True` if `total_measurement_order` is compatible with `self.partial_order_layers`, `False` otherwise.
        """
        non_outputs_set = set(self.og.graph.nodes) - set(self.og.output_nodes)

        if set(total_measurement_order) != non_outputs_set:
            print("The input total measurement order does not contain all non-output nodes.")
            return False

        if len(total_measurement_order) != len(non_outputs_set):
            print("The input total measurement order contains duplicates.")
            return False

        layer = len(self.partial_order_layers) - 1  # First layer to be measured.

        for node in total_measurement_order:
            while True:
                if node in self.partial_order_layers[layer]:
                    break
                layer -= 1
                if layer == 0:  # Layer 0 only contains output nodes.
                    return False

        return True

    # def is_wellformed(self) -> bool:
    #     """Verify if `Corrections` object is well formed.

    #     Returns
    #     -------
    #     bool
    #         `True` if `self` is well formed, `False` otherwise.

    #     Notes
    #     -----
    #     This method verifies that:
    #         - Corrected nodes belong to the underlying open graph.
    #         - Nodes in domain set are measured.
    #         - Corrections are runnable. This amounts to verifying that the corrections-induced directed graph does not have loops.
    #     """
    #     for corr_type in ["X", "Z"]:
    #         corrections = getattr(self, f"{corr_type.lower()}_corrections")
    #         for node, domain in corrections.items():
    #             if node not in self.og.graph.nodes:
    #                 print(
    #                     f"Cannot apply {corr_type} correction. Corrected node {node} does not belong to the open graph."
    #                 )
    #                 return False
    #             if not domain.issubset(self.og.measurements):
    #                 print(f"Cannot apply {corr_type} correction. Domain nodes {domain} are not measured.")
    #                 return False
    #     if nx.is_directed_acyclic_graph(self.extract_dag()):
    #         print("Corrections are not runnable since the induced directed graph contains cycles.")
    #         return False

    #     return True


@dataclass(frozen=True)
class PauliFlow(Generic[_M_co]):
    og: OpenGraph[_M_co]
    correction_function: Mapping[int, set[int]]
    partial_order_layers: Sequence[AbstractSet[int]]

    @classmethod
    def from_correction_matrix(cls, correction_matrix: CorrectionMatrix) -> Self | None:
        correction_function = correction_matrix.to_correction_function()
        partial_order_layers = compute_partial_order_layers(correction_matrix)
        if partial_order_layers is None:
            return None

        return cls(correction_matrix.aog.og, correction_function, partial_order_layers)

    def to_corrections(self) -> XZCorrections[_M_co]:
        """Compute the X and Z corrections induced by the Pauli flow encoded in `self`.

        Returns
        -------
        Corrections[_M_co]

        Notes
        -----
        This function partially implements Theorem 4 of Browne et al., NJP 9, 250 (2007). The generated X and Z corrections can be used to obtain a robustly deterministic pattern on the underlying open graph.
        """
        future = self.partial_order_layers[0]
        x_corrections: dict[int, set[int]] = {}  # {domain: nodes}
        z_corrections: dict[int, set[int]] = {}  # {domain: nodes}

        for layer in self.partial_order_layers[1:]:
            for measured_node in layer:
                correcting_set = self.correction_function[measured_node]
                # Conditionals avoid storing empty correction sets
                if x_corrected_nodes := correcting_set & future:
                    x_corrections[measured_node] = x_corrected_nodes
                if z_corrected_nodes := self.og.odd_neighbors(correcting_set) & future:
                    z_corrections[measured_node] = z_corrected_nodes

            future |= layer

        return XZCorrections(self.og, x_corrections, z_corrections, self.partial_order_layers)

    def is_well_formed(self) -> bool:
        r"""Verify if flow object is well formed.

        Returns
        -------
        bool
            `True` if `self` is well formed, `False` otherwise.

        Notes
        -----
        This method verifies that:
            - The correction function's domain and codomain respectively are non-output and non-input nodes.
            - The product of the flow-demand and the correction matrices is the identity matrix, :math:`MC = \mathbb{1}`.
            - The product of the order-demand and the correction matrices is the adjacency matrix of a DAG compatible with `self.partial_order_layers`.
        """
        domain = set(self.correction_function)
        if not domain.intersection(self.og.output_nodes):
            print("Invalid flow. Domain of the correction function includes output nodes.")
            return False

        codomain = set().union(*self.correction_function.values())
        if not codomain.intersection(self.og.input_nodes):
            print("Invalid flow. Codomain of the correction function includes input nodes.")
            return False

        correction_matrix = CorrectionMatrix.from_correction_function(self.og, self.correction_function)

        aog, c_matrix = correction_matrix

        identity = MatGF2(np.eye(len(aog.non_outputs), dtype=np.uint8))
        mc_matrix = aog.flow_demand_matrix.mat_mul(c_matrix)
        if not np.all(mc_matrix == identity):
            print(
                "Invalid flow. The product of the flow-demand and the correction matrices is not the identity matrix, MC ≠ 1"
            )
            return False

        partial_order_layers = compute_partial_order_layers(correction_matrix)
        if partial_order_layers is None:
            print(
                "Invalid flow. The correction function is not compatible with a partial order on the open graph. The product of the order-demand and the correction matrices NC does not form a DAG."
            )
            return False

        # TODO: Verify that self.partial_order_layers is compatible with partial_order_layers

        return True


@dataclass(frozen=True)
class GFlow(PauliFlow[_PM_co], Generic[_PM_co]):
    @override
    def to_corrections(self) -> XZCorrections[_PM_co]:
        r"""Compute the X and Z corrections induced by the generalised flow encoded in `self`.

        Returns
        -------
        Corrections[Plane]

        Notes
        -----
        - This function partially implements Theorem 2 of Browne et al., NJP 9, 250 (2007). The generated X and Z corrections can be used to obtain a robustly deterministic pattern on the underlying open graph.

        - Contrary to the overridden method in the parent class, here we do not need any information on the partial order to build the corrections since a valid correction function :math:`g` guarantees that both :math:`g(i)\setminus \{i\}` and :math:`Odd(g(i))` are in the future of :math:`i`.
        """
        x_corrections: dict[int, set[int]] = {}  # {domain: nodes}
        z_corrections: dict[int, set[int]] = {}  # {domain: nodes}

        for measured_node, correcting_set in self.correction_function.items():
            # Conditionals avoid storing empty correction sets
            if x_corrected_nodes := correcting_set - {measured_node}:
                x_corrections[measured_node] = x_corrected_nodes
            if z_corrected_nodes := self.og.odd_neighbors(correcting_set) - {measured_node}:
                z_corrections[measured_node] = z_corrected_nodes

        return XZCorrections(self.og, x_corrections, z_corrections, self.partial_order_layers)


@dataclass(frozen=True)
class CausalFlow(
    GFlow[_PM_co], Generic[_PM_co]
):  # TODO: change parametric type to Plane.XY. Requires defining Plane.XY as subclasses of Plane
    @override
    @classmethod
    def from_correction_matrix(cls, correction_matrix: CorrectionMatrix) -> None:
        raise NotImplementedError("Initialization of a causal flow from a correction matrix is not supported.")

    @override
    def to_corrections(self) -> XZCorrections[_PM_co]:
        r"""Compute the X and Z corrections induced by the causal flow encoded in `self`.

        Returns
        -------
        Corrections[Plane]

        Notes
        -----
        This function partially implements Theorem 1 of Browne et al., NJP 9, 250 (2007). The generated X and Z corrections can be used to obtain a robustly deterministic pattern on the underlying open graph.
        """
        x_corrections: dict[int, set[int]] = {}  # {domain: nodes}
        z_corrections: dict[int, set[int]] = {}  # {domain: nodes}

        for measured_node, correcting_set in self.correction_function.items():
            # Conditionals avoid storing empty correction sets
            if x_corrected_nodes := correcting_set:
                x_corrections[measured_node] = x_corrected_nodes
            if z_corrected_nodes := self.og.neighbors(correcting_set) - {measured_node}:
                z_corrections[measured_node] = z_corrected_nodes

        return XZCorrections(self.og, x_corrections, z_corrections, self.partial_order_layers)


def _corrections_to_dag(
    x_corrections: Mapping[int, AbstractSet[int]], z_corrections: Mapping[int, AbstractSet[int]]
) -> nx.DiGraph[int]:
    relations: set[tuple[int, int]] = set()

    for measured_node, corrected_nodes in x_corrections.items():
        relations.update(product([measured_node], corrected_nodes))

    for measured_node, corrected_nodes in z_corrections.items():
        relations.update(product([measured_node], corrected_nodes))

    return nx.DiGraph(relations)


def _dag_to_partial_order_layers(dag: nx.DiGraph[int]) -> list[set[int]]:

    try:
        topo_gen = reversed(list(nx.topological_generations(dag)))
    except nx.NetworkXUnfeasible:
        raise ValueError(
            "XZ-corrections are not runnable since the induced directed graph contains closed loops."
        ) from nx.NetworkXUnfeasible

    return [set(layer) for layer in topo_gen]


###########
# OLD functions
###########


# @dataclass(frozen=True)
# class PartialOrder:
#     """Class for storing and manipulating the partial order in a flow.

#     Attributes
#     ----------
#     dag: nx.DiGraph[int]
#         Directed Acyclical Graph (DAG) representing the partial order. The transitive closure of `dag` yields all the relations in the partial order.

#     layers: Mapping[int, AbstractSet[int]]
#         Mapping storing the partial order in a layer structure.
#         The pair `(key, value)` corresponds to the layer and the set of nodes in that layer.
#         Layer 0 corresponds to the largest nodes in the partial order. In general, if `i > j`, then nodes in `layers[j]` are in the future of nodes in `layers[i]`.

#     """

#     dag: nx.DiGraph[int]
#     layers: Mapping[int, AbstractSet[int]]

#     @classmethod
#     def from_adj_matrix(cls, adj_mat: npt.NDArray[np.uint8], nodelist: Collection[int] | None = None) -> PartialOrder:
#         """Construct a partial order from an adjacency matrix representing a DAG.

#         Parameters
#         ----------
#         adj_mat: npt.NDArray[np.uint8]
#             Adjacency matrix of the DAG. A nonzero element `adj_mat[i,j]` represents a link `i -> j`.
#         node_list: Collection[int] | None
#             Mapping between matrix indices and node labels. Optional, defaults to `None`.

#         Returns
#         -------
#         PartialOrder

#         Notes
#         -----
#         The `layers` attribute of the `PartialOrder` attribute is obtained by performing a topological sort on the DAG. This routine verifies that the input directed graph is indeed acyclical. See :func:`_compute_layers_from_dag` for more details.
#         """
#         dag = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph, nodelist=nodelist)
#         layers = _compute_layers_from_dag(dag)
#         return cls(dag=dag, layers=layers)

#     @classmethod
#     def from_relations(cls, relations: Collection[tuple[int, int]]) -> PartialOrder:
#         """Construct a partial order from the order relations.

#         Parameters
#         ----------
#         relations: Collection[tuple[int, int]]
#             Collection of relations in the partial order. A tuple `(a, b)` represents `a > b` in the partial order.

#         Returns
#         -------
#         PartialOrder

#         Notes
#         -----
#         The `layers` attribute of the `PartialOrder` attribute is obtained by performing a topological sort on the DAG. This routine verifies that the input directed graph is indeed acyclical. See :func:`_compute_layers_from_dag` for more details.
#         """
#         dag = nx.DiGraph(relations)
#         layers = _compute_layers_from_dag(dag)
#         return cls(dag=dag, layers=layers)

#     @classmethod
#     def from_layers(cls, layers: Mapping[int, AbstractSet[int]]) -> PartialOrder:
#         dag = _compute_dag_from_layers(layers)
#         return cls(dag=dag, layers=layers)

#     @classmethod
#     def from_corrections(cls, corrections: XZCorrections) -> PartialOrder:
#         relations: set[tuple[int, int]] = set()

#         for node, domain in corrections.x_corrections.items():
#             relations.update(product([node], domain))

#         for node, domain in corrections.z_corrections.items():
#             relations.update(product([node], domain))

#         return cls.from_relations(relations)

#     @property
#     def nodes(self) -> set[int]:
#         """Return nodes in the partial order."""
#         return set(self.dag.nodes)

#     @property
#     def node_layer_mapping(self) -> dict[int, int]:
#         """Return layers in the form `{node: layer}`."""
#         mapping: dict[int, int] = {}
#         for layer, nodes in self.layers.items():
#             mapping.update(dict.fromkeys(nodes, layer))

#         return mapping

#     @cached_property
#     def transitive_closure(self) -> set[tuple[int, int]]:
#         """Return the transitive closure of the Directed Acyclic Graph (DAG) encoding the partial order.

#         Returns
#         -------
#         set[tuple[int, int]]
#             A tuple `(i, j)` belongs to the transitive closure of the DAG if `i > j` according to the partial order.
#         """
#         return set(nx.transitive_closure_dag(self.dag).edges())

#     def greater(self, a: int, b: int) -> bool:
#         """Verify order between two nodes.

#         Parameters
#         ----------
#         a : int
#         b : int

#         Returns
#         -------
#         bool
#             `True` if `a > b` in the partial order, `False` otherwise.

#         Raises
#         ------
#         ValueError
#             If either node `a` or `b` is not included in the definition of the partial order.
#         """
#         if a not in self.nodes:
#             raise ValueError(f"Node a = {a} is not included in the partial order.")
#         if b not in self.nodes:
#             raise ValueError(f"Node b = {b} is not included in the partial order.")
#         return (a, b) in self.transitive_closure

#     def compute_future(self, node: int) -> set[int]:
#         """Compute the future of `node`.

#         Parameters
#         ----------
#         node : int
#             Node for which the future is computed.

#         Returns
#         -------
#         set[int]
#             Set of nodes `i` such that `i > node` in the partial order.
#         """
#         if node not in self.nodes:
#             raise ValueError(f"Node {node} is not included in the partial order.")

#         return {i for i, j in self.transitive_closure if j == node}

#     def is_compatible(self, other: PartialOrder) -> bool:
#         r"""Verify compatibility between two partial orders.

#         Parameters
#         ----------
#         other : PartialOrder

#         Returns
#         -------
#         bool
#             `True` if partial order `self` is compatible with partial order `other`, `False` otherwise.

#         Notes
#         -----
#         We define partial-order compatibility as follows:
#             A partial order :math:`<_P` on a set :math:`U` is compatible with a partial order :math:`<_Q` on a set :math:`V` iff :math:`a <_P b \rightarrow a <_Q b \forall a, b \in U`.
#             This definition of compatibility requires that :math:`U \subseteq V`.
#             Further, it is not symmetric.
#         """
#         return self.transitive_closure.issubset(other.transitive_closure)


# def _compute_layers_from_dag(dag: nx.DiGraph[int]) -> dict[int, set[int]]:
#     try:
#         generations = reversed(list(nx.topological_generations(dag)))
#         return {layer: set(generation) for layer, generation in enumerate(generations)}
#     except nx.NetworkXUnfeasible as exc:
#         raise ValueError("Partial order contains loops.") from exc


# def _compute_dag_from_layers(layers: Mapping[int, AbstractSet[int]]) -> nx.DiGraph[int]:
#     max_layer = max(layers)
#     relations: list[tuple[int, int]] = []
#     visited_nodes: set[int] = set()

#     for i, j in pairwise(reversed(range(max_layer + 1))):
#         layer_curr, layer_next = layers[i], layers[j]
#         if layer_curr & visited_nodes:
#             raise ValueError(f"Layer {i} contains nodes in previous layers.")
#         visited_nodes |= layer_curr
#         relations.extend(product(layer_curr, layer_next))

#     if layers[0] & visited_nodes:
#         raise ValueError(f"Layer {i} contains nodes in previous layers.")

#     return nx.DiGraph(relations)
