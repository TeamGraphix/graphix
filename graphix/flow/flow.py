"""Module for flow classes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise, product
from typing import TYPE_CHECKING, Generic, Self, override

import networkx as nx

from graphix.command import E, M, N, X, Z
from graphix.flow._find_pflow import compute_correction_function, compute_partial_order_layers
from graphix.fundamentals import Axis, Plane, Sign
from graphix.graphix._linalg import MatGF2
from graphix.graphix.flow._find_pflow import AlgebraicOpenGraph
from graphix.opengraph_ import _M, OpenGraph
from graphix.pattern import Pattern

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from collections.abc import Set as AbstractSet
    from typing import Self

    import numpy as np
    import numpy.typing as npt

    from graphix._linalg import MatGF2
    from graphix.flow._find_pflow import AlgebraicOpenGraph
    from graphix.measurements import ExpressionOrFloat

TotalOrder = Sequence[int]


@dataclass(frozen=True)
class Corrections(Generic[_M]):
    og: OpenGraph[_M]
    x_corrections: dict[int, set[int]]   # {node: domain}
    z_corrections: dict[int, set[int]]   # {node: domain}

    def extract_dag(self) -> nx.DiGraph[int]:
        """Extract directed graph induced by the corrections.

        Returns
        -------
        nx.DiGraph[int]
            Directed graph in which an edge `i -> j` represents a correction applied to qubit `i`, conditioned on the measurement outcome of qubit `j`.

        Notes
        -----
        - Not all nodes of the underlying open graph are nodes of the returned directed graph, but only those involved in a correction, either as corrected qubits or belonging to a correction domain.
        - Despite the name, the output of this method is not guranteed to be a directed acyclical graph (i.e., a directed graph without any loops). This is only the case if the `Corrections` object is well formed, which is verified by the method :func:`Corrections.is_wellformed`.
        """
        relations: set[tuple[int, int]] = set()

        for node, domain in self.x_corrections.items():
            relations.update(product([node], domain))

        for node, domain in self.z_corrections.items():
            relations.update(product([node], domain))

        return nx.DiGraph(relations)

    def is_wellformed(self, verbose: bool = True) -> bool:
        """Verify if `Corrections` object is well formed.

        Parameters
        ----------
        verbose : bool
            Optional flag that indicates the source of the issue when `self` is malformed. Defaults to `True`.

        Returns
        -------
        bool
            `True` if `self` is well formed, `False` otherwise.

        Notes
        -----
        This method verifies that:
            - Corrected nodes belong to the underlying open graph.
            - Nodes in domain set are measured.
            - Corrections are runnable. This amounts to verifying that the corrections-induced directed graph does not have loops.
        """
        for corr_type in ['X', 'Z']:
            corrections = getattr(self, f"{corr_type.lower()}_corrections")
            for node, domain in corrections.items():
                if node not in self.og.graph.nodes:
                    if verbose:
                        print(f"Cannot apply {corr_type} correction. Corrected node {node} does not belong to the open graph.")
                    return False
                if not domain.issubset(self.og.measurements):
                    if verbose:
                        print(f"Cannot apply {corr_type} correction. Domain nodes {domain} are not measured.")
                    return False
        if nx.is_directed_acyclic_graph(self.extract_dag()):
            if verbose:
                print("Corrections are not runnable since the induced directed graph contains cycles.")
            return False

        return True

    def is_compatible(self, total_order: TotalOrder) -> bool:
        # Verify compatibility
        # Verify nodes are in open graph
        return True

    def to_pattern(self, angles: Mapping[int, ExpressionOrFloat | Sign], total_order: TotalOrder | None = None) -> Pattern:

        # TODO: Should we verify thar corrections are well formed ? If we did so, and the total order is inferred from the corrections, we are doing a topological sort twice

        # TODO: Do we want to raise an error or just a warning and assign 0 by default ?
        if not angles.keys() == self.og.measurements.keys():
            raise ValueError("All measured nodes in the open graph must have an assigned angle label.")

        if total_order is None:
            total_order = list(reversed(list(nx.topological_sort(self.extract_dag()))))
        elif not self.is_compatible(total_order):
            raise ValueError("The input total order is not compatible with the partial order induced by the correction sets.")

        pattern = Pattern(input_nodes=self.og.input_nodes)
        non_input_nodes = set(self.og.graph.nodes) - set(self.og.input_nodes)

        for i in non_input_nodes:
            pattern.add(N(node=i))
        for e in self.og.graph.edges:
            pattern.add(E(nodes=e))

        for node in total_order:
            if node in self.og.output_nodes:
                break

            # TODO: the following block is hideous.
            # Refactor Plane and Axis ?
            # Abstract class Plane, Plane.XY, .XZ, .YZ subclasses ?
            # Axis X subclass of Plane.XY, Plane.XZ, etc. ?
            # Method Axis, Sign -> Plane, angle

            meas_label = self.og.measurements[node]
            angle_label = angles[node]

            if isinstance(meas_label, Plane):
                assert not isinstance(angle_label, Sign)
                pattern.add(M(node=node, plane=meas_label, angle=angle_label))
            else:
                assert isinstance(angle_label, Sign)
                if meas_label == Axis.X:
                    plane = Plane.XY
                    angle = 0 if angle_label is Sign.PLUS else 1
                elif meas_label == Axis.Y:
                    plane = Plane.XY
                    angle = 0.5 if angle_label is Sign.PLUS else 1.5
                elif meas_label == Axis.Z:
                    plane = Plane.XZ
                    angle = 0 if angle_label is Sign.PLUS else 1

                pattern.add(M(node=node, plane=plane, angle=angle))

            if node in self.z_corrections:
                pattern.add(Z(node=node, domain=self.z_corrections[node]))
            if node in self.x_corrections:
                pattern.add(X(node=node, domain=self.x_corrections[node]))

        pattern.reorder_output_nodes(self.og.output_nodes)
        return pattern


@dataclass(frozen=True)
class PauliFlow(Generic[_M]):
    og: OpenGraph[_M]
    correction_function: Mapping[int, set[int]]
    partial_order_layers: Sequence[AbstractSet[int]]

    # TODO: Add parametric dependence of AlgebraicOpenGraph
    @classmethod
    def from_correction_matrix(cls, aog: AlgebraicOpenGraph, correction_matrix: MatGF2) -> Self | None:
        correction_function = compute_correction_function(aog, correction_matrix)
        partial_order_layers = compute_partial_order_layers(aog, correction_matrix)
        if partial_order_layers is None:
            return None

        return cls(aog.og, correction_function, partial_order_layers)

    def to_corrections(self) -> Corrections[_M]:
        """Compute the X and Z corrections induced by the Pauli flow encoded in `self`.

        Returns
        -------
        Corrections[_M]

        Notes
        -----
        This function partially implements Theorem 4 of Browne et al., NJP 9, 250 (2007). The generated X and Z corrections can be used to obtain a robustly deterministic pattern on the underlying open graph.
        """
        future: set[int] = self.partial_order_layers[0]
        x_corrections: dict[int, set[int]] = defaultdict(set)  # {node: domain}
        z_corrections: dict[int, set[int]] = defaultdict(set)  # {node: domain}

        for layer in self.partial_order_layers[1:]:
            for node in layer:
                corr_set = self.correction_function[node]
                x_corrections[node].update(corr_set & future)
                z_corrections[node].update(self.og.odd_neighbors(corr_set) & future)

            future |= layer

        return Corrections(self.og, x_corrections, z_corrections)

    # TODO
    # def is_well_formed(self) -> bool:

    # TODO: for compatibility with previous encoding of layers.
    # def node_layer_mapping(self) -> dict[int, int]:
    #     """Return layers in the form `{node: layer}`."""
    #     mapping: dict[int, int] = {}
    #     for layer, nodes in self.layers.items():
    #         mapping.update(dict.fromkeys(nodes, layer))

    #     return mapping


@dataclass(frozen=True)
class GFlow(PauliFlow[Plane]):

    @override
    def to_corrections(self) -> Corrections[Plane]:
        r"""Compute the X and Z corrections induced by the generalised flow encoded in `self`.

        Returns
        -------
        Corrections[Plane]

        Notes
        -----
        - This function partially implements Theorem 2 of Browne et al., NJP 9, 250 (2007). The generated X and Z corrections can be used to obtain a robustly deterministic pattern on the underlying open graph.

        - Contrary to the overridden method in the parent class, here we do not need any information on the partial order to build the corrections since a valid correction function :math:`g` guarantees that both :math:`g(i)\setminus \{i\}` and :math:`Odd(g(i))` are in the future of :math:`i`.
        """
        x_corrections: dict[int, set[int]] = defaultdict(set)  # {node: domain}
        z_corrections: dict[int, set[int]] = defaultdict(set)  # {node: domain}

        for node, corr_set in self.correction_function.items():
            x_corrections[node].update(corr_set - {node})
            z_corrections[node].update(self.og.odd_neighbors(corr_set))

        return Corrections(self.og, x_corrections, z_corrections)


@dataclass(frozen=True)
class CausalFlow(GFlow):  # TODO: change parametric type to Plane.XY. Requires defining Plane.XY as subclasses of Plane

    @override
    @staticmethod
    def from_correction_matrix() -> None:
        raise NotImplementedError("Initialization of a causal flow from a correction matrix is not supported.")

    @override
    def to_corrections(self) -> Corrections[Plane]:
        r"""Compute the X and Z corrections induced by the causal flow encoded in `self`.

        Returns
        -------
        Corrections[Plane]

        Notes
        -----
        This function partially implements Theorem 1 of Browne et al., NJP 9, 250 (2007). The generated X and Z corrections can be used to obtain a robustly deterministic pattern on the underlying open graph.
        """
        x_corrections: dict[int, set[int]] = defaultdict(set)  # {node: domain}
        z_corrections: dict[int, set[int]] = defaultdict(set)  # {node: domain}

        for node, corr_set in self.correction_function.items():
            x_corrections[node].update(corr_set)
            z_corrections[node].update(self.og.neighbors(corr_set) - {node})

        return Corrections(self.og, x_corrections, z_corrections)


@dataclass(frozen=True)
class PartialOrder:
    """Class for storing and manipulating the partial order in a flow.

    Attributes
    ----------
    dag: nx.DiGraph[int]
        Directed Acyclical Graph (DAG) representing the partial order. The transitive closure of `dag` yields all the relations in the partial order.

    layers: Mapping[int, AbstractSet[int]]
        Mapping storing the partial order in a layer structure.
        The pair `(key, value)` corresponds to the layer and the set of nodes in that layer.
        Layer 0 corresponds to the largest nodes in the partial order. In general, if `i > j`, then nodes in `layers[j]` are in the future of nodes in `layers[i]`.

    """

    dag: nx.DiGraph[int]
    layers: Mapping[int, AbstractSet[int]]

    @classmethod
    def from_adj_matrix(cls, adj_mat: npt.NDArray[np.uint8], nodelist: Collection[int] | None = None) -> PartialOrder:
        """Construct a partial order from an adjacency matrix representing a DAG.

        Parameters
        ----------
        adj_mat: npt.NDArray[np.uint8]
            Adjacency matrix of the DAG. A nonzero element `adj_mat[i,j]` represents a link `i -> j`.
        node_list: Collection[int] | None
            Mapping between matrix indices and node labels. Optional, defaults to `None`.

        Returns
        -------
        PartialOrder

        Notes
        -----
        The `layers` attribute of the `PartialOrder` attribute is obtained by performing a topological sort on the DAG. This routine verifies that the input directed graph is indeed acyclical. See :func:`_compute_layers_from_dag` for more details.
        """
        dag = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph, nodelist=nodelist)
        layers = _compute_layers_from_dag(dag)
        return cls(dag=dag, layers=layers)

    @classmethod
    def from_relations(cls, relations: Collection[tuple[int, int]]) -> PartialOrder:
        """Construct a partial order from the order relations.

        Parameters
        ----------
        relations: Collection[tuple[int, int]]
            Collection of relations in the partial order. A tuple `(a, b)` represents `a > b` in the partial order.

        Returns
        -------
        PartialOrder

        Notes
        -----
        The `layers` attribute of the `PartialOrder` attribute is obtained by performing a topological sort on the DAG. This routine verifies that the input directed graph is indeed acyclical. See :func:`_compute_layers_from_dag` for more details.
        """
        dag = nx.DiGraph(relations)
        layers = _compute_layers_from_dag(dag)
        return cls(dag=dag, layers=layers)

    @classmethod
    def from_layers(cls, layers: Mapping[int, AbstractSet[int]]) -> PartialOrder:
        dag = _compute_dag_from_layers(layers)
        return cls(dag=dag, layers=layers)

    @classmethod
    def from_corrections(cls, corrections: Corrections) -> PartialOrder:
        relations: set[tuple[int, int]] = set()

        for node, domain in corrections.x_corrections.items():
            relations.update(product([node], domain))

        for node, domain in corrections.z_corrections.items():
            relations.update(product([node], domain))

        return cls.from_relations(relations)

    @property
    def nodes(self) -> set[int]:
        """Return nodes in the partial order."""
        return set(self.dag.nodes)

    @property
    def node_layer_mapping(self) -> dict[int, int]:
        """Return layers in the form `{node: layer}`."""
        mapping: dict[int, int] = {}
        for layer, nodes in self.layers.items():
            mapping.update(dict.fromkeys(nodes, layer))

        return mapping

    @cached_property
    def transitive_closure(self) -> set[tuple[int, int]]:
        """Return the transitive closure of the Directed Acyclic Graph (DAG) encoding the partial order.

        Returns
        -------
        set[tuple[int, int]]
            A tuple `(i, j)` belongs to the transitive closure of the DAG if `i > j` according to the partial order.
        """
        return set(nx.transitive_closure_dag(self.dag).edges())

    def greater(self, a: int, b: int) -> bool:
        """Verify order between two nodes.

        Parameters
        ----------
        a : int
        b : int

        Returns
        -------
        bool
            `True` if `a > b` in the partial order, `False` otherwise.

        Raises
        ------
        ValueError
            If either node `a` or `b` is not included in the definition of the partial order.
        """
        if a not in self.nodes:
            raise ValueError(f"Node a = {a} is not included in the partial order.")
        if b not in self.nodes:
            raise ValueError(f"Node b = {b} is not included in the partial order.")
        return (a, b) in self.transitive_closure

    def compute_future(self, node: int) -> set[int]:
        """Compute the future of `node`.

        Parameters
        ----------
        node : int
            Node for which the future is computed.

        Returns
        -------
        set[int]
            Set of nodes `i` such that `i > node` in the partial order.
        """
        if node not in self.nodes:
            raise ValueError(f"Node {node} is not included in the partial order.")

        return {i for i, j in self.transitive_closure if j == node}

    def is_compatible(self, other: PartialOrder) -> bool:
        r"""Verify compatibility between two partial orders.

        Parameters
        ----------
        other : PartialOrder

        Returns
        -------
        bool
            `True` if partial order `self` is compatible with partial order `other`, `False` otherwise.

        Notes
        -----
        We define partial-order compatibility as follows:
            A partial order :math:`<_P` on a set :math:`U` is compatible with a partial order :math:`<_Q` on a set :math:`V` iff :math:`a <_P b \rightarrow a <_Q b \forall a, b \in U`.
            This definition of compatibility requires that :math:`U \subseteq V`.
            Further, it is not symmetric.
        """
        return self.transitive_closure.issubset(other.transitive_closure)


###########
# OLD functions

def _compute_layers_from_dag(dag: nx.DiGraph[int]) -> dict[int, set[int]]:
    try:
        generations = reversed(list(nx.topological_generations(dag)))
        return {layer: set(generation) for layer, generation in enumerate(generations)}
    except nx.NetworkXUnfeasible as exc:
        raise ValueError("Partial order contains loops.") from exc


def _compute_dag_from_layers(layers: Mapping[int, AbstractSet[int]]) -> nx.DiGraph[int]:
    max_layer = max(layers)
    relations: list[tuple[int, int]] = []
    visited_nodes: set[int] = set()

    for i, j in pairwise(reversed(range(max_layer + 1))):
        layer_curr, layer_next = layers[i], layers[j]
        if layer_curr & visited_nodes:
            raise ValueError(f"Layer {i} contains nodes in previous layers.")
        visited_nodes |= layer_curr
        relations.extend(product(layer_curr, layer_next))

    if layers[0] & visited_nodes:
        raise ValueError(f"Layer {i} contains nodes in previous layers.")

    return nx.DiGraph(relations)
