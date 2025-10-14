"""Module for flow classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise, product
from typing import TYPE_CHECKING, Generic

import networkx as nx
import numpy as np
import numpy.typing as npt

from graphix.opengraph_ import OpenGraph, _MeasurementLabel_T

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from collections.abc import Set as AbstractSet


@dataclass
class Corrections_(Generic[_MeasurementLabel_T]):
    og: OpenGraph[_MeasurementLabel_T]
    _x_corrections: dict[int, set[int]] = field(default_factory=dict)  # {node: domain}
    _z_corrections: dict[int, set[int]] = field(default_factory=dict)  # {node: domain}

    @property
    def x_corrections(self) -> dict[int, set[int]]:
        return self._x_corrections

    @property
    def z_corrections(self) -> dict[int, set[int]]:
        return self._z_corrections

    # TODO: This may be a cached_property. In this case we would have to clear the cache after adding an X or a Z correction.
    # TODO: Strictly speaking, this function returns a directed graph (i.e., we don't check that it's acyclical at this level). This is done by the function `is_wellformed`
    @property
    def dag(self) -> nx.DiGraph[int]:

        relations: set[tuple[int, int]] = set()

        for node, domain in self.x_corrections.items():
            relations.update(product([node], domain))

        for node, domain in self.z_corrections.items():
            relations.update(product([node], domain))

        return nx.DiGraph(relations)

    # TODO: There's a bit a of duplicity between X and Z, can we do better?

    def add_x_correction(self, node: int, domain: set[int]) -> None:
        if node not in self.og.graph.nodes:
            raise ValueError(f"Cannot apply X correction. Corrected node {node} does not belong to the open graph.")

        if not domain.issubset(self.og.measurements):
            raise ValueError(f"Cannot apply X correction. Domain nodes {domain} are not measured.")

        if node in self._x_corrections:
            self._x_corrections[node] |= domain
        else:
            self._x_corrections.update({node: domain})

    def add_z_correction(self, node: int, domain: set[int]) -> None:
        if node not in self.og.graph.nodes:
            raise ValueError(f"Cannot apply Z correction. Corrected node {node} does not belong to the open graph.")

        if not domain.issubset(self.og.measurements):
            raise ValueError(f"Cannot apply Z correction. Domain nodes {domain} are not measured.")

        if node in self._z_corrections:
            self._z_corrections[node] |= domain
        else:
            self._z_corrections.update({node: domain})

    def is_wellformed(self) -> bool:
        return nx.is_directed_acyclic_graph(self.dag)


@dataclass(frozen=True)
class Corrections(Generic[_MeasurementLabel_T]):
    og: OpenGraph[_MeasurementLabel_T]
    x_corrections: dict[int, set[int]]   # {node: domain}
    z_corrections: dict[int, set[int]]   # {node: domain}

    def __post_init__(self):
        for corr_type in ['X', 'Z']:
            corrections = self.__getattribute__(f"{corr_type.lower()}_corrections")
            for node, domain in corrections.items():
                if node not in self.og.graph.nodes:
                    raise ValueError(f"Cannot apply {corr_type} correction. Corrected node {node} does not belong to the open graph.")
                if not domain.issubset(self.og.measurements):
                    raise ValueError(f"Cannot apply {corr_type} correction. Domain nodes {domain} are not measured.")
        if nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Corrections are not runnable since the induced directed graph contains cycles.")

    @property
    def dag(self) -> nx.DiGraph[int]:

        relations: set[tuple[int, int]] = set()

        for node, domain in self.x_corrections.items():
            relations.update(product([node], domain))

        for node, domain in self.z_corrections.items():
            relations.update(product([node], domain))

        return nx.DiGraph(relations)


@dataclass(frozen=True)
class PauliFlow(Corrections[_MeasurementLabel_T]):
    og: OpenGraph[_MeasurementLabel_T]
    correction_function: Mapping[int, set[int]]

    # TODO: Not needed atm
    # @classmethod
    # def from_correction_function(cls, og, pf) -> Self:
    #     x_corrections: dict[int, set[int]] = {}
    #     z_corrections: dict[int, set[int]] = {}

    #     return cls(og, x_corrections, z_corrections, pf)

    @classmethod
    def from_c_matrix(cls, aog, c_matrix) -> Self:
        x_corrections: dict[int, set[int]] = {}
        z_corrections: dict[int, set[int]] = {}
        pf: dict[int, set[int]] = {}

        return cls(aog, x_corrections, z_corrections, pf)


@dataclass(frozen=True)
class GFlow(PauliFlow[_MeasurementLabel_T]):
    pass


@dataclass(frozen=True)
class CausalFlow(GFlow[_MeasurementLabel_T]):
    pass


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


def _compute_corrections(og: OpenGraph, corr_func: Mapping[int, set[int]]) -> Corrections:

    for node, corr_set in corr_func.items():
        domain_x = corr_set - {node}
        domain_z = og.odd_neighbors(corr_set)


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
