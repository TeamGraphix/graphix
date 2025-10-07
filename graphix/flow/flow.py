"""Module for flow classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Generic

import networkx as nx
import numpy as np
import numpy.typing as npt

from graphix.opengraph_ import OpenGraph, _MeasurementLabel_T

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


@dataclass(frozen=True)
class PauliFlow(Generic[_MeasurementLabel_T]):
    og: OpenGraph[_MeasurementLabel_T]
    correction_function: Mapping[int, set[int]]
    partial_order: PartialOrder

    # we might want to pass some order ?
    def compute_corrections(self) -> Corrections[_MeasurementLabel_T]:
        corrections = Corrections(self.og)

        # TODO: implement Browne et al. Theorems. (Need to have defiend Partial Order)
        # I think we only need to override in CausalFlow.

        # for layer in self.partial_order.layers: #TODO Define partial order iter

        return corrections


@dataclass(frozen=True)
class GFlow(PauliFlow[_MeasurementLabel_T]):
    pass


@dataclass(frozen=True)
class CausalFlow(GFlow[_MeasurementLabel_T]):
    pass


@dataclass(frozen=True)
class PartialOrder:
    dag: nx.DiGraph[int]
    layers: dict[int, set[int]] = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.layers = {
                layer: set(generation)
                for layer, generation in enumerate(reversed(nx.topological_generations(self.dag)))
            }
        except nx.NetworkXUnfeasible:
            raise ValueError("Partial order contains loops.")

    @staticmethod
    def from_adj_matrix(adj_mat: npt.NDArray[np.uint8], nodelist: Collection[int] | None = None) -> PartialOrder:
        return PartialOrder(nx.from_numpy_array(adj_mat, create_using=nx.DiGraph, nodelist=nodelist))

    @staticmethod
    def from_relations(relations: Collection[tuple[int, int]]) -> PartialOrder:
        return PartialOrder(nx.DiGraph(relations))

    @staticmethod
    def from_layers(layers: Mapping[int, set[int]]) -> PartialOrder:
        pass

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


@dataclass
class Corrections(Generic[_MeasurementLabel_T]):
    og: OpenGraph[_MeasurementLabel_T]
    _x_corrections: dict[int, set[int]] = field(default_factory=dict)  # {node: domain}
    _z_corrections: dict[int, set[int]] = field(default_factory=dict)  # {node: domain}

    @property
    def x_corrections(self) -> dict[int, set[int]]:
        return self._x_corrections

    @property
    def z_corrections(self) -> dict[int, set[int]]:
        return self._z_corrections

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

    # TODO: There's a bit a of duplicity between X and Z, can we do better?
