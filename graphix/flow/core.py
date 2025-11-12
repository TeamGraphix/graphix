"""Class for flow objects and XZ-corrections."""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

import networkx as nx

# override introduced in Python 3.12
from typing_extensions import override

import graphix.pattern
from graphix.command import E, M, N, X, Z
from graphix.flow._find_gpflow import CorrectionMatrix, _M_co, _PM_co, compute_partial_order_layers

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet
    from typing import Self

    from graphix.measurements import Measurement
    from graphix.opengraph import OpenGraph
    from graphix.pattern import Pattern

TotalOrder = Sequence[int]


@dataclass(frozen=True)
class XZCorrections(Generic[_M_co]):
    """An unmutable dataclass providing a representation of XZ-corrections.

    Attributes
    ----------
    og : OpenGraph[_M_co]
        The open graph with respect to which the XZ-corrections are defined.
    x_corrections : Mapping[int, AbstractSet[int]]
        Mapping of X-corrections: in each (`key`, `value`) pair, `key` is a measured node, and `value` is the set of nodes on which an X-correction must be applied depending on the measurement result of `key`.
    z_corrections : Mapping[int, AbstractSet[int]]
        Mapping of Z-corrections: in each (`key`, `value`) pair, `key` is a measured node, and `value` is the set of nodes on which an Z-correction must be applied depending on the measurement result of `key`.
    partial_order_layers : Sequence[AbstractSet[int]]
        Partial order between the open graph's nodes in a layer form determined by the corrections. The set `layers[i]` comprises the nodes in layer `i`. Nodes in layer `i` are "larger" in the partial order than nodes in layer `i+1`. If the open graph has output nodes, they are always in layer 0. Non-corrected, measured nodes are always in the last layer.

    Notes
    -----
    The XZ-corrections mappings define a partial order, therefore, only `og`, `x_corrections` and `z_corrections` are necessary to initialize an `XZCorrections` instance (see :func:`XZCorrections.from_measured_nodes_mapping`). However, XZ-corrections are often extracted from a flow whose partial order is known and can be used to construct a pattern, so it can also be passed as an argument to the `dataclass` constructor. The correctness of the input parameters is not verified automatically.

    """

    og: OpenGraph[_M_co]
    x_corrections: Mapping[int, AbstractSet[int]]  # {domain: nodes}
    z_corrections: Mapping[int, AbstractSet[int]]  # {domain: nodes}
    partial_order_layers: Sequence[AbstractSet[int]]

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
        outputs_set = frozenset(og.output_nodes)
        non_outputs_set = nodes_set - outputs_set

        if not non_outputs_set.issuperset(x_corrections):
            raise ValueError("Keys of input X-corrections contain non-measured nodes.")
        if not set(z_corrections).issubset(non_outputs_set):
            raise ValueError("Keys of input Z-corrections contain non-measured nodes.")

        dag = _corrections_to_dag(x_corrections, z_corrections)
        partial_order_layers = _dag_to_partial_order_layers(dag)

        if partial_order_layers is None:
            raise ValueError(
                "Input XZ-corrections are not runnable since the induced directed graph contains closed loops."
            )

        # If there're no corrections, the partial order has 2 layers only: outputs and measured nodes.
        if len(partial_order_layers) == 0:
            partial_order_layers = [outputs_set] if outputs_set else []
            if non_outputs_set:
                partial_order_layers.append(frozenset(non_outputs_set))
            return XZCorrections(og, x_corrections, z_corrections, tuple(partial_order_layers))

        # If the open graph has outputs, the first element in the output of `_dag_to_partial_order_layers(dag)` may or may not contain output nodes.
        if outputs_set:
            partial_order_layers = [
                outputs_set,
                frozenset(partial_order_layers[0] - outputs_set),
                *partial_order_layers[1:],
            ]

        ordered_nodes = {node for layer in partial_order_layers for node in layer}
        if not ordered_nodes.issubset(nodes_set):
            raise ValueError("Values of input mapping contain labels which are not nodes of the input open graph.")

        # We append to the last layer (first measured nodes) all the non-output nodes not involved in the corrections.
        if unordered_nodes := frozenset(nodes_set - ordered_nodes):
            partial_order_layers.append(unordered_nodes)

        return XZCorrections(og, x_corrections, z_corrections, tuple(partial_order_layers))

    def to_pattern(
        self: XZCorrections[Measurement],
        total_measurement_order: TotalOrder | None = None,
    ) -> Pattern:
        """Generate a unique pattern from an instance of `XZCorrections[Measurement]`.

        Parameters
        ----------
        total_measurement_order : TotalOrder | None
            Ordered sequence of all the non-output nodes in the open graph indicating the measurement order. This parameter must be compatible with the partial order induced by the XZ-corrections.
            Optional, defaults to `None`. If `None` an arbitrary total order compatible with `self.partial_order_layers` is generated.

        Returns
        -------
        Pattern

        Notes
        -----
        - The `XZCorrections` instance must be of parametric type `Measurement` to allow for a pattern extraction, otherwise the underlying open graph does not contain information about the measurement angles.

        - The resulting pattern is guaranteed to be runnable if the `XZCorrections` object is well formed, but does not need to be deterministic. It will be deterministic if the XZ-corrections were inferred from a flow. In this case, this routine follows the recipe in Theorems 1, 2 and 4 in Ref. [1].

        References
        ----------
        [1] Browne et al., NJP 9, 250 (2007).
        """
        if total_measurement_order is None:
            total_measurement_order = self.generate_total_measurement_order()
        elif not self.is_compatible(total_measurement_order):
            raise ValueError(
                "The input total measurement order is not compatible with the partial order induced by the XZ-corrections."
            )

        pattern = graphix.pattern.Pattern(input_nodes=self.og.input_nodes)
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
        shift = 1 if self.og.output_nodes else 0
        total_order = [node for layer in reversed(self.partial_order_layers[shift:]) for node in layer]

        assert set(total_order) == set(self.og.graph.nodes) - set(self.og.output_nodes)
        return total_order

    def extract_dag(self) -> nx.DiGraph[int]:
        """Extract the directed graph induced by the XZ-corrections.

        Returns
        -------
        nx.DiGraph[int]
            Directed graph in which an edge `i -> j` represents a correction applied to qubit `j`, conditioned on the measurement outcome of qubit `i`.

        Notes
        -----
        - Not all nodes of the underlying open graph are nodes of the returned directed graph, but only those involved in a correction, either as corrected qubits or belonging to a correction domain.
        - The output of this method is not guaranteed to be a directed acyclical graph (i.e., a directed graph without any loops). This is only the case if the `XZCorrections` object is well formed, which is verified by the method :func:`XZCorrections.is_wellformed`.
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

        shift = 1 if self.og.output_nodes else 0
        measured_layers = list(reversed(self.partial_order_layers[shift:]))

        i = 0
        n_measured_layers = len(measured_layers)
        layer = measured_layers[0]

        for node in total_measurement_order:
            while node not in layer:
                i += 1
                if i == n_measured_layers:
                    return False
                layer = measured_layers[i]

        return True


@dataclass(frozen=True)
class PauliFlow(Generic[_M_co]):
    """An unmutable dataclass providing a representation of a Pauli flow.

    Attributes
    ----------
    og : OpenGraph[_M_co]
        The open graph with respect to which the Pauli flow is defined.
    correction_function : Mapping[int, AbstractSet[int]
        Pauli flow correction function. `correction_function[i]` is the set of qubits correcting the measurement of qubit `i`.
    partial_order_layers : Sequence[AbstractSet[int]]
        Partial order between the open graph's nodes in a layer form. The set `layers[i]` comprises the nodes in layer `i`. Nodes in layer `i` are "larger" in the partial order than nodes in layer `i+1`. Output nodes are always in layer 0.

    Notes
    -----
    - See Definition 5 in Ref. [1] for a definition of Pauli flow.

    - The flow's correction function defines a partial order (see Def. 2.8 and 2.9, Lemma 2.11 and Theorem 2.12 in Ref. [2]), therefore, only `og` and `correction_function` are necessary to initialize an `PauliFlow` instance (see :func:`PauliFlow.from_correction_matrix`). However, flow-finding algorithms generate a partial order in a layer form, which is necessary to extract the flow's XZ-corrections, so it is stored as an attribute.

    - A correct flow can only exist on an open graph with output nodes, so `layers[0]` always contains a finite set of nodes.

    References
    ----------
    [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
    [2] Mitosek and Backens, 2024 (arXiv:2410.23439).

    """

    og: OpenGraph[_M_co]
    correction_function: Mapping[int, AbstractSet[int]]
    partial_order_layers: Sequence[AbstractSet[int]]

    @classmethod
    def from_correction_matrix(cls, correction_matrix: CorrectionMatrix[_M_co]) -> Self | None:
        """Initialize a Pauli flow object from a matrix encoding a correction function.

        Attributes
        ----------
        correction_matrix : CorrectionMatrix[_M_co]
            Algebraic representation of the correction function.

        Returns
        -------
        Self | None
            A Pauli flow if it exists, `None` otherwise.

        Notes
        -----
        This method verifies if there exists a partial measurement order on the input open graph compatible with the input correction matrix. See Lemma 3.12, and Theorem 3.1 in Ref. [1]. Failure to find a partial order implies the non-existence of a Pauli flow if the correction matrix was calculated by means of Algorithms 2 and 3 in [1].

        References
        ----------
        [1] Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        correction_function = correction_matrix.to_correction_function()
        partial_order_layers = compute_partial_order_layers(correction_matrix)
        if partial_order_layers is None:
            return None

        return cls(correction_matrix.aog.og, correction_function, partial_order_layers)

    def to_corrections(self) -> XZCorrections[_M_co]:
        """Compute the X and Z corrections induced by the Pauli flow encoded in `self`.

        Returns
        -------
        XZCorrections[_M_co]

        Notes
        -----
        This method partially implements Theorem 4 in [1]. The generated X and Z corrections can be used to obtain a robustly deterministic pattern on the underlying open graph.

        References
        ----------
        [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
        """
        future = copy(self.partial_order_layers[0])  # Sets are mutable
        x_corrections: dict[int, AbstractSet[int]] = {}  # {domain: nodes}
        z_corrections: dict[int, AbstractSet[int]] = {}  # {domain: nodes}

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


@dataclass(frozen=True)
class GFlow(PauliFlow[_PM_co], Generic[_PM_co]):
    """An unmutable subclass of `PauliFlow` providing a representation of a generalised flow (gflow).

    This class differs from its parent class in the following:
        - It cannot be constructed from `OpenGraph[Axis]` instances, since the gflow is only defined for planar measurements.
        - The extraction of XZ-corrections from the gflow does not require knowledge on the partial order.
        - The method :func:`GFlow.is_well_formed` verifies the definition of gflow (Definition 2.36 in Ref. [1]).

    References
    ----------
    [1] Backens et al., Quantum 5, 421 (2021), doi.org/10.22331/q-2021-03-25-421

    """

    @override
    def to_corrections(self) -> XZCorrections[_PM_co]:
        r"""Compute the XZ-corrections induced by the generalised flow encoded in `self`.

        Returns
        -------
        XZCorrections[_PM_co]

        Notes
        -----
        - This function partially implements Theorem 2 in Ref. [1]. The generated XZ-corrections can be used to obtain a robustly deterministic pattern on the underlying open graph.

        - Contrary to the overridden method in the parent class, here we do not need any information on the partial order to build the corrections since a valid correction function :math:`g` guarantees that both :math:`g(i)\setminus \{i\}` and :math:`Odd(g(i))` are in the future of :math:`i`.

        References
        ----------
        [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
        """
        x_corrections: dict[int, AbstractSet[int]] = {}  # {domain: nodes}
        z_corrections: dict[int, AbstractSet[int]] = {}  # {domain: nodes}

        for measured_node, correcting_set in self.correction_function.items():
            # Conditionals avoid storing empty correction sets
            if x_corrected_nodes := correcting_set - {measured_node}:
                x_corrections[measured_node] = x_corrected_nodes
            if z_corrected_nodes := self.og.odd_neighbors(correcting_set) - {measured_node}:
                z_corrections[measured_node] = z_corrected_nodes

        return XZCorrections(self.og, x_corrections, z_corrections, self.partial_order_layers)


@dataclass(frozen=True)
class CausalFlow(GFlow[_PM_co], Generic[_PM_co]):
    """An unmutable subclass of `GFlow` providing a representation of a causal flow.

    This class differs from its parent class in the following:
        - The extraction of XZ-corrections from the causal flow does assumes that correction sets have one element only.
        - The method :func:`CausalFlow.is_well_formed` verifies the definition of causal flow (Definition 2 in Ref. [1]).

    References
    ----------
    [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).

    """

    @override
    @classmethod
    def from_correction_matrix(cls, correction_matrix: CorrectionMatrix[_PM_co]) -> None:
        raise NotImplementedError("Initialization of a causal flow from a correction matrix is not supported.")

    @override
    def to_corrections(self) -> XZCorrections[_PM_co]:
        r"""Compute the XZ-corrections induced by the causal flow encoded in `self`.

        Returns
        -------
        XZCorrections[_PM_co]

        Notes
        -----
        This function partially implements Theorem 1 in Ref. [1]. The generated XZ-corrections can be used to obtain a robustly deterministic pattern on the underlying open graph.

        References
        ----------
        [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
        """
        x_corrections: dict[int, AbstractSet[int]] = {}  # {domain: nodes}
        z_corrections: dict[int, AbstractSet[int]] = {}  # {domain: nodes}

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
    """Convert an XZ-corrections mapping into a directed graph.

    Parameters
    ----------
    x_corrections : Mapping[int, AbstractSet[int]]
        Mapping of X-corrections: in each (`key`, `value`) pair, `key` is a measured node, and `value` is the set of nodes on which an X-correction must be applied depending on the measurement result of `key`.
    z_corrections : Mapping[int, AbstractSet[int]]
        Mapping of Z-corrections: in each (`key`, `value`) pair, `key` is a measured node, and `value` is the set of nodes on which an Z-correction must be applied depending on the measurement result of `key`.

    Returns
    -------
    nx.DiGraph[int]
        Directed graph in which an edge `i -> j` represents a correction applied to qubit `j`, conditioned on the measurement outcome of qubit `i`.

    Notes
    -----
    See :func:`XZCorrections.extract_dag`.
    """
    relations = (
        (measured_node, corrected_node)
        for corrections in (x_corrections, z_corrections)
        for measured_node, corrected_nodes in corrections.items()
        for corrected_node in corrected_nodes
    )

    return nx.DiGraph(relations)


def _dag_to_partial_order_layers(dag: nx.DiGraph[int]) -> list[frozenset[int]] | None:
    """Return the partial order encoded in a directed graph in a layer form if it exists.

    Parameters
    ----------
    dag : nx.DiGraph[int]
        A directed graph.

    Returns
    -------
    list[set[int]] | None
        Partial order between corrected qubits in a layer form or `None` if the input directed graph is not acyclical.
        The set `layers[i]` comprises the nodes in layer `i`. Nodes in layer `i` are "larger" in the partial order than nodes in layer `i+1`.
    """
    try:
        topo_gen = reversed(list(nx.topological_generations(dag)))
    except nx.NetworkXUnfeasible:
        return None

    return [frozenset(layer) for layer in topo_gen]
