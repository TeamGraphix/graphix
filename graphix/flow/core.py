"""Class for flow objects and XZ-corrections."""

from __future__ import annotations

import enum
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Generic, TypeVar

import networkx as nx

# `override` introduced in Python 3.12, `assert_never` introduced in Python 3.11
from typing_extensions import assert_never, override

import graphix.pattern
from graphix.command import E, M, N, X, Z
from graphix.flow._find_gpflow import (
    CorrectionMatrix,
    _M_co,
    _PM_co,
    compute_partial_order_layers,
)
from graphix.fundamentals import Axis, Plane

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

        # If the open graph has outputs, the first element in the output of `_dag_to_partial_order_layers(dag)` may or may not contain all or some output nodes.
        if outputs_set:
            if measured_layer_0 := partial_order_layers[0] - outputs_set:
                # `partial_order_layers[0]` contains (some or all) outputs and measured nodes
                partial_order_layers = [
                    outputs_set,
                    frozenset(measured_layer_0),
                    *partial_order_layers[1:],
                ]
            else:
                # `partial_order_layers[0]` contains only (some or all) outputs
                partial_order_layers = [
                    outputs_set,
                    *partial_order_layers[1:],
                ]

        ordered_nodes = frozenset.union(*partial_order_layers)

        if not ordered_nodes.issubset(nodes_set):
            raise ValueError("Values of input mapping contain labels which are not nodes of the input open graph.")

        # We include all the non-output nodes not involved in the corrections in the last layer (first measured nodes).
        if unordered_nodes := frozenset(nodes_set - ordered_nodes):
            partial_order_layers[-1] |= unordered_nodes

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

    - The flow's correction function defines a partial order (see Def. 2.8 and 2.9, Lemma 2.11 and Theorem 2.12 in Ref. [2]), therefore, only `og` and `correction_function` are necessary to initialize an `PauliFlow` instance (see :func:`PauliFlow.try_from_correction_matrix`). However, flow-finding algorithms generate a partial order in a layer form, which is necessary to extract the flow's XZ-corrections, so it is stored as an attribute.

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
    def try_from_correction_matrix(cls, correction_matrix: CorrectionMatrix[_M_co]) -> Self | None:
        """Initialize a `PauliFlow` object from a matrix encoding a correction function.

        Parameters
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

    def is_well_formed(self) -> bool:
        """Verify if flow is well formed.

        This method is a wrapper over :func:`self.check_well_formed` catching the `FlowError` exceptions.

        Returns
        -------
        ``True`` if ``self`` is a well-formed  flow, ``False`` otherwise.
        """
        try:
            self.check_well_formed()
        except FlowError:
            return False
        return True

    def check_well_formed(self) -> None:
        r"""Verify if the Pauli flow is well formed.

        Raises
        ------
        FlowError
            if the Pauli flow is not well formed.

        Notes
        -----
        General properties of flows:
            - The domain of the correction function is :math:`O^c`, the non-output nodes of the open graph.
            - The image of the correction function is a subset of :math:`I^c`, the non-input nodes of the open graph.
            - The nodes in the partial order are the nodes in the open graph.
            - The first layer of the partial order layers is :math:`O`, the output nodes of the open graph. This is guaranteed because open graphs without outputs do not have flow.

        Specific properties of Pauli flows:
            - If :math:`j \in p(i), i \neq j, \lambda(j) \notin \{X, Y\}`, then :math:`i \prec j` (P1).
            - If :math:`j \in Odd(p(i)), i \neq j, \lambda(j) \notin \{Y, Z\}`, then :math:`i \prec j` (P2).
            - If :math:`neg i \prec j, i \neq j, \lambda(j) = Y`, then either :math:`j \notin p(i)` and :math:`j \in Odd((p(i)))` or :math:`j \in p(i)` and :math:`j \notin Odd((p(i)))` (P3).
            - If :math:`\lambda(i) = XY`, then :math:`i \notin p(i)` and :math:`i \in Odd((p(i)))` (P4).
            - If :math:`\lambda(i) = XZ`, then :math:`i \in p(i)` and :math:`i \in Odd((p(i)))` (P5).
            - If :math:`\lambda(i) = YZ`, then :math:`i \in p(i)` and :math:`i \notin Odd((p(i)))` (P6).
            - If :math:`\lambda(i) = X`, then :math:`i \in Odd((p(i)))` (P7).
            - If :math:`\lambda(i) = Z`, then :math:`i \in p(i)` (P8).
            - If :math:`\lambda(i) = Y`, then either :math:`i \notin p(i)` and :math:`i \in Odd((p(i)))` or :math:`i \in p(i)` and :math:`i \notin Odd((p(i)))` (P9),
        where :math:`i \in O^c`, :math:`c` is the correction function, :math:`prec` denotes the partial order, :math:`\lambda(i)` is the measurement plane or axis of node :math:`i`, and :math:`Odd(s)` is the odd neighbourhood of the set :math:`s` in the open graph.

        See Definition 5 in Ref. [1] or Definition 2.4 in Ref. [2].

        References
        ----------
        [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
        [2] Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        _check_flow_general_properties(self)

        o_set = set(self.og.output_nodes)
        oc_set = set(self.og.measurements)

        past_and_present_nodes: set[int] = set()
        past_and_present_nodes_y_meas: set[int] = set()

        layer_idx = len(self.partial_order_layers) - 1
        for layer in reversed(self.partial_order_layers[1:]):
            if not oc_set.issuperset(layer) or not layer or layer & past_and_present_nodes:
                raise PartialOrderLayerError(PartialOrderLayerErrorReason.NthLayer, layer_index=layer_idx, layer=layer)

            past_and_present_nodes.update(layer)
            past_and_present_nodes_y_meas.update(
                node for node in layer if self.og.measurements[node].to_plane_or_axis() == Axis.Y
            )
            for node in layer:
                correction_set = set(self.correction_function[node])

                meas = self.get_measurement_label(node)

                for i in (correction_set - {node}) & past_and_present_nodes:
                    if self.og.measurements[i].to_plane_or_axis() not in {Axis.X, Axis.Y}:
                        raise FlowPropositionOrderError(
                            FlowPropositionOrderErrorReason.P1,
                            node=node,
                            correction_set=correction_set,
                            past_and_present_nodes=past_and_present_nodes,
                        )

                odd_neighbors = self.og.odd_neighbors(correction_set)

                for i in (odd_neighbors - {node}) & past_and_present_nodes:
                    if self.og.measurements[i].to_plane_or_axis() not in {Axis.Y, Axis.Z}:
                        raise FlowPropositionOrderError(
                            FlowPropositionOrderErrorReason.P2,
                            node=node,
                            correction_set=correction_set,
                            past_and_present_nodes=past_and_present_nodes,
                        )

                closed_odd_neighbors = (odd_neighbors | correction_set) - (odd_neighbors & correction_set)

                if (past_and_present_nodes_y_meas - {node}) & closed_odd_neighbors:
                    raise FlowPropositionOrderError(
                        FlowPropositionOrderErrorReason.P3,
                        node=node,
                        correction_set=correction_set,
                        past_and_present_nodes=past_and_present_nodes_y_meas,
                    )

                if meas == Plane.XY:
                    if not (node not in correction_set and node in odd_neighbors):
                        raise FlowPropositionError(
                            FlowPropositionErrorReason.P4, node=node, correction_set=correction_set
                        )
                elif meas == Plane.XZ:
                    if not (node in correction_set and node in odd_neighbors):
                        raise FlowPropositionError(
                            FlowPropositionErrorReason.P5, node=node, correction_set=correction_set
                        )
                elif meas == Plane.YZ:
                    if not (node in correction_set and node not in odd_neighbors):
                        raise FlowPropositionError(
                            FlowPropositionErrorReason.P6, node=node, correction_set=correction_set
                        )
                elif meas == Axis.X:
                    if node not in odd_neighbors:
                        raise FlowPropositionError(
                            FlowPropositionErrorReason.P7, node=node, correction_set=correction_set
                        )
                elif meas == Axis.Z:
                    if node not in correction_set:
                        raise FlowPropositionError(
                            FlowPropositionErrorReason.P8, node=node, correction_set=correction_set
                        )
                elif meas == Axis.Y:
                    if node not in closed_odd_neighbors:
                        raise FlowPropositionError(
                            FlowPropositionErrorReason.P9, node=node, correction_set=correction_set
                        )
                else:
                    assert_never(meas)

            layer_idx -= 1

        if {*o_set, *past_and_present_nodes} != set(self.og.graph.nodes):
            raise PartialOrderError(PartialOrderErrorReason.IncorrectNodes)

    def get_measurement_label(self, node: int) -> Plane | Axis:
        """Get the measurement label of a given node in the open graph.

        This method interprets measurements with a Pauli angle as `Axis` instances, in consistence with the Pauli flow extraction routine.

        Parameters
        ----------
        node : int

        Returns
        -------
        Plane | Axis
        """
        return self.og.measurements[node].to_plane_or_axis()


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
    @classmethod
    def try_from_correction_matrix(cls, correction_matrix: CorrectionMatrix[_PM_co]) -> Self | None:
        """Initialize a `GFlow` object from a matrix encoding a correction function.

        Parameters
        ----------
        correction_matrix : CorrectionMatrix[_PM_co]
            Algebraic representation of the correction function.

        Returns
        -------
        Self | None
            A gflow if it exists, ``None`` otherwise.

        Notes
        -----
        This method verifies if there exists a partial measurement order on the input open graph compatible with the input correction matrix. See Lemma 3.12, and Theorem 3.1 in Ref. [1]. Failure to find a partial order implies the non-existence of a generalised flow if the correction matrix was calculated by means of Algorithms 2 and 3 in [1].

        References
        ----------
        [1] Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        return super().try_from_correction_matrix(correction_matrix)

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

    def check_well_formed(self) -> None:
        r"""Verify if the generalised flow is well formed.

        Raises
        ------
        FlowError
            if the gflow is not well formed.

        Notes
        -----
        General properties of flows:
            - The domain of the correction function is :math:`O^c`, the non-output nodes of the open graph.
            - The image of the correction function is a subset of :math:`I^c`, the non-input nodes of the open graph.
            - The nodes in the partial order are the nodes in the open graph.
            - The first layer of the partial order layers is :math:`O`, the output nodes of the open graph. This is guaranteed because open graphs without outputs do not have flow.

        Specific properties of gflows:
            - If :math:`j \in g(i), i \neq j`, then :math:`i \prec j` (G1).
            - If :math:`j \in Odd(g(i)), i \neq j`, then :math:`i \prec j` (G2).
            - If :math:`\lambda(i) = XY`, then :math:`i \notin g(i)` and :math:`i \in Odd((g(i)))` (G3).
            - If :math:`\lambda(i) = XZ`, then :math:`i \in g(i)` and :math:`i \in Odd((g(i)))` (G4).
            - If :math:`\lambda(i) = YZ`, then :math:`i \in g(i)` and :math:`i \notin Odd((g(i)))` (G5),
        where :math:`i \in O^c`, :math:`g` is the correction function, :math:`prec` denotes the partial order, :math:`\lambda(i)` is the measurement plane of node :math:`i`, and :math:`Odd(s)` is the odd neighbourhood of the set :math:`s` in the open graph.

        See Definition 2.36 in Ref. [1].

        References
        ----------
        [1] Backens et al., Quantum 5, 421 (2021), doi.org/10.22331/q-2021-03-25-421
        """
        _check_flow_general_properties(self)

        o_set = set(self.og.output_nodes)
        oc_set = set(self.og.measurements)

        layer_idx = len(self.partial_order_layers) - 1
        past_and_present_nodes: set[int] = set()
        for layer in reversed(self.partial_order_layers[1:]):
            if not oc_set.issuperset(layer) or not layer or layer & past_and_present_nodes:
                raise PartialOrderLayerError(PartialOrderLayerErrorReason.NthLayer, layer_index=layer_idx, layer=layer)

            past_and_present_nodes.update(layer)

            for node in layer:
                correction_set = set(self.correction_function[node])

                if (correction_set - {node}) & past_and_present_nodes:
                    raise FlowPropositionOrderError(
                        FlowPropositionOrderErrorReason.G1,
                        node=node,
                        correction_set=correction_set,
                        past_and_present_nodes=past_and_present_nodes,
                    )

                odd_neighbors = self.og.odd_neighbors(correction_set)

                if (odd_neighbors - {node}) & past_and_present_nodes:
                    raise FlowPropositionOrderError(
                        FlowPropositionOrderErrorReason.G2,
                        node=node,
                        correction_set=correction_set,
                        past_and_present_nodes=past_and_present_nodes,
                    )

                plane = self.get_measurement_label(node)

                if plane == Plane.XY:
                    if not (node not in correction_set and node in odd_neighbors):
                        raise FlowPropositionError(
                            FlowPropositionErrorReason.G3, node=node, correction_set=correction_set
                        )
                elif plane == Plane.XZ:
                    if not (node in correction_set and node in odd_neighbors):
                        raise FlowPropositionError(
                            FlowPropositionErrorReason.G4, node=node, correction_set=correction_set
                        )
                elif plane == Plane.YZ:
                    if not (node in correction_set and node not in odd_neighbors):
                        raise FlowPropositionError(
                            FlowPropositionErrorReason.G5, node=node, correction_set=correction_set
                        )
                else:
                    assert_never(plane)

            layer_idx -= 1

        if {*o_set, *past_and_present_nodes} != set(self.og.graph.nodes):
            raise PartialOrderError(PartialOrderErrorReason.IncorrectNodes)

    @override
    def get_measurement_label(self, node: int) -> Plane:
        """Get the measurement label of a given node in the open graph.

        This method interprets measurements with a Pauli angle as `Plane` instances, in consistence with the gflow extraction routine.

        Parameters
        ----------
        node : int

        Returns
        -------
        Plane
        """
        return self.og.measurements[node].to_plane()


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
    def try_from_correction_matrix(cls, correction_matrix: CorrectionMatrix[_PM_co]) -> None:
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

    def check_well_formed(self) -> None:
        r"""Verify if the causal flow is well formed.

        Raises
        ------
        FlowError
            if the causal flow is not well formed.

        Notes
        -----
        General properties of flows:
            - The domain of the correction function is :math:`O^c`, the non-output nodes of the open graph.
            - The image of the correction function is a subset of :math:`I^c`, the non-input nodes of the open graph.
            - The nodes in the partial order are the nodes in the open graph.
            - The first layer of the partial order layers is :math:`O`, the output nodes of the open graph. This is guaranteed because open graphs without outputs do not have flow.

        Specific properties of causal flows:
            - Correction sets have one element only (C0),
            - :math:`i \sim c(i)` (C1),
            - :math:`i \prec c(i)` (C2),
            - :math:`\forall k \in N_G(c(i)) \setminus \{i\}, i \prec k` (C3),
        where :math:`i \in O^c`, :math:`c` is the correction function and :math:`prec` denotes the partial order.

        Causal flows are defined on open graphs with XY measurements only.

        See Definition 2 in Ref. [1].

        References
        ----------
        [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
        """
        _check_flow_general_properties(self)

        o_set = set(self.og.output_nodes)
        oc_set = set(self.og.measurements)

        layer_idx = len(self.partial_order_layers) - 1
        past_and_present_nodes: set[int] = set()
        for layer in reversed(self.partial_order_layers[1:]):
            if not oc_set.issuperset(layer) or not layer or layer & past_and_present_nodes:
                raise PartialOrderLayerError(PartialOrderLayerErrorReason.NthLayer, layer_index=layer_idx, layer=layer)

            past_and_present_nodes.update(layer)

            for node in layer:
                correction_set = set(self.correction_function[node])

                if len(correction_set) != 1:
                    raise FlowPropositionError(FlowPropositionErrorReason.C0, node=node, correction_set=correction_set)

                meas = self.get_measurement_label(node)
                if meas != Plane.XY:
                    raise FlowError("Causal flow is only defined on open graphs with XY measurements.")

                neighbors = self.og.neighbors(correction_set)

                if node not in neighbors:
                    raise FlowPropositionError(FlowPropositionErrorReason.C1, node=node, correction_set=correction_set)

                # If some nodes of the correction set are in the past or in the present of the current node, they cannot be in its future, so the flow is incorrrect.
                if correction_set & past_and_present_nodes:
                    raise FlowPropositionOrderError(
                        FlowPropositionOrderErrorReason.C2,
                        node=node,
                        correction_set=correction_set,
                        past_and_present_nodes=past_and_present_nodes,
                    )

                if (neighbors - {node}) & past_and_present_nodes:
                    raise FlowPropositionOrderError(
                        FlowPropositionOrderErrorReason.C3,
                        node=node,
                        correction_set=correction_set,
                        past_and_present_nodes=past_and_present_nodes,
                    )

            layer_idx -= 1

        if {*o_set, *past_and_present_nodes} != set(self.og.graph.nodes):
            raise PartialOrderError(PartialOrderErrorReason.IncorrectNodes)


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


def _check_correction_function_domain(
    og: OpenGraph[_M_co], correction_function: Mapping[int, AbstractSet[int]]
) -> bool:
    """Verify that the domain of the correction function is the set of non-output nodes of the open graph."""
    oc_set = og.graph.nodes - set(og.output_nodes)
    return correction_function.keys() == oc_set


def _check_correction_function_image(og: OpenGraph[_M_co], correction_function: Mapping[int, AbstractSet[int]]) -> bool:
    """Verify that the image of the correction function is a subset of non-input nodes of the open graph."""
    ic_set = og.graph.nodes - set(og.input_nodes)
    image = set().union(*correction_function.values())
    return image.issubset(ic_set)


def _check_flow_general_properties(flow: PauliFlow[_M_co]) -> None:
    """Verify the general properties of a flow.

    Parameters
    ----------
    flow : PauliFlow[_M_co]

    Raises
    ------
    FlowError
        If the causal flow is not well formed.

    Notes
    -----
    General properties of flows:
        - The domain of the correction function is :math:`O^c`, the non-output nodes of the open graph.
        - The image of the correction function is a subset of :math:`I^c`, the non-input nodes of the open graph.
        - The nodes in the partial order are the nodes in the open graph.
        - The first layer of the partial order layers is :math:`O`, the output nodes of the open graph. This is guaranteed because open graphs without outputs do not have flow.
    """
    if not _check_correction_function_domain(flow.og, flow.correction_function):
        raise CorrectionFunctionError(CorrectionFunctionErrorReason.IncorrectDomain)

    if not _check_correction_function_image(flow.og, flow.correction_function):
        raise CorrectionFunctionError(CorrectionFunctionErrorReason.IncorrectImage)

    if len(flow.partial_order_layers) == 0:
        raise PartialOrderError(PartialOrderErrorReason.Empty)

    first_layer = flow.partial_order_layers[0]
    o_set = set(flow.og.output_nodes)
    if first_layer != o_set or not first_layer:
        raise PartialOrderLayerError(PartialOrderLayerErrorReason.FirstLayer, layer_index=0, layer=first_layer)


class FlowErrorReason:
    """Describe the reason of a `FlowError`."""


class CorrectionFunctionErrorReason(FlowErrorReason, Enum):
    """Describe the reason of a `CorrectionFunctionError` exception."""

    IncorrectDomain = enum.auto()
    """The domain of the correction function is not the set of non-output nodes (measured qubits) of the open graph."""

    IncorrectImage = enum.auto()
    """The image of the correction function is not a subset of non-input nodes (prepared qubits) of the open graph."""


class FlowPropositionErrorReason(FlowErrorReason, Enum):
    """Describe the reason of a `FlowPropositionError` exception."""

    C0 = enum.auto()
    """A correction set in a causal flow has more than one element."""

    C1 = enum.auto()
    """Causal flow (C1). A node and its corrector must be neighbors."""

    G3 = enum.auto()
    """Gflow (G3). Nodes measured on plane XY cannot be in their own correcting set and must belong to the odd neighbourhood of their own correcting set."""

    G4 = enum.auto()
    """Gflow (G4). Nodes measured on plane XZ must belong to their own correcting set and its odd neighbourhood."""

    G5 = enum.auto()
    """Gflow (G5). Nodes measured on plane YZ must belong to their own correcting set and cannot be in the odd neighbourhood of their own correcting set."""

    P4 = enum.auto()
    """Pauli flow (P4). Equivalent to (G3) but for Pauli flows."""

    P5 = enum.auto()
    """Pauli flow (P5). Equivalent to (G4) but for Pauli flows."""

    P6 = enum.auto()
    """Pauli flow (P6). Equivalent to (G5) but for Pauli flows."""

    P7 = enum.auto()
    """Pauli flow (P7). Nodes measured along axis X must belong to the odd neighbourhood of their own correcting set."""

    P8 = enum.auto()
    """Pauli flow (P8). Nodes measured along axis Z must belong to their own correcting set."""

    P9 = enum.auto()
    """Pauli flow (P9). Nodes measured along axis Y must belong to the closed odd neighbourhood of their own correcting set."""


class FlowPropositionOrderErrorReason(FlowErrorReason, Enum):
    """Describe the reason of a `FlowPropositionOrderError` exception."""

    C2 = enum.auto()
    """Causal flow (C2). Nodes must be in the past of their correction set."""

    C3 = enum.auto()
    """Causal flow (C3). Neighbors of the correcting nodes (except the corrected node) must be in the future of the corrected node."""

    G1 = enum.auto()
    """Gflow (G1). Equivalent to (C2) but for gflows."""

    G2 = enum.auto()
    """Gflow (G2). The odd neighbourhood (except the corrected node) of the correcting nodes must be in the future of the corrected node."""

    P1 = enum.auto()
    """Pauli flow (P1). Nodes must be in the past of their correcting nodes that are not measured along the X or the Y axes."""

    P2 = enum.auto()
    """Pauli flow (P2). The odd neighbourhood (except the corrected node and nodes measured along axes Y or Z) of the correcting nodes must be in the future of the corrected node."""

    P3 = enum.auto()
    """Pauli flow (P3). Nodes that are measured along axis Y and that are not in the future of the corrected node (except the corrected node itself) cannot be in the closed odd neighbourhood of the correcting set."""


# NOTE: In the near future, this class may inherit from `XZCorrectionsErrorReason` too.
class PartialOrderErrorReason(FlowErrorReason, Enum):
    """Describe the reason of a `PartialOrderError` exception."""

    Empty = enum.auto()
    """The partial order is empty."""

    IncorrectNodes = enum.auto()
    """The partial order does not contain all the nodes of the open graph or contains nodes that are not in the open graph."""


class PartialOrderLayerErrorReason(FlowErrorReason, Enum):
    """Describe the reason of a `PartialOrderLayerError` exception."""

    FirstLayer = enum.auto()
    """The first layer of the partial order is not the set of output nodes (non-measured qubits) of the open graph or is empty."""  # A well-defined flow cannot exist on an open graph without outputs.

    NthLayer = enum.auto()
    """Nodes in the partial order beyond the first layer are not non-output nodes (measured qubits) of the open graph, layer is empty or contains duplicates."""


# We bind `_Reason` to `str` to allow passing generic strings to a `FlowError` exception.
_Reason = TypeVar("_Reason", bound=FlowErrorReason | str)


@dataclass
class FlowError(Exception, Generic[_Reason]):
    """Exception subclass to handle flow errors."""

    reason: _Reason


@dataclass
class CorrectionFunctionError(FlowError[CorrectionFunctionErrorReason]):
    """Exception subclass to handle general flow errors in the correction function."""

    def __str__(self) -> str:
        """Explain the error."""
        if self.reason == CorrectionFunctionErrorReason.IncorrectDomain:
            return "The domain of the correction function must be the set of non-output nodes (measured qubits) of the open graph."

        if self.reason == CorrectionFunctionErrorReason.IncorrectImage:
            return "The image of the correction function must be a subset of non-input nodes (prepared qubits) of the open graph."

        assert_never(self.reason)


@dataclass
class FlowPropositionError(FlowError[FlowPropositionErrorReason]):
    """Exception subclass to handle violations of the flow-definition propositions which concern the correction function only (C0, C1, G1, G3, G4, G5, P4, P5, P6, P7, P8, P9)."""

    node: int
    correction_set: AbstractSet[int]

    def __str__(self) -> str:
        """Explain the error."""
        error_help = f"Error found at c({self.node}) = {self.correction_set}."

        if self.reason == FlowPropositionErrorReason.C0:
            return f"Correction set c({self.node}) = {self.correction_set} has more than one element."

        if self.reason == FlowPropositionErrorReason.C1:
            return f"{self.reason.name}: a node and its corrector must be neighbors. {error_help}"

        if self.reason == FlowPropositionErrorReason.G3 or self.reason == FlowPropositionErrorReason.P4:  # noqa: PLR1714
            return f"{self.reason.name}: nodes measured on plane XY cannot be in their own correcting set and must belong to the odd neighbourhood of their own correcting set.\n{error_help}"

        if self.reason == FlowPropositionErrorReason.G4 or self.reason == FlowPropositionErrorReason.P5:  # noqa: PLR1714
            return f"{self.reason.name}: nodes measured on plane XZ must belong to their own correcting set and its odd neighbourhood.\n{error_help}"

        if self.reason == FlowPropositionErrorReason.G5 or self.reason == FlowPropositionErrorReason.P6:  # noqa: PLR1714
            return f"{self.reason.name}: nodes measured on plane YZ must belong to their own correcting set and cannot be in the odd neighbourhood of their own correcting set.\n{error_help}"

        if self.reason == FlowPropositionErrorReason.P7:
            return f"{self.reason.name}: nodes measured along axis X must belong to the odd neighbourhood of their own correcting set.\n{error_help}"

        if self.reason == FlowPropositionErrorReason.P8:
            return f"{self.reason.name}: nodes measured along axis Z must belong to their own correcting set.\n{error_help}"

        if self.reason == FlowPropositionErrorReason.P9:
            return f"{self.reason.name}: nodes measured along axis Y must belong to the closed odd neighbourhood of their own correcting set.\n{error_help}"

        assert_never(self.reason)


@dataclass
class FlowPropositionOrderError(FlowError[FlowPropositionOrderErrorReason]):
    """Exception subclass to handle violations of the flow-definition propositions which concern the correction function and the partial order (C2, C3, G1, G2, P1, P2, P3)."""

    node: int
    correction_set: AbstractSet[int]
    past_and_present_nodes: AbstractSet[int]

    def __str__(self) -> str:
        """Explain the error."""
        error_help = f"Error found at c({self.node}) = {self.correction_set}. Partial order: {self.past_and_present_nodes} ≼ {self.node}."

        if self.reason == FlowPropositionOrderErrorReason.C2 or self.reason == FlowPropositionOrderErrorReason.G1:  # noqa: PLR1714
            return f"{self.reason.name}: nodes must be in the past of their correction set.\n{error_help}"

        if self.reason == FlowPropositionOrderErrorReason.C3:
            return f"{self.reason.name}: neighbors of the correcting nodes (except the corrected node) must be in the future of the corrected node.\n{error_help}"

        if self.reason == FlowPropositionOrderErrorReason.G2:
            return f"{self.reason.name}: the odd neighbourhood (except the corrected node) of the correcting nodes must be in the future of the corrected node.\n{error_help}"

        if self.reason == FlowPropositionOrderErrorReason.P1:
            return f"{self.reason.name}: nodes must be in the past of their correcting nodes unless these are measured along the X or the Y axes.\n{error_help}"

        if self.reason == FlowPropositionOrderErrorReason.P2:
            return f"{self.reason.name}: the odd neighbourhood (except the corrected node and nodes measured along axes Y or Z) of the correcting nodes must be in the future of the corrected node.\n{error_help}"

        if self.reason == FlowPropositionOrderErrorReason.P3:
            return f"{self.reason.name}: nodes that are measured along axis Y and that are not in the future of the corrected node (except the corrected node itself) cannot be in the closed odd neighbourhood of the correcting set.\nError found at c({self.node}) = {self.correction_set}. Partial order for Y-measured nodes: {self.past_and_present_nodes} ≼ {self.node}."

        assert_never(self.reason)


@dataclass
class PartialOrderError(FlowError[PartialOrderErrorReason]):
    """Exception subclass to handle general flow errors in the partial order."""

    def __str__(self) -> str:
        """Explain the error."""
        if self.reason == PartialOrderErrorReason.Empty:
            return "The partial order cannot be empty."

        if self.reason == PartialOrderErrorReason.IncorrectNodes:
            return "The partial order does not contain all the nodes of the open graph or contains nodes that are not in the open graph."
        assert_never(self.reason)


@dataclass
class PartialOrderLayerError(FlowError[PartialOrderLayerErrorReason]):
    """Exception subclass to handle flow errors concerning a specific layer of the partial order."""

    layer_index: int
    layer: AbstractSet[int]

    def __str__(self) -> str:
        """Explain the error."""
        if self.reason == PartialOrderLayerErrorReason.FirstLayer:
            return f"The first layer of the partial order must contain all the output nodes of the open graph and cannot be empty. First layer: {self.layer}"

        if self.reason == PartialOrderLayerErrorReason.NthLayer:
            return f"Partial order layer {self.layer_index} = {self.layer} contains non-measured nodes of the open graph, is empty or contains nodes in previous layers."
        assert_never(self.reason)
