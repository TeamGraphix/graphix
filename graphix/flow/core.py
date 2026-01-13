"""Class for flow objects and XZ-corrections."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import networkx as nx

# `override` introduced in Python 3.12, `assert_never` introduced in Python 3.11
from typing_extensions import assert_never, override

# `override` introduced in Python 3.12, `assert_never` introduced in Python 3.11
import graphix.pattern
from graphix.command import E, M, N, X, Z
from graphix.flow._find_gpflow import (
    CorrectionMatrix,
    _M_co,
    _PM_co,
    compute_partial_order_layers,
)
from graphix.flow.exceptions import (
    FlowError,
    FlowGenericError,
    FlowGenericErrorReason,
    FlowPropositionError,
    FlowPropositionErrorReason,
    FlowPropositionOrderError,
    FlowPropositionOrderErrorReason,
    PartialOrderError,
    PartialOrderErrorReason,
    PartialOrderLayerError,
    PartialOrderLayerErrorReason,
    XZCorrectionsGenericError,
    XZCorrectionsGenericErrorReason,
    XZCorrectionsOrderError,
    XZCorrectionsOrderErrorReason,
)
from graphix.fundamentals import Axis, Plane
from graphix.pretty_print import OutputFormat, flow_to_str, xzcorr_to_str

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet
    from typing import Self

    from graphix.measurements import Measurement
    from graphix.opengraph import OpenGraph
    from graphix.parameter import ExpressionOrSupportsFloat, Parameter
    from graphix.pattern import Pattern

TotalOrder = Sequence[int]

_T_PauliFlowMeasurement = TypeVar("_T_PauliFlowMeasurement", bound="PauliFlow[Measurement]")


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

        Raises
        ------
        XZCorrectionsError
            If the input dictionaries are not well formed. In well-formed correction dictionaries:
                - Keys are a subset of the measured nodes.
                - Values correspond to nodes of the open graph.
                - Corrections do not form closed loops.

        Notes
        -----
        This method computes the partial order induced by the XZ-corrections.
        """
        x_corrections = x_corrections or {}
        z_corrections = z_corrections or {}

        nodes_set = set(og.graph.nodes)
        outputs_set = frozenset(og.output_nodes)
        non_outputs_set = set(og.measurements)

        if not non_outputs_set.issuperset(x_corrections.keys() | z_corrections.keys()):
            raise XZCorrectionsGenericError(XZCorrectionsGenericErrorReason.IncorrectKeys)

        dag = _corrections_to_dag(x_corrections, z_corrections)
        partial_order_layers = _dag_to_partial_order_layers(dag)

        if partial_order_layers is None:
            raise XZCorrectionsGenericError(XZCorrectionsGenericErrorReason.ClosedLoop)

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
            raise XZCorrectionsGenericError(XZCorrectionsGenericErrorReason.IncorrectValues)

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

        Raises
        ------
        XZCorrectionsError
            If the input total measurement order is not compatible with the partial order induced by the XZ-corrections.

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
            raise XZCorrectionsGenericError(XZCorrectionsGenericErrorReason.IncompatibleOrder)

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

    def check_well_formed(self) -> None:
        r"""Verify if the XZ-corrections are well formed.

        Raises
        ------
        XZCorrectionsError
            if the XZ-corrections are not well formed.

        Notes
        -----
        A correct `XZCorrections` instance verifies the following properties:
            - Keys of the correction dictionaries are measured nodes, i.e., a subset of :math:`O^c`.
            - Corrections respect the partial order.
            - The first layer of the partial order contains all the output nodes if there are any.
            - The partial order contains all the nodes (without duplicates) and it does not have empty layers.

        This method assumes that the open graph is well formed.
        """
        if len(self.partial_order_layers) == 0:
            if not (self.og.graph or self.x_corrections or self.z_corrections):
                return
            raise PartialOrderError(PartialOrderErrorReason.Empty)

        o_set = set(self.og.output_nodes)
        oc_set = set(self.og.measurements)

        if not oc_set.issuperset(self.x_corrections.keys() | self.z_corrections.keys()):
            raise XZCorrectionsGenericError(XZCorrectionsGenericErrorReason.IncorrectKeys)

        first_layer = self.partial_order_layers[0]

        # Unlike for flows, XZCorrections can be well defined on open graphs without outputs
        if o_set:
            if first_layer != o_set:
                raise PartialOrderLayerError(PartialOrderLayerErrorReason.FirstLayer, layer_index=0, layer=first_layer)
            shift = 1
        else:
            shift = 0

        measured_layers = reversed(self.partial_order_layers[shift:])
        layer_idx = (
            len(self.partial_order_layers) - 1
        )  # To keep track of the layer index when iterating `self.partial_order_layers` in reverse order.
        past_and_present_nodes: set[int] = set()

        for layer in measured_layers:
            if not oc_set.issuperset(layer) or not layer or layer & past_and_present_nodes:
                raise PartialOrderLayerError(PartialOrderLayerErrorReason.NthLayer, layer_index=layer_idx, layer=layer)

            past_and_present_nodes.update(layer)

            for node in layer:
                for corrections, reason in zip(
                    [self.x_corrections, self.z_corrections], XZCorrectionsOrderErrorReason, strict=True
                ):
                    correction_set = corrections.get(node, set())
                    if correction_set & past_and_present_nodes:
                        raise XZCorrectionsOrderError(
                            reason,
                            node=node,
                            correction_set=correction_set,
                            past_and_present_nodes=past_and_present_nodes,
                        )

            layer_idx -= 1

        if {*o_set, *past_and_present_nodes} != set(self.og.graph.nodes):
            raise PartialOrderError(PartialOrderErrorReason.IncorrectNodes)

    def __str__(self) -> str:
        """Return a human-readable string representing the XZCorrections' mappings and partial order layers."""
        return self.to_ascii()

    def to_ascii(self, multiline: bool = False) -> str:
        """Return an ASCII string representing the XZCorrections' mappings and partial order layers.

        Parameters
        ----------
        multiline : bool (optional)
            Flag to format the output. If ``True`` a new line is printed after each correction set. Defaults to ``False``.

        Returns
        -------
        str
        """
        return xzcorr_to_str(self, output=OutputFormat.ASCII, multiline=multiline)

    def to_latex(self, multiline: bool = False) -> str:
        """Return a string containing the LaTeX representation of the XZCorrections' mappings and partial order layers.

        See notes in :meth:`to_ascii` for additional information.
        """
        return xzcorr_to_str(self, output=OutputFormat.LaTeX, multiline=multiline)

    def to_unicode(self, multiline: bool = False) -> str:
        """Return a Unicode string representing the XZCorrections' mappings and partial order layers.

        See notes in :meth:`to_ascii` for additional information.
        """
        return xzcorr_to_str(self, output=OutputFormat.Unicode, multiline=multiline)

    def subs(
        self: XZCorrections[Measurement], variable: Parameter, substitute: ExpressionOrSupportsFloat
    ) -> XZCorrections[Measurement]:
        """Substitute a parameter with a value or expression in all measurement angles of the open graph.

        Parameters
        ----------
        variable : Parameter
            The symbolic expression to be replaced within the measurement angles.
        substitute : ExpressionOrSupportsFloat
            The value or symbolic expression to substitute in place of `variable`.

        Returns
        -------
        XZCorrections[Measurement]
            A new instance of the XZCorrections object with the updated measurement parameters.

        Notes
        -----
        See notes and examples in :func:`OpenGraph.subs`.
        """
        new_og = self.og.subs(variable, substitute)
        return dataclasses.replace(self, og=new_og)

    def xreplace(
        self: XZCorrections[Measurement], assignment: Mapping[Parameter, ExpressionOrSupportsFloat]
    ) -> XZCorrections[Measurement]:
        """Perform parallel substitution of multiple parameters in measurement angles of the open graph.

        Parameters
        ----------
        assignment : Mapping[Parameter, ExpressionOrSupportsFloat]
            A dictionary-like mapping where keys are the `Parameter` objects to be replaced and values are the new expressions or numerical values.

        Returns
        -------
        XZCorrections[Measurement]
            A new instance of the XZCorrections object with the updated measurement angles.

        Notes
        -----
        See notes and examples in :func:`OpenGraph.xreplace`.
        """
        new_og = self.og.xreplace(assignment)
        return dataclasses.replace(self, og=new_og)


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

    _CF_PREFIX: str = "p"  # Correction function prefix for printing

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
                    x_corrections[measured_node] = frozenset(x_corrected_nodes)
                if z_corrected_nodes := self.og.odd_neighbors(correcting_set) & future:
                    z_corrections[measured_node] = frozenset(z_corrected_nodes)

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

    def __str__(self) -> str:
        """Return a human-readable string representing the flow's correction function and partial order layers."""
        return self.to_ascii()

    def to_ascii(self, multiline: bool = False) -> str:
        """Return an ASCII string representing the flow's correction function and partial order layers.

        Parameters
        ----------
        multiline : bool (optional)
            Flag to format the output. If ``True`` a new line is printed after each correction set. Defaults to ``False``.

        Returns
        -------
        str
        """
        return flow_to_str(self, output=OutputFormat.ASCII, multiline=multiline)

    def to_latex(self, multiline: bool = False) -> str:
        """Return a string containing the LaTeX representation of the flow's correction function and partial order layers.

        See notes in :meth:`to_ascii` for additional information.
        """
        return flow_to_str(self, output=OutputFormat.LaTeX, multiline=multiline)

    def to_unicode(self, multiline: bool = False) -> str:
        """Return a Unicode string representing the flow's correction function and partial order layers.

        See notes in :meth:`to_ascii` for additional information.
        """
        return flow_to_str(self, output=OutputFormat.Unicode, multiline=multiline)

    def subs(  # noqa: PYI019 Annotating with `Self` is not possible since `self` must be of parametric type `Measurement`.
        self: _T_PauliFlowMeasurement, variable: Parameter, substitute: ExpressionOrSupportsFloat
    ) -> _T_PauliFlowMeasurement:
        """Substitute a parameter with a value or expression in all measurement angles of the open graph.

        Parameters
        ----------
        variable : Parameter
            The symbolic expression to be replaced within the measurement angles.
        substitute : ExpressionOrSupportsFloat
            The value or symbolic expression to substitute in place of `variable`.

        Returns
        -------
        _T_PauliFlowMeasurement
            A new instance of the flow object with the updated measurement parameters.

        Notes
        -----
        See notes and examples in :func:`OpenGraph.subs`.
        """
        new_og = self.og.subs(variable, substitute)
        return dataclasses.replace(self, og=new_og)

    def xreplace(  # noqa: PYI019
        self: _T_PauliFlowMeasurement, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]
    ) -> _T_PauliFlowMeasurement:
        """Perform parallel substitution of multiple parameters in measurement angles of the open graph.

        Parameters
        ----------
        assignment : Mapping[Parameter, ExpressionOrSupportsFloat]
            A dictionary-like mapping where keys are the `Parameter` objects to be replaced and values are the new expressions or numerical values.

        Returns
        -------
        _T_PauliFlowMeasurement
            A new instance of the flow object with the updated measurement angles.

        Notes
        -----
        See notes and examples in :func:`OpenGraph.xreplace`.
        """
        new_og = self.og.xreplace(assignment)
        return dataclasses.replace(self, og=new_og)


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

    _CF_PREFIX: str = "g"  # Correction function prefix for printing

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
                x_corrections[measured_node] = frozenset(x_corrected_nodes)
            if z_corrected_nodes := self.og.odd_neighbors(correcting_set) - {measured_node}:
                z_corrections[measured_node] = frozenset(z_corrected_nodes)

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

    _CF_PREFIX: str = "c"  # Correction function prefix for printing

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
                x_corrections[measured_node] = frozenset(x_corrected_nodes)
            if z_corrected_nodes := self.og.neighbors(correcting_set) - {measured_node}:
                z_corrections[measured_node] = frozenset(z_corrected_nodes)

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
                    raise FlowGenericError(FlowGenericErrorReason.XYPlane)

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
        raise FlowGenericError(FlowGenericErrorReason.IncorrectCorrectionFunctionDomain)

    if not _check_correction_function_image(flow.og, flow.correction_function):
        raise FlowGenericError(FlowGenericErrorReason.IncorrectCorrectionFunctionImage)

    if len(flow.partial_order_layers) == 0:
        raise PartialOrderError(PartialOrderErrorReason.Empty)

    first_layer = flow.partial_order_layers[0]
    o_set = set(flow.og.output_nodes)
    if first_layer != o_set or not first_layer:
        raise PartialOrderLayerError(PartialOrderLayerErrorReason.FirstLayer, layer_index=0, layer=first_layer)
