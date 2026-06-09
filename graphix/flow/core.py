"""Class for flow objects and XZ-corrections."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

import networkx as nx
import numpy as np

# `override` introduced in Python 3.12, `assert_never` introduced in Python 3.11
from typing_extensions import assert_never, override

from graphix._linalg import MatGF2, solve_f2_linear_system
from graphix.circ_ext.extraction import (
    CliffordMap,
    ExtractionResult,
    PauliExponentialDAG,
    extraction_ps_from_corrected_node,
)
from graphix.command import C, E, M, N, X, Z
from graphix.flow._find_gpflow import (
    CorrectionMatrix,
    compute_partial_order_layers,
)
from graphix.flow._partial_order import compute_topological_generations
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
from graphix.fundamentals import AbstractMeasurement, AbstractPlanarMeasurement, Axis, Plane
from graphix.measurements import Measurement
from graphix.pretty_print import OutputFormat, flow_to_str, xzcorr_to_str

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet
    from typing import Self

    # Unpack introduced in Python 3.12
    from typing_extensions import Unpack

    from graphix.circ_ext.extraction import PauliString
    from graphix.measurements import BlochMeasurement
    from graphix.opengraph import OpenGraph
    from graphix.parameter import ExpressionOrSupportsFloat, Parameter
    from graphix.pattern import Pattern
    from graphix.visualization import DrawKwargs


TotalOrder = Sequence[int]

_T_PauliFlowMeasurement = TypeVar("_T_PauliFlowMeasurement", bound="PauliFlow[Measurement]")
_M = TypeVar("_M", bound=Measurement)
_AM_co = TypeVar("_AM_co", bound=AbstractMeasurement, covariant=True)
_PM_co = TypeVar("_PM_co", bound=AbstractPlanarMeasurement, covariant=True)


@dataclass(frozen=True)
class XZCorrections(Generic[_AM_co]):
    """An unmutable dataclass providing a representation of XZ-corrections.

    Attributes
    ----------
    og : OpenGraph[_AM_co]
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

    og: OpenGraph[_AM_co]
    x_corrections: Mapping[int, AbstractSet[int]]  # {domain: nodes}
    z_corrections: Mapping[int, AbstractSet[int]]  # {domain: nodes}
    partial_order_layers: Sequence[AbstractSet[int]]

    @staticmethod
    def from_measured_nodes_mapping(
        og: OpenGraph[_AM_co],
        x_corrections: Mapping[int, AbstractSet[int]] | None = None,
        z_corrections: Mapping[int, AbstractSet[int]] | None = None,
    ) -> XZCorrections[_AM_co]:
        """Create an `XZCorrections` instance from the XZ-corrections mappings.

        Parameters
        ----------
        og : OpenGraph[_AM_co]
            Open graph with respect to which the corrections are defined.
        x_corrections : Mapping[int, AbstractSet[int]] | None
            Mapping of X-corrections: in each (`key`, `value`) pair, `key` is a measured node, and `value` is the set of nodes on which an X-correction must be applied depending on the measurement result of `key`.
        z_corrections : Mapping[int, AbstractSet[int]] | None
            Mapping of X-corrections: in each (`key`, `value`) pair, `key` is a measured node, and `value` is the set of nodes on which an X-correction must be applied depending on the measurement result of `key`.

        Returns
        -------
        XZCorrections[_AM_co]

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

        non_outputs_set = set(og.measurements)

        if not non_outputs_set.issuperset(x_corrections.keys() | z_corrections.keys()):
            raise XZCorrectionsGenericError(XZCorrectionsGenericErrorReason.IncorrectKeys)

        partial_order_layers = _corrections_to_partial_order_layers(
            og, x_corrections, z_corrections
        )  # Raises an `XZCorrectionsError` if mappings are not well formed.

        return XZCorrections(og, x_corrections, z_corrections, partial_order_layers)

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
        from graphix.pattern import Pattern  # noqa: PLC0415

        if total_measurement_order is None:
            total_measurement_order = self.generate_total_measurement_order()
        elif not self.is_compatible(total_measurement_order):
            raise XZCorrectionsGenericError(XZCorrectionsGenericErrorReason.IncompatibleOrder)

        pattern = Pattern(input_nodes=self.og.input_nodes)
        non_input_nodes = set(self.og.graph.nodes) - set(self.og.input_nodes)

        for i in non_input_nodes:
            pattern.add(N(node=i))
        for e in self.og.graph.edges:
            pattern.add(E(nodes=e))

        for measured_node in total_measurement_order:
            measurement = self.og.measurements[measured_node]
            pattern.add(M(node=measured_node, measurement=measurement))

            for corrected_node in self.z_corrections.get(measured_node, []):
                pattern.add(Z(node=corrected_node, domain={measured_node}))

            for corrected_node in self.x_corrections.get(measured_node, []):
                pattern.add(X(node=corrected_node, domain={measured_node}))

        for output_node, clifford in self.og.output_cliffords.items():
            pattern.add(C(node=output_node, clifford=clifford))

        pattern.reorder_output_nodes(self.og.output_nodes)
        return pattern

    def to_causal_flow(self: XZCorrections[_PM_co]) -> CausalFlow[_PM_co]:
        r"""Extract a causal flow from XZ-corrections.

        This method does not invoke the flow-extraction routine on the underlying open graph.
        Instead, it assigns the ``x_corrections`` mapping to the flow's correction function
        and verifies that it is compatible with the intrinsic partial order of the XZ-corrections.
        If the resulting correction function is incompatible with this partial order,
        or the open graph contains measurements in XZ or YZ planes, a ``FlowError`` is raised.

        Returns
        -------
        CausalFlow[_PM_co]

        Notes
        -----
        See Theorem 1 in Ref. [1].

        References
        ----------
        [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
        """
        cf = CausalFlow(self.og, self.x_corrections, self.partial_order_layers)
        cf.check_well_formed()  # Raises a `FlowError` if the partial order and the correction function are not compatible, if a measured node is corrected by more than one node, or if nodes are not measured on the XY plane.
        return cf

    def to_gflow(self: XZCorrections[_PM_co]) -> GFlow[_PM_co]:
        r"""Extract a gflow from XZ-corrections.

        This method does not invoke the flow-extraction routine on the underlying open graph.
        Instead, it assigns the ``x_corrections`` mapping to the flow's correction function
        and verifies that it is compatible with the intrinsic partial order of the XZ-corrections.
        Nodes measured in planes XZ or YZ are assigned to their correcting set. If the resulting
        correction function is incompatible with this partial order ``FlowError`` is raised.

        Returns
        -------
        GFlow[_PM_co]

        Notes
        -----
        See Theorem 2 in Ref. [1].

        References
        ----------
        [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
        """
        correction_function: dict[int, set[int]] = {}

        for i, meas in self.og.measurements.items():
            corrections = set(self.x_corrections.get(i, set()))

            if meas.to_plane() in {Plane.XZ, Plane.YZ}:
                corrections.add(i)

            correction_function[i] = corrections

        gf = GFlow(self.og, correction_function, self.partial_order_layers)
        gf.check_well_formed()  # Raises a `FlowError` if the partial order and the correction function are not compatible.
        return gf

    def to_pauli_flow(self) -> PauliFlow[_AM_co]:
        r"""Extract a Pauli flow from XZ-corrections.

        Contrary to :meth:`to_causal_flow` and :meth:`to_gflow`, the correction function of a
        Pauli flow cannot be read off the XZ-corrections directly. Corrections applied to
        Pauli-measured qubits in the past or the present of the corrected node ("anachronical"
        corrections) are absorbed by the measurement (:math:`M^X X = M^X`, :math:`M^Y Y = M^Y`
        and :math:`M^Z Z = M^Z`) and therefore leave no trace in the pattern; compare the
        pattern of Theorem 2 (gflow), where the corrections coincide with the correction
        function, with that of Theorem 4 (Pauli flow), where only the future part
        :math:`p(i) \cap \{j : j > i\}` of each correcting set appears (Ref. [1]).

        The reconstruction is performed node by node. For a measured node :math:`i`, the future
        part of the correcting set is fixed by the X-corrections, and the anachronical part is
        recovered by solving the linear system over :math:`\mathbb{F}_2` whose equations are the
        Pauli-flow conditions (P1)-(P9) together with the requirement that the induced
        XZ-corrections match ``self`` (i.e. :math:`Odd(p(i)) \cap \{j : j > i\}` equals the
        Z-corrections of :math:`i`). The resulting flow is then validated with
        :meth:`PauliFlow.check_well_formed`.

        Returns
        -------
        PauliFlow[_AM_co]

        Raises
        ------
        FlowError
            If the XZ-corrections are not induced by any Pauli flow, or if the partial order is
            incompatible with the reconstructed correction function.

        Notes
        -----
        See Definition 5 and Theorem 4 in Ref. [1]. The induced corrections are recovered by
        :meth:`PauliFlow.to_corrections`, which is the left inverse of this method on
        XZ-corrections extracted from a Pauli flow.

        References
        ----------
        [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
        """
        correction_function = _reconstruct_pauli_correction_function(self)
        pf = PauliFlow(self.og, correction_function, self.partial_order_layers)
        # The reconstruction enforces propositions (P1)-(P9) by construction, but `check_well_formed`
        # additionally validates the user-supplied partial order: it rejects, for instance,
        # XZ-corrections without output nodes (which admit no flow) or a partial order that does not
        # cover every measured node. It raises a `FlowError` in those cases.
        pf.check_well_formed()
        return pf

    def to_bloch(self: XZCorrections[Measurement]) -> XZCorrections[BlochMeasurement]:
        """Return the XZ-corrections where all measurements in the open graph are converted to Bloch.

        See :meth:`OpenGraph.to_bloch` for additional information.
        """
        return XZCorrections(self.og.to_bloch(), self.x_corrections, self.z_corrections, self.partial_order_layers)

    def downcast_bloch(self: XZCorrections[Measurement]) -> XZCorrections[BlochMeasurement]:
        """Return the open graph if all measurements are described as Bloch measurements; raise `TypeError` otherwise.

        See :meth:`OpenGraph.downcast_bloch` for additional information.
        """
        return XZCorrections(
            self.og.downcast_bloch(), self.x_corrections, self.z_corrections, self.partial_order_layers
        )

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

    def draw(self, **options: Unpack[DrawKwargs]) -> None:
        """Visualize the opengraph, correction structure and partial order.

        Parameters
        ----------
        options: Unpack[DrawKwargs]
            Options controlling graph visualization. See :class:`VisualizationOptions`.
        """
        from graphix.visualization import GraphVisualizer  # noqa: PLC0415  Avoid circular imports

        gv = GraphVisualizer.from_xzcorrections(xz_corr=self, **options)
        gv.visualize()

    def subs(self: XZCorrections[_M], variable: Parameter, substitute: ExpressionOrSupportsFloat) -> XZCorrections[_M]:
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
        self: XZCorrections[_M], assignment: Mapping[Parameter, ExpressionOrSupportsFloat]
    ) -> XZCorrections[_M]:
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
class PauliFlow(Generic[_AM_co]):
    """An unmutable dataclass providing a representation of a Pauli flow.

    Attributes
    ----------
    og : OpenGraph[_AM_co]
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

    og: OpenGraph[_AM_co]
    correction_function: Mapping[int, AbstractSet[int]]
    partial_order_layers: Sequence[AbstractSet[int]]

    _CF_PREFIX: str = "p"  # Correction function prefix for printing

    @classmethod
    def try_from_correction_matrix(cls, correction_matrix: CorrectionMatrix[_AM_co]) -> Self | None:
        """Initialize a `PauliFlow` object from a matrix encoding a correction function.

        Parameters
        ----------
        correction_matrix : CorrectionMatrix[_AM_co]
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

    def to_corrections(self) -> XZCorrections[_AM_co]:
        """Compute the X and Z corrections induced by the Pauli flow encoded in `self`.

        Returns
        -------
        XZCorrections[_AM_co]

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

                meas = self.node_measurement_label(node)

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

                match meas:
                    case Plane.XY:
                        if not (node not in correction_set and node in odd_neighbors):
                            raise FlowPropositionError(
                                FlowPropositionErrorReason.P4, node=node, correction_set=correction_set
                            )
                    case Plane.XZ:
                        if not (node in correction_set and node in odd_neighbors):
                            raise FlowPropositionError(
                                FlowPropositionErrorReason.P5, node=node, correction_set=correction_set
                            )
                    case Plane.YZ:
                        if not (node in correction_set and node not in odd_neighbors):
                            raise FlowPropositionError(
                                FlowPropositionErrorReason.P6, node=node, correction_set=correction_set
                            )
                    case Axis.X:
                        if node not in odd_neighbors:
                            raise FlowPropositionError(
                                FlowPropositionErrorReason.P7, node=node, correction_set=correction_set
                            )
                    case Axis.Z:
                        if node not in correction_set:
                            raise FlowPropositionError(
                                FlowPropositionErrorReason.P8, node=node, correction_set=correction_set
                            )
                    case Axis.Y:
                        if node not in closed_odd_neighbors:
                            raise FlowPropositionError(
                                FlowPropositionErrorReason.P9, node=node, correction_set=correction_set
                            )
                    case _:
                        assert_never(meas)

            layer_idx -= 1

        if {*o_set, *past_and_present_nodes} != set(self.og.graph.nodes):
            raise PartialOrderError(PartialOrderErrorReason.IncorrectNodes)

    def node_measurement_label(self, node: int) -> Plane | Axis:
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

    def draw(self, **options: Unpack[DrawKwargs]) -> None:
        """Visualize the opengraph, correction structure and partial order.

        Parameters
        ----------
        options: Unpack[DrawKwargs]
            Options controlling graph visualization. See :class:`VisualizationOptions`.
        """
        from graphix.visualization import GraphVisualizer  # noqa: PLC0415  Avoid circular imports

        gv = GraphVisualizer.from_flow(flow=self, **options)
        gv.visualize()

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

    def is_focused(self) -> bool:
        """Verify if the input Pauli flow is focused.

        Returns
        -------
        bool
            ``True`` if the input Pauli flow is focused, ``False`` otherwise.

        Notes
        -----
        This function verifies Definition 4.3 in Ref. [1].

        References
        ----------
        [1] Simmons, 2021 (arXiv:2109.05654).
        """
        oc_set = self.og.measurements.keys()

        for corrected_node, correction_set in self.correction_function.items():
            odd_correction_set = self.og.odd_neighbors(correction_set)
            symdiff_set = odd_correction_set.symmetric_difference(correction_set)
            for node in oc_set - {corrected_node}:
                meas_label = self.node_measurement_label(node)
                if node in correction_set and meas_label not in {Plane.XY, Axis.X, Axis.Y}:
                    return False
                if node in odd_correction_set and meas_label not in {Plane.XZ, Plane.YZ, Axis.Y, Axis.Z}:
                    return False
                if meas_label == Axis.Y and node in symdiff_set:
                    return False
        return True

    @cached_property
    def extraction_pauli_strings(self: PauliFlow[Measurement]) -> dict[int, PauliString]:
        """Compute the extraction Pauli strings associated with each node in the correction function.

        This property requires the flow to be focused.

        Returns
        -------
        dict[int, PauliString]
            A dictionary where the keys are node indices (from the correction function) and the values are the computed `PauliString` objects.

        Raises
        ------
        ValueError
            If the flow is not focused (i.e., ``self.is_focused()`` is False).

        Notes
        -----
        This property is cached; the dictionary is computed only once upon the first access and stored for subsequent calls.
        See notes in `PauliString.from_measured_node` for additional information.
        """
        if not self.is_focused():
            raise ValueError("Flow is not focused.")
        return {node: extraction_ps_from_corrected_node(self, node) for node in self.correction_function}

    def extract_circuit(self: PauliFlow[Measurement]) -> ExtractionResult:
        """Extract a circuit from a flow.

        This routine assumes that the flow ``self`` is focused (see Notes).

        Returns
        -------
        ExtractionResult
            Wrapper over a Pauli-exponential DAG and a Clifford map encoding the linear transformation implemented by the input flow.

        Notes
        -----
        - This method implements the algorithm in [1].

        - Flows are guaranteed to be focused if obtained from :func:`OpenGraph.extract_pauli_flow` or :func:`OpenGraph.extract_gflow` (see [2]).

        References
        ----------
        [1] Simmons, 2021 (arXiv:2109.05654).
        [2] Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        if self.og.output_cliffords:
            raise NotImplementedError("Circuit extraction is not supported for open graphs with Clifford decorations.")
        pexp_dag = PauliExponentialDAG.from_focused_flow(self)
        clifford_map = CliffordMap.from_focused_flow(self)

        return ExtractionResult(pexp_dag=pexp_dag, clifford_map=clifford_map)


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

                plane = self.node_measurement_label(node)

                match plane:
                    case Plane.XY:
                        if not (node not in correction_set and node in odd_neighbors):
                            raise FlowPropositionError(
                                FlowPropositionErrorReason.G3, node=node, correction_set=correction_set
                            )
                    case Plane.XZ:
                        if not (node in correction_set and node in odd_neighbors):
                            raise FlowPropositionError(
                                FlowPropositionErrorReason.G4, node=node, correction_set=correction_set
                            )
                    case Plane.YZ:
                        if not (node in correction_set and node not in odd_neighbors):
                            raise FlowPropositionError(
                                FlowPropositionErrorReason.G5, node=node, correction_set=correction_set
                            )
                    case _:
                        assert_never(plane)

            layer_idx -= 1

        if {*o_set, *past_and_present_nodes} != set(self.og.graph.nodes):
            raise PartialOrderError(PartialOrderErrorReason.IncorrectNodes)

    @override
    def node_measurement_label(self, node: int) -> Plane:
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

                meas = self.node_measurement_label(node)
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


def _corrections_to_partial_order_layers(
    og: OpenGraph[_AM_co], x_corrections: Mapping[int, AbstractSet[int]], z_corrections: Mapping[int, AbstractSet[int]]
) -> tuple[frozenset[int], ...]:
    """Return the partial order encoded in the correction mappings in a layer form if it exists.

    Parameters
    ----------
    og : OpenGraph[_AM_co]
        The open graph with respect to which the XZ-corrections are defined.
    x_corrections : Mapping[int, AbstractSet[int]]
        Mapping of X-corrections: in each (`key`, `value`) pair, `key` is a measured node, and `value` is the set of nodes on which an X-correction must be applied depending on the measurement result of `key`.
    z_corrections : Mapping[int, AbstractSet[int]]
        Mapping of Z-corrections: in each (`key`, `value`) pair, `key` is a measured node, and `value` is the set of nodes on which an Z-correction must be applied depending on the measurement result of `key`.

    Returns
    -------
    tuple[frozenset[int], ...]
        Partial order between the open graph's in a layer form.
        The set `layers[i]` comprises the nodes in layer `i`. Nodes in layer `i` are "larger" in the partial order than nodes in layer `i+1`.

    Raises
    ------
    XZCorrectionsError
        If the input dictionaries are not well formed. In well-formed correction dictionaries:
            - Keys are a subset of the measured nodes.
            - Values correspond to nodes of the open graph.
            - Corrections do not form closed loops.
    """
    oset = frozenset(og.output_nodes)  # First layer by convention if not empty
    dag: defaultdict[int, set[int]] = defaultdict(
        set
    )  # `i: {j}` represents `i -> j`, i.e., a correction applied to qubit `j`, conditioned on the measurement outcome of qubit `i`.
    indegree_map: dict[int, int] = {}

    for corrections in [x_corrections, z_corrections]:
        for measured_node, corrected_nodes in corrections.items():
            if measured_node not in oset:
                for corrected_node in corrected_nodes - oset:
                    if corrected_node not in dag[measured_node]:  # Don't include multiple edges in the dag.
                        dag[measured_node].add(corrected_node)
                        indegree_map[corrected_node] = indegree_map.get(corrected_node, 0) + 1

    zero_indegree = og.graph.nodes - oset - indegree_map.keys()
    generations = compute_topological_generations(dag, indegree_map, zero_indegree)
    if generations is None:
        raise XZCorrectionsGenericError(XZCorrectionsGenericErrorReason.ClosedLoop)

    if len(generations) == 0:
        if oset:
            return (oset,)
        return ()

    ordered_nodes = frozenset.union(*generations)

    if not ordered_nodes.issubset(og.graph.nodes):
        raise XZCorrectionsGenericError(XZCorrectionsGenericErrorReason.IncorrectValues)

    # We include all the non-output nodes not involved in the corrections in the last layer (first measured nodes).
    if unordered_nodes := frozenset(og.graph.nodes - ordered_nodes - oset):
        generations = *generations[:-1], frozenset(generations[-1] | unordered_nodes)

    if oset:
        return oset, *generations[::-1]
    return generations[::-1]


def _check_correction_function_domain(
    og: OpenGraph[_AM_co], correction_function: Mapping[int, AbstractSet[int]]
) -> bool:
    """Verify that the domain of the correction function is the set of non-output nodes of the open graph."""
    oc_set = og.graph.nodes - set(og.output_nodes)
    return correction_function.keys() == oc_set


def _check_correction_function_image(
    og: OpenGraph[_AM_co], correction_function: Mapping[int, AbstractSet[int]]
) -> bool:
    """Verify that the image of the correction function is a subset of non-input nodes of the open graph."""
    ic_set = og.graph.nodes - set(og.input_nodes)
    image = set().union(*correction_function.values())
    return image.issubset(ic_set)


def _check_flow_general_properties(flow: PauliFlow[_AM_co]) -> None:
    """Verify the general properties of a flow.

    Parameters
    ----------
    flow : PauliFlow[_AM_co]

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


def _solve_f2(rows: list[tuple[set[int], int]], n_vars: int) -> list[int] | None:
    """Return one solution of a linear system over GF(2), or ``None`` if it is inconsistent.

    The equations are assembled into an augmented matrix ``[A | b]`` which is reduced to row
    echelon form with :meth:`graphix._linalg.MatGF2.gauss_elimination` and solved with
    :func:`graphix._linalg.solve_f2_linear_system`. A row that is zero on the coefficient side but
    non-zero on the constant side witnesses an inconsistent system.

    Parameters
    ----------
    rows : list[tuple[set[int], int]]
        Each equation is a pair ``(coefficients, rhs)`` where ``coefficients`` is the set of
        variable indices whose coefficient is ``1`` and ``rhs`` is the right-hand-side bit.
    n_vars : int
        Number of variables.

    Returns
    -------
    list[int] | None
        A solution vector of length ``n_vars`` (free variables set to ``0``), or ``None`` if the
        system is inconsistent.
    """
    if n_vars == 0:
        # No variables: the system is consistent if and only if every constant vanishes.
        return [] if all(rhs % 2 == 0 for _, rhs in rows) else None
    # With at least one variable there is always at least one equation (each free variable is
    # constrained by a self, (P2) or (P3) proposition), so the augmented matrix is never empty.
    augmented = np.zeros((len(rows), n_vars + 1), dtype=np.uint8)
    for i, (coefficients, rhs) in enumerate(rows):
        for j in coefficients:
            augmented[i, j] = 1
        augmented[i, n_vars] = rhs & 1
    reduced = augmented.view(MatGF2).gauss_elimination(ncols=n_vars, copy=True)
    coefficients_block = np.asarray(reduced[:, :n_vars])
    constants_block = np.asarray(reduced[:, n_vars])
    if (constants_block.astype(bool) & ~coefficients_block.any(axis=1)).any():
        return None
    solution = solve_f2_linear_system(MatGF2(coefficients_block), MatGF2(constants_block))
    return [int(bit) for bit in solution]


def _odd_neighbourhood_equation(
    graph: nx.Graph[int],
    inputs: AbstractSet[int],
    free_index: Mapping[int, int],
    fixed: Mapping[int, int],
    target: int,
) -> tuple[set[int], int]:
    r"""Express :math:`target \in Odd(p)` as a linear form over the free membership variables.

    Parameters
    ----------
    graph : nx.Graph[int]
        Graph of the open graph.
    inputs : AbstractSet[int]
        Input nodes (never belong to a correcting set).
    free_index : Mapping[int, int]
        Map from a node with free membership to its variable index.
    fixed : Mapping[int, int]
        Map from a node with fixed membership to its value (``0`` or ``1``).
    target : int
        Node whose odd-neighbourhood membership is expressed.

    Returns
    -------
    tuple[set[int], int]
        ``(coefficients, constant)`` such that ``[target in Odd(p)]`` equals the parity of the
        selected free variables XORed with ``constant``.
    """
    coefficients: set[int] = set()
    constant = 0
    for neighbor in graph.neighbors(target):
        if neighbor in free_index:
            coefficients ^= {free_index[neighbor]}
        elif neighbor not in inputs:
            constant ^= fixed[neighbor]
    return coefficients, constant


def _membership_coefficients(free_index: Mapping[int, int], target: int) -> set[int]:
    """Return the coefficients of the membership variable ``[target in p]``.

    This helper is only used for nodes measured along axis Y, which are always free variables
    (when non-input) or inputs (which can never be corrected and contribute ``0``). It therefore
    never needs to handle a node whose membership is fixed to ``1``.

    Parameters
    ----------
    free_index : Mapping[int, int]
        Map from a node with free membership to its variable index.
    target : int
        Node whose membership is expressed.

    Returns
    -------
    set[int]
        ``{index}`` if ``target`` is a free variable, the empty set otherwise.
    """
    index = free_index.get(target)
    return set() if index is None else {index}


def _reconstruct_pauli_correction_function(xz: XZCorrections[_AM_co]) -> dict[int, frozenset[int]]:
    r"""Reconstruct the correction function of a Pauli flow inducing the given XZ-corrections.

    See :meth:`XZCorrections.to_pauli_flow` for the rationale. For every measured node, the
    future part of the correcting set is fixed by the X-corrections and the anachronical part is
    recovered by solving over :math:`\mathbb{F}_2` the Pauli-flow conditions (P1)-(P9) together
    with the constraint that the induced Z-corrections match ``xz``.

    Parameters
    ----------
    xz : XZCorrections[_AM_co]

    Returns
    -------
    dict[int, frozenset[int]]
        The reconstructed correction function.

    Raises
    ------
    FlowError
        If no Pauli flow induces the XZ-corrections of ``xz``.

    References
    ----------
    [1] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212).
    """
    og = xz.og
    graph = og.graph
    inputs = set(og.input_nodes)
    measurements = og.measurements
    layers = xz.partial_order_layers

    # future(node): the nodes lying in strictly earlier-measured layers (i.e. smaller index).
    cumulative: list[frozenset[int]] = []
    accumulated: set[int] = set()
    for layer in layers:
        cumulative.append(frozenset(accumulated))
        accumulated |= set(layer)
    layer_of = {node: k for k, layer in enumerate(layers) for node in layer}

    correction_function: dict[int, frozenset[int]] = {}

    for node, measurement in measurements.items():
        if node not in layer_of:
            # The partial order does not cover this measured node; the reconstructed correction
            # function will be incomplete and rejected by `PauliFlow.check_well_formed`.
            continue
        label = measurement.to_plane_or_axis()
        x_future = xz.x_corrections.get(node, frozenset())
        z_future = xz.z_corrections.get(node, frozenset())
        future = cumulative[layer_of[node]]

        # Membership of each non-input node in p(node): either fixed (0/1) or a free variable.
        # (P1) restricts anachronical correctors to nodes measured along the X or Y axes.
        fixed: dict[int, int] = {}
        free_index: dict[int, int] = {}
        for candidate in graph.nodes:
            if candidate in inputs:
                continue
            if candidate in future:
                fixed[candidate] = 1 if candidate in x_future else 0
            elif candidate == node:
                if label == Plane.XY:  # (P4): node ∉ p.
                    fixed[candidate] = 0
                elif label in {Plane.XZ, Plane.YZ, Axis.Z}:  # (P5)/(P6)/(P8): node ∈ p.
                    fixed[candidate] = 1
                else:  # Axis.X or Axis.Y: self-membership is free.
                    free_index[candidate] = len(free_index)
            elif (other := measurements.get(candidate)) is not None and other.to_plane_or_axis() in {Axis.X, Axis.Y}:
                free_index[candidate] = len(free_index)
            else:
                fixed[candidate] = 0

        rows: list[tuple[set[int], int]] = []

        # Induced Z-corrections must match: Odd(p(node)) ∩ future = z_future.
        for future_node in future:
            coefficients, constant = _odd_neighbourhood_equation(graph, inputs, free_index, fixed, future_node)
            rows.append((coefficients, (1 if future_node in z_future else 0) ^ constant))

        # Self conditions on the odd neighbourhood (P4)-(P9).
        if label in {Plane.XY, Plane.XZ, Axis.X}:  # (P4)/(P5)/(P7): node ∈ Odd(p).
            coefficients, constant = _odd_neighbourhood_equation(graph, inputs, free_index, fixed, node)
            rows.append((coefficients, 1 ^ constant))
        elif label == Plane.YZ:  # (P6): node ∉ Odd(p).
            coefficients, constant = _odd_neighbourhood_equation(graph, inputs, free_index, fixed, node)
            rows.append((coefficients, constant))
        elif label == Axis.Y:  # (P9): exactly one of node ∈ p and node ∈ Odd(p).
            coefficients, constant = _odd_neighbourhood_equation(graph, inputs, free_index, fixed, node)
            rows.append((coefficients ^ _membership_coefficients(free_index, node), 1 ^ constant))
        # Axis.Z: (P8) only constrains the fixed membership; no condition on the odd neighbourhood.

        # Conditions on the other non-future nodes (P2) and (P3).
        for other_node in graph.nodes:
            if other_node == node or other_node in future or other_node not in measurements:
                continue
            other_label = measurements[other_node].to_plane_or_axis()
            if other_label in {Plane.XY, Plane.XZ, Plane.YZ, Axis.X}:  # (P2): other_node ∉ Odd(p).
                coefficients, constant = _odd_neighbourhood_equation(graph, inputs, free_index, fixed, other_node)
                rows.append((coefficients, constant))
            elif other_label == Axis.Y:  # (P3): other_node ∈ p ⇔ other_node ∈ Odd(p).
                coefficients, constant = _odd_neighbourhood_equation(graph, inputs, free_index, fixed, other_node)
                rows.append((coefficients ^ _membership_coefficients(free_index, other_node), constant))
            # Axis.Z: no constraint (an anachronical Z-correction is absorbed by the measurement).

        # A measured input node cannot belong to its own correcting set, contradicting (P5)/(P6)/(P8).
        if node in inputs and label in {Plane.XZ, Plane.YZ, Axis.Z}:
            rows.append((set(), 1))

        solution = _solve_f2(rows, len(free_index))
        if solution is None:
            # The local linear system has no solution: no Pauli flow induces these corrections.
            raise FlowGenericError(FlowGenericErrorReason.NoCompatiblePauliFlow)

        correcting_set = {member for member, bit in fixed.items() if bit}
        correcting_set |= {member for member, index in free_index.items() if solution[index]}
        correction_function[node] = frozenset(correcting_set)

    return correction_function
