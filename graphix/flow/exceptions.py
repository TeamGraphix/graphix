"""Module for flows and XZ-corrections exceptions."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

# `override` introduced in Python 3.12, `assert_never` introduced in Python 3.11
from typing_extensions import assert_never

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet


class CorrectionFunctionErrorReason(Enum):
    """Describe the reason of a `CorrectionFunctionError` exception."""

    IncorrectDomain = enum.auto()
    """The domain of the correction function is not the set of non-output nodes (measured qubits) of the open graph."""

    IncorrectImage = enum.auto()
    """The image of the correction function is not a subset of non-input nodes (prepared qubits) of the open graph."""


class FlowPropositionErrorReason(Enum):
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


class FlowPropositionOrderErrorReason(Enum):
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


class FlowGenericErrorReason(Enum):
    """Describe the reason of a `FlowGenericError`."""

    XYPlane = enum.auto()
    "A causal flow is defined on an open graphs with non-XY measurements."


class XZCorrectionsOrderErrorReason(Enum):
    """Describe the reason of an `XZCorrectionsOrderError` exception."""

    X = enum.auto()
    """An X-correction set contains nodes in the present or the past of the corrected node."""

    Z = enum.auto()
    """An X-correction set contains nodes in the present or the past of the corrected node."""


class XZCorrectionsGenericErrorReason(Enum):
    """Describe the reason of an `XZCorrectionsGenericError`."""

    IncorrectKeys = enum.auto()
    """Keys of correction dictionaries are not a subset of the measured nodes."""

    IncorrectValues = enum.auto()
    """Values of correction dictionaries contain labels which are not nodes of the open graph."""

    ClosedLoop = enum.auto()
    """XZ-corrections are not runnable since the induced directed graph contains closed loops."""

    IncompatibleOrder = enum.auto()
    """The input total measurement order is not compatible with the partial order induced by the XZ-corrections."""


class PartialOrderErrorReason(Enum):
    """Describe the reason of a `PartialOrderError` exception."""

    Empty = enum.auto()
    """The partial order is empty."""

    IncorrectNodes = enum.auto()
    """The partial order does not contain all the nodes of the open graph or contains nodes that are not in the open graph."""


class PartialOrderLayerErrorReason(Enum):
    """Describe the reason of a `PartialOrderLayerError` exception."""

    FirstLayer = enum.auto()
    """The first layer of the partial order is not the set of output nodes (non-measured qubits) of the open graph or is empty.

    XZ-corrections can be defined on open graphs without outputs. That is not the case for correct flows.
    """

    NthLayer = enum.auto()
    """Nodes in the partial order beyond the first layer are not non-output nodes (measured qubits) of the open graph, layer is empty or contains duplicates."""


@dataclass
class FlowError(Exception):
    """Exception subclass to handle flow errors."""


@dataclass
class XZCorrectionsError(Exception):
    """Exception subclass to handle XZ-corrections errors."""


@dataclass
class CorrectionFunctionError(FlowError):
    """Exception subclass to handle general flow errors in the correction function."""

    reason: CorrectionFunctionErrorReason

    def __str__(self) -> str:
        """Explain the error."""
        if self.reason == CorrectionFunctionErrorReason.IncorrectDomain:
            return "The domain of the correction function must be the set of non-output nodes (measured qubits) of the open graph."

        if self.reason == CorrectionFunctionErrorReason.IncorrectImage:
            return "The image of the correction function must be a subset of non-input nodes (prepared qubits) of the open graph."

        assert_never(self.reason)


@dataclass
class FlowPropositionError(FlowError):
    """Exception subclass to handle violations of the flow-definition propositions which concern the correction function only (C0, C1, G1, G3, G4, G5, P4, P5, P6, P7, P8, P9)."""

    reason: FlowPropositionErrorReason
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
class FlowPropositionOrderError(FlowError):
    """Exception subclass to handle violations of the flow-definition propositions which concern the correction function and the partial order (C2, C3, G1, G2, P1, P2, P3)."""

    reason: FlowPropositionOrderErrorReason
    node: int
    correction_set: AbstractSet[int]
    past_and_present_nodes: AbstractSet[int]

    def __str__(self) -> str:
        """Explain the error."""
        error_help = f"The flow's partial order implies that {self.past_and_present_nodes - {self.node}} ≼ {self.node}. This is incompatible with the correction set c({self.node}) = {self.correction_set}."

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
            return f"{self.reason.name}: nodes that are measured along axis Y and that are not in the future of the corrected node (except the corrected node itself) cannot be in the closed odd neighbourhood of the correcting set.\n{error_help}"

        assert_never(self.reason)


@dataclass
class FlowGenericError(FlowError):
    """Exception subclass to handle generic flow errors."""

    reason: FlowGenericErrorReason

    def __str__(self) -> str:
        """Explain the error."""
        if self.reason == FlowGenericErrorReason.XYPlane:
            return "Causal flow is only defined on open graphs with XY measurements."

        assert_never(self.reason)


@dataclass
class PartialOrderError(FlowError, XZCorrectionsError):
    """Exception subclass to handle general flow and XZ-corrections errors in the partial order."""

    reason: PartialOrderErrorReason

    def __str__(self) -> str:
        """Explain the error."""
        if self.reason == PartialOrderErrorReason.Empty:
            return "The partial order cannot be empty."

        if self.reason == PartialOrderErrorReason.IncorrectNodes:
            return "The partial order does not contain all the nodes of the open graph or contains nodes that are not in the open graph."
        assert_never(self.reason)


@dataclass
class PartialOrderLayerError(FlowError, XZCorrectionsError):
    """Exception subclass to handle flow and XZ-corrections errors concerning a specific layer of the partial order."""

    reason: PartialOrderLayerErrorReason
    layer_index: int
    layer: AbstractSet[int]

    def __str__(self) -> str:
        """Explain the error."""
        if self.reason == PartialOrderLayerErrorReason.FirstLayer:
            return f"The first layer of the partial order must contain all the output nodes of the open graph and cannot be empty. First layer: {self.layer}"

        # Note: A flow defined on an open graph without outputs will trigger this error. This is not the case for an XZ-corrections object.

        if self.reason == PartialOrderLayerErrorReason.NthLayer:
            return f"Partial order layer {self.layer_index} = {self.layer} contains non-measured nodes of the open graph, is empty or contains nodes in previous layers."
        assert_never(self.reason)


@dataclass
class XZCorrectionsOrderError(XZCorrectionsError):
    """Exception subclass to handle incorrect XZ-corrections objects where the error concerns the correction dictionaries and the partial order."""

    reason: XZCorrectionsOrderErrorReason
    node: int
    correction_set: AbstractSet[int]
    past_and_present_nodes: AbstractSet[int]

    def __str__(self) -> str:
        """Explain the error."""
        if self.reason == XZCorrectionsOrderErrorReason.X:
            return "The X-correction set {self.node} -> {self.correction_set} is incompatible with the partial order: {self.past_and_present_nodes - {self.node}} ≼ {self.node}."

        if self.reason == XZCorrectionsOrderErrorReason.Z:
            return "The Z-correction set {self.node} -> {self.correction_set} is incompatible with the partial order: {self.past_and_present_nodes - {self.node}} ≼ {self.node}."

        assert_never(self.reason)


@dataclass
class XZCorrectionsGenericError(XZCorrectionsError):
    """Exception subclass to handle generic flow errors."""

    reason: XZCorrectionsGenericErrorReason

    def __str__(self) -> str:
        """Explain the error."""
        if self.reason == XZCorrectionsGenericErrorReason.IncorrectKeys:
            return "Keys of correction dictionaries must be a subset of the measured nodes."
        if self.reason == XZCorrectionsGenericErrorReason.IncorrectValues:
            return "Values of correction dictionaries must contain labels which are nodes of the open graph."
        if self.reason == XZCorrectionsGenericErrorReason.ClosedLoop:
            return "XZ-corrections are not runnable since the induced directed graph contains closed loops."
        if self.reason == XZCorrectionsGenericErrorReason.IncompatibleOrder:
            return "The input total measurement order is not compatible with the partial order induced by the XZ-corrections."

        assert_never(self.reason)
