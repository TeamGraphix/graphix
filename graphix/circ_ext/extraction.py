"""Tools for circuit extraction."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

from graphix.fundamentals import ParameterizedAngle, Plane, Sign
from graphix.measurements import BlochMeasurement, Measurement, PauliMeasurement

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphix.circ_ext.compilation import CompilationPass
    from graphix.command import Node
    from graphix.flow.core import PauliFlow
    from graphix.opengraph import OpenGraph
    from graphix.transpiler import Circuit


@dataclass(frozen=True)
class ExtractionResult:
    """Dataclass to represent the output of the circuit-extraction algorithm introduced in Ref. [1].

    Attributes
    ----------
    pexp_dag: PauliExponentialDAG
        Pauli exponential directed acyclical graph (DAG) representing a sequence multi-qubit rotations.

    clifford_map: CliffordMap
        Clifford transformation.

    Notes
    -----
    See Definition 3.3 in Ref. [1].

    References
    ----------
    [1] Simmons, 2021 (arXiv:2109.05654).
    """

    pexp_dag: PauliExponentialDAG
    clifford_map: CliffordMap

    def to_circuit(self, cp: CompilationPass) -> Circuit:
        """Transpile the extraction result to circuit.

        Transpilation is only supported when the pair Pauli-exponential DAG and Clifford map represents a unitary transformation.

        Parameters
        ----------
        cp : CompilationPass
            Compilation pass to synthesize the Pauli exponential DAG and the Clifford map in the extraction result.

        Returns
        -------
        Circuit
            Quantum circuit represented as a set of instructions.
        """
        return cp.er_to_circuit(self)


@dataclass(frozen=True)
class PauliString:
    """Dataclass representing a Pauli string over a set of MBQC nodes.

    Attributes
    ----------
    x_nodes : AbstractSet[int]
        Nodes on which a Pauli X operator is applied.
    y_nodes : AbstractSet[int]
        Nodes on which a Pauli Y operator is applied.
    z_nodes : AbstractSet[int]
        Nodes on which a Pauli Z operator is applied.
    sign : Sign
        Phase of the Pauli string.
    """

    x_nodes: AbstractSet[int] = dataclasses.field(default_factory=frozenset)
    y_nodes: AbstractSet[int] = dataclasses.field(default_factory=frozenset)
    z_nodes: AbstractSet[int] = dataclasses.field(default_factory=frozenset)
    sign: Sign = dataclasses.field(default_factory=lambda: Sign.PLUS)

    @staticmethod
    def from_measured_node(flow: PauliFlow[Measurement], node: Node) -> PauliString:
        """Extract the Pauli string of a measured node and its focused correction set.

        Parameters
        ----------
        flow : PauliFlow[Measurement]
            A focused Pauli flow. The resulting Pauli string is extracted from its correction function.
        node : int
            A measured node whose associated Pauli string is computed.

        Returns
        -------
        PauliString
            Primary extraction string associated to the input measured nodes. The sets in the returned `PauliString` instance are disjoint.

        Notes
        -----
        See Eq. (13) and Lemma 4.4 in Ref. [1]. The phase of the Pauli string is given by Eq. (37).

        References
        ----------
        [1] Simmons, 2021 (arXiv:2109.05654).
        """
        og = flow.og
        c_set = set(flow.correction_function[node])
        odd_c_set = og.odd_neighbors(c_set)
        inter_c_odd_set = c_set & odd_c_set

        x_corrections = frozenset((c_set - odd_c_set).intersection(og.output_nodes))
        y_corrections = frozenset(inter_c_odd_set.intersection(og.output_nodes))
        z_corrections = frozenset((odd_c_set - c_set).intersection(og.output_nodes))

        # Sign computation.
        negative_sign = False

        # One phase flip per edge between adjacent vertices in the correction set.
        negative_sign ^= og.graph.subgraph(c_set).number_of_edges() % 2 == 1

        # One phase flip per two Ys in the graph state stabilizer.
        negative_sign ^= bool(len(inter_c_odd_set) // 2 % 2)

        # One phase flip per node in the graph state stabilizer that is absorbed from a Pauli measurement with angle π.
        for n in c_set | odd_c_set:
            meas = og.measurements.get(n, None)
            if isinstance(meas, PauliMeasurement):
                negative_sign ^= meas.sign == Sign.MINUS

        # One phase flip if measured on the YZ plane.
        negative_sign ^= flow.node_measurement_label(node) == Plane.YZ

        return PauliString(x_corrections, y_corrections, z_corrections, Sign.minus_if(negative_sign))

    def remap(self, outputs_mapping: Callable[[int], int]) -> Self:
        """Remap nodes to qubit indices.

        Parameters
        ----------
        outputs_mapping: Callable[[int], int]
            Mapping between node numbers of the original MBQC pattern or open graph and qubit indices of a quantum circuit.

        Returns
        -------
        PauliString
            Pauli string defined on qubit indices.
        """
        x_nodes = {outputs_mapping(n) for n in self.x_nodes}
        y_nodes = {outputs_mapping(n) for n in self.y_nodes}
        z_nodes = {outputs_mapping(n) for n in self.z_nodes}
        return replace(self, x_nodes=frozenset(x_nodes), y_nodes=frozenset(y_nodes), z_nodes=frozenset(z_nodes))


@dataclass(frozen=True)
class PauliExponential:
    r"""Dataclass representing a Pauli exponential over a set of MBQC nodes.

    A Pauli exponential corresponds to the unitary operator

    .. math::

        U(\alpha) = \exp \left(i \frac{alpha}{2} P\right),

    where :math:`\alpha` is a real-valued angle and :math:`P` is a Pauli string.

    Attributes
    ----------
    angle : ParameterizedAngle
        The Pauli exponential angle :math:`\alpha` in units of :math:`\pi`. When extracted from a corrected node, it corresponds to the node's measurement divided by two.
    pauli_string : PauliString
        The signed Pauli string :math:`P` specifying the tensor product of Pauli operators acting on the corresponding MBQC nodes.
    """

    angle: ParameterizedAngle
    pauli_string: PauliString

    @staticmethod
    def from_measured_node(flow: PauliFlow[Measurement], node: Node) -> PauliExponential:
        """Extract the Pauli exponential of a measured node and its focused correction set.

        Parameters
        ----------
        flow : PauliFlow[Measurement]
            A focused Pauli flow. The resulting Pauli string is extracted from its correction function.
        node : int
            A measured node whose associated Pauli string is computed.

        Returns
        -------
        PauliExponential
            Primary extraction string associated to the input measured nodes. The sets in the returned `PauliString` instance are disjoint.

        Notes
        -----
        See Eq. (13) and Lemma 4.4 in Ref. [1]. The phase of the Pauli string is given by Eq. (37).

        References
        ----------
        [1] Simmons, 2021 (arXiv:2109.05654).
        """
        pauli_string = flow.pauli_strings[node]
        meas = flow.og.measurements[node]
        # We don't extract any rotation from Pauli Measurements. This is equivalent to setting the angle to 0.
        angle = meas.angle / 2 if isinstance(meas, BlochMeasurement) else 0

        return PauliExponential(angle, pauli_string)

    def remap(self, outputs_mapping: Callable[[int], int]) -> Self:
        """Remap nodes to qubit indices.

        See documentation in :meth:`PauliString.remap` for additional information.
        """
        return replace(self, pauli_string=self.pauli_string.remap(outputs_mapping))


@dataclass(frozen=True)
class PauliExponentialDAG:
    """Dataclass to represent a multi-qubit rotation formed by a sequence of Pauli exponentials extracted from a pattern.

    Attributes
    ----------
    pauli_exponentials: Mapping[int, PauliExponential]
        Mapping between measured nodes (``keys``) and Pauli exponentials (``values``).
    partial_order_layers: Sequence[AbstractSet[int]]
        Partial order between the Pauli exponentials in a layer form. The set `layers[i]` comprises the nodes in layer `i`. Nodes in layer `i` are "larger" in the partial order than nodes in layer `i+1`. The pattern's output nodes are always in layer 0.
    output_nodes: Sequence[int]
        Output nodes on which the Pauli exponential rotation acts.

    Notes
    -----
    See Definition 3.3 in Ref. [1].

    References
    ----------
    [1] Simmons, 2021 (arXiv:2109.05654).
    """

    pauli_exponentials: Mapping[int, PauliExponential]
    partial_order_layers: Sequence[AbstractSet[int]]
    output_nodes: Sequence[int]

    @staticmethod
    def from_focused_flow(flow: PauliFlow[Measurement]) -> PauliExponentialDAG:
        """Extract a Pauli exponential rotation from a focused Pauli flow.

        This routine associates a Pauli exponential to each measured node in ``flow``. The flow's partial order defines a partial order between the Pauli exponentials such that Pauli exponentials in the same layer commute.

        Parameters
        ----------
        flow : PauliFlow[Measurement]
            A focused Pauli flow.

        Returns
        -------
        PauliExponentialRotation

        Notes
        -----
        See Definition 3.3 and Example C.13 in Ref. [1].

        References
        ----------
        [1] Simmons, 2021 (arXiv:2109.05654).
        """
        pauli_strings = {node: PauliExponential.from_measured_node(flow, node) for node in flow.correction_function}

        return PauliExponentialDAG(pauli_strings, flow.partial_order_layers, flow.og.output_nodes)

    def remap(self, outputs_mapping: Callable[[int], int]) -> Self:
        """Remap nodes to qubit indices.

        See documentation in :meth:`PauliString.remap` for additional information.
        """
        pauli_exponentials = {node: pexp.remap(outputs_mapping) for node, pexp in self.pauli_exponentials.items()}
        return replace(self, pauli_exponentials=pauli_exponentials)


@dataclass(frozen=True)
class CliffordMap:
    """Dataclass to represent a Clifford map.

    A Clifford map describes a linear transformation between the space of input qubits and the space of output qubits. It is encoded as a map from the Pauli-group generators (X and Z) over the input nodes to Pauli strings over the output nodes.

    Attributes
    ----------
    x_map: Mapping[int, PauliString]
        Map for the X generators. ``keys`` correspond to input nodes and ``values`` to their corresponding Pauli string over the outputs nodes.
    z_map: Mapping[int, PauliString]
        Map for the Z generators. ``keys`` correspond to input nodes and ``values`` to their corresponding Pauli string over the outputs nodes.
    input_nodes: Sequence[int]
        Sequence of inputs nodes.
    output_nodes: Sequence[int]
        Sequence of outputs nodes.

    Notes
    -----
    See Definition 3.3 in Ref. [1].

    References
    ----------
    [1] Simmons, 2021 (arXiv:2109.05654).
    """

    x_map: Mapping[int, PauliString]
    z_map: Mapping[int, PauliString]
    input_nodes: Sequence[int]
    output_nodes: Sequence[int]

    @staticmethod
    def from_focused_flow(flow: PauliFlow[Measurement]) -> CliffordMap:
        """Extract a Clifford map from a focused Pauli flow.

        This routine associates a two Pauli strings (one per generator of the Pauli group, X and Z) to each input node in ``flow.og``.

        Parameters
        ----------
        flow : PauliFlow[Measurement]
            A focused Pauli flow.

        Returns
        -------
        CliffordMap

        Notes
        -----
        See Definition 3.3 and Example C.13 in Ref. [1].

        References
        ----------
        [1] Simmons, 2021 (arXiv:2109.05654).
        """
        z_map = clifford_z_map_from_focused_flow(flow)
        x_map = clifford_x_map_from_focused_flow(flow)
        return CliffordMap(x_map, z_map, flow.og.input_nodes, flow.og.output_nodes)

    def remap(self, outputs_mapping: Callable[[int], int]) -> Self:
        """Remap nodes to qubit indices.

        See documentation in :meth:`PauliString.remap` for additional information.
        """
        x_map = {node: ps.remap(outputs_mapping) for node, ps in self.x_map.items()}
        z_map = {node: ps.remap(outputs_mapping) for node, ps in self.z_map.items()}
        return replace(self, x_map=x_map, z_map=z_map)


def extend_input(og: OpenGraph[Measurement]) -> tuple[OpenGraph[Measurement], dict[int, int]]:
    r"""Extend the inputs of a given open graph.

    For every input node :math:`v`, a new node :math:`u` and edge :math:`(u, v)` are added to the open graph. Node :math:`u` is measured in plane :math:`XY` with angle :math:`\alpha = 0` and replaces :math:`v` in the open graph's sequence of input nodes.

    Parameters
    ----------
    og: OpenGraph[Measurement]
        Open graph whose input nodes are extended.

    Returns
    -------
    OpenGraph[Measurement]
        Open graph with the extended inputs.
    dict[int, int]
        Mapping between previous (``key``) and new (``value``) input nodes.

    Notes
    -----
    This operation preserves the Pauli flow.
    """
    ancillary_inputs_map: dict[int, int] = {}
    fresh_node = max(og.graph.nodes) + 1
    graph = og.graph.copy()
    input_nodes = list(og.input_nodes)
    new_input_nodes: list[int] = []
    while input_nodes:
        input_node = input_nodes.pop()
        graph.add_edge(input_node, fresh_node)
        ancillary_inputs_map[input_node] = fresh_node
        new_input_nodes.append(fresh_node)
        fresh_node += 1

    measurements = {**og.measurements, **dict.fromkeys(new_input_nodes, Measurement.X)}

    # We reverse the inputs order to match the order of initial inputs.
    return replace(og, graph=graph, input_nodes=new_input_nodes[::-1], measurements=measurements), ancillary_inputs_map


def clifford_z_map_from_focused_flow(flow: PauliFlow[Measurement]) -> dict[int, PauliString]:
    """Extract a map between Z over the input nodes and Pauli strings over the output nodes from a focused Pauli flow.

    If the input node is a measured node, the resulting Pauli string is given by the correction set. If the input node is also an output node, the resulting Pauli string is Z (representing the identity map).

    Parameters
    ----------
    flow : PauliFlow[Measurement]
        A focused Pauli flow.

    Returns
    -------
    dict[int, PauliString]
        Map between input nodes (``keys``) and Pauli strings over the output nodes (``values``).

    Notes
    -----
    See Definition 3.3 and Example C.13 in Ref. [1].

    References
    ----------
    [1] Simmons, 2021 (arXiv:2109.05654).
    """
    # Nodes are either measured or outputs.
    return {
        node: flow.pauli_strings[node] if node in flow.og.measurements else PauliString(z_nodes=frozenset({node}))
        for node in flow.og.input_nodes
    }


def clifford_x_map_from_focused_flow(flow: PauliFlow[Measurement]) -> Mapping[int, PauliString]:
    """Extract a map between X over the input nodes and Pauli strings over the output nodes from a focused Pauli flow.

    The resulting Pauli string is given by the correction set of a focused flow of the extended open graph.

    Parameters
    ----------
    flow : PauliFlow[Measurement]
        A focused Pauli flow.

    Returns
    -------
    dict[int, PauliString]
        Map between input nodes (``keys``) and Pauli strings over the output nodes (``values``).

    Notes
    -----
    See Definition 3.3 and Example C.13 in Ref. [1].

    References
    ----------
    [1] Simmons, 2021 (arXiv:2109.05654).
    """
    og = flow.og
    og_extended, ancillary_inputs_map = extend_input(og)

    # Here it's crucial to not infer Pauli measurements to avoid converting measurements inadvertently.
    flow_extended = og_extended.extract_pauli_flow()

    # `flow_extended` is guaranteed to be focused if `flow` is focused.
    # This function assumes that `flow` is focused and does not check it.
    # In the context for `CliffordMap.from_focused_flow` the check is performed when accessing the cached property `flow.pauli_strings` in the function `clifford_z_map_from_focused_flow`.

    # It's better to call the `PauliString` constructor instead of the cached property `flow_extended.pauli_strings` since the latter will compute a `PauliString` for _every_ node in the correction function and we just need it for the input nodes.
    x_map_ancillas = {node: PauliString.from_measured_node(flow_extended, node) for node in og_extended.input_nodes}

    return {input_node: x_map_ancillas[ancillary_inputs_map[input_node]] for input_node in og.input_nodes}
