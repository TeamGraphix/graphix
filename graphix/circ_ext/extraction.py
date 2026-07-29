"""Tools for circuit extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from graphix._linalg import MatGF2
from graphix.fundamentals import Axis, ParameterizedAngle, Plane, Sign
from graphix.measurements import BlochMeasurement, Measurement, PauliMeasurement
from graphix.sim.base_backend import NodeIndex

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from collections.abc import Set as AbstractSet

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

    def to_circuit(
        self,
        pexp_cp: Callable[[PauliExponentialDAG, Circuit], None] | None = None,
        cm_cp: Callable[[CliffordMap, Circuit], None] | None = None,
    ) -> Circuit:
        """Transpile the extraction result to circuit.

        Transpilation is only supported when the pair Pauli-exponential DAG and Clifford map represents a unitary transformation.

        Parameters
        ----------
        pexp_cp: Callable[[PauliExponentialDAG, Circuit], None] | None, default
            Compilation pass to synthesize a Pauli exponential DAG. If ``None`` (default), :func:`graphix.circ_ext.compilation.pexp_ladder_pass` is employed.
        cm_cp: Callable[[CliffordMap, Circuit], None] | None
            Compilation pass to synthesize a Clifford map. If ``None`` (default), :func:`graphix.circ_ext.compilation.cm_berg_pass` is employed. This pass only handles unitaries so far (Clifford maps with the same number of input and output nodes).

        Returns
        -------
        Circuit
            Quantum circuit represented as a set of instructions.
        """
        from graphix.circ_ext.compilation import er_to_circuit  # noqa: PLC0415

        return er_to_circuit(self, pexp_cp=pexp_cp, cm_cp=cm_cp)


@dataclass(frozen=True)
class PauliString:
    """Dataclass representing a Pauli string.

    Attributes
    ----------
    dim : int
        Dimension of the Hilbert space on which the Pauli string acts.
    axes : Mapping[int, Axis]
        Mapping between qubit indices and the applied Pauli operator.
    sign : Sign
        Sign of the Pauli string.

    Notes
    -----
    The identity operators in the Pauli string are omitted in ``axes``, but they can be inferred from the dimension of the Hilbert space ``dim``.
    """

    dim: int
    axes: Mapping[int, Axis]
    sign: Sign = Sign.PLUS

    @staticmethod
    def from_str(ps: str) -> PauliString:
        """Construct a PauliString from its string representation.

        The sign prefix (``'+'`` or ``'-'``) is optional; if omitted, the sign
        defaults to ``Sign.PLUS``.

        Parameters
        ----------
        ps : str
            String encoding of a Pauli string. Optionally starts with ``'+'``
            or ``'-'``, followed by zero or more single-character Pauli operators
            (``'X'``, ``'Y'``, ``'Z'``, or ``'I'``).

        Returns
        -------
        PauliString
            The PauliString instance corresponding to the input string.

        Raises
        ------
        ValueError
            If the first character is not ``'+'`` or ``'-'`` or a valid operator character,
            or any operator character is not one of ``'X'``, ``'Y'``, ``'Z'``, ``'I'``.

        Examples
        --------
        >>> PauliString.from_str("+XYZ")
        PauliString(dim=3, axes={0: Axis.X, 1: Axis.Y, 2: Axis.Z}, sign=Sign.PLUS)
        >>> PauliString.from_str("-IXI")
        PauliString(dim=3, axes={1: Axis.X}, sign=Sign.MINUS)
        >>> PauliString.from_str("X")
        PauliString(dim=1, axes={0: Axis.X}, sign=Sign.PLUS)
        >>> PauliString.from_str("")
        PauliString(dim=0, axes={}, sign=Sign.PLUS)
        """
        if len(ps) == 0:
            return PauliString(0, {})

        first_char, *operator_chars = iter(ps)

        match first_char:
            case "+":
                sign = Sign.PLUS
            case "-":
                sign = Sign.MINUS
            case "X" | "Y" | "Z" | "I":
                sign = Sign.PLUS
                operator_chars = [first_char, *operator_chars]
            case _:
                raise ValueError(
                    f"First character must be '+', '-', or a valid operator ('X', 'Y', 'Z', or 'I'), got '{first_char}'."
                )

        invalid = {op for op in operator_chars if op not in Axis.__members__ and op != "I"}
        if invalid:
            raise ValueError(f"Invalid Pauli operator(s): {invalid}. Each operator must be 'X', 'Y', 'Z', or 'I'.")

        return PauliString(
            sign=sign,
            dim=len(operator_chars),
            axes={i: Axis[op] for i, op in enumerate(operator_chars) if op != "I"},
        )

    def __str__(self) -> str:
        """Return a string representation of the Pauli string."""
        pauli_str = (
            str(self.sign),
            *(axis.name if (axis := self.axes.get(node)) else "I" for node in range(self.dim)),
        )

        return "".join(pauli_str)

    @staticmethod
    def from_tableau(tab: MatGF2) -> PauliString:
        r"""Construct a `PauliString` from a one-dimensional tableau representation.

        The tableau encodes a Pauli operator of the form
        :math:`\pm P_0 \otimes P_1 \otimes \cdots \otimes P_{n-1}`,
        where each single-qubit Pauli is stored as an (x, z) bit pair and the final
        element encodes the sign.

        Layout of ``tab`` (length ``2n + 1``)::

            [ x_0, x_1, …, x_{n-1} | z_0, z_1, …, z_{n-1} | sign ]

        Encoding conventions:

        * ``(x=1, z=0)`` → X
        * ``(x=0, z=1)`` → Z
        * ``(x=1, z=1)`` → Y
        * ``(x=0, z=0)`` → I (identity, qubit absent in ``axes``)
        * ``sign = 0``   → +1
        * ``sign = 1``   → -1

        Parameters
        ----------
        tab : MatGF2
            A one-dimensional GF(2) array of odd length ``2n + 1``
            representing an n-qubit Pauli operator.

        Returns
        -------
        PauliString
            The Pauli operator encoded by ``tab``.

        Raises
        ------
        ValueError
            If ``tab`` is not one-dimensional or ``len(tab)`` is even.

        Examples
        --------
        >>> tab = MatGF2(np.array([1, 1, 1]))
        >>> PauliString.from_tableau(tab)
        PauliString(dim=1, axes={0: Axis.Y}, sign=Sign.MINUS)
        >>> tab = MatGF2(np.array([0, 0, 0, 1, 0]))
        >>> PauliString.from_tableau(tab)
        PauliString(dim=2, axes={1: Axis.Z}, sign=Sign.PLUS)
        """
        if tab.ndim != 1:
            raise ValueError(
                f"Attempted to initialise a PauliString from a {tab.ndim}-dimensional tableau. `PauliString.from_tableau` expects a one-dimensional array."
            )
        if len(tab) % 2 == 0:
            raise ValueError(
                f"`PauliString.from_tableau` expects an array with an odd number of elements (got {len(tab)})."
            )

        dim = len(tab) // 2
        sign = Sign.minus_if(tab[-1])

        axes: dict[int, Axis] = {}
        for i, (x, z) in enumerate(zip(tab[:dim], tab[dim:-1], strict=True)):
            if (x, z) == (1, 0):
                axes[i] = Axis.X
            elif (x, z) == (0, 1):
                axes[i] = Axis.Z
            elif (x, z) == (1, 1):
                axes[i] = Axis.Y

        return PauliString(dim, axes, sign)

    def to_tableau(self) -> MatGF2:
        """Serialise this PauliString into a one-dimensional tableau representation.

        Produces the inverse of :meth:`from_tableau`: a ``MatGF2`` of length
        ``2n + 1`` whose layout is::

            [ x_0, x_1, …, x_{n-1} | z_0, z_1, …, z_{n-1} | sign ]

        Encoding conventions:

        * X → ``(x=1, z=0)``
        * Z → ``(x=0, z=1)``
        * Y → ``(x=1, z=1)``
        * I → ``(x=0, z=0)`` (absent in ``self.axes``)
        * ``+`` sign → ``0``
        * ``-`` sign → ``1``

        Returns
        -------
        MatGF2
            A one-dimensional GF(2) array of length ``2 * self.dim + 1``.

        Examples
        --------
        >>> ps = PauliString.from_str("-XY")
        >>> ps.to_tableau()
        MatGF2([1, 1, 0, 1, 1], dtype=uint8)
        """
        tab = MatGF2(np.zeros(2 * self.dim + 1, dtype=np.uint8))

        for i, ax in self.axes.items():
            if ax in {Axis.X, Axis.Y}:
                tab[i] = 1
            if ax in {Axis.Y, Axis.Z}:
                tab[i + self.dim] = 1

        if self.sign is Sign.MINUS:
            tab[2 * self.dim] = 1

        return tab


@dataclass(frozen=True)
class PauliExponential:
    r"""Dataclass representing a Pauli exponential.

    A Pauli exponential corresponds to the unitary operator

    .. math::

        U(\alpha) = \exp \left(i \frac{\alpha}{2} P\right),

    where :math:`\alpha` is a real-valued angle and :math:`P` is a Pauli string.

    Attributes
    ----------
    angle : ParameterizedAngle
        The Pauli exponential angle :math:`\alpha` in units of :math:`\pi`. When extracted from a corrected node, it corresponds to the node's measurement divided by two.
    pauli_string : PauliString
        The signed Pauli string :math:`P` specifying the tensor product of Pauli operators acting on qubit indices.
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
            Primary extraction string associated to the input measured nodes. The sets in the returned ``PauliString`` instance are disjoint.

        Notes
        -----
        See Eq. (13) and Lemma 4.4 in Ref. [1]. The phase of the Pauli string is given by Eq. (37).

        References
        ----------
        [1] Simmons, 2021 (arXiv:2109.05654).
        """
        pauli_string = flow.extraction_pauli_strings[node]
        meas = flow.og.measurements[node]
        # We don't extract any rotation from Pauli Measurements. This is equivalent to setting the angle to 0.
        angle = meas.angle / 2 if isinstance(meas, BlochMeasurement) else 0

        return PauliExponential(angle, pauli_string)


@dataclass(frozen=True)
class PauliExponentialDAG:
    """Dataclass to represent a multi-qubit rotation formed by a sequence of Pauli exponentials extracted from a pattern.

    Attributes
    ----------
    pauli_exponentials: Mapping[int, PauliExponential]
        Mapping between measured nodes (``keys``) and Pauli exponentials (``values``).
    partial_order_layers: Sequence[AbstractSet[int]]
        Partial order between the Pauli exponentials in a layer form. The set ``layers[i]`` comprises the nodes in layer ``i``. Nodes in layer ``i`` are "larger" in the partial order than nodes in layer ``i+1``. The pattern's output nodes are always in layer 0.
    output_nodes: Sequence[int]
        Output nodes on which the Pauli exponential rotation acts.

    Notes
    -----
    See Definition 3.3 in Ref. [1].

    The Pauli strings in the Pauli exponentials are defined on qubit indices
    which correspond to the indices of the sequence ``output_nodes``.

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
        PauliExponentialDAG

        Notes
        -----
        See Definition 3.3 and Example C.13 in Ref. [1].

        References
        ----------
        [1] Simmons, 2021 (arXiv:2109.05654).
        """
        pauli_strings = {node: PauliExponential.from_measured_node(flow, node) for node in flow.correction_function}

        return PauliExponentialDAG(pauli_strings, flow.partial_order_layers, flow.og.output_nodes)


@dataclass(frozen=True)
class CliffordMap:
    r"""Dataclass to represent a Clifford map.

    A Clifford map encodes the action of a Clifford operator on Pauli generators.
    It describes how single-qubit Pauli operators on the input qubits are mapped,
    under conjugation, to Pauli strings on the output qubits.

    For each input qubit :math:`i`, the map specifies:

    .. math::

        P_i = C X_i C^\dagger

    .. math::

        P_i = C Z_i C^\dagger

    where the resulting operators :math:`P_i` are Pauli strings over the output qubits.

    The sequences ``input_nodes`` and ``output_nodes`` define the correspondence
    between qubit indices and node labels in the input and output spaces.

    Attributes
    ----------
    x_map: Sequence[PauliString]
        Images of the :math:`X` generators. The :math:`i`-th element is the Pauli
        string corresponding to :math:`C X_i C^\dagger`.
    z_map: Sequence[PauliString]
        Images of the :math:`Z` generators. The :math:`i`-th element is the Pauli
        string corresponding to :math:`C Z_i C^\dagger`.
    input_nodes: Sequence[int]
        Sequence of inputs nodes.
    output_nodes: Sequence[int]
        Sequence of outputs nodes.

    Notes
    -----
    See Definition 3.3 in Ref. [1].

    Elements of ``x_map`` and ``z_map`` are in one-to-one correspondance with ``input_nodes``. Each Pauli string is defined over qubit indices corresponding to positions in ``output_nodes``.

    References
    ----------
    [1] Simmons, 2021 (arXiv:2109.05654).
    """

    x_map: Sequence[PauliString]
    z_map: Sequence[PauliString]
    input_nodes: Sequence[int]
    output_nodes: Sequence[int]

    @staticmethod
    def from_focused_flow(flow: PauliFlow[Measurement]) -> CliffordMap:
        """Extract a Clifford map from a focused Pauli flow.

        This routine associates two Pauli strings (one per generator of the Pauli group, X and Z) to each input node in ``flow.og``.

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

    def to_tableau(self) -> MatGF2:
        """Convert the CliffordMap into its binary tableau representation.

        The returned tableau is a ``(2n, 2n + 1)`` binary matrix over GF(2),
        where ``n`` is the number of qubits. The first ``n`` rows correspond
        to the images of X generators, and the next ``n`` rows correspond to
        the images of Z generators. Columns encode the X and Z components of
        the resulting Pauli strings, along with a sign column.

        Each PauliString in ``x_map`` and ``z_map`` is decomposed into its
        X/Z support:
        - X contributes a 1 to the X block.
        - Z contributes a 1 to the Z block.
        - Y contributes a 1 to both X and Z blocks.
        The sign of the Pauli string is stored in the final column
        (0 for ``Sign.PLUS`` and 1 for ``Sign.MINUS``).

        Returns
        -------
        MatGF2
            A binary matrix of shape ``(2n, 2n + 1)`` representing the
            Clifford tableau.

        Raises
        ------
        NotImplementedError
            If the number of input nodes differs from the number of output
            nodes (i.e., the map is an isometry instead of a square Clifford).
        """
        n = len(self.input_nodes)
        if n != len(self.output_nodes):
            raise NotImplementedError(
                f"Isometries are not supported yet: # of inputs ({len(self.input_nodes)}) must be equal to the # of outputs ({len(self.output_nodes)})."
            )

        return MatGF2(np.vstack([ps.to_tableau() for psmap in (self.x_map, self.z_map) for ps in psmap]))


def extraction_ps_from_corrected_node(flow: PauliFlow[Measurement], node: Node) -> PauliString:
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
        Primary extraction string associated to the input measured nodes. The Pauli string is defined over qubit indices corresponding to positions in ``output_nodes``.

    Notes
    -----
    See Eq. (13) and Lemma 4.4 in Ref. [1]. The phase of the Pauli string is given by Eq. (37).

    References
    ----------
    [1] Simmons, 2021 (arXiv:2109.05654).
    """
    og = flow.og
    dim = len(flow.og.output_nodes)
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

    axes_dict: dict[int, Axis] = {}
    output_to_qubit_mapping = NodeIndex()
    output_to_qubit_mapping.extend(og.output_nodes)

    # Sets `x_corrections`, `y_corrections` and `z_corrections` are disjoint.
    corrections = (x_corrections, y_corrections, z_corrections)
    for correction, axis in zip(corrections, Axis, strict=True):
        for cnode in correction:
            qubit = output_to_qubit_mapping.index(cnode)
            axes_dict[qubit] = axis

    return PauliString(dim, axes_dict, Sign.minus_if(negative_sign))


def extend_input(og: OpenGraph[Measurement]) -> tuple[OpenGraph[Measurement], dict[int, int]]:
    r"""Extend the inputs of a given open graph.

    For every input node :math:`v`, a new node :math:`u` and edge :math:`(u, v)` are added to the open graph. Node :math:`u` is measured in Pauli axis :math:`X` and replaces :math:`v` in the open graph's sequence of input nodes.

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


def clifford_z_map_from_focused_flow(flow: PauliFlow[Measurement]) -> tuple[PauliString, ...]:
    r"""Extract the images of the Z generators of a Clifford map from a focused Pauli flow.

    If the input node is a measured node, the resulting Pauli string is given by the correction set. If the input node is also an output node, the resulting Pauli string is Z (representing the identity map).

    Parameters
    ----------
    flow : PauliFlow[Measurement]
        A focused Pauli flow.

    Returns
    -------
    tuple[PauliString,...]
        Images of the :math:`Z` generators. The :math:`i`-th element is the Pauli string
        corresponding to :math:`C Z_i C^\dagger`, where :math:`C` is the Clifford map.

    Notes
    -----
    See Definition 3.3 and Example C.13 in Ref. [1].

    References
    ----------
    [1] Simmons, 2021 (arXiv:2109.05654).
    """
    dim = len(flow.og.output_nodes)
    output_to_qubit_mapping = NodeIndex()
    output_to_qubit_mapping.extend(flow.og.output_nodes)
    # Input nodes are either measured or outputs.
    return tuple(
        flow.extraction_pauli_strings[node]
        if node in flow.og.measurements
        else PauliString(dim, {output_to_qubit_mapping.index(node): Axis.Z})
        for node in flow.og.input_nodes
    )


def clifford_x_map_from_focused_flow(flow: PauliFlow[Measurement]) -> tuple[PauliString, ...]:
    r"""Extract the images of the X generators of a Clifford map from a focused Pauli flow.

    The resulting Pauli string is given by the correction set of a focused flow of the extended open graph.

    Parameters
    ----------
    flow : PauliFlow[Measurement]
        A focused Pauli flow.

    Returns
    -------
    tuple[PauliString,...]
        Images of the :math:`X` generators. The :math:`i`-th element is the Pauli string
        corresponding to :math:`C X_i C^\dagger`, where :math:`C` is the Clifford map.

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
    flow_extended = og_extended.to_pauliflow()

    # `flow_extended` is guaranteed to be focused if `flow` is focused.
    # This function assumes that `flow` is focused and does not check it.
    # In the context for `CliffordMap.from_focused_flow` the check is performed when accessing the cached property `flow.pauli_strings` in the function `clifford_z_map_from_focused_flow`.

    # It's better to call the `PauliString` constructor instead of the cached property `flow_extended.pauli_strings` since the latter will compute a `PauliString` for _every_ node in the correction function and we just need it for the input nodes.
    x_map_ancillas = {node: extraction_ps_from_corrected_node(flow_extended, node) for node in og_extended.input_nodes}

    return tuple(x_map_ancillas[ancillary_inputs_map[input_node]] for input_node in og.input_nodes)
