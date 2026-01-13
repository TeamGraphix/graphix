"""Class for open graph states."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import networkx as nx

from graphix import parameter
from graphix.flow._find_cflow import find_cflow
from graphix.flow._find_gpflow import AlgebraicOpenGraph, PlanarAlgebraicOpenGraph, compute_correction_matrix
from graphix.flow.core import GFlow, PauliFlow
from graphix.fundamentals import AbstractMeasurement, AbstractPlanarMeasurement
from graphix.measurements import Measurement

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence

    from graphix.flow.core import CausalFlow
    from graphix.parameter import ExpressionOrSupportsFloat, Parameter
    from graphix.pattern import Pattern

# TODO: Maybe move these definitions to graphix.fundamentals and graphix.measurements ? Now they are redefined in graphix.flow._find_gpflow, not very elegant.
_M_co = TypeVar("_M_co", bound=AbstractMeasurement, covariant=True)
_PM_co = TypeVar("_PM_co", bound=AbstractPlanarMeasurement, covariant=True)


@dataclass(frozen=True)
class OpenGraph(Generic[_M_co]):
    """An unmutable dataclass providing a representation of open graph states.

    Attributes
    ----------
    graph : networkx.Graph[int]
        The underlying resource-state graph. Nodes represent qubits and edges represent the application of :math:`CZ` gate on the linked nodes.
    input_nodes : Sequence[int]
        An ordered sequence of node labels corresponding to the open graph inputs.
    output_nodes : Sequence[int]
        An ordered sequence of node labels corresponding to the open graph outputs.
    measurements : Mapping[int, _M_co]
        A mapping between the non-output nodes of the open graph (``key``) and their corresponding measurement label (``value``). Measurement labels can be specified as `Measurement`, `Plane` or `Axis` instances.

    Notes
    -----
    The inputs and outputs of `OpenGraph` instances in Graphix are defined as ordered sequences of node labels. This contrasts the usual definition of open graphs in the literature, where inputs and outputs are unordered sets of nodes labels. This restriction facilitates the interplay with `Pattern` objects, where the order of input and output nodes represents a choice of Hilbert space basis.

    Example
    -------
    >>> import networkx as nx
    >>> from graphix.fundamentals import Plane
    >>> from graphix.opengraph import OpenGraph
    >>> from graphix.measurements import Measurement
    >>>
    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> measurements = {i: Measurement(0.5 * i, Plane.XY) for i in range(2)}
    >>> input_nodes = [0]
    >>> output_nodes = [2]
    >>> og = OpenGraph(graph, input_nodes, output_nodes, measurements)
    """

    graph: nx.Graph[int]
    input_nodes: Sequence[int]
    output_nodes: Sequence[int]
    measurements: Mapping[int, _M_co]

    def __post_init__(self) -> None:
        """Validate the correctness of the open graph."""
        all_nodes = set(self.graph.nodes)
        inputs = set(self.input_nodes)
        outputs = set(self.output_nodes)

        if not set(self.measurements).issubset(all_nodes):
            raise OpenGraphError("All measured nodes must be part of the graph's nodes.")
        if not inputs.issubset(all_nodes):
            raise OpenGraphError("All input nodes must be part of the graph's nodes.")
        if not outputs.issubset(all_nodes):
            raise OpenGraphError("All output nodes must be part of the graph's nodes.")
        if outputs & self.measurements.keys():
            raise OpenGraphError("Output nodes cannot be measured.")
        if all_nodes - outputs != self.measurements.keys():
            raise OpenGraphError("All non-output nodes must be measured.")
        if len(inputs) != len(self.input_nodes):
            raise OpenGraphError("Input nodes contain duplicates.")
        if len(outputs) != len(self.output_nodes):
            raise OpenGraphError("Output nodes contain duplicates.")

    def to_pattern(self: OpenGraph[Measurement]) -> Pattern:
        """Extract a deterministic pattern from an `OpenGraph[Measurement]` if it exists.

        Returns
        -------
        Pattern
            A deterministic pattern on the open graph.

        Raises
        ------
        OpenGraphError
            If the open graph does not have flow.

        Notes
        -----
        - The open graph instance must be of parametric type `Measurement` to allow for a pattern extraction, otherwise it does not contain information about the measurement angles.

        - This method proceeds by searching a flow on the open graph and converting it into a pattern as prescripted in Ref. [1].
        It first attempts to find a causal flow because the corresponding flow-finding algorithm has lower complexity. If it fails, it attemps to find a Pauli flow because this property is more general than a generalised flow, and the corresponding flow-finding algorithms have the same complexity in the current implementation.

        References
        ----------
        [1] Browne et al., NJP 9, 250 (2007)
        """
        for extractor in (self.find_causal_flow, self.find_pauli_flow):
            flow = extractor()
            if flow is not None:
                return flow.to_corrections().to_pattern()

        raise OpenGraphError("The open graph does not have flow. It does not support a deterministic pattern.")

    def isclose(self, other: OpenGraph[_M_co], rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        """Check if two open graphs are equal within a given tolerance.

        Parameters
        ----------
        other : OpenGraph[_M_co]
        rel_tol : float
            Relative tolerance. Optional, defaults to ``1e-09``.
        abs_tol : float
            Absolute tolerance. Optional, defaults to ``0.0``.

        Returns
        -------
        bool
            ``True`` if the two open graphs are approximately equal.

        Notes
        -----
        This method verifies the open graphs have:
            - Truly equal underlying graphs (not up to an isomorphism).
            - Equal input and output nodes.
            - Same measurement planes or axes and approximately equal measurement angles if the open graph is of parametric type `Measurement`.

        The static typer does not allow an ``isclose`` comparison of two open graphs with different parametric type. For a structural comparison, see :func:`OpenGraph.is_equal_structurally`.
        """
        return self.is_equal_structurally(other) and all(
            m.isclose(other.measurements[node], rel_tol=rel_tol, abs_tol=abs_tol)
            for node, m in self.measurements.items()
        )

    def is_equal_structurally(self, other: OpenGraph[AbstractMeasurement]) -> bool:
        """Compare the underlying structure of two open graphs.

        Parameters
        ----------
        other : OpenGraph[AbstractMeasurement]

        Returns
        -------
        bool
        ``True`` if ``self`` and ``og`` have the same structure.

        Notes
        -----
        This method verifies the open graphs have:
            - Truly equal underlying graphs (not up to an isomorphism).
            - Equal input and output nodes. This assumes equal types as well, i.e., if ``self.input_nodes`` is a ``list`` and ``other.input_nodes`` is a ``tuple``, this method will return ``False``.
        It assumes the open graphs are well formed.

        The static typer allows comparing the structure of two open graphs with different parametric type.
        """
        return (
            nx.utils.graphs_equal(self.graph, other.graph)
            and self.input_nodes == other.input_nodes
            and self.output_nodes == other.output_nodes
        )

    def neighbors(self, nodes: Collection[int]) -> set[int]:
        """Return the set containing the neighborhood of a set of nodes in the open graph.

        Parameters
        ----------
        nodes : Collection[int]
            Set of nodes whose neighborhood is to be found

        Returns
        -------
        neighbors_set : set[int]
            Neighborhood of set `nodes`.
        """
        neighbors_set: set[int] = set()
        for node in nodes:
            neighbors_set |= set(self.graph.neighbors(node))
        return neighbors_set

    def odd_neighbors(self, nodes: Collection[int]) -> set[int]:
        """Return the set containing the odd neighborhood of a set of nodes in the open graph.

        Parameters
        ----------
        nodes : Collection[int]
            Set of nodes whose odd neighborhood is to be found

        Returns
        -------
        odd_neighbors_set : set[int]
            Odd neighborhood of set `nodes`.
        """
        odd_neighbors_set: set[int] = set()
        for node in nodes:
            odd_neighbors_set ^= self.neighbors([node])
        return odd_neighbors_set

    def extract_causal_flow(self: OpenGraph[_PM_co]) -> CausalFlow[_PM_co]:
        """Try to extract a causal flow on the open graph.

        This method is a wrapper over :func:`OpenGraph.find_causal_flow` with a single return type.

        Returns
        -------
        CausalFlow[_PM_co]
            A causal flow object if the open graph has causal flow.

        Raises
        ------
        OpenGraphError
            If the open graph does not have a causal flow.

        See Also
        --------
        :func:`OpenGraph.find_causal_flow`
        """
        cf = self.find_causal_flow()
        if cf is None:
            raise OpenGraphError("The open graph does not have a causal flow.")
        return cf

    def extract_gflow(self: OpenGraph[_PM_co]) -> GFlow[_PM_co]:
        r"""Try to extract a maximally delayed generalised flow (gflow) on the open graph.

        This method is a wrapper over :func:`OpenGraph.find_gflow` with a single return type.

        Returns
        -------
        GFlow[_PM_co]
            A gflow object if the open graph has gflow.

        Raises
        ------
        OpenGraphError
            If the open graph does not have a gflow.

        See Also
        --------
        :func:`OpenGraph.find_gflow`
        """
        gf = self.find_gflow()
        if gf is None:
            raise OpenGraphError("The open graph does not have a gflow.")
        return gf

    def extract_pauli_flow(self: OpenGraph[_M_co]) -> PauliFlow[_M_co]:
        r"""Try to extract a maximally delayed Pauli on the open graph.

        This method is a wrapper over :func:`OpenGraph.find_pauli_flow` with a single return type.

        Returns
        -------
        PauliFlow[_M_co]
            A Pauli flow object if the open graph has Pauli flow.

        Raises
        ------
        OpenGraphError
            If the open graph does not have a Pauli flow.

        See Also
        --------
        :func:`OpenGraph.find_pauli_flow`
        """
        pf = self.find_pauli_flow()
        if pf is None:
            raise OpenGraphError("The open graph does not have a Pauli flow.")
        return pf

    def find_causal_flow(self: OpenGraph[_PM_co]) -> CausalFlow[_PM_co] | None:
        """Return a causal flow on the open graph if it exists.

        Returns
        -------
        CausalFlow[_PM_co] | None
            A causal flow object if the open graph has causal flow  or ``None`` otherwise.

        See Also
        --------
        :func:`OpenGraph.extract_causal_flow`

        Notes
        -----
        - The open graph instance must be of parametric type `Measurement` or `Plane` since the causal flow is only defined on open graphs with :math:`XY` measurements.
        - This function implements the algorithm presented in Ref. [1] with polynomial complexity on the number of nodes, :math:`O(N^2)`.

        References
        ----------
        [1] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
        """
        return find_cflow(self)

    def find_gflow(self: OpenGraph[_PM_co]) -> GFlow[_PM_co] | None:
        r"""Return a maximally delayed Pauli on the open graph if it exists.

        Returns
        -------
        GFlow[_PM_co] | None
            A gflow object if the open graph has gflow or ``None`` otherwise.

        See Also
        --------
        :func:`OpenGraph.extract_gflow`

        Notes
        -----
        - The open graph instance must be of parametric type `Measurement` or `Plane` since the gflow is only defined on open graphs with planar measurements. Measurement instances with a Pauli angle (integer multiple of :math:`\pi/2`) are interpreted as `Plane` instances, in contrast with :func:`OpenGraph.find_pauli_flow`.
        - This function implements the algorithm presented in Ref. [1] with polynomial complexity on the number of nodes, :math:`O(N^3)`.

        References
        ----------
        [1] Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        aog = PlanarAlgebraicOpenGraph(self)
        correction_matrix = compute_correction_matrix(aog)
        if correction_matrix is None:
            return None
        return GFlow.try_from_correction_matrix(
            correction_matrix
        )  # The constructor returns `None` if the correction matrix is not compatible with any partial order on the open graph.

    def find_pauli_flow(self: OpenGraph[_M_co]) -> PauliFlow[_M_co] | None:
        r"""Return a maximally delayed Pauli on the open graph if it exists.

        Returns
        -------
        PauliFlow[_M_co] | None
            A Pauli flow object if the open graph has Pauli flow or ``None`` otherwise.

        See Also
        --------
        :func:`OpenGraph.extract_pauli_flow`

        Notes
        -----
        - Measurement instances with a Pauli angle (integer multiple of :math:`\pi/2`) are interpreted as `Axis` instances, in contrast with :func:`OpenGraph.find_gflow`.
        - This function implements the algorithm presented in Ref. [1] with polynomial complexity on the number of nodes, :math:`O(N^3)`.

        References
        ----------
        [1] Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        aog = AlgebraicOpenGraph(self)
        correction_matrix = compute_correction_matrix(aog)
        if correction_matrix is None:
            return None
        return PauliFlow.try_from_correction_matrix(
            correction_matrix
        )  # The constructor returns `None` if the correction matrix is not compatible with any partial order on the open graph.

    def compose(self, other: OpenGraph[_M_co], mapping: Mapping[int, int]) -> tuple[OpenGraph[_M_co], dict[int, int]]:
        r"""Compose two open graphs by merging subsets of nodes from ``self`` and ``other``, and relabeling the nodes of ``other`` that were not merged.

        Parameters
        ----------
        other : OpenGraph[_M_co]
            Open graph to be composed with ``self``.
        mapping: dict[int, int]
            Partial relabelling of the nodes in ``other``, with ``keys`` and ``values`` denoting the old and new node labels, respectively.

        Returns
        -------
        og: OpenGraph[_M_co]
            Composed open graph.
        mapping_complete: dict[int, int]
            Complete relabelling of the nodes in ``other``, with ``keys`` and ``values`` denoting the old and new node label, respectively.

        Notes
        -----
        Let's denote :math:`\{G(V_1, E_1), I_1, O_1\}` the open graph `self`, :math:`\{G(V_2, E_2), I_2, O_2\}` the open graph `other`, :math:`\{G(V, E), I, O\}` the resulting open graph `og` and `{v:u}` an element of `mapping`.

        We define :math:`V, U` the set of nodes in `mapping.keys()` and `mapping.values()`, and :math:`M = U \cap V_1` the set of merged nodes.

        The open graph composition requires that
        - :math:`V \subseteq V_2`.
        - If both `v` and `u` are measured, the corresponding measurements must have the same plane and angle.
         The returned open graph follows this convention:
        - :math:`I = (I_1 \cup I_2) \setminus M \cup (I_1 \cap I_2 \cap M)`,
        - :math:`O = (O_1 \cup O_2) \setminus M \cup (O_1 \cap O_2 \cap M)`,
        - If only one node of the pair `{v:u}` is measured, this measure is assigned to :math:`u \in V` in the resulting open graph.
        - Input (and, respectively, output) nodes in the returned open graph have the order of the open graph `self` followed by those of the open graph `other`. Merged nodes are removed, except when they are input (or output) nodes in both open graphs, in which case, they appear in the order they originally had in the graph `self`.
        """
        if not (mapping.keys() <= other.graph.nodes):
            raise ValueError("Keys of mapping must be correspond to nodes of other.")
        if len(mapping) != len(set(mapping.values())):
            raise ValueError("Values in mapping contain duplicates.")

        for v, u in mapping.items():
            if (
                (vm := other.measurements.get(v)) is not None
                and (um := self.measurements.get(u)) is not None
                and not vm.isclose(um)
            ):
                raise OpenGraphError(f"Attempted to merge nodes with different measurements: {v, vm} -> {u, um}.")

        shift = max(*self.graph.nodes, *mapping.values()) + 1

        mapping_sequential = {
            node: i for i, node in enumerate(sorted(other.graph.nodes - mapping.keys()), start=shift)
        }  # assigns new labels to nodes in other not specified in mapping

        mapping_complete = {**mapping, **mapping_sequential}

        g2_shifted = nx.relabel_nodes(other.graph, mapping_complete)
        g = nx.compose(self.graph, g2_shifted)

        merged = set(mapping_complete.values()) & self.graph.nodes

        def merge_ports(p1: Iterable[int], p2: Iterable[int]) -> list[int]:
            p2_mapped = [mapping_complete[node] for node in p2]
            p2_set = set(p2_mapped)
            part1 = [node for node in p1 if node not in merged or node in p2_set]
            part2 = [node for node in p2_mapped if node not in merged]
            return part1 + part2

        inputs = merge_ports(self.input_nodes, other.input_nodes)
        outputs = merge_ports(self.output_nodes, other.output_nodes)

        measurements_shifted = {mapping_complete[i]: meas for i, meas in other.measurements.items()}
        measurements = {**self.measurements, **measurements_shifted}

        return OpenGraph(g, inputs, outputs, measurements), mapping_complete

    def subs(
        self: OpenGraph[Measurement], variable: Parameter, substitute: ExpressionOrSupportsFloat
    ) -> OpenGraph[Measurement]:
        """Substitute a parameter with a value or expression in all measurement angles.

        Creates a new open graph where every measurement angle containing the specified variable is updated using the provided substitution. The original open graph instance remains unmodified.

        Parameters
        ----------
        variable : Parameter
            The symbolic expression to be replaced within the measurement angles.
        substitute : ExpressionOrSupportsFloat
            The value or symbolic expression to substitute in place of `variable`.

        Returns
        -------
        OpenGraph[Measurement]
            A new instance of OpenGraph with the updated measurement parameters.

        Notes
        -----
        Substitution relies on object identity. You must provide the exact parameter object instance currently stored in the measurements. Passing a new object with the same name will not trigger a substitution if it is not the same instance in memory.

        Examples
        --------
        >>> import networkx as nx
        >>> from graphix.fundamentals import Plane
        >>> from graphix.measurements import Measurement
        >>> from graphix.opengraph import OpenGraph
        >>> from graphix.parameter import Placeholder
        >>> # Initialize placeholders and open graph
        >>> parametric_angles = [Placeholder(f"alpha{i}") for i in range(2)]
        >>> measurements = {node: Measurement(angle, Plane.XY) for node, angle in enumerate(parametric_angles)}
        >>> og = OpenGraph(
        ...     graph=nx.Graph([(0, 1), (1, 2)]),
        ...     input_nodes=[0],
        ...     output_nodes=[2],
        ...     measurements=measurements,
        ... )
        >>> # To perform substitution, use the actual object in memory
        >>> new_og = og.subs(parametric_angles[0], 0.3)
        >>> # Note: og.subs(Placeholder("alpha0"), 0.3) would not trigger any substitution.
        """
        measurements = {
            node: Measurement(parameter.subs(meas.angle, variable, substitute), meas.plane)
            for node, meas in self.measurements.items()
        }
        return dataclasses.replace(self, measurements=measurements)

    def xreplace(
        self: OpenGraph[Measurement], assignment: Mapping[Parameter, ExpressionOrSupportsFloat]
    ) -> OpenGraph[Measurement]:
        """Perform parallel substitution of multiple parameters in measurement angles.

        Creates a new open graph where occurrences of parameters defined in the assignment mapping are replaced by their corresponding values. The original open graph instance remains unmodified.

        Parameters
        ----------
        assignment : Mapping[Parameter, ExpressionOrSupportsFloat]
            A dictionary-like mapping where keys are the `Parameter` objects to be replaced and values are the new expressions or numerical values.

        Returns
        -------
        OpenGraph[Measurement]
            A new instance of OpenGraph with the updated measurement angles.

        Notes
        -----
        The notes provided in :func:`self.subs` apply here as well.

        Examples
        --------
        >>> import networkx as nx
        >>> from graphix.fundamentals import Plane
        >>> from graphix.measurements import Measurement
        >>> from graphix.opengraph import OpenGraph
        >>> from graphix.parameter import Placeholder
        >>> # Initialize placeholders
        >>> alpha = Placeholder("alpha")
        >>> beta = Placeholder("beta")
        >>> measurements = {0: Measurement(alpha, Plane.XY), 1: Measurement(beta, Plane.XY)}
        >>> og = OpenGraph(nx.Graph([(0, 1)]), [0], [], measurements)
        >>> # Substitute multiple parameters at once
        >>> subs_map = {alpha: 0.5, beta: 1.2}
        >>> new_og = og.xreplace(subs_map)
        """
        measurements = {
            node: Measurement(parameter.xreplace(meas.angle, assignment), meas.plane)
            for node, meas in self.measurements.items()
        }
        return dataclasses.replace(self, measurements=measurements)


class OpenGraphError(Exception):
    """Exception subclass to handle open graphs errors."""
