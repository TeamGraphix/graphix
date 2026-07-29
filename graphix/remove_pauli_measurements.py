"""Remove Pauli measurements.

This module provides procedures for pushing Pauli measurements in
front of a pattern and for subsequently removing them from the
pattern.

Pauli pushing uses commutation rules of Pauli measurements to move
them before other measurements while appropriately shifting their
signals, so that all Pauli measurements end up with empty
domains. This step is required before the actual removal can be
performed.

For the removal itself, this module implements the algorithm described
in [BMBdF+21], Theorem 4.12 (Section 4.3: Removing Clifford
vertices).

[BMBdF+21] Miriam Backens, Hector Miller-Bakewell, Giovanni de Felice,
           Leo Lobski, and John van de Wetering,
           There and back again: A circuit extraction tale, Quantum, 2021,
           https://doi.org/10.22331/q-2021-03-25-421
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
from warnings import warn

import networkx as nx
from typing_extensions import assert_never

from graphix.clifford import Clifford, Domains
from graphix.command import Command
from graphix.fundamentals import Axis, Sign
from graphix.measurements import PauliMeasurement
from graphix.optimization import StandardizedPattern

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet
    from typing import TypeAlias

    from graphix.command import Node

    Graph: TypeAlias = nx.Graph[int]
else:
    Graph = nx.Graph


@dataclass(frozen=True, slots=True)
class PauliPushingCut:
    """Cut of the pattern measurements into Pauli and non-Pauli measurements."""

    original_pattern: StandardizedPattern

    pauli_measurements: tuple[Command.M, ...]
    """Pauli measurements: they are all applied before non-Pauli measurements and their domains are empty."""

    non_pauli_measurements: tuple[Command.M, ...]

    shifted_domains: dict[int, set[int]]
    """The shifted domains.

    The output of the original pattern can be retrieved by using
    :func:`~graphix.pattern.shift_outcomes` with these domains.
    """

    @property
    def measurements(self) -> tuple[Command.M, ...]:
        """Return the list of measurements, where Pauli measurements appear first and without signal."""
        return self.pauli_measurements + self.non_pauli_measurements

    @classmethod
    def from_standardizedpattern(
        cls, pattern: StandardizedPattern, leave_nodes: AbstractSet[Node] | None = None, *, stacklevel: int = 1
    ) -> PauliPushingCut:
        """Move Pauli measurements before the other measurements and return the cut between Pauli measurements and non-Pauli measurements.

        If you only need the resulting pattern, you can use
        :meth:`~graphix.optimization.StandardizedPattern.perform_pauli_pushing` or
        :meth:`~graphix.pattern.Pattern.perform_pauli_pushing` instead.

        Parameters
        ----------
        pattern: StandardizedPattern
            The pattern to reorder.
        leave_nodes : AbstractSet[Node], optional
            Nodes that should not be moved. This constraint only
            applies to Pauli nodes and has no effect on non-Pauli nodes.
        stacklevel : int, optional
            Stack level to use for warnings. Defaults to 1, meaning that warnings
            are reported at this function's call site.

        Returns
        -------
        PauliPushingCut
            The cut between Pauli measurements and non-Pauli measurements.
        """
        pattern._warn_non_inferred_pauli_measurements(stacklevel=stacklevel + 1)

        if leave_nodes:
            leave_non_pauli_nodes = [
                cmd.node
                for cmd in pattern.m_list
                if not isinstance(cmd.measurement, PauliMeasurement) and cmd.node in leave_nodes
            ]
            if leave_non_pauli_nodes:
                warn(
                    f"`leave_nodes` contains nodes that are not Pauli: {leave_non_pauli_nodes}. The constraint has no effect on these nodes.",
                    stacklevel=stacklevel + 1,
                )

        shifted_domains: dict[int, set[int]] = {}

        pauli_measurements: list[Command.M] = []
        non_pauli_measurements: list[Command.M] = []
        for cmd in pattern.m_list:
            s_domain = _expand_domain(shifted_domains, cmd.s_domain)
            t_domain = _expand_domain(shifted_domains, cmd.t_domain)
            if not isinstance(cmd.measurement, PauliMeasurement) or (leave_nodes and cmd.node in leave_nodes):
                non_pauli_measurements.append(
                    Command.M(node=cmd.node, measurement=cmd.measurement, s_domain=s_domain, t_domain=t_domain)
                )
            else:
                match cmd.measurement.axis:
                    case Axis.X:
                        # M^X X^s Z^t = M^{XY,0} X^s Z^t
                        #             = M^{XY,(-1)^s·0+tπ}
                        #             = S^t M^X
                        # M^{-X} X^s Z^t = M^{XY,π} X^s Z^t
                        #                = M^{XY,(-1)^s·π+tπ}
                        #                = S^t M^{-X}
                        shifted_domains[cmd.node] = t_domain
                    case Axis.Y:
                        # M^Y X^s Z^t = M^{XY,π/2} X^s Z^t
                        #             = M^{XY,(-1)^s·π/2+tπ}
                        #             = M^{XY,π/2+(s+t)π}      (since -π/2 = π/2 - π ≡ π/2 + π (mod 2π))
                        #             = S^{s+t} M^Y
                        # M^{-Y} X^s Z^t = M^{XY,-π/2} X^s Z^t
                        #                = M^{XY,(-1)^s·(-π/2)+tπ}
                        #                = M^{XY,-π/2+(s+t)π}  (since π/2 = -π/2 + π)
                        #                = S^{s+t} M^{-Y}
                        shifted_domains[cmd.node] = s_domain ^ t_domain
                    case Axis.Z:
                        # M^Z X^s Z^t = M^{XZ,0} X^s Z^t
                        #             = M^{XZ,(-1)^t((-1)^s·0+sπ)}
                        #             = M^{XZ,(-1)^t·sπ}
                        #             = M^{XZ,sπ}              (since (-1)^t·π ≡ π (mod 2π))
                        #             = S^s M^Z
                        # M^{-Z} X^s Z^t = M^{XZ,π} X^s Z^t
                        #                = M^{XZ,(-1)^t((-1)^s·π+sπ)}
                        #                = M^{XZ,(s+1)π}
                        #                = S^s M^{-Z}
                        shifted_domains[cmd.node] = s_domain
                    case _:  # pragma: no cover
                        assert_never(cmd.measurement.axis)
                pauli_measurements.append(Command.M(node=cmd.node, measurement=cmd.measurement))
        return cls(pattern, tuple(pauli_measurements), tuple(non_pauli_measurements), shifted_domains)

    def to_standardizedpattern(self) -> StandardizedPattern:
        """Return the standardized pattern where all Pauli measurements have been pushed."""
        return StandardizedPattern(
            self.original_pattern.input_nodes,
            self.original_pattern.output_nodes,
            self.original_pattern.n_list,
            self.original_pattern.e_set,
            self.measurements,
            _expand_corrections(self.shifted_domains, self.original_pattern.z_dict),
            _expand_corrections(self.shifted_domains, self.original_pattern.x_dict),
            self.original_pattern.c_dict,
        )


def _expand_domain(shifted_domains: Mapping[Node, AbstractSet[Node]], domain: AbstractSet[Node]) -> set[Node]:
    """Merge previously shifted domains into ``domain``.

    Parameters
    ----------
    shifted_domains: Mapping[Node, AbstractSet[Node]]
        Shifted domains
    domain : AbstractSet[Node]
        Domain to update with any accumulated shift information.
    """
    new_domain = set(domain)
    for node in domain & shifted_domains.keys():
        new_domain ^= shifted_domains[node]
    return new_domain


def _expand_corrections(
    shifted_domains: Mapping[Node, AbstractSet[Node]], corrections: Mapping[Node, AbstractSet[Node]]
) -> dict[Node, set[Node]]:
    return {node: _expand_domain(shifted_domains, domain) for node, domain in corrections.items()}


@dataclass(slots=True)
class _NodeSpec:
    """Annotations attached to every node of the graph state."""

    src: Node
    """The corresponding node in the original pattern."""

    domains: Domains = dataclasses.field(default_factory=lambda: Domains(set(), set()))
    """Correction domains (the nodes refer to the numbering of the original pattern)."""

    clifford: Clifford = Clifford.I

    pauli_measurement: PauliMeasurement | None = None
    """Pauli measurement if the node is measured with a Pauli measurement.

    ``None`` if the node is an output or measured with a non-Pauli measurement.
    """


class _RemovePauliMeasurements:
    """Processing structure for Pauli measurement removal.

    This class is instantiated from a Pauli-pushing cut and can be
    converted back to a standardized pattern with the method
    :meth:`to_standardizedpattern`. The public methods preserve the
    pattern semantics as invariant, such that an equivalent
    standardized pattern can be obtained at any stage of the process.
    """

    cut: PauliPushingCut
    """Cut of the pattern measurements obtained by Pauli-pushing."""

    graph: Graph
    node_specs: dict[Node, _NodeSpec]

    measurements: tuple[Command.M, ...]
    """List of the original measurements after Pauli-pushing."""

    pauli_measurements: dict[Axis, set[Node]]
    """For each axis, the set of non-input nodes that have a Pauli measurement on that axis.

    Nodes are given with the indexing of the original pattern: use ``node_map`` to retrieve the index in the graph."""

    input_node_set: set[Node]
    """Set of input nodes: inputs nodes are never pivoted, therefore their indexing is preserved."""

    output_node_set: set[Node]
    """Set of output nodes, using the new indexing."""

    node_map: dict[Node, Node]
    """Mapping from the nodes of the original pattern to the corresponding nodes in the (possibly pivoted) graph.

    Nodes that have been removed from the graph are absent from this
    mapping.

    The following invariant is maintained for all node ``u``:
    ``node_specs[node_map[u]].src == u``.
    """

    def __init__(self, cut: PauliPushingCut) -> None:
        self.cut = cut
        self.graph = cut.original_pattern.graph()
        self.node_specs = {node: _NodeSpec(node) for node in self.graph.nodes()}
        for node, domain in cut.original_pattern.x_dict.items():
            self.node_specs[node].domains.s_domain = _expand_domain(cut.shifted_domains, domain)
        for node, domain in cut.original_pattern.z_dict.items():
            self.node_specs[node].domains.t_domain = _expand_domain(cut.shifted_domains, domain)
        for node, clifford in cut.original_pattern.c_dict.items():
            self.node_specs[node].clifford = clifford
        self.measurements = cut.measurements
        self.pauli_measurements = {axis: set() for axis in Axis}
        self.input_node_set = set(cut.original_pattern.input_nodes)
        self.output_node_set = set(cut.original_pattern.output_nodes)
        for cmd_m in self.cut.pauli_measurements:
            if not isinstance(cmd_m.measurement, PauliMeasurement):  # pragma: no cover
                msg = "Pauli measurement expected."
                raise TypeError(msg)
            self.node_specs[cmd_m.node].pauli_measurement = cmd_m.measurement
            if cmd_m.node not in self.input_node_set:
                self.pauli_measurements[cmd_m.measurement.axis].add(cmd_m.node)
        self.node_map = {node: node for node in self.graph.nodes()}

    def _apply_clifford(self, node: Node, clifford: Clifford) -> None:
        """Apply a single-qubit Clifford gate to a node.

        This internal method breaks the semantics invariant: the
        semantics of the pattern is not preserved.
        """
        spec = self.node_specs[node]
        spec.clifford @= clifford
        spec.domains = clifford.commute_domains(spec.domains)
        if spec.pauli_measurement is not None:
            axis = spec.pauli_measurement.axis
            spec.pauli_measurement = spec.pauli_measurement.clifford(clifford)
            # Update pauli_measurements: sets in `pauli_measurements`
            # dict only cover non-input nodes, so this update is
            # skipped when the node is an input.
            if node in self.input_node_set:
                return
            new_axis = spec.pauli_measurement.axis
            if new_axis != axis:
                self.pauli_measurements[axis].remove(spec.src)
                self.pauli_measurements[new_axis].add(spec.src)

    def local_complement(self, u: Node) -> None:
        """
        Local complement.

        This implements Lemma 2.31 and 4.3 [BMBdF+21].
        """
        n_u = list(self.graph.neighbors(u))
        _complement_subgraph(self.graph, n_u)
        # |+⟩⟨+| + exp(-iπ/2) |-⟩⟨-| = H S† H
        self._apply_clifford(u, Clifford.H @ Clifford.SDG @ Clifford.H)
        for node in n_u:
            # |0⟩⟨0| + exp(iπ/2) |1⟩⟨1| = S
            self._apply_clifford(node, Clifford.S)

    def pivot_edge(self, u: Node, v: Node) -> None:
        """
        Pivot an edge.

        Prerequisite (not checked):
        - (u, v) is a graph edge;
        - u and v are not input nodes.

        This implements Lemmas 2.32 and 4.5 [BMBdF+21].
        """
        n_u = set(self.graph.neighbors(u))
        n_v = set(self.graph.neighbors(v))

        inter = n_u & n_v
        only_u = n_u - inter - {v}
        only_v = n_v - inter - {u}

        _complement_edges(self.graph, only_u, only_v)
        _complement_edges(self.graph, only_u, inter)
        _complement_edges(self.graph, only_v, inter)

        spec_u = self.node_specs[u]
        spec_v = self.node_specs[v]
        self.node_specs[v] = spec_u
        self.node_specs[u] = spec_v
        self.node_map[spec_u.src] = v
        self.node_map[spec_v.src] = u

        self._apply_clifford(u, Clifford.H)
        self._apply_clifford(v, Clifford.H)

        for node in inter:
            self._apply_clifford(node, Clifford.Z)

        u_output = u in self.output_node_set
        v_output = v in self.output_node_set
        if u_output != v_output:
            if u_output:
                old_output, new_output = u, v
            else:
                old_output, new_output = v, u
            self.output_node_set.remove(old_output)
            self.output_node_set.add(new_output)

    def _remove_node(self, u: Node) -> None:
        """Remove a node from the graph.

        This internal method breaks the semantics invariant: the
        semantics of the pattern is not preserved.
        """
        spec = self.node_specs[u]
        if spec.pauli_measurement is not None:
            self.pauli_measurements[spec.pauli_measurement.axis].remove(spec.src)
        del self.node_map[spec.src]
        del self.node_specs[u]
        self.graph.remove_node(u)

    def remove_z(self, u: Node, sign: Sign) -> None:
        """
        Remove Z/-Z measurement.

        Prerequisite (not checked):
        - u measured in Z (sign==PLUS) or -Z (sign==MINUS);
        - u is not an input node.

        This implements Lemma 4.7 [BMBdF+21].
        """
        if sign == Sign.MINUS:
            for node in self.graph.neighbors(u):
                self._apply_clifford(node, Clifford.Z)
        self._remove_node(u)

    def remove_y(self, u: Node, sign: Sign) -> None:
        """
        Remove Y/-Y measurement.

        Prerequisite (not checked):
        - u measured in Y (sign==PLUS) or -Y (sign=MINUS);
        - u is not an input node.

        This implements Lemma 4.8 [BMBdF+21].
        """
        self.local_complement(u)
        self.remove_z(u, sign)

    def remove_x_with_internal_neighbor(self, u: Node, v: Node, sign: Sign) -> None:
        """
        Remove X/-X measurement.

        Prerequisite (not checked):
        - u measured in X (sign==PLUS) or -X (sign=MINUS);
        - (u, v) is a graph edge;
        - u and v are internal nodes.

        This implements Lemma 4.9 [BMBdF+21].
        """
        self.pivot_edge(u, v)
        # `pivot_edge` swaps the endpoints `u` and `v`, so the
        # original node `u` becomes `v`.
        self.remove_z(v, sign)

    def remove_all_y_or_z(self) -> None:
        """
        Remove all Y and Z measurements, repeatedly.

        This implements Theorem 4.12, Steps 1 and 2.
        """
        for axis, remove in (
            (Axis.Y, self.remove_y),  # Step 1: remove any non-input Y measured node
            (Axis.Z, self.remove_z),  # Step 2: remove any non-input Z measured node
        ):
            while (node := next(iter(self.pauli_measurements[axis]), None)) is not None:
                new_node = self.node_map[node]
                spec = self.node_specs[new_node]
                if spec.pauli_measurement is None:  # pragma: no cover
                    msg = "Pauli measurement expected."
                    raise RuntimeError(msg)
                remove(new_node, spec.pauli_measurement.sign)

    def try_remove_x_with_internal_neighbor(self) -> bool:
        """Find an X measurement connected to an internal neighbor and remove it if any.

        This implements Theorem 4.12, Step 3, except for the final "go
        back to step 1", which is handled by the ``while`` loop in
        :func:`remove_pauli_measurements`.

        Returns
        -------
        bool
            ``True`` if a suitable node was found and removed, ``False`` otherwise.
        """
        for node in self.pauli_measurements[Axis.X]:
            new_node = self.node_map[node]
            internal_neighbors = set(self.graph.neighbors(new_node)) - self.input_node_set - self.output_node_set
            v = next(iter(internal_neighbors), None)
            if v is None:
                continue
            spec = self.node_specs[new_node]
            if spec.pauli_measurement is None:  # pragma: no cover
                msg = "Pauli measurement expected."
                raise RuntimeError(msg)
            self.remove_x_with_internal_neighbor(new_node, v, spec.pauli_measurement.sign)
            return True
        return False

    def try_pivot_x_with_output_node(self) -> bool:
        """Find an X-measurement connected to an output node that is not also an input and pivot it if any.

        This implements Lemma 4.11 and Theorem 4.12, Step 4, except
        for the final "go back to step 1", which is handled by
        the ``while`` loop in :func:`remove_pauli_measurements`.

        This method does not remove any node: it only performs the
        pivot, converting the X-measurement into a Z-measurement,
        which will be removed by Step 1 on the next iteration.

        Returns
        -------
        bool
            ``True`` if a suitable node was found and pivoted, ``False`` otherwise.
        """
        for node in self.pauli_measurements[Axis.X]:
            new_node = self.node_map[node]
            non_input_output_nodes = set(self.graph.neighbors(new_node)) & self.output_node_set - self.input_node_set
            v = next(iter(non_input_output_nodes), None)
            if v is None:
                continue
            self.pivot_edge(new_node, v)
            return True
        return False

    def remove_isolated_internal_nodes(self, *, stacklevel: int = 1) -> None:
        """Remove isolated internal nodes.

        Parameters
        ----------
        stacklevel : int, optional
            Stack level to use for warnings. Defaults to 1, meaning that warnings
            are reported at this function's call site.
        """
        # Construct the list first since the graph should not be
        # modified while enumerating isolated nodes.
        for node in list(nx.isolates(self.graph)):
            if node not in self.input_node_set and node not in self.output_node_set:
                if node not in self.pauli_measurements:
                    warn("Non-Pauli measurement on an isolated node was removed.", stacklevel=stacklevel + 1)
                self._remove_node(node)

    def _create_new_m(self, original_m: Command.M) -> Command.M | None:
        node = self.node_map.get(original_m.node)
        if node is None:
            return None
        spec = self.node_specs[node]
        new_m = original_m.clifford(spec.clifford)
        new_m.node = node
        new_m.s_domain = _map_domain(self.node_map, new_m.s_domain)
        new_m.t_domain = _map_domain(self.node_map, new_m.t_domain)
        return new_m

    def to_standardizedpattern(self) -> StandardizedPattern:
        # Prepare only nodes that have not been removed (i.e., that still appear in `node_specs`)
        n_list = tuple(cmd_n for cmd_n in self.cut.original_pattern.n_list if cmd_n.node in self.node_specs)
        output_nodes = tuple(self.node_map[node] for node in self.cut.original_pattern.output_nodes)
        measurements = tuple(new_m for original_m in self.measurements if (new_m := self._create_new_m(original_m)))
        z_dict = {
            node: t_domain
            for node in output_nodes
            if (t_domain := _map_domain(self.node_map, self.node_specs[node].domains.t_domain))
        }
        x_dict = {
            node: s_domain
            for node in output_nodes
            if (s_domain := _map_domain(self.node_map, self.node_specs[node].domains.s_domain))
        }
        c_dict = {node: clifford for node in output_nodes if (clifford := self.node_specs[node].clifford) != Clifford.I}
        return StandardizedPattern(
            self.cut.original_pattern.input_nodes,
            output_nodes,
            n_list,
            self.graph.edges(),
            measurements,
            z_dict,
            x_dict,
            c_dict,
        )


def _complement_subgraph(graph: nx.Graph[Node], s: Iterable[Node]) -> None:
    """Complement edges in a given subgraph."""
    all_pairs = set(itertools.combinations(s, 2))
    existing = all_pairs & graph.edges()
    graph.remove_edges_from(existing)
    graph.add_edges_from(all_pairs - existing)


def _complement_edges(graph: nx.Graph[Node], s: set[Node], t: set[Node]) -> None:
    """Complement edges between two set of nodes.

    ``s`` and ``t`` are supposed to be disjoint.
    """
    all_pairs = {(u, v) for u in s for v in t}
    existing = {(u, v) for u, v in graph.edges(s) if v in t}
    graph.remove_edges_from(existing)
    graph.add_edges_from(all_pairs - existing)


def _map_domain(node_map: Mapping[Node, Node], domain: Iterable[Node]) -> set[Node]:
    """Translate a set of nodes from the original numbering to the new numbering.

    Parameters
    ----------
    node_map : Mapping[Node, Node]
        Mapping from the nodes of the original pattern to the
        corresponding nodes in the (possibly pivoted) graph.  Nodes
        that have been removed from the graph are absent from this
        mapping.
    domain : Iterable[Node]
        The collection of nodes expressed with the original numbering.

    Returns
    -------
    set[Node]
        A new set containing the translated nodes.  Any node in
        ``domain`` that does not appear in ``node_map`` (i.e., has
        been removed) is omitted from the result, i.e. we treat such a
        node as having been measured with the outcome 0.
    """
    return {v for node in domain if (v := node_map.get(node)) is not None}


def remove_pauli_measurements(cut: PauliPushingCut, *, stacklevel: int = 1) -> StandardizedPattern:
    """Remove non-input Pauli measurements from the given pattern.

    This function implements the algorithm described in [BMBdF+21],
    Theorem 4.12 (Section 4.3: Removing Clifford vertices).

    This function removes all non-input Y and Z measured nodes and all
    non-input X measured nodes connected to any other internal vertex.
    Furthermore, if any non-input X measured node is connected to an
    output node that is not also an input, pivoting these nodes
    enables eliminating further nodes.  In particular, if the pattern
    has flow, all non-input Pauli measurements are removed.

    Note that if the pattern is nondeterministic, only the 0-branch is
    preserved.

    Parameters
    ----------
    cut: PauliPushingCut
        The Pauli-pushed pattern to optimize.
    stacklevel : int, optional
        Stack level to use for warnings. Defaults to 1, meaning that warnings
        are reported at this function's call site.

    Returns
    -------
    StandardizedPattern
        The pattern in which Pauli measurements have been removed.
    """
    process = _RemovePauliMeasurements(cut)
    while True:
        process.remove_all_y_or_z()  # Steps 1 and 2
        if (
            not process.try_remove_x_with_internal_neighbor()  # Step 3
            and not process.try_pivot_x_with_output_node()  # Step 4
        ):
            break
    process.remove_isolated_internal_nodes(stacklevel=stacklevel + 1)
    return process.to_standardizedpattern()
