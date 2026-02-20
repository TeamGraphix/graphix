"""Optimization procedures for patterns."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING
from warnings import warn

import networkx as nx

# assert_never added in Python 3.11
from typing_extensions import assert_never

import graphix.pattern
from graphix import command
from graphix.clifford import Clifford, Domains
from graphix.command import CommandKind, Node
from graphix.flow._partial_order import compute_topological_generations
from graphix.flow.core import CausalFlow, GFlow, XZCorrections
from graphix.flow.exceptions import (
    FlowGenericError,
    FlowGenericErrorReason,
)
from graphix.fundamentals import Axis, Plane, Sign
from graphix.measurements import BlochMeasurement, Measurement, Outcome, PauliMeasurement
from graphix.opengraph import OpenGraph
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from collections.abc import Set as AbstractSet
    from typing import Self

    from graphix import Pattern


def standardize(pattern: Pattern) -> Pattern:
    """Return a standardized form to the given pattern.

    A standardized form is an equivalent pattern where the commands
    appear in the following order: ``N``, ``E``, ``M``, ``Z``, ``X``,
    ``C``.

    Note that a standardized form does not always exist in presence of
    ``C`` commands. For instance, there is no standardized form for the
    following pattern (written in the right-to-left convention):
    ``E(0, 1) C(0, H) N(1) N(0)``.

    The function raises ``NotImplementedError`` if there is no
    standardized form. This behavior can change in the future.


    Parameters
    ----------
    pattern : Pattern
        The original pattern.

    Returns
    -------
    standardized : Pattern
        The standardized pattern, if it exists.

    """
    return StandardizedPattern.from_pattern(pattern).to_pattern()


@dataclass(frozen=True, slots=True)
class _StandardizedPattern:
    """Immutable internal storage for standardized patterns.

    It is defined as a superclass to allow
    :class:`StandardizedPattern` to define a custom ``__init__``
    method accepting any compatible collections for initialization,
    while keeping the dataclass frozen and avoiding direct
    ``setattr``-based field initialization.
    """

    input_nodes: tuple[Node, ...]
    output_nodes: tuple[Node, ...]
    results: Mapping[Node, Outcome]
    n_list: tuple[command.N, ...]
    e_set: frozenset[frozenset[Node]]
    m_list: tuple[command.M, ...]
    c_dict: Mapping[Node, Clifford]
    z_dict: Mapping[Node, frozenset[Node]]
    x_dict: Mapping[Node, frozenset[Node]]


class StandardizedPattern(_StandardizedPattern):
    """Pattern in standardized form.

    Use the method :meth:`to_pattern()` to get the standardized pattern.

    This class uses immutable data structures for its fields
    (``tuple``, ``frozenset`` and ``Mapping``).

    Instances can be generated with the constructor from any
    compatible data structures, and an instance can be generated
    directly from a pattern with the class method :meth:`from_pattern`.

    The constructor instantiates the ``Mapping`` fields as
    ``MappingProxyType`` objects over fresh dictionaries, ensuring
    their immutability. We expose the type ``Mapping`` instead of
    ``MappingProxyType`` for the readability, as they provide the same
    interface.

    Attributes
    ----------
    input_nodes: tuple[Node, ...]
        Input nodes.
    output_nodes: tuple[Node, ...]
        Output nodes.
    results: Mapping[Node, Outcome]
        Already measured nodes (by Pauli presimulation).
    n_list: tuple[command.N]
        The N commands.
    e_set: frozenset[frozenset[Node]]
        Set of edges. Each edge is a set with two elements.
    m_list: tuple[command.M]
        The M commands.
    c_dict: Mapping[Node, Clifford]
        Mapping associating Clifford corrections to some nodes.
    z_dict: Mapping[Node, frozenset[Node]]
        Mapping associating Z-domains to some nodes.
    x_dict: Mapping[Node, frozenset[Node]]
        Mapping associating X-domains to some nodes.

    """

    def __init__(
        self,
        input_nodes: Iterable[Node],
        output_nodes: Iterable[Node],
        results: Mapping[Node, Outcome],
        n_list: Iterable[command.N],
        e_set: Iterable[Iterable[Node]],
        m_list: Iterable[command.M],
        c_dict: Mapping[Node, Clifford],
        z_dict: Mapping[Node, Iterable[Node]],
        x_dict: Mapping[Node, Iterable[Node]],
    ) -> None:
        """Return a new StandardizedPattern with immutable data structures."""
        super().__init__(
            tuple(input_nodes),
            tuple(output_nodes),
            MappingProxyType(dict(results)),
            tuple(n_list),
            frozenset(frozenset(edge) for edge in e_set),
            tuple(m_list),
            MappingProxyType(dict(c_dict)),
            MappingProxyType({node: frozenset(nodes) for node, nodes in z_dict.items()}),
            MappingProxyType({node: frozenset(nodes) for node, nodes in x_dict.items()}),
        )

    @classmethod
    def from_pattern(cls, pattern: Pattern) -> Self:
        """Compute the standardized form of the given pattern."""
        s_domain: set[Node]
        t_domain: set[Node]
        s_domain_opt: set[Node] | None
        t_domain_opt: set[Node] | None

        n_list: list[command.N] = []
        e_set: set[frozenset[Node]] = set()
        m_list: list[command.M] = []
        c_dict: dict[Node, Clifford] = {}
        z_dict: dict[Node, set[Node]] = {}
        x_dict: dict[Node, set[Node]] = {}

        # Standardization could turn non-runnable patterns into
        # runnable ones, so we check runnability first to avoid hiding
        # code-logic errors.
        # For example, the non-runnable pattern E(0,1) M(0) N(1) N(0) would
        # become M(0) E(0,1) N(1) N(0), which is runnable.
        pattern.check_runnability()

        for cmd in pattern:
            match cmd.kind:
                case CommandKind.N:
                    n_list.append(cmd)
                case CommandKind.E:
                    for side in (0, 1):
                        i, j = cmd.nodes[side], cmd.nodes[1 - side]
                        if clifford_gate := c_dict.get(i):
                            _commute_clifford(clifford_gate, c_dict, i, j)
                        if s_domain_opt := x_dict.get(i):
                            _add_correction_domain(z_dict, j, s_domain_opt)
                    edge = frozenset(cmd.nodes)
                    e_set.symmetric_difference_update((edge,))
                case CommandKind.M:
                    new_cmd = None
                    if clifford_gate := c_dict.pop(cmd.node, None):
                        new_cmd = cmd.clifford(clifford_gate)
                    if t_domain_opt := z_dict.pop(cmd.node, None):
                        if new_cmd is None:
                            new_cmd = copy(cmd)
                        # The original domain should not be mutated
                        new_cmd.t_domain = new_cmd.t_domain ^ t_domain_opt  # noqa: PLR6104
                    if s_domain_opt := x_dict.pop(cmd.node, None):
                        if new_cmd is None:
                            new_cmd = copy(cmd)
                        # The original domain should not be mutated
                        new_cmd.s_domain = new_cmd.s_domain ^ s_domain_opt  # noqa: PLR6104
                    if new_cmd is None:
                        m_list.append(cmd)
                    else:
                        m_list.append(new_cmd)
                case CommandKind.X | CommandKind.Z:
                    if cmd.kind == CommandKind.X:
                        s_domain = cmd.domain
                        t_domain = set()
                    else:
                        s_domain = set()
                        t_domain = cmd.domain
                    domains = c_dict.get(cmd.node, Clifford.I).commute_domains(Domains(s_domain, t_domain))
                    if domains.t_domain:
                        _add_correction_domain(z_dict, cmd.node, domains.t_domain)
                    if domains.s_domain:
                        _add_correction_domain(x_dict, cmd.node, domains.s_domain)
                case CommandKind.C:
                    # Each pattern command is applied by left multiplication: if a clifford `C`
                    # has been already applied to a node, applying a clifford `C'` to the same
                    # node is equivalent to apply `C'C` to a fresh node.
                    c_dict[cmd.node] = cmd.clifford @ c_dict.get(cmd.node, Clifford.I)
        return cls(
            pattern.input_nodes, pattern.output_nodes, pattern.results, n_list, e_set, m_list, c_dict, z_dict, x_dict
        )

    def extract_graph(self) -> nx.Graph[int]:
        """Return the graph state from the command sequence, extracted from 'N' and 'E' commands.

        Returns
        -------
        graph_state: nx.Graph
        """
        graph: nx.Graph[int] = nx.Graph()
        graph.add_nodes_from(self.input_nodes)
        for cmd_n in self.n_list:
            graph.add_node(cmd_n.node)
        for u, v in self.e_set:
            graph.add_edge(u, v)
        return graph

    def perform_pauli_pushing(self, leave_nodes: set[Node] | None = None, *, stacklevel: int = 1) -> Self:
        """Move Pauli measurements before the other measurements.

        Parameters
        ----------
        leave_nodes : set[Node], optional
            Nodes that should not be moved. This constraint only
            applies to Pauli nodes and has no effect on non-Pauli nodes.
        stacklevel : int, optional
            Stack level to use for warnings. Defaults to 1, meaning that warnings
            are reported at this function's call site.

        Returns
        -------
        Pattern
            The pattern in which Pauli measurements have been moved
            before the other measurements.
        """
        self._warn_non_inferred_pauli_measurements(stacklevel=stacklevel + 1)

        if leave_nodes:
            leave_non_pauli_nodes = [
                cmd.node
                for cmd in self.m_list
                if not isinstance(cmd.measurement, PauliMeasurement) and cmd.node in leave_nodes
            ]
            if leave_non_pauli_nodes:
                warn(
                    f"`leave_nodes` contains nodes that are not Pauli: {leave_non_pauli_nodes}. The constraint has no effect on these nodes.",
                    stacklevel=stacklevel + 1,
                )

        if leave_nodes is None:
            leave_nodes = set()
        shift_domains: dict[int, set[int]] = {}

        def expand_domain(domain: AbstractSet[int]) -> set[int]:
            """Merge previously shifted domains into ``domain``.

            Parameters
            ----------
            domain : set[int]
                Domain to update with any accumulated shift information.
            """
            new_domain = set(domain)
            for node in domain & shift_domains.keys():
                new_domain ^= shift_domains[node]
            return new_domain

        pauli_list = []
        non_pauli_list = []
        for cmd in self.m_list:
            s_domain = expand_domain(cmd.s_domain)
            t_domain = expand_domain(cmd.t_domain)
            if not isinstance(cmd.measurement, PauliMeasurement) or (leave_nodes and cmd.node in leave_nodes):
                non_pauli_list.append(
                    command.M(node=cmd.node, measurement=cmd.measurement, s_domain=s_domain, t_domain=t_domain)
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
                        shift_domains[cmd.node] = t_domain
                    case Axis.Y:
                        # M^Y X^s Z^t = M^{XY,π/2} X^s Z^t
                        #             = M^{XY,(-1)^s·π/2+tπ}
                        #             = M^{XY,π/2+(s+t)π}      (since -π/2 = π/2 - π ≡ π/2 + π (mod 2π))
                        #             = S^{s+t} M^Y
                        # M^{-Y} X^s Z^t = M^{XY,-π/2} X^s Z^t
                        #                = M^{XY,(-1)^s·(-π/2)+tπ}
                        #                = M^{XY,-π/2+(s+t)π}  (since π/2 = -π/2 + π)
                        #                = S^{s+t} M^{-Y}
                        shift_domains[cmd.node] = s_domain ^ t_domain
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
                        shift_domains[cmd.node] = s_domain
                    case _:
                        assert_never(cmd.measurement.axis)
                pauli_list.append(command.M(node=cmd.node, measurement=cmd.measurement))
        return self.__class__(
            self.input_nodes,
            self.output_nodes,
            self.results,
            self.n_list,
            self.e_set,
            pauli_list + non_pauli_list,
            self.c_dict,
            {node: expand_domain(domain) for node, domain in self.z_dict.items()},
            {node: expand_domain(domain) for node, domain in self.x_dict.items()},
        )

    def to_pattern(self) -> Pattern:
        """Return the standardized pattern."""
        pattern = graphix.pattern.Pattern(input_nodes=self.input_nodes)
        pattern.results = dict(self.results)
        pattern.extend(
            self.n_list,
            (command.E((u, v)) for u, v in self.e_set),
            self.m_list,
            (command.Z(node=node, domain=set(domain)) for node, domain in self.z_dict.items()),
            (command.X(node=node, domain=set(domain)) for node, domain in self.x_dict.items()),
            (command.C(node=node, clifford=clifford_gate) for node, clifford_gate in self.c_dict.items()),
        )
        pattern.reorder_output_nodes(self.output_nodes)
        return pattern

    def to_space_optimal_pattern(self) -> Pattern:
        """Return a pattern that is space-optimal for the given measurement order."""
        pattern = graphix.pattern.Pattern(input_nodes=self.input_nodes)
        pattern.results = dict(self.results)
        active = set(self.input_nodes)
        done: set[Node] = set()
        n_dict = {n.node: n for n in self.n_list}
        graph = self.extract_graph()

        def ensure_active(node: Node) -> None:
            """Initialize node in pattern if it has not been initialized before."""
            if node not in active:
                pattern.add(n_dict[node])
                active.add(node)

        def ensure_neighborhood(node: Node) -> None:
            """Initialize and entangle the inactive nodes in the neighbourhood of ``node``."""
            ensure_active(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in done:
                    ensure_active(neighbor)
                    pattern.add(command.E((node, neighbor)))

        for m in self.m_list:
            ensure_neighborhood(m.node)
            pattern.add(m)
            done.add(m.node)
        for node in self.output_nodes:
            ensure_neighborhood(node)
            if domain := self.z_dict.get(node):
                pattern.add(command.Z(node, set(domain)))
            if domain := self.x_dict.get(node):
                pattern.add(command.X(node, set(domain)))
            if clifford_gate := self.c_dict.get(node):
                pattern.add(command.C(node, clifford_gate))
            done.add(node)
        return pattern

    def extract_opengraph(self) -> OpenGraph[Measurement]:
        """Extract the underlying resource-state open graph from the pattern.

        Returns
        -------
        OpenGraph[Measurement]

        Raises
        ------
        ValueError
            If `N` commands in the pattern do not represent a |+⟩ state.

        Notes
        -----
        This operation loses all the information on the Clifford commands.
        """
        for n in self.n_list:
            if n.state != BasicStates.PLUS:
                raise ValueError(
                    f"Open graph construction in flow extraction requires N commands to represent a |+⟩ state. Error found in {n}."
                )
        measurements = {m.node: m.measurement for m in self.m_list}
        return OpenGraph(self.extract_graph(), self.input_nodes, self.output_nodes, measurements)

    def extract_partial_order_layers(self) -> tuple[frozenset[int], ...]:
        """Extract the measurement order of the pattern in the form of layers.

        This method builds a directed acyclical graph (DAG) from the pattern and then performs a topological sort.

        Returns
        -------
        tuple[frozenset[int], ...]
            Measurement partial order between the pattern's nodes in a layer form.

        Raises
        ------
        ValueError
            If the correction domains in the pattern form closed loops. This implies that the pattern is not runnable.

        Notes
        -----
        The returned object follows the same conventions as the ``partial_order_layers`` attribute of :class:`PauliFlow` and :class:`XZCorrections` objects:
            - Nodes in the same layer can be measured simultaneously.
            - Nodes in layer ``i`` must be measured after nodes in layer ``i + 1``.
            - All output nodes (if any) are in the first layer.
            - There cannot be any empty layers.
        """
        oset = frozenset(self.output_nodes)  # First layer by convention.
        pre_measured_nodes = self.results.keys()  # Not included in the partial order layers.
        excluded_nodes = oset | pre_measured_nodes

        zero_indegree = set(self.input_nodes).union(n.node for n in self.n_list) - excluded_nodes
        dag: dict[int, set[int]] = {
            node: set() for node in zero_indegree
        }  # `i: {j}` represents `i -> j` which means that node `i` must be measured before node `j`.
        indegree_map: dict[int, int] = {}

        def process_domain(node: Node, domain: AbstractSet[Node]) -> None:
            for dep_node in domain:
                if (
                    not {node, dep_node} & excluded_nodes and node not in dag[dep_node]
                ):  # Don't include multiple edges in the dag.
                    dag[dep_node].add(node)
                    indegree_map[node] = indegree_map.get(node, 0) + 1

        domain: AbstractSet[Node]

        for m in self.m_list:
            node, domain = m.node, m.s_domain | m.t_domain
            process_domain(node, domain)

        for corrections in [self.z_dict, self.x_dict]:
            for node, domain in corrections.items():
                process_domain(node, domain)

        zero_indegree -= indegree_map.keys()

        generations = compute_topological_generations(dag, indegree_map, zero_indegree)
        if generations is None:
            raise ValueError("Pattern domains form closed loops.")

        if oset:
            return oset, *generations[::-1]
        return generations[::-1]

    def extract_causal_flow(self) -> CausalFlow[BlochMeasurement]:
        """Extract the causal flow structure from the current measurement pattern.

        This method does not call the flow-extraction routine on the underlying open graph, but constructs the flow from the pattern corrections instead.

        Returns
        -------
        flow.CausalFlow[Measurement]
            The causal flow associated with the current pattern.

        Raises
        ------
        FlowError
            If the pattern:
            - Contains measurements in forbidden planes (XZ or YZ),
            - Is empty, or
            - Induces a correction function and a partial order which fail the well-formedness checks for a valid causal flow.
        ValueError
            If `N` commands in the pattern do not represent a |+⟩ state or if the pattern corrections form closed loops.

        Notes
        -----
        This method makes use of :func:`StandardizedPattern.extract_partial_order_layers` which computes the pattern's direct acyclical graph (DAG) induced by the corrections and returns a particular layer stratification (obtained by doing a topological sort on the DAG). Further, it constructs the pattern's induced correction function from :math:`M` and :math:`X` commands.
        In general, there may exist various layerings which represent the corrections of the pattern. To ensure that a given layering is compatible with the pattern's induced correction function, the partial order must be extracted from a standardized pattern. Commutation of entanglement commands with X and Z corrections in the standardization procedure may generate new corrections, which guarantees that all the topological information of the underlying graph is encoded in the extracted partial order.
        """
        correction_function: dict[int, set[int]] = defaultdict(set)
        pre_measured_nodes = self.results.keys()  # Not included in the flow.

        for m in self.m_list:
            try:
                bloch = m.measurement.downcast_bloch()
            except TypeError:
                valid = False
            else:
                valid = bloch.plane == Plane.XY
            if not valid:
                raise FlowGenericError(FlowGenericErrorReason.XYPlane)
            _update_corrections(m.node, m.s_domain - pre_measured_nodes, correction_function)

        for node, domain in self.x_dict.items():
            _update_corrections(node, domain - pre_measured_nodes, correction_function)

        og = (
            self.extract_opengraph()
        )  # Raises a `ValueError` if `N` commands in the pattern do not represent a |+⟩ state.
        partial_order_layers = (
            self.extract_partial_order_layers()
        )  # Raises a `ValueError` if the pattern corrections form closed loops.
        cf = CausalFlow(og.downcast_bloch(), dict(correction_function), partial_order_layers)
        cf.check_well_formed()  # Raises a `FlowError` if the partial order and the correction function are not compatible, or if a measured node is corrected by more than one node.
        return cf

    def extract_gflow(self) -> GFlow[BlochMeasurement]:
        """Extract the generalized flow (gflow) structure from the current measurement pattern.

        This method does not call the flow-extraction routine on the underlying open graph, but constructs the gflow from the pattern corrections instead.

        Returns
        -------
        flow.GFlow[Measurement]
            The gflow associated with the current pattern.

        Raises
        ------
        FlowError
            If the pattern is empty or if the extracted structure does not satisfy
            the well-formedness conditions required for a valid gflow.
        TypeError
            If the pattern contains a Pauli measurement
        ValueError
            If `N` commands in the pattern do not represent a |+⟩ state or if the pattern corrections form closed loops.

        Notes
        -----
        The notes provided in :func:`self.extract_causal_flow` apply here as well.
        """
        correction_function: dict[int, set[int]] = {}
        pre_measured_nodes = self.results.keys()  # Not included in the flow.

        for m in self.m_list:
            # Raises a `TypeError` if the measurement is not represented as a Bloch measurement
            if m.measurement.downcast_bloch().plane in {Plane.XZ, Plane.YZ}:
                correction_function.setdefault(m.node, set()).add(m.node)
            _update_corrections(m.node, m.s_domain - pre_measured_nodes, correction_function)

        for node, domain in self.x_dict.items():
            _update_corrections(node, domain - pre_measured_nodes, correction_function)

        og = (
            self.extract_opengraph()
        )  # Raises a `ValueError` if `N` commands in the pattern do not represent a |+⟩ state.
        partial_order_layers = (
            self.extract_partial_order_layers()
        )  # Raises a `ValueError` if the pattern corrections form closed loops.
        gf = GFlow(og.downcast_bloch(), correction_function, partial_order_layers)
        gf.check_well_formed()
        return gf

    def extract_xzcorrections(self) -> XZCorrections[Measurement]:
        """Extract the XZ-corrections from the current measurement pattern.

        Returns
        -------
        XZCorrections[Measurement]
            The XZ-corrections associated with the current pattern.

        Raises
        ------
        XZCorrectionsError
            If the extracted correction dictionaries are not well formed.
        ValueError
            If `N` commands in the pattern do not represent a |+⟩ state or if the pattern corrections form closed loops.
        """
        x_corr: dict[int, set[int]] = {}
        z_corr: dict[int, set[int]] = {}

        pre_measured_nodes = self.results.keys()  # Not included in the xz-corrections.

        for m in self.m_list:
            _update_corrections(m.node, m.s_domain - pre_measured_nodes, x_corr)
            _update_corrections(m.node, m.t_domain - pre_measured_nodes, z_corr)

        for node, domain in self.x_dict.items():
            _update_corrections(node, domain - pre_measured_nodes, x_corr)

        for node, domain in self.z_dict.items():
            _update_corrections(node, domain - pre_measured_nodes, z_corr)

        og = (
            self.extract_opengraph()
        )  # Raises a `ValueError` if `N` commands in the pattern do not represent a |+⟩ state.

        return XZCorrections.from_measured_nodes_mapping(
            og, x_corr, z_corr
        )  # Raises a `XZCorrectionsError` if the input dictionaries are not well formed.

    def _warn_non_inferred_pauli_measurements(self, stacklevel: int) -> None:
        for m in self.m_list:
            if isinstance(m.measurement, BlochMeasurement) and m.measurement.try_to_pauli() is not None:
                warn("Pattern with non-inferred Pauli measurements.", stacklevel=stacklevel + 1)
                return


def _add_correction_domain(domain_dict: dict[Node, set[Node]], node: Node, domain: set[Node]) -> None:
    """Merge a correction domain into ``domain_dict`` for ``node``.

    Parameters
    ----------
    domain_dict : dict[int, Command]
        Mapping from node index to accumulated domain.
    node : int
        Target node whose domain should be updated.
    domain : set[int]
        Domain to merge with the existing one.
    """
    if previous_domain := domain_dict.get(node):
        previous_domain ^= domain
    else:
        domain_dict[node] = domain.copy()


def _commute_clifford(clifford_gate: Clifford, c_dict: dict[int, Clifford], i: int, j: int) -> None:
    """Commute a Clifford with an entanglement command.

    Parameters
    ----------
    clifford_gate : Clifford
        Clifford gate before the entanglement command
    c_dict : dict[int, Clifford]
        Mapping from the node index to accumulated Clifford commands.
    i : int
        First node of the entanglement command where the Clifford is applied.
    j : int
        Second node of the entanglement command where the Clifford is applied.
    """
    if clifford_gate in {Clifford.I, Clifford.Z, Clifford.S, Clifford.SDG}:
        # Clifford gate commutes with the entanglement command.
        pass
    elif clifford_gate in {Clifford.X, Clifford.Y, Clifford(9), Clifford(10)}:
        # Clifford gate commutes with the entanglement command up to a Z Clifford on the other index.
        c_dict[j] = Clifford.Z @ c_dict.get(j, Clifford.I)
    else:
        # Clifford gate commutes with the entanglement command up to a two-qubit Clifford
        raise NotImplementedError(
            f"Pattern contains a Clifford followed by an E command on qubit {i} which only commute up to a two-qubit Clifford. Standarization is not supported."
        )


def _incorporate_pauli_results_in_domain(
    results: Mapping[int, int], domain: AbstractSet[int]
) -> tuple[bool, set[int]] | None:
    if not (results.keys() & domain):
        return None
    new_domain = set(domain - results.keys())
    odd_outcome = sum(outcome for node, outcome in results.items() if node in domain) % 2
    return odd_outcome == 1, new_domain


def _update_corrections(node: Node, domain: AbstractSet[Node], correction: dict[Node, set[Node]]) -> None:
    """Update the correction mapping by adding a node to all entries in a domain.

    Parameters
    ----------
    node : Node
        The node to add as a correction.
    domain : AbstractSet[Node]
        A set of measured nodes whose corresponding correction sets should be updated.
    correction : dict[Node, set[Node]]
        A mapping from measured nodes to sets of nodes on which corrections are applied. This
        dictionary is modified in place.

    Notes
    -----
    This function is used to extract the correction function from :math:`X`, :math:`Z` and :math:`M` commands when constructing a flow.
    """
    for measured_node in domain:
        correction.setdefault(measured_node, set()).add(node)


def incorporate_pauli_results(pattern: Pattern) -> Pattern:
    """Return an equivalent pattern where results from Pauli presimulation are integrated in corrections."""
    result = graphix.pattern.Pattern(input_nodes=pattern.input_nodes)
    for cmd in pattern:
        match cmd.kind:
            case CommandKind.M:
                s = _incorporate_pauli_results_in_domain(pattern.results, cmd.s_domain)
                t = _incorporate_pauli_results_in_domain(pattern.results, cmd.t_domain)
                if s or t:
                    if s:
                        apply_x, new_s_domain = s
                    else:
                        apply_x = False
                        new_s_domain = cmd.s_domain
                    if t:
                        apply_z, new_t_domain = t
                    else:
                        apply_z = False
                        new_t_domain = cmd.t_domain
                    new_cmd = command.M(cmd.node, cmd.measurement, new_s_domain, new_t_domain)
                    if apply_x:
                        new_cmd = new_cmd.clifford(Clifford.X)
                    if apply_z:
                        new_cmd = new_cmd.clifford(Clifford.Z)
                    result.add(new_cmd)
                else:
                    result.add(cmd)
            case CommandKind.X | CommandKind.Z:
                signal = _incorporate_pauli_results_in_domain(pattern.results, cmd.domain)
                if signal:
                    apply_c, new_domain = signal
                    if new_domain:
                        cmd_cstr = command.X if cmd.kind == CommandKind.X else command.Z
                        result.add(cmd_cstr(cmd.node, new_domain))
                    if apply_c:
                        c = Clifford.X if cmd.kind == CommandKind.X else Clifford.Z
                        result.add(command.C(cmd.node, c))
                else:
                    result.add(cmd)
            case _:
                result.add(cmd)
    result.reorder_output_nodes(pattern.output_nodes)
    return result


def remove_useless_domains(pattern: Pattern) -> Pattern:
    """Return an equivalent pattern where measurement domains that are not used given the specific measurement angles and planes are removed."""
    new_pattern = graphix.pattern.Pattern(input_nodes=pattern.input_nodes)
    new_pattern.results = pattern.results
    for cmd in pattern:
        if cmd.kind == CommandKind.M:
            match cmd.measurement:
                case PauliMeasurement(Axis.X, Sign.PLUS):
                    new_cmd = dataclasses.replace(cmd, s_domain=set())
                case PauliMeasurement(Axis.Z, Sign.PLUS):
                    new_cmd = dataclasses.replace(cmd, t_domain=set())
                case _:
                    new_cmd = cmd
            new_pattern.add(new_cmd)
        else:
            new_pattern.add(cmd)
    new_pattern.reorder_output_nodes(pattern.output_nodes)
    return new_pattern


def single_qubit_domains(pattern: Pattern) -> Pattern:
    """Return an equivalent pattern where domains contains at most one qubit."""
    new_pattern = graphix.pattern.Pattern(input_nodes=pattern.input_nodes)
    new_pattern.results = pattern.results

    def decompose_domain(cmd: Callable[[int, set[int]], command.Command], node: int, domain: AbstractSet[int]) -> bool:
        if len(domain) <= 1:
            return False
        for src in domain:
            new_pattern.add(cmd(node, {src}))
        return True

    for cmd in pattern:
        match cmd.kind:
            case CommandKind.M:
                replaced_s_domain = decompose_domain(command.X, cmd.node, cmd.s_domain)
                replaced_t_domain = decompose_domain(command.Z, cmd.node, cmd.t_domain)
                if replaced_s_domain or replaced_t_domain:
                    new_s_domain = set() if replaced_s_domain else cmd.s_domain
                    new_t_domain = set() if replaced_t_domain else cmd.t_domain
                    new_cmd = dataclasses.replace(cmd, s_domain=new_s_domain, t_domain=new_t_domain)
                    new_pattern.add(new_cmd)
                    continue
            case CommandKind.X:
                if decompose_domain(command.X, cmd.node, cmd.domain):
                    continue
            case CommandKind.Z:
                if decompose_domain(command.Z, cmd.node, cmd.domain):
                    continue
        new_pattern.add(cmd)

    new_pattern.reorder_output_nodes(pattern.output_nodes)
    return new_pattern
