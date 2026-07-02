"""Optimization procedures for patterns."""

from __future__ import annotations

import dataclasses
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING
from warnings import warn

import networkx as nx

# assert_never added in Python 3.11
from graphix import command
from graphix.clifford import Clifford, Domains
from graphix.command import CommandKind, Node
from graphix.flow._partial_order import compute_topological_generations
from graphix.flow.core import XZCorrections
from graphix.fundamentals import Axis, Sign
from graphix.measurements import BlochMeasurement, Measurement, PauliMeasurement
from graphix.opengraph import OpenGraph
from graphix.space_minimization import (
    minimize_space,
    standardized_pattern_max_space,
    standardized_to_space_optimal_pattern,
)
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from collections.abc import Set as AbstractSet
    from typing import Self

    from graphix.pattern import Pattern
    from graphix.space_minimization import SpaceMinimizationHeuristic


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
    n_list: tuple[command.N, ...]
    e_set: frozenset[frozenset[Node]]
    m_list: tuple[command.M, ...]
    z_dict: Mapping[Node, frozenset[Node]]
    x_dict: Mapping[Node, frozenset[Node]]
    c_dict: Mapping[Node, Clifford]


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
    n_list: tuple[command.N]
        The N commands.
    e_set: frozenset[frozenset[Node]]
        Set of edges. Each edge is a set with two elements.
    m_list: tuple[command.M]
        The M commands.
    z_dict: Mapping[Node, frozenset[Node]]
        Mapping associating Z-domains to some nodes.
    x_dict: Mapping[Node, frozenset[Node]]
        Mapping associating X-domains to some nodes.
    c_dict: Mapping[Node, Clifford]
        Mapping associating Clifford corrections to some output nodes.

    """

    def __init__(
        self,
        input_nodes: Iterable[Node],
        output_nodes: Iterable[Node],
        n_list: Iterable[command.N],
        e_set: Iterable[Iterable[Node]],
        m_list: Iterable[command.M],
        z_dict: Mapping[Node, Iterable[Node]],
        x_dict: Mapping[Node, Iterable[Node]],
        c_dict: Mapping[Node, Clifford],
    ) -> None:
        """Return a new StandardizedPattern with immutable data structures."""
        super().__init__(
            tuple(input_nodes),
            tuple(output_nodes),
            tuple(n_list),
            frozenset(frozenset(edge) for edge in e_set),
            tuple(m_list),
            MappingProxyType({node: frozenset(nodes) for node, nodes in z_dict.items()}),
            MappingProxyType({node: frozenset(nodes) for node, nodes in x_dict.items()}),
            MappingProxyType(dict(c_dict)),
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
        z_dict: dict[Node, set[Node]] = {}
        x_dict: dict[Node, set[Node]] = {}
        c_dict: dict[Node, Clifford] = {}

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
                        new_cmd.t_domain = new_cmd.t_domain ^ t_domain_opt
                    if s_domain_opt := x_dict.pop(cmd.node, None):
                        if new_cmd is None:
                            new_cmd = copy(cmd)
                        # The original domain should not be mutated
                        new_cmd.s_domain = new_cmd.s_domain ^ s_domain_opt
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
        return cls(pattern.input_nodes, pattern.output_nodes, n_list, e_set, m_list, z_dict, x_dict, c_dict)

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

    def perform_pauli_pushing(
        self, leave_nodes: AbstractSet[Node] | None = None, *, stacklevel: int = 1
    ) -> StandardizedPattern:
        """Move Pauli measurements before the other measurements.

        If you need to recover the cut between Pauli measurements and
        non-Pauli measurements or the shifted signal, you can use
        :meth:`~graphix.remove_pauli_measurements.PauliPushingCut.from_standardized_pattern` instead.

        Parameters
        ----------
        leave_nodes : AbstractSet[Node], optional
            Nodes that should not be moved. This constraint only
            applies to Pauli nodes and has no effect on non-Pauli nodes.
        stacklevel : int, optional
            Stack level to use for warnings. Defaults to 1, meaning that warnings
            are reported at this function's call site.

        Returns
        -------
        StandardizedPattern
            The pattern in which Pauli measurements have been moved
            before the other measurements.
        """
        from graphix.remove_pauli_measurements import PauliPushingCut  # noqa: PLC0415

        return PauliPushingCut.from_standardized_pattern(
            self, leave_nodes, stacklevel=stacklevel + 1
        ).to_standardized_pattern()

    def max_space(self) -> int:
        """Compute the maximum number of nodes that must be present in the graph (graph space) during the execution of the space-optimal pattern for the given measurement order.

        This is equivalent to ``to_space_optimal_pattern().max_space()``.

        Returns
        -------
        n_nodes : int
            Maximum number of nodes present in the graph during space-optimal
            pattern execution.
        """
        return standardized_pattern_max_space(self)

    def minimize_space(self, heuristics: Iterable[SpaceMinimizationHeuristic] | None = None) -> StandardizedPattern:
        """Return a pattern with an optimized measurement order that reduces the maximal space, i.e. the number of qubits simultaneously required to execute the pattern.

        Note that standardized patterns always have a maximal space
        equal to the total number of nodes in the open graph, because
        standardization requires the entire graph to be prepared
        before measurement.

        Space reduction is specifically realized when the optimized
        order is applied via :meth:`to_space_optimal_pattern()`.

        See :func:`graphix.space_minimization.minimize_space` for default heuristics.

        Parameters
        ----------
        heuristics : Iterable[~graphix.space_minimization.SpaceMinimizationHeuristic] | None, default None
            The heuristics to apply sequentially. Defaults to
            :const:`~graphix.space_minimization.DEFAULT_HEURISTICS`.

        Returns
        -------
        StandardizedPattern
            The optimized pattern.
        """
        return minimize_space(self, heuristics)

    def to_pattern(self) -> Pattern:
        """Return the standardized pattern."""
        from graphix.pattern import Pattern  # noqa: PLC0415

        pattern = Pattern(input_nodes=self.input_nodes)
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
        """Return a pattern that is space-optimal for the given measurement order.

        This method treats the measurement order as fixed, performing
        node preparations (``N``) and entanglements (``E``) as late as
        possible to minimize space usage. While the resulting pattern
        is guaranteed to be optimal for this specific order, the
        method does not explore alternative orders.

        To find an alternative measurement order that further reduces
        space, use :meth:`minimize_space`.
        """
        return standardized_to_space_optimal_pattern(self)

    def extract_opengraph(self) -> OpenGraph[Measurement]:
        r"""Extract the underlying resource-state open graph from the pattern.

        Returns
        -------
        OpenGraph[Measurement]

        Raises
        ------
        ValueError
            If ``N`` commands in the pattern do not represent a :math:`\ket{+}` state.
        """
        for n in self.n_list:
            if n.state != BasicStates.PLUS:
                raise ValueError(
                    f"Open graph construction in flow extraction requires N commands to represent a |+⟩ state. Error found in {n}."
                )
        measurements = {m.node: m.measurement for m in self.m_list}
        return OpenGraph(self.extract_graph(), self.input_nodes, self.output_nodes, measurements, self.c_dict)

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
        oset = frozenset(self.output_nodes)  # First layer by convention if not empty.
        excluded_nodes = oset

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

    def extract_xzcorrections(self) -> XZCorrections[Measurement]:
        r"""Extract the XZ-corrections from the current measurement pattern.

        Returns
        -------
        XZCorrections[Measurement]
            The XZ-corrections associated with the current pattern.

        Raises
        ------
        XZCorrectionsError
            If the extracted correction dictionaries are not well formed.
        ValueError
            If ``N`` commands in the pattern do not represent a :math:`\ket{+}` state or if the pattern corrections form closed loops.
        """
        x_corr: dict[int, set[int]] = {}
        z_corr: dict[int, set[int]] = {}

        for m in self.m_list:
            _update_corrections(m.node, m.s_domain, x_corr)
            _update_corrections(m.node, m.t_domain, z_corr)

        for node, domain in self.x_dict.items():
            _update_corrections(node, domain, x_corr)

        for node, domain in self.z_dict.items():
            _update_corrections(node, domain, z_corr)

        og = (
            self.extract_opengraph()
        )  # Raises a `ValueError` if `N` commands in the pattern do not represent a |+⟩ state.

        return XZCorrections.from_measured_nodes_mapping(
            og, x_corr, z_corr
        )  # Raises a `XZCorrectionsError` if the input dictionaries are not well formed.

    def map(self, f: Callable[[Measurement], Measurement]) -> StandardizedPattern:
        """Return a pattern where the function ``f`` has been applied to each measurement.

        Parameters
        ----------
        f: Callable[[Measurement], Measurement]
            Function applied to each measurement.

        Returns
        -------
        StandardizedPattern
            The resulting pattern.
        """
        m_list = tuple(cmd_m.map(f) for cmd_m in self.m_list)
        return StandardizedPattern(
            self.input_nodes,
            self.output_nodes,
            self.n_list,
            self.e_set,
            m_list,
            self.z_dict,
            self.x_dict,
            self.c_dict,
        )

    def to_bloch(self) -> StandardizedPattern:
        """Return an equivalent pattern in which all measurements are represented as Bloch measurements."""
        return self.map(lambda m: m.to_bloch())

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


def remove_useless_domains(pattern: Pattern) -> Pattern:
    """Return an equivalent pattern where measurement domains that are not used given the specific measurement angles and planes are removed."""
    from graphix.pattern import Pattern  # noqa: PLC0415

    new_pattern = Pattern(input_nodes=pattern.input_nodes)
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
    from graphix.pattern import Pattern  # noqa: PLC0415

    new_pattern = Pattern(input_nodes=pattern.input_nodes)

    def decompose_domain(
        cmd: Callable[[int, set[int]], command.CommandType], node: int, domain: AbstractSet[int]
    ) -> bool:
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


def remove_local_clifford_commands(pattern: Pattern) -> Pattern:
    """Return an equivalent pattern where local Clifford commands have been replaced by MBQC commands.

    This function transpiles MBQC+LC patterns into MBQC patterns.
    """
    from graphix.pattern import Pattern  # noqa: PLC0415

    nodes = pattern.extract_nodes()
    if not nodes:
        return pattern
    max_node = max(nodes)
    new_pattern = Pattern(input_nodes=pattern.input_nodes)
    mapping: dict[Node, Node] = {}

    def reindex(node: Node) -> Node:
        return mapping.get(node, node)

    for cmd in pattern:
        match cmd.kind:
            case CommandKind.C:
                cmd_node = reindex(cmd.node)
                clifford_pattern = cmd.clifford.to_opengraph().to_pattern()
                (output_node,) = clifford_pattern.output_nodes
                # We avoid using `new_pattern.compose` here because
                # pattern composition is linear in the size of each
                # pattern, which would make transpilation run in
                # quadratic time.
                # clifford_pattern satisfies the following properties:
                # - The set of input nodes is {0}.
                # - The output node is the highest-indexed node.
                new_pattern.extend(clifford_pattern.reindex(lambda node: cmd_node if node == 0 else node + max_node))  # noqa: B023
                max_node += output_node
                mapping[cmd.node] = max_node
            case _:
                new_cmd = cmd.reindex(reindex)
                new_pattern.add(new_cmd)
    new_pattern.reorder_output_nodes(map(reindex, pattern.output_nodes))
    return new_pattern
