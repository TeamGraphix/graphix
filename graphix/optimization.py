"""Optimization procedures for patterns."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import networkx as nx

# assert_never added in Python 3.11
from typing_extensions import assert_never

import graphix.pattern
from graphix import command
from graphix.clifford import Clifford
from graphix.command import CommandKind, Node
from graphix.fundamentals import Axis
from graphix.measurements import Domains, Outcome, PauliMeasurement

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet
    from typing import Self

    from graphix import Pattern


def standardize(pattern: Pattern) -> Pattern:
    """Return a standardized form to the given pattern.

    A standardized form is an equivalent pattern where the commands
    appear in the following order: `N`, `E`, `M`, `Z`, `X`, `C`.

    Note that a standardized form does not always exist in presence of
    `C` commands. For instance, there is no standardized form for the
    following pattern (written in the right-to-left convention):
    `E(0, 1) C(0, H) N(1) N(0)`.

    The function raises `NotImplementedError` if there is no
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
    directly from a pattern with the class method `from_pattern`.

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
            if cmd.kind == CommandKind.N:
                n_list.append(cmd)
            elif cmd.kind == CommandKind.E:
                for side in (0, 1):
                    i, j = cmd.nodes[side], cmd.nodes[1 - side]
                    if clifford_gate := c_dict.get(i):
                        _commute_clifford(clifford_gate, c_dict, i, j)
                    if s_domain_opt := x_dict.get(i):
                        _add_correction_domain(z_dict, j, s_domain_opt)
                edge = frozenset(cmd.nodes)
                e_set.symmetric_difference_update((edge,))
            elif cmd.kind == CommandKind.M:
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
            # Use of `==` here for mypy
            elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
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
            elif cmd.kind == CommandKind.C:
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

    def perform_pauli_pushing(self, leave_nodes: set[Node] | None = None) -> Self:
        """Move all Pauli measurements before the other measurements (except nodes in `leave_nodes`)."""
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
            pm = PauliMeasurement.try_from(
                cmd.plane, cmd.angle
            )  # None returned if the measurement is not in Pauli basis
            if pm is None or cmd.node in leave_nodes:
                non_pauli_list.append(
                    command.M(node=cmd.node, angle=cmd.angle, plane=cmd.plane, s_domain=s_domain, t_domain=t_domain)
                )
            else:
                if pm.axis == Axis.X:
                    # M^X X^s Z^t = M^{XY,0} X^s Z^t
                    #             = M^{XY,(-1)^s·0+tπ}
                    #             = S^t M^X
                    # M^{-X} X^s Z^t = M^{XY,π} X^s Z^t
                    #                = M^{XY,(-1)^s·π+tπ}
                    #                = S^t M^{-X}
                    shift_domains[cmd.node] = t_domain
                elif pm.axis == Axis.Y:
                    # M^Y X^s Z^t = M^{XY,π/2} X^s Z^t
                    #             = M^{XY,(-1)^s·π/2+tπ}
                    #             = M^{XY,π/2+(s+t)π}      (since -π/2 = π/2 - π ≡ π/2 + π (mod 2π))
                    #             = S^{s+t} M^Y
                    # M^{-Y} X^s Z^t = M^{XY,-π/2} X^s Z^t
                    #                = M^{XY,(-1)^s·(-π/2)+tπ}
                    #                = M^{XY,-π/2+(s+t)π}  (since π/2 = -π/2 + π)
                    #                = S^{s+t} M^{-Y}
                    shift_domains[cmd.node] = s_domain ^ t_domain
                elif pm.axis == Axis.Z:
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
                else:
                    assert_never(pm.axis)
                pauli_list.append(command.M(node=cmd.node, angle=cmd.angle, plane=cmd.plane))
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


def incorporate_pauli_results(pattern: Pattern) -> Pattern:
    """Return an equivalent pattern where results from Pauli presimulation are integrated in corrections."""
    result = graphix.pattern.Pattern(input_nodes=pattern.input_nodes)
    for cmd in pattern:
        if cmd.kind == CommandKind.M:
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
                new_cmd = command.M(cmd.node, cmd.plane, cmd.angle, new_s_domain, new_t_domain)
                if apply_x:
                    new_cmd = new_cmd.clifford(Clifford.X)
                if apply_z:
                    new_cmd = new_cmd.clifford(Clifford.Z)
                result.add(new_cmd)
            else:
                result.add(cmd)
        # Use == for mypy
        elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
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
        else:
            result.add(cmd)
    result.reorder_output_nodes(pattern.output_nodes)
    return result
