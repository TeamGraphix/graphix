"""Space minimization procedures for patterns."""

from __future__ import annotations

import dataclasses
from itertools import chain
from typing import TYPE_CHECKING

import graphix
from graphix import command
from graphix.command import CommandKind
from graphix.flow.exceptions import FlowError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from collections.abc import Set as AbstractSet

    from graphix.command import Node
    from graphix.optimization import StandardizedPattern
    from graphix.pattern import Pattern

    SpaceMinimizationHeuristic = Callable[[StandardizedPattern], tuple[tuple[Node, ...], bool] | None]
    """Define a space minimization heuristic: return a new measurement order or ``None`` if the heuristic cannot handle the given pattern.

    A space minimization heuristic provides a measurement order for a given pattern.

    Parameters
    ----------
    pattern: StandardizedPattern
        The pattern to optimize for space.

    Returns
    -------
    new_order: tuple[Node, ...]
        The new measurement order.

    known_optimal: bool
        ``True`` if the new measurement order is known to be optimal.
        This will interrupt the search for a better heuristic.
    """


def pattern_max_space(pattern: Pattern) -> int:
    """Compute the maximum number of nodes that must be present in the graph (graph space) during the execution of the pattern.

    For statevector simulation, this is equivalent to the maximum memory
    needed for classical simulation.

    Parameters
    ----------
    pattern : Pattern
        The pattern.

    Returns
    -------
    n_nodes : int
        Maximum number of nodes present in the graph during pattern execution.
    """
    num_active = len(pattern.input_nodes)
    max_active = num_active
    for cmd in pattern:
        match cmd.kind:
            case CommandKind.N:
                num_active += 1
                max_active = max(num_active, max_active)
            case CommandKind.M:
                num_active -= 1
    return max_active


def standardized_pattern_max_space(pattern: StandardizedPattern) -> int:
    """Compute the maximum number of nodes that must be present in the graph (graph space) during the execution of the space-optimal pattern for the given measurement order.

    This is equivalent to ``pattern.to_space_optimal_pattern().max_space()``.

    Returns
    -------
    n_nodes : int
        Maximum number of nodes present in the graph during space-optimal
        pattern execution.
    """
    initialized = set(pattern.input_nodes)
    graph = pattern.extract_graph()
    num_active = len(pattern.input_nodes)
    max_active = num_active

    def activate(node: Node) -> None:
        """Add ``node`` to the graph if it has not been added before."""
        nonlocal num_active, max_active
        if node not in initialized:
            initialized.add(node)
            num_active += 1
            max_active = max(num_active, max_active)

    def activate_neighborhood(node: Node) -> None:
        """Activate ``node`` and all its uninitialized neighbours."""
        activate(node)
        for neighbor in graph.neighbors(node):
            activate(neighbor)

    for m in pattern.m_list:
        activate_neighborhood(m.node)
        num_active -= 1
    for node in pattern.output_nodes:
        activate_neighborhood(node)
    return max_active


def standardized_to_space_optimal_pattern(pattern: StandardizedPattern) -> Pattern:
    """Return a pattern that is space-optimal for the given measurement order."""
    target = graphix.Pattern(input_nodes=pattern.input_nodes)
    target.results = dict(pattern.results)
    initialized = set(pattern.input_nodes)
    done: set[Node] = set()
    n_dict = {n.node: n for n in pattern.n_list}
    graph = pattern.extract_graph()

    def ensure_active(node: Node) -> None:
        """Initialize node in pattern if it has not been initialized before."""
        if node not in initialized:
            target.add(n_dict[node])
            initialized.add(node)

    def ensure_neighborhood(node: Node) -> None:
        """Initialize and entangle the inactive nodes in the neighbourhood of ``node``."""
        ensure_active(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in done:
                ensure_active(neighbor)
                target.add(command.E((node, neighbor)))

    for m in pattern.m_list:
        ensure_neighborhood(m.node)
        target.add(m)
        done.add(m.node)
    for node in pattern.output_nodes:
        ensure_neighborhood(node)
        if domain := pattern.z_dict.get(node):
            target.add(command.Z(node, set(domain)))
        if domain := pattern.x_dict.get(node):
            target.add(command.X(node, set(domain)))
        if clifford_gate := pattern.c_dict.get(node):
            target.add(command.C(node, clifford_gate))
        done.add(node)
    return target


def minimize_space(
    pattern: StandardizedPattern, heuristics: Iterable[SpaceMinimizationHeuristic] | None = None
) -> StandardizedPattern:
    """Return a pattern that reduces the maximal space, i.e. the number of qubits simultaneously required to execute the pattern.

    The default heuristics are as follows:

    - If the pattern has a causal flow, then the layers of the flow
      are used to provide an optimal measurement order for the
      pattern. See :func:`minimization_using_causal_flow`.

    - If the pattern has no causal flow, then a greedy algorithm is
      applied to choose at each step to measure the node with the
      minimal degree.  This particular heuristic has no guarantee to
      produce an optimal pattern and can even increase the maximal
      space. See :func:`greedy_minimization_by_degree`.

    - If the above heuristic does not give a pattern with a smaller
      maximal space than the original pattern, then the original
      measurement order is unchanged. See
      :func:`do_nothing_for_space_minimization`.

    Parameters
    ----------
    pattern: StandardizedPattern
        The pattern to optimize.

    heuristics: Iterable[SpaceMinimizationHeuristic] | None = None
        The heuristics to try in order.
        By default:
        ``[minimization_using_causal_flow, greedy_minimization_by_degree, do_nothing_for_space_minimization]``

    Returns
    -------
    StandardizedPattern
        The optimized pattern.
    """
    if heuristics is None:
        heuristics = [minimization_using_causal_flow, greedy_minimization_by_degree, do_nothing_for_space_minimization]
    m_dict = {m.node: m for m in pattern.m_list}
    best_pattern_with_max_space: tuple[StandardizedPattern, int] | None = None
    for f in heuristics:
        heuristics_result = f(pattern)
        if heuristics_result is not None:
            new_order, known_optimal = heuristics_result
            new_m_list = tuple(m_dict[node] for node in new_order)
            tentative_pattern = dataclasses.replace(pattern, m_list=new_m_list)
            if known_optimal:
                return tentative_pattern
            tentative_max_space = tentative_pattern.max_space()
            if best_pattern_with_max_space is not None:
                _best_pattern, best_max_space = best_pattern_with_max_space
                if best_max_space <= tentative_max_space:
                    continue
            best_pattern_with_max_space = tentative_pattern, tentative_max_space
    if best_pattern_with_max_space is None:
        return pattern
    return best_pattern_with_max_space[0]


def minimization_using_causal_flow(pattern: StandardizedPattern) -> tuple[tuple[Node, ...], bool] | None:
    """Use the causal flow layer to minimize space.

    This minimization heuristic is optimal but requires the pattern to have a causal flow.
    """
    try:
        cf = pattern.extract_causal_flow()
    except FlowError:
        return None
    else:
        meas_order = tuple(chain(*reversed(cf.partial_order_layers[1:])))
        return meas_order, True


def do_nothing_for_space_minimization(pattern: StandardizedPattern) -> tuple[tuple[Node, ...], bool] | None:
    """Leave the pattern unchanged.

    This minimization heuristic does not optimize anything so it
    cannot return a pattern that has worse max space than the original
    pattern.

    Usually, this heuristic is included in the list of the heuristic
    considered to ensure that the result of space minimization cannot
    have a worse max space than the original pattern.  This is the
    case by default, but the user can remove this heuristic from the
    list to see the effect of the other heuristic even if they don't
    reduce the max space of the pattern.
    """
    return tuple(m.node for m in pattern.m_list), False


def _extract_dependency(pattern: StandardizedPattern) -> dict[Node, set[Node]]:
    """Get dependency (byproduct correction & dependent measurement) structure of nodes in the graph (resource) state, according to the pattern.

    This is used to determine the optimum measurement order.

    Returns
    -------
    dependency : dict[Node, set[Node]]
        index is node number. all nodes in the each set must be measured before measuring
    """
    nodes = chain(pattern.input_nodes, (n.node for n in pattern.n_list))
    dependency: dict[int, set[int]] = {i: set() for i in nodes}
    for m in pattern.m_list:
        dependency[m.node] |= m.s_domain | m.t_domain
    for node, domain in chain(pattern.z_dict.items(), pattern.x_dict.items()):
        dependency[node] |= domain
    return dependency


def _update_dependency(measured: AbstractSet[Node], dependency: Mapping[Node, set[Node]]) -> None:
    """Remove measured nodes from the 'dependency'.

    Parameters
    ----------
    measured: AbstractSet[Node]
        measured nodes.
    dependency: Mapping[Node, set[Node]]
        which is produced by `_extract_dependency`
    """
    for s in dependency.values():
        s.difference_update(measured)


def greedy_minimization_by_degree(pattern: StandardizedPattern) -> tuple[tuple[Node, ...], bool] | None:
    """Choose greedily the nodes by minimal degree.

    This minimization heuristic can worsen the max space in some
    situations.
    """
    graph = pattern.extract_graph()
    nodes = set(graph.nodes)
    edges = set(graph.edges)
    not_measured = nodes - set(pattern.output_nodes)
    dependency = _extract_dependency(pattern)
    # keys() should be converted into `set` because it is transient.
    _update_dependency(set(pattern.results.keys()), dependency)
    meas_order = []
    removable_edges: set[tuple[Node, Node]] = set()
    while not_measured:
        min_edges = len(nodes) + 1
        next_node: Node | None = None
        for i in not_measured:
            if not dependency[i]:
                connected_edges = set(graph.edges(i)) & edges
                if min_edges > len(connected_edges):
                    min_edges = len(connected_edges)
                    next_node = i
                    removable_edges = connected_edges
        assert next_node is not None
        meas_order.append(next_node)
        _update_dependency({next_node}, dependency)
        not_measured -= {next_node}
        edges -= removable_edges
    return tuple(meas_order), False
