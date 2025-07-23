"""Optimization procedures for patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import graphix
from graphix import command
from graphix.clifford import Clifford
from graphix.command import CommandKind, Node
from graphix.measurements import Domains

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

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
    n_list: list[command.N] = []
    e_list: list[command.E] = []
    m_list: list[command.M] = []
    c_dict: dict[int, Clifford] = {}
    z_dict: dict[int, set[Node]] = {}
    x_dict: dict[int, set[Node]] = {}

    def add_correction_domain(domain_dict: dict[Node, set[Node]], node: Node, domain: set[Node]) -> None:
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

    def commute_clifford(clifford_gate: Clifford, c_dict: dict[int, Clifford], i: int, j: int) -> None:
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

    s_domain: set[Node]
    t_domain: set[Node]
    s_domain_opt: set[Node] | None
    t_domain_opt: set[Node] | None

    for cmd in pattern:
        if cmd.kind == CommandKind.N:
            n_list.append(cmd)
        elif cmd.kind == CommandKind.E:
            for side in (0, 1):
                i, j = cmd.nodes[side], cmd.nodes[1 - side]
                if clifford_gate := c_dict.get(i):
                    commute_clifford(clifford_gate, c_dict, i, j)
                if s_domain_opt := x_dict.get(i):
                    add_correction_domain(z_dict, j, s_domain_opt)
            e_list.append(cmd)
        elif cmd.kind == CommandKind.M:
            new_cmd = cmd
            if clifford_gate := c_dict.pop(cmd.node, None):
                new_cmd = new_cmd.clifford(clifford_gate)
            if t_domain_opt := z_dict.pop(cmd.node, None):
                # The original domain should not be mutated
                new_cmd.t_domain = new_cmd.t_domain ^ t_domain_opt  # noqa: PLR6104
            if s_domain_opt := x_dict.pop(cmd.node, None):
                # The original domain should not be mutated
                new_cmd.s_domain = new_cmd.s_domain ^ s_domain_opt  # noqa: PLR6104
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
                add_correction_domain(z_dict, cmd.node, domains.t_domain)
            if domains.s_domain:
                add_correction_domain(x_dict, cmd.node, domains.s_domain)
        elif cmd.kind == CommandKind.C:
            # Each pattern command is applied by left multiplication: if a clifford `C`
            # has been already applied to a node, applying a clifford `C'` to the same
            # node is equivalent to apply `C'C` to a fresh node.
            c_dict[cmd.node] = cmd.clifford @ c_dict.get(cmd.node, Clifford.I)
    result = graphix.Pattern(input_nodes=pattern.input_nodes)
    result.results = pattern.results
    result.extend(
        [
            *n_list,
            *e_list,
            *m_list,
            *(command.Z(node=node, domain=domain) for node, domain in z_dict.items()),
            *(command.X(node=node, domain=domain) for node, domain in x_dict.items()),
            *(command.C(node=node, clifford=clifford_gate) for node, clifford_gate in c_dict.items()),
        ]
    )
    return result


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
    result = graphix.Pattern(input_nodes=pattern.input_nodes)
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
