"""MBQC pattern according to Measurement Calculus.

ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)
"""

from __future__ import annotations

import copy
import dataclasses
import warnings
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, SupportsFloat, TypeVar

import networkx as nx
from typing_extensions import assert_never, override

from graphix import command, optimization, parameter
from graphix.clifford import Clifford
from graphix.command import Command, CommandKind
from graphix.fundamentals import Axis, Plane, Sign
from graphix.gflow import find_flow, find_gflow, get_layers
from graphix.graphsim import GraphState
from graphix.measurements import Outcome, PauliMeasurement, toggle_outcome
from graphix.pretty_print import OutputFormat, pattern_to_str
from graphix.simulator import PatternSimulator
from graphix.states import BasicStates
from graphix.visualization import GraphVisualizer

if TYPE_CHECKING:
    from collections.abc import Container, Iterator, Mapping
    from collections.abc import Set as AbstractSet
    from typing import Any

    from numpy.random import Generator

    from graphix.parameter import ExpressionOrFloat, ExpressionOrSupportsFloat, Parameter
    from graphix.sim import Backend, BackendState, Data


_StateT_co = TypeVar("_StateT_co", bound="BackendState", covariant=True)


@dataclass(frozen=True)
class NodeAlreadyPreparedError(Exception):
    """Exception raised if a node is already prepared."""

    node: int

    @override
    def __str__(self) -> str:
        """Return the message of the error."""
        return f"Node already prepared: {self.node}"


class Pattern:
    """
    MBQC pattern class.

    Pattern holds a sequence of commands to operate the MBQC (Pattern.seq),
    and provide modification strategies to improve the structure and simulation
    efficiency of the pattern accoring to measurement calculus.

    ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)

    Attributes
    ----------
    list(self) :
        list of commands.

        .. line-block::
            each command is a list [type, nodes, attr] which will be applied in the order of list indices.
            type: one of {'N', 'M', 'E', 'X', 'Z', 'S', 'C'}
            nodes: int for {'N', 'M', 'X', 'Z', 'S', 'C'} commands, tuple (i, j) for {'E'} command
            attr for N: none
            attr for M: meas_plane, angle, s_domain, t_domain
            attr for X: signal_domain
            attr for Z: signal_domain
            attr for S: signal_domain
            attr for C: clifford_index, as defined in :py:mod:`graphix.clifford`
    n_node : int
        total number of nodes in the resource state
    """

    results: dict[int, Outcome]
    __seq: list[Command]

    def __init__(
        self,
        input_nodes: Iterable[int] | None = None,
        cmds: Iterable[Command] | None = None,
        output_nodes: Iterable[int] | None = None,
    ) -> None:
        """
        Construct a pattern.

        Parameters
        ----------
        input_nodes : Iterable[int] | None
            Optional. List of input qubits.
        cmds : Iterable[Command] | None
            Optional. List of initial commands.
        output_nodes : Iterable[int] | None
            Optional. List of output qubits.
        """
        self.results = {}  # measurement results from the graph state simulator
        if input_nodes is None:
            self.__input_nodes = []
        else:
            self.__input_nodes = list(input_nodes)  # input nodes (list() makes our own copy of the list)
        self.__n_node = len(self.__input_nodes)  # total number of nodes in the graph state
        self._pauli_preprocessed = False  # flag for `measure_pauli` preprocessing completion

        self.__seq = []
        # output nodes are initially a copy input nodes, since none are measured yet
        self.__output_nodes = list(self.__input_nodes)

        if cmds is not None:
            self.extend(cmds)

        if output_nodes is not None:
            self.reorder_output_nodes(output_nodes)

    def add(self, cmd: Command) -> None:
        """Add command to the end of the pattern.

        An MBQC command is an instance of :class:`graphix.command.Command`.

        Parameters
        ----------
        cmd : :class:`graphix.command.Command`
            MBQC command.
        """
        if cmd.kind == CommandKind.N:
            if cmd.node in self.__output_nodes:
                raise NodeAlreadyPreparedError(cmd.node)
            self.__n_node += 1
            self.__output_nodes.append(cmd.node)
        elif cmd.kind == CommandKind.M:
            self.__output_nodes.remove(cmd.node)
        self.__seq.append(cmd)

    def extend(self, *cmds: Command | Iterable[Command]) -> None:
        """Add sequences of commands.

        :param cmds: sequences of commands
        """
        for item in cmds:
            if isinstance(item, Iterable):
                for cmd in item:
                    self.add(cmd)
            else:
                self.add(item)

    def clear(self) -> None:
        """Clear the sequence of pattern commands."""
        self.__n_node = len(self.__input_nodes)
        self.__seq = []
        self.__output_nodes = list(self.__input_nodes)

    def replace(self, cmds: list[Command], input_nodes: list[int] | None = None) -> None:
        """Replace pattern with a given sequence of pattern commands.

        :param cmds: list of commands

        :param input_nodes: optional, list of input qubits (by default, keep the same input nodes as before)
        """
        if input_nodes is not None:
            self.__input_nodes = list(input_nodes)
        self.clear()
        self.extend(cmds)

    def compose(
        self, other: Pattern, mapping: Mapping[int, int], preserve_mapping: bool = False
    ) -> tuple[Pattern, dict[int, int]]:
        r"""Compose two patterns by merging subsets of outputs from `self` and a subset of inputs of `other`, and relabeling the nodes of `other` that were not merged.

        Parameters
        ----------
        other : Pattern
            Pattern to be composed with `self`.
        mapping: Mapping[int, int]
            Partial relabelling of the nodes in `other`, with `keys` and `values` denoting the old and new node labels, respectively.
        preserve_mapping: bool
            Boolean flag controlling the ordering of the output nodes in the returned pattern.

        Returns
        -------
        p: Pattern
            composed pattern
        mapping_complete: dict[int, int]
            Complete relabelling of the nodes in `other`, with `keys` and `values` denoting the old and new node label, respectively.

        Notes
        -----
        Let's denote :math:`(I_j, O_j, V_j, S_j)` the ordered set of inputs and outputs, the computational space and the sequence of commands of pattern :math:`P_j`, respectively, with :math:`j = 1` for pattern `self` and :math:`j = 2` for pattern `other`. Let's denote :math:`P` the resulting pattern with :math:`(I, O, V, S)`.
        Let's denote :math:`K, U` the sets of `keys` and `values` of `mapping`, :math:`M_1 = O_1 \cap U` the set of merged outputs, and :math:`M_2 = \{k \in I_2 \cap K | k \rightarrow v, v \in M_1 \}` the set of merged inputs.

        The pattern composition requires that
        - :math:`K \subseteq V_2`.
        - For a pair :math:`(k, v) \in (K, U)`
            - :math:`U \cap V_1 \setminus O_1 = \emptyset`. If :math:`v \in O_1`, then :math:`k \in I_2`, otherwise an error is raised.
            - :math:`v` can always satisfy :math:`v \notin V_1`, thereby allowing a custom relabelling.

        The returned pattern follows this convention:
        - Nodes of pattern `other` not specified in `mapping` (i.e., :math:`V_2 \cap K^c`) are relabelled in ascending order.
        - The sequence of the resulting pattern is :math:`S = S_2 S_1`, where nodes in :math:`S_2` are relabelled according to `mapping`.
        - :math:`I = I_1 \cup (I_2 \setminus M_2)`.
        - :math:`O = (O_1 \setminus M_1) \cup O_2`.
        - Input (and, respectively, output) nodes in the returned pattern have the order of the pattern `self` followed by those of the pattern `other`. Merged nodes are removed.
        - If `preserve_mapping = True` and :math:`|M_1| = |I_2| = |O_2|`, then the outputs of the returned pattern are the outputs of pattern `self`, where the nth merged output is replaced by the output of pattern `other` corresponding to its nth input instead.
        """
        nodes_p1_lst, _ = self.get_graph()
        nodes_p1: set[int] = set(nodes_p1_lst) | self.results.keys()  # Results contain preprocessed Pauli nodes
        nodes_p2_lst, _ = other.get_graph()
        nodes_p2: set[int] = set(nodes_p2_lst) | other.results.keys()

        if not mapping.keys() <= nodes_p2:
            raise ValueError("Keys of `mapping` must correspond to the nodes of `other`.")

        # Cast to set for improved performance in membership test
        mapping_values_set = set(mapping.values())
        o1_set = set(self.__output_nodes)
        i2_set = set(other.input_nodes)

        if len(mapping) != len(mapping_values_set):
            raise ValueError("Values of `mapping` contain duplicates.")

        if mapping_values_set & nodes_p1 - o1_set:
            raise ValueError("Values of `mapping` must not contain measured nodes of pattern `self`.")

        for k, v in mapping.items():
            if v in o1_set and k not in i2_set:
                raise ValueError(
                    f"Mapping {k} -> {v} is not valid. {v} is an output of pattern `self` but {k} is not an input of pattern `other`."
                )

        # Check if resulting pattern will have C commands before E commands
        if any(cmd.kind == CommandKind.C for cmd in self.__seq) and any(cmd.kind == CommandKind.E for cmd in other):
            warnings.warn(
                r"Pattern `self` contains Clifford commands and pattern `other` contains E commands. Standardization might not be possible for the resulting composed pattern.",
                stacklevel=2,
            )

        shift = max(*nodes_p1, *mapping.values()) + 1
        mapping_sequential = {
            node: i for i, node in enumerate(sorted(nodes_p2 - mapping.keys()), start=shift)
        }  # assigns new labels to nodes in other not specified in mapping

        mapping_complete = {**mapping, **mapping_sequential}

        mapped_inputs = [mapping_complete[n] for n in other.input_nodes]
        mapped_outputs = [mapping_complete[n] for n in other.output_nodes]
        mapped_results: dict[int, Outcome] = {mapping_complete[n]: m for n, m in other.results.items()}

        merged = mapping_values_set.intersection(self.__output_nodes)

        inputs = self.__input_nodes + [n for n in mapped_inputs if n not in merged]

        if preserve_mapping and not (len(merged) == len(other.input_nodes) == len(other.output_nodes)):
            warnings.warn(
                "`preserve_mapping = True` ignored because the number of merged nodes, inputs, and outputs of pattern `other` are different.",
                stacklevel=2,
            )
            preserve_mapping = False

        if preserve_mapping:
            io_mapping = {mapping[i]: mapping_complete[o] for i, o in zip(other.input_nodes, other.output_nodes)}
            outputs = [io_mapping[n] if n in merged else n for n in self.__output_nodes]
        else:
            outputs = [n for n in self.__output_nodes if n not in merged] + mapped_outputs

        def update_command(cmd: Command) -> Command:
            # Shallow copy is enough since the mutable attributes of cmd_new susceptible to change are reassigned
            cmd_new = copy.copy(cmd)

            if cmd_new.kind is CommandKind.E:
                i, j = cmd_new.nodes
                cmd_new.nodes = (mapping_complete[i], mapping_complete[j])
            elif cmd_new.kind is not CommandKind.T:
                cmd_new.node = mapping_complete[cmd_new.node]
                if cmd_new.kind is CommandKind.M:
                    cmd_new.s_domain = {mapping_complete[i] for i in cmd_new.s_domain}
                    cmd_new.t_domain = {mapping_complete[i] for i in cmd_new.t_domain}
                # Use of `==` here for mypy
                elif cmd_new.kind == CommandKind.X or cmd_new.kind == CommandKind.Z or cmd_new.kind == CommandKind.S:  # noqa: PLR1714
                    cmd_new.domain = {mapping_complete[i] for i in cmd_new.domain}

            return cmd_new

        seq = self.__seq + [update_command(c) for c in other]

        results: dict[int, Outcome] = {**self.results, **mapped_results}
        p = Pattern(input_nodes=inputs, output_nodes=outputs, cmds=seq)
        p.results = results

        return p, mapping_complete

    @property
    def input_nodes(self) -> list[int]:
        """List input nodes."""
        return list(self.__input_nodes)  # copy for preventing modification

    @property
    def output_nodes(self) -> list[int]:
        """List all nodes that are either `input_nodes` or prepared with `N` commands and that have not been measured with an `M` command."""
        return list(self.__output_nodes)  # copy for preventing modification

    def __len__(self) -> int:
        """Return the length of command sequence."""
        return len(self.__seq)

    def __iter__(self) -> Iterator[Command]:
        """Iterate over commands."""
        return iter(self.__seq)

    def __getitem__(self, index: int) -> Command:
        """Get the command at a given index."""
        return self.__seq[index]

    @property
    def n_node(self) -> int:
        """Count of nodes that are either `input_nodes` or prepared with `N` commands."""
        return self.__n_node

    def reorder_output_nodes(self, output_nodes: Iterable[int]) -> None:
        """Arrange the order of output_nodes.

        Parameters
        ----------
        output_nodes: iterable of int
            output nodes order determined by user. each index corresponds to that of logical qubits.
        """
        output_nodes = list(output_nodes)  # make our own copy (allow iterators to be passed)
        assert_permutation(self.__output_nodes, output_nodes)
        self.__output_nodes = output_nodes

    def reorder_input_nodes(self, input_nodes: Iterable[int]) -> None:
        """Arrange the order of input_nodes.

        Parameters
        ----------
        input_nodes: iterable of int
            input nodes order determined by user. each index corresponds to that of logical qubits.
        """
        input_nodes = list(input_nodes)  # make our own copy (allow iterators to be passed)
        assert_permutation(self.__input_nodes, input_nodes)
        self.__input_nodes = input_nodes

    def __repr__(self) -> str:
        """Return a representation string of the pattern."""
        arguments = []
        if self.__input_nodes:
            arguments.append(f"input_nodes={self.__input_nodes}")
        if self.__seq:
            arguments.append(f"cmds={self.__seq}")
        if self.__output_nodes:
            arguments.append(f"output_nodes={self.__output_nodes}")
        return f"Pattern({', '.join(arguments)})"

    def __str__(self) -> str:
        """Return a human-readable string of the pattern."""
        return self.to_ascii()

    def __eq__(self, other: object) -> bool:
        """Return `True` if the two patterns are equal, `False` otherwise."""
        if not isinstance(other, Pattern):
            return NotImplemented
        return (
            self.__seq == other.__seq
            and self.__input_nodes == other.__input_nodes
            and self.__output_nodes == other.__output_nodes
            and self.results == other.results
        )

    def to_ascii(
        self, left_to_right: bool = False, limit: int = 40, target: Container[command.CommandKind] | None = None
    ) -> str:
        """Return the ASCII string representation of the pattern."""
        return pattern_to_str(self, OutputFormat.ASCII, left_to_right, limit, target)

    def to_latex(
        self, left_to_right: bool = False, limit: int = 40, target: Container[command.CommandKind] | None = None
    ) -> str:
        """Return a string containing the LaTeX representation of the pattern."""
        return pattern_to_str(self, OutputFormat.LaTeX, left_to_right, limit, target)

    def to_unicode(
        self, left_to_right: bool = False, limit: int = 40, target: Container[command.CommandKind] | None = None
    ) -> str:
        """Return the Unicode string representation of the pattern."""
        return pattern_to_str(self, OutputFormat.Unicode, left_to_right, limit, target)

    def print_pattern(self, lim: int = 40, target: Container[CommandKind] | None = None) -> None:
        """Print the pattern sequence (Pattern.seq).

        This method is deprecated.
        See :meth:`to_ascii`, :meth:`to_latex`, :meth:`to_unicode` and :func:`graphix.pretty_print.pattern_to_str`.

        Parameters
        ----------
        lim: int, optional
            maximum number of commands to show
        target : list of CommandKind, optional
            show only specified commands, e.g. [CommandKind.M, CommandKind.X, CommandKind.Z]
        """
        warnings.warn(
            "Method `print_pattern` is deprecated. Use one of the methods `to_ascii`, `to_latex`, `to_unicode`, or the function `graphix.pretty_print.pattern_to_str`.",
            DeprecationWarning,
            stacklevel=1,
        )
        print(pattern_to_str(self, OutputFormat.ASCII, left_to_right=True, limit=lim, target=target))

    def standardize(self) -> None:
        """Execute standardization of the pattern.

        'standard' pattern is one where commands are sorted in the
        order of 'N', 'E', 'M' and then byproduct commands ('X' and
        'Z') and finally Clifford commands ('C').
        """
        self.__seq = optimization.standardize(self).__seq

    def is_standard(self, strict: bool = False) -> bool:
        """Determine whether the command sequence is standard.

        Parameters
        ----------
        strict : bool, optional
            If True, ensures that C commands are the last ones.

        Returns
        -------
        is_standard : bool
            True if the pattern is standard
        """
        it = iter(self)
        try:
            kind = next(it).kind
            while kind == CommandKind.N:
                kind = next(it).kind
            while kind == CommandKind.E:
                kind = next(it).kind
            while kind == CommandKind.M:
                kind = next(it).kind
            if strict:
                xz = {CommandKind.X, CommandKind.Z}
                while kind in xz:
                    kind = next(it).kind
                while kind == CommandKind.C:
                    kind = next(it).kind
            else:
                xzc = {CommandKind.X, CommandKind.Z, CommandKind.C}
                while kind in xzc:
                    kind = next(it).kind
        except StopIteration:
            return True
        else:
            return False

    def shift_signals(self, method: str = "direct") -> dict[int, set[int]]:
        """Perform signal shifting procedure.

        Extract the t-dependence of the measurement into 'S' commands
        and commute them to the end of the command sequence where it can be removed.
        This procedure simplifies the dependence structure of the pattern.

        Ref for the original 'mc' method:
            V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)

        Parameters
        ----------
        method : str, optional
            'direct' shift_signals is executed on a conventional Pattern sequence.
            'mc' shift_signals is done using the original algorithm on the measurement calculus paper.

        Returns
        -------
        signal_dict : dict[int, set[int]]
            For each node, the signal that have been shifted.
        """
        if method == "direct":
            return self.shift_signals_direct()
        if method == "mc":
            signal_dict = self.extract_signals()
            target = self._find_op_to_be_moved(CommandKind.S, rev=True)
            while target is not None:
                if target == len(self.__seq) - 1:
                    self.__seq.pop(target)
                    target = self._find_op_to_be_moved(CommandKind.S, rev=True)
                    continue
                cmd = self.__seq[target + 1]
                kind = cmd.kind
                if kind == CommandKind.X:
                    self._commute_xs(target)
                elif kind == CommandKind.Z:
                    self._commute_zs(target)
                elif kind == CommandKind.M:
                    self._commute_ms(target)
                elif kind == CommandKind.S:
                    self._commute_ss(target)
                else:
                    self._commute_with_following(target)
                target += 1
            return signal_dict
        raise ValueError("Invalid method")

    def shift_signals_direct(self) -> dict[int, set[int]]:
        """Perform signal shifting procedure."""
        signal_dict: dict[int, set[int]] = {}

        def expand_domain(domain: set[command.Node]) -> None:
            """Expand ``domain`` with previously shifted signals.

            Parameters
            ----------
            domain : set[int]
                Set of nodes representing the current domain. This set is
                modified in place by XORing any previously shifted domains.
            """
            for node in domain & signal_dict.keys():
                domain ^= signal_dict[node]

        for i, cmd in enumerate(self):
            if cmd.kind == CommandKind.M:
                s_domain = set(cmd.s_domain)
                t_domain = set(cmd.t_domain)
                expand_domain(s_domain)
                expand_domain(t_domain)
                plane = cmd.plane
                if plane == Plane.XY:
                    # M^{XY,α} X^s Z^t = M^{XY,(-1)^s·α+tπ}
                    #                  = S^t M^{XY,(-1)^s·α}
                    #                  = S^t M^{XY,α} X^s
                    if t_domain:
                        signal_dict[cmd.node] = t_domain
                        t_domain = set()
                elif plane == Plane.XZ:
                    # M^{XZ,α} X^s Z^t = M^{XZ,(-1)^t((-1)^s·α+sπ)}
                    #                  = M^{XZ,(-1)^{s+t}·α+(-1)^t·sπ}
                    #                  = M^{XZ,(-1)^{s+t}·α+sπ         (since (-1)^t·π ≡ π (mod 2π))
                    #                  = S^s M^{XZ,(-1)^{s+t}·α}
                    #                  = S^s M^{XZ,α} Z^{s+t}
                    if s_domain:
                        signal_dict[cmd.node] = s_domain
                        t_domain ^= s_domain
                        s_domain = set()
                elif plane == Plane.YZ and s_domain:
                    # M^{YZ,α} X^s Z^t = M^{YZ,(-1)^t·α+sπ)}
                    #                  = S^s M^{YZ,(-1)^t·α}
                    #                  = S^s M^{YZ,α} Z^t
                    signal_dict[cmd.node] = s_domain
                    s_domain = set()
                if s_domain != cmd.s_domain or t_domain != cmd.t_domain:
                    self.__seq[i] = dataclasses.replace(cmd, s_domain=s_domain, t_domain=t_domain)
            # Use of `==` here for mypy
            elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
                domain = set(cmd.domain)
                expand_domain(domain)
                if domain != cmd.domain:
                    self.__seq[i] = dataclasses.replace(cmd, domain=domain)
        return signal_dict

    def _find_op_to_be_moved(self, op: CommandKind, rev: bool = False, skipnum: int = 0) -> int | None:
        """Find a command.

        Parameters
        ----------
        op : CommandKind, N, E, M, X, Z, S
            command types to be searched
        rev : bool
            search from the end (true) or start (false) of seq
        skipnum : int
            skip the detected command by specified times
        """
        if not rev:  # Search from the start
            start_index, end_index, step = 0, len(self.__seq), 1
        else:  # Search from the end
            start_index, end_index, step = len(self.__seq) - 1, -1, -1

        num_ops = 0
        for index in range(start_index, end_index, step):
            if self.__seq[index].kind == op:
                num_ops += 1
                if num_ops == skipnum + 1:
                    return index

        # If no target found
        return None

    def _commute_ex(self, target: int) -> bool:
        """Perform the commutation of E and X.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a X command followed by E command
        """
        x = self.__seq[target]
        e = self.__seq[target + 1]
        assert x.kind == CommandKind.X
        assert e.kind == CommandKind.E
        if e.nodes[0] == x.node:
            z = command.Z(node=e.nodes[1], domain=x.domain)
            self.__seq.pop(target + 1)  # del E
            self.__seq.insert(target, z)  # add Z in front of X
            self.__seq.insert(target, e)  # add E in front of Z
            return True
        if e.nodes[1] == x.node:
            z = command.Z(node=e.nodes[0], domain=x.domain)
            self.__seq.pop(target + 1)  # del E
            self.__seq.insert(target, z)  # add Z in front of X
            self.__seq.insert(target, e)  # add E in front of Z
            return True
        self._commute_with_following(target)
        return False

    def _commute_mx(self, target: int) -> bool:
        """Perform the commutation of M and X.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a X command followed by M command
        """
        x = self.__seq[target]
        m = self.__seq[target + 1]
        assert x.kind == CommandKind.X
        assert m.kind == CommandKind.M
        if x.node == m.node:
            m.s_domain ^= x.domain
            self.__seq.pop(target)  # del X
            return True
        self._commute_with_following(target)
        return False

    def _commute_mz(self, target: int) -> bool:
        """Perform the commutation of M and Z.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a Z command followed by M command
        """
        z = self.__seq[target]
        m = self.__seq[target + 1]
        assert z.kind == CommandKind.Z
        assert m.kind == CommandKind.M
        if z.node == m.node:
            m.t_domain ^= z.domain
            self.__seq.pop(target)  # del Z
            return True
        self._commute_with_following(target)
        return False

    def _commute_xs(self, target: int) -> None:
        """Perform the commutation of X and S.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by X command
        """
        s = self.__seq[target]
        x = self.__seq[target + 1]
        assert s.kind == CommandKind.S
        assert x.kind == CommandKind.X
        if s.node in x.domain:
            x.domain ^= s.domain
        self._commute_with_following(target)

    def _commute_zs(self, target: int) -> None:
        """Perform the commutation of Z and S.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by Z command
        """
        s = self.__seq[target]
        z = self.__seq[target + 1]
        assert s.kind == CommandKind.S
        assert z.kind == CommandKind.Z
        if s.node in z.domain:
            z.domain ^= s.domain
        self._commute_with_following(target)

    def _commute_ms(self, target: int) -> None:
        """Perform the commutation of M and S.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by M command
        """
        s = self.__seq[target]
        m = self.__seq[target + 1]
        assert s.kind == CommandKind.S
        assert m.kind == CommandKind.M
        if s.node in m.s_domain:
            m.s_domain ^= s.domain
        if s.node in m.t_domain:
            m.t_domain ^= s.domain
        self._commute_with_following(target)

    def _commute_ss(self, target: int) -> None:
        """Perform the commutation of two S commands.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by S command
        """
        s1 = self.__seq[target]
        s2 = self.__seq[target + 1]
        assert s1.kind == CommandKind.S
        assert s2.kind == CommandKind.S
        if s1.node in s2.domain:
            s2.domain ^= s1.domain
        self._commute_with_following(target)

    def _commute_with_following(self, target: int) -> None:
        """Perform the commutation of two consecutive commands that commutes.

        commutes the target command with the following command.

        Parameters
        ----------
        target : int
            target command index
        """
        a = self.__seq[target + 1]
        self.__seq.pop(target + 1)
        self.__seq.insert(target, a)

    def _commute_with_preceding(self, target: int) -> None:
        """Perform the commutation of two consecutive commands that commutes.

        commutes the target command with the preceding command.

        Parameters
        ----------
        target : int
            target command index
        """
        a = self.__seq[target - 1]
        self.__seq.pop(target - 1)
        self.__seq.insert(target, a)

    def _move_n_to_left(self) -> None:
        """Move all 'N' commands to the start of the sequence.

        N can be moved to the start of sequence without the need of considering
        commutation relations.
        """
        new_seq = []
        n_list = []
        for cmd in self.__seq:
            if cmd.kind == CommandKind.N:
                n_list.append(cmd)
            else:
                new_seq.append(cmd)
        n_list.sort(key=lambda n_cmd: n_cmd.node)
        self.__seq = n_list + new_seq

    def _move_byproduct_to_right(self) -> None:
        """Move the byproduct commands to the end of sequence, using the commutation relations implemented in graphix.Pattern class."""
        # First, we move all X commands to the end of sequence
        index = len(self.__seq) - 1
        x_limit = len(self.__seq) - 1
        while index > 0:
            if self.__seq[index].kind == CommandKind.X:
                index_x = index
                while index_x < x_limit:
                    cmd = self.__seq[index_x + 1]
                    kind = cmd.kind
                    if kind == CommandKind.E:
                        move = self._commute_ex(index_x)
                        if move:
                            x_limit += 1  # addition of extra Z means target must be increased
                            index_x += 1
                    elif kind == CommandKind.M:
                        search = self._commute_mx(index_x)
                        if search:
                            x_limit -= 1  # XM commutation rule removes X command
                            break
                    else:
                        self._commute_with_following(index_x)
                    index_x += 1
                else:
                    x_limit -= 1
            index -= 1
        # then, move Z to the end of sequence in front of X
        index = x_limit
        z_limit = x_limit
        while index > 0:
            if self.__seq[index].kind == CommandKind.Z:
                index_z = index
                while index_z < z_limit:
                    cmd = self.__seq[index_z + 1]
                    if cmd.kind == CommandKind.M:
                        search = self._commute_mz(index_z)
                        if search:
                            z_limit -= 1  # ZM commutation rule removes Z command
                            break
                    else:
                        self._commute_with_following(index_z)
                    index_z += 1
            index -= 1

    def _move_e_after_n(self) -> None:
        """Move all E commands to the start of sequence, before all N commands. assumes that _move_n_to_left() method was called."""
        moved_e = 0
        target = self._find_op_to_be_moved(CommandKind.E, skipnum=moved_e)
        while target is not None:
            if (target == 0) or (
                self.__seq[target - 1].kind == CommandKind.N or self.__seq[target - 1].kind == CommandKind.E
            ):
                moved_e += 1
                target = self._find_op_to_be_moved(CommandKind.E, skipnum=moved_e)
                continue
            self._commute_with_preceding(target)
            target -= 1

    def extract_signals(self) -> dict[int, set[int]]:
        """Extract 't' domain of measurement commands, turn them into signal 'S' commands and add to the command sequence.

        This is used for shift_signals() method.
        """
        signal_dict = {}
        pos = 0
        while pos < len(self.__seq):
            cmd = self.__seq[pos]
            if cmd.kind == CommandKind.M:
                extracted_signal = extract_signal(cmd.plane, cmd.s_domain, cmd.t_domain)
                if extracted_signal.signal:
                    self.__seq.insert(pos + 1, command.S(node=cmd.node, domain=extracted_signal.signal))
                    cmd.s_domain = extracted_signal.s_domain
                    cmd.t_domain = extracted_signal.t_domain
                    pos += 1
                signal_dict[cmd.node] = extracted_signal.signal
            pos += 1
        return signal_dict

    def _get_dependency(self) -> dict[int, set[int]]:
        """Get dependency (byproduct correction & dependent measurement) structure of nodes in the graph (resource) state, according to the pattern.

        This is used to determine the optimum measurement order.

        Returns
        -------
        dependency : dict of set
            index is node number. all nodes in the each set must be measured before measuring
        """
        nodes, _ = self.get_graph()
        dependency: dict[int, set[int]] = {i: set() for i in nodes}
        for cmd in self.__seq:
            if cmd.kind == CommandKind.M:
                dependency[cmd.node] |= cmd.s_domain | cmd.t_domain
            # Use of `==` here for mypy
            elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
                dependency[cmd.node] |= cmd.domain
        return dependency

    @staticmethod
    def update_dependency(measured: AbstractSet[int], dependency: dict[int, set[int]]) -> None:
        """Remove measured nodes from the 'dependency'.

        Parameters
        ----------
        measured: set of int
            measured nodes.
        dependency: dict of set
            which is produced by `_get_dependency`

        Returns
        -------
        dependency: dict of set
            updated dependency information
        """
        for i in dependency:
            dependency[i] -= measured

    def get_layers(self) -> tuple[int, dict[int, set[int]]]:
        """Construct layers(l_k) from dependency information.

        kth layer must be measured before measuring k+1th layer
        and nodes in the same layer can be measured simultaneously.

        Returns
        -------
        depth : int
            depth of graph
        layers : dict of set
            nodes grouped by layer index(k)
        """
        dependency = self._get_dependency()
        measured = self.results.keys()
        self.update_dependency(measured, dependency)
        not_measured = set(self.__input_nodes)
        for cmd in self.__seq:
            if cmd.kind == CommandKind.N and cmd.node not in self.output_nodes:
                not_measured |= {cmd.node}
        depth = 0
        l_k: dict[int, set[int]] = {}
        k = 0
        while not_measured:
            l_k[k] = set()
            for i in not_measured:
                if not dependency[i]:
                    l_k[k] |= {i}
            self.update_dependency(l_k[k], dependency)
            not_measured -= l_k[k]
            k += 1
            depth = k
        return depth, l_k

    def _measurement_order_depth(self) -> list[int]:
        """Obtain a measurement order which reduces the depth of a pattern.

        Returns
        -------
        meas_order: list of int
            optimal measurement order for parallel computing
        """
        d, l_k = self.get_layers()
        meas_order: list[int] = []
        for i in range(d):
            meas_order.extend(l_k[i])
        return meas_order

    @staticmethod
    def connected_edges(node: int, edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
        """Search not activated edges connected to the specified node.

        Returns
        -------
        connected: set of tuple
                set of connected edges
        """
        connected = set()
        for edge in edges:
            if edge[0] == node or edge[1] == node:
                connected |= {edge}
        return connected

    def _measurement_order_space(self) -> list[int]:
        """Determine measurement order that heuristically optimises the max_space of a pattern.

        Returns
        -------
        meas_order: list of int
            sub-optimal measurement order for classical simulation
        """
        # NOTE calling get_graph
        nodes_list, edges_list = self.get_graph()
        nodes = set(nodes_list)
        edges = set(edges_list)
        not_measured = nodes - set(self.output_nodes)
        dependency = self._get_dependency()
        self.update_dependency(self.results.keys(), dependency)
        meas_order = []
        removable_edges = set()
        while not_measured:
            min_edges = len(nodes) + 1
            next_node = -1
            for i in not_measured:
                if not dependency[i]:
                    connected_edges = self.connected_edges(i, edges)
                    if min_edges > len(connected_edges):
                        min_edges = len(connected_edges)
                        next_node = i
                        removable_edges = connected_edges
            if not (next_node > -1):
                print(next_node)
            assert next_node > -1
            meas_order.append(next_node)
            self.update_dependency({next_node}, dependency)
            not_measured -= {next_node}
            edges -= removable_edges
        return meas_order

    def get_measurement_order_from_flow(self) -> list[int] | None:
        """Return a measurement order generated from flow. If a graph has flow, the minimum 'max_space' of a pattern is guaranteed to width+1.

        Returns
        -------
        meas_order: list of int
            measurement order
        """
        # NOTE calling get_graph
        nodes_list, edges_list = self.get_graph()
        g: nx.Graph[int] = nx.Graph()
        g.add_nodes_from(nodes_list)
        g.add_edges_from(edges_list)
        vin = set(self.input_nodes) if self.input_nodes is not None else set()
        vout = set(self.output_nodes)
        meas_planes = self.get_meas_plane()
        f, l_k = find_flow(g, vin, vout, meas_planes=meas_planes)
        if f is None:
            return None
        depth, layer = get_layers(l_k)
        meas_order: list[int] = []
        for i in range(depth):
            k = depth - i
            nodes = layer[k]
            meas_order += nodes  # NOTE this is list concatenation
        return meas_order

    def get_measurement_order_from_gflow(self) -> list[int]:
        """Return a list containing the node indices, in the order of measurements which can be performed with minimum depth.

        Returns
        -------
        meas_order : list of int
            measurement order
        """
        # NOTE calling get_graph
        nodes, edges = self.get_graph()
        g: nx.Graph[int] = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        isolated = list(nx.isolates(g))
        if isolated:
            raise ValueError("The input graph must be connected")
        vin = set(self.input_nodes) if self.input_nodes is not None else set()
        vout = set(self.output_nodes)
        meas_planes = self.get_meas_plane()
        flow, l_k = find_gflow(g, vin, vout, meas_planes=meas_planes)
        if flow is None or l_k is None:  # We check both to avoid typing issues with `get_layers`.
            raise ValueError("No gflow found")
        k, layers = get_layers(l_k)
        meas_order: list[int] = []
        while k > 0:
            meas_order.extend(layers[k])
            k -= 1
        return meas_order

    def sort_measurement_commands(self, meas_order: list[int]) -> list[command.M]:
        """Convert measurement order to sequence of measurement commands.

        Parameters
        ----------
        meas_order: list of int
            optimal measurement order.

        Returns
        -------
        meas_cmds: list of command
            sorted measurement commands
        """
        meas_cmds = []
        for i in meas_order:
            target = 0
            while True:
                cmd = self.__seq[target]
                if cmd.kind == CommandKind.M and (cmd.node == i):
                    meas_cmds.append(cmd)
                    break
                target += 1
        return meas_cmds

    def get_measurement_commands(self) -> list[command.M]:
        """Return the list containing the measurement commands, in the order of measurements.

        Returns
        -------
        meas_cmds : list
            list of measurement commands in the order of meaurements
        """
        if not self.is_standard():
            self.standardize()
        meas_cmds = []
        ind = self._find_op_to_be_moved(CommandKind.M)
        if ind is None:
            return []
        while True:
            try:
                cmd = self.__seq[ind]
            except IndexError:
                break
            if cmd.kind != CommandKind.M:
                break
            meas_cmds.append(cmd)
            ind += 1
        return meas_cmds

    def get_meas_plane(self) -> dict[int, Plane]:
        """Get measurement plane from the pattern.

        Returns
        -------
        meas_plane: dict of graphix.pauli.Plane
            list of planes representing measurement plane for each node.
        """
        meas_plane = {}
        for cmd in self.__seq:
            if cmd.kind == CommandKind.M:
                meas_plane[cmd.node] = cmd.plane
        return meas_plane

    def get_angles(self) -> dict[int, ExpressionOrFloat]:
        """Get measurement angles of the pattern.

        Returns
        -------
        angles : dict
            measurement angles of the each node.
        """
        angles = {}
        for cmd in self.__seq:
            if cmd.kind == CommandKind.M:
                angles[cmd.node] = cmd.angle
        return angles

    def get_max_degree(self) -> int:
        """Get max degree of a pattern.

        Returns
        -------
        max_degree : int
            max degree of a pattern
        """
        nodes, edges = self.get_graph()
        g: nx.Graph[int] = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        degree = g.degree()
        assert isinstance(degree, nx.classes.reportviews.DiDegreeView)
        return int(max(list(dict(degree).values())))

    def get_graph(self) -> tuple[list[int], list[tuple[int, int]]]:
        """Return the list of nodes and edges from the command sequence, extracted from 'N' and 'E' commands.

        Returns
        -------
        node_list : list
            list of node indices.
        edge_list : list
            list of tuples (i,j) specifying edges
        """
        # We rely on the fact that self.input_nodes returns a copy:
        # self.input_nodes is equivalent to list(self.__input_nodes)
        node_list, edge_list = self.input_nodes, []
        for cmd in self.__seq:
            if cmd.kind == CommandKind.N:
                assert cmd.node not in node_list
                node_list.append(cmd.node)
            elif cmd.kind == CommandKind.E:
                edge_list.append(cmd.nodes)
        return node_list, edge_list

    def get_isolated_nodes(self) -> set[int]:
        """Get isolated nodes.

        Returns
        -------
        isolated_nodes : set of int
            set of the isolated nodes
        """
        nodes, edges = self.get_graph()
        node_set = set(nodes)
        connected_node_set = set()
        for edge in edges:
            connected_node_set |= set(edge)
        return node_set - connected_node_set

    def get_vops(self, conj: bool = False, include_identity: bool = False) -> dict[int, Clifford]:
        """Get local-Clifford decorations from measurement or Clifford commands.

        Parameters
        ----------
            conj (False) : bool, optional
                Apply conjugations to all local Clifford operators.
            include_identity (False) : bool, optional
                Whether or not to include identity gates in the output

        Returns
        -------
            vops : dict
        """
        vops = {}
        for cmd in self.__seq:
            if cmd.kind == CommandKind.M:
                if include_identity:
                    vops[cmd.node] = Clifford.I
            elif cmd.kind == CommandKind.C:
                if cmd.clifford == Clifford.I:
                    if include_identity:
                        vops[cmd.node] = cmd.clifford
                elif conj:
                    vops[cmd.node] = cmd.clifford.conj
                else:
                    vops[cmd.node] = cmd.clifford
        for out in self.output_nodes:
            if out not in vops and include_identity:
                vops[out] = Clifford.I
        return vops

    def connected_nodes(self, node: int, prepared: set[int] | None = None) -> list[int]:
        """Find nodes that are connected to a specified node.

        These nodes must be in the statevector when the specified
        node is measured, to ensure correct computation.
        If connected nodes already exist in the statevector (prepared),
        then they will be ignored as they do not need to be prepared again.

        Parameters
        ----------
        node : int
            node index
        prepared : list
            list of node indices, which are to be ignored

        Returns
        -------
        node_list : list
            list of nodes that are entangled with specified node
        """
        if not self.is_standard():
            self.standardize()
        if prepared is None:
            prepared = set()
        node_list = []
        ind = self._find_op_to_be_moved(CommandKind.E)
        if ind is not None:  # end -> 'node' is isolated
            cmd = self.__seq[ind]
            while cmd.kind == CommandKind.E:
                if cmd.nodes[0] == node:
                    if cmd.nodes[1] not in prepared:
                        node_list.append(cmd.nodes[1])
                elif cmd.nodes[1] == node and cmd.nodes[0] not in prepared:
                    node_list.append(cmd.nodes[0])
                ind += 1
                cmd = self.__seq[ind]
        return node_list

    def correction_commands(self) -> list[command.X | command.Z]:
        """Return the list of byproduct correction commands."""
        assert self.is_standard()
        # Use of `==` here for mypy
        return [seqi for seqi in self.__seq if seqi.kind == CommandKind.X or seqi.kind == CommandKind.Z]  # noqa: PLR1714

    def parallelize_pattern(self) -> None:
        """Optimize the pattern to reduce the depth of the computation by gathering measurement commands that can be performed simultaneously.

        This optimized pattern runs efficiently on GPUs and quantum hardwares with
        depth (e.g. coherence time) limitations.
        """
        if not self.is_standard():
            self.standardize()
        meas_order = self._measurement_order_depth()
        self._reorder_pattern(self.sort_measurement_commands(meas_order))

    def minimize_space(self) -> None:
        """Optimize the pattern to minimize the max_space property of the pattern.

        The optimized pattern has significantly
        reduced space requirement (memory space for classical simulation,
        and maximum simultaneously prepared qubits for quantum hardwares).
        """
        if not self.is_standard():
            self.standardize()
        meas_order = None
        if not self._pauli_preprocessed:
            meas_order = self.get_measurement_order_from_flow()
        if meas_order is None:
            meas_order = self._measurement_order_space()
        self._reorder_pattern(self.sort_measurement_commands(meas_order))

    def _reorder_pattern(self, meas_commands: list[command.M]) -> None:
        """Reorder the command sequence.

        Parameters
        ----------
        meas_commands : list of command
            list of measurement ('M') commands
        """
        prepared = set(self.input_nodes)
        measured: set[int] = set()
        new: list[Command] = []
        cmd: Command

        for cmd in meas_commands:
            node = cmd.node
            if node not in prepared:
                new.append(command.N(node=node))
                prepared.add(node)
            node_list = self.connected_nodes(node, measured)
            for add_node in node_list:
                if add_node not in prepared:
                    new.append(command.N(node=add_node))
                    prepared.add(add_node)
                new.append(command.E(nodes=(node, add_node)))
            new.append(cmd)
            measured.add(node)

        # add isolated nodes
        for cmd in self.__seq:
            if cmd.kind == CommandKind.N and cmd.node not in prepared:
                new.append(command.N(node=cmd.node))
            elif (
                (cmd.kind == CommandKind.E and all(node in self.output_nodes for node in cmd.nodes))
                or cmd.kind == CommandKind.C
                or cmd.kind in {CommandKind.Z, CommandKind.X}
            ):
                new.append(cmd)

        self.__seq = new

    def max_space(self) -> int:
        """Compute the maximum number of nodes that must be present in the graph (graph space) during the execution of the pattern.

        For statevector simulation, this is equivalent to the maximum memory
        needed for classical simulation.

        Returns
        -------
        n_nodes : int
            max number of nodes present in the graph during pattern execution.
        """
        nodes = len(self.input_nodes)
        max_nodes = nodes
        for cmd in self.__seq:
            if cmd.kind == CommandKind.N:
                nodes += 1
            elif cmd.kind == CommandKind.M:
                nodes -= 1
            max_nodes = max(nodes, max_nodes)
        return max_nodes

    def space_list(self) -> list[int]:
        """Return the list of the number of nodes present in the graph (space) during each step of execution of the pattern (for N and M commands).

        Returns
        -------
        N_list : list
            time evolution of 'space' at each 'N' and 'M' commands of pattern.
        """
        nodes = 0
        n_list = []
        for cmd in self.__seq:
            if cmd.kind == CommandKind.N:
                nodes += 1
                n_list.append(nodes)
            elif cmd.kind == CommandKind.M:
                nodes -= 1
                n_list.append(nodes)
        return n_list

    def simulate_pattern(
        self,
        backend: Backend[_StateT_co] | str = "statevector",
        input_state: Data = BasicStates.PLUS,
        rng: Generator | None = None,
        **kwargs: Any,
    ) -> BackendState:
        """Simulate the execution of the pattern by using :class:`graphix.simulator.PatternSimulator`.

        Available backend: ['statevector', 'densitymatrix', 'tensornetwork']

        Parameters
        ----------
        backend : str
            optional parameter to select simulator backend.
        rng: Generator, optional
            Random-number generator for measurements.
            This generator is used only in case of random branch selection
            (see :class:`RandomBranchSelector`).
        kwargs: keyword args for specified backend.

        Returns
        -------
        state :
            quantum state representation for the selected backend.

        .. seealso:: :class:`graphix.simulator.PatternSimulator`
        """
        sim = PatternSimulator(self, backend=backend, **kwargs)
        sim.run(input_state, rng=rng)
        return sim.backend.state

    def perform_pauli_measurements(self, leave_input: bool = False, ignore_pauli_with_deps: bool = False) -> None:
        """Perform Pauli measurements in the pattern using efficient stabilizer simulator.

        Parameters
        ----------
        leave_input : bool
            Optional (*False* by default).
            If *True*, measurements on input nodes are preserved as-is in the pattern.
        ignore_pauli_with_deps : bool
            Optional (*False* by default).
            If *True*, Pauli measurements with domains depending on other measures are preserved as-is in the pattern.
            If *False*, all Pauli measurements are preprocessed. Formally, measurements are swapped so that all Pauli measurements are applied first, and domains are updated accordingly.

        .. seealso:: :func:`measure_pauli`

        """
        if not ignore_pauli_with_deps:
            self.move_pauli_measurements_to_the_front()
        measure_pauli(self, leave_input, copy=False)

    def draw_graph(
        self,
        flow_from_pattern: bool = True,
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurement_planes: bool = False,
        show_loop: bool = True,
        node_distance: tuple[int, int] = (1, 1),
        figsize: tuple[int, int] | None = None,
        save: bool = False,
        filename: str | None = None,
    ) -> None:
        """Visualize the underlying graph of the pattern with flow or gflow structure.

        Parameters
        ----------
        flow_from_pattern : bool
            If True, the command sequence of the pattern is used to derive flow or gflow structure. If False, only the underlying graph is used.
        show_pauli_measurement : bool
            If True, the nodes with Pauli measurement angles are colored light blue.
        show_local_clifford : bool
            If True, indexes of the local Clifford operator are displayed adjacent to the nodes.
        show_measurement_planes : bool
            If True, measurement planes are displayed adjacent to the nodes.
        show_loop : bool
            whether or not to show loops for graphs with gflow. defaulted to True.
        node_distance : tuple
            Distance multiplication factor between nodes for x and y directions.
        figsize : tuple
            Figure size of the plot.
        save : bool
            If True, the plot is saved as a png file.
        filename : str
            Filename of the saved plot.
        """
        nodes, edges = self.get_graph()
        g: nx.Graph[int] = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        vin = self.input_nodes if self.input_nodes is not None else []
        vout = self.output_nodes
        meas_planes = self.get_meas_plane()
        meas_angles = self.get_angles()
        local_clifford = self.get_vops()

        vis = GraphVisualizer(g, vin, vout, meas_planes, meas_angles, local_clifford)

        if flow_from_pattern:
            vis.visualize_from_pattern(
                pattern=self.copy(),
                show_pauli_measurement=show_pauli_measurement,
                show_local_clifford=show_local_clifford,
                show_measurement_planes=show_measurement_planes,
                show_loop=show_loop,
                node_distance=node_distance,
                figsize=figsize,
                save=save,
                filename=filename,
            )
        else:
            vis.visualize(
                show_pauli_measurement=show_pauli_measurement,
                show_local_clifford=show_local_clifford,
                show_measurement_planes=show_measurement_planes,
                show_loop=show_loop,
                node_distance=node_distance,
                figsize=figsize,
                save=save,
                filename=filename,
            )

    def to_qasm3(self, filename: Path | str) -> None:
        """Export measurement pattern to OpenQASM 3.0 file.

        Parameters
        ----------
        filename : Path | str
            file name to export to. example: "filename.qasm"
        """
        with Path(filename).with_suffix(".qasm").open("w", encoding="utf-8") as file:
            file.write("// generated by graphix\n")
            file.write("OPENQASM 3;\n")
            file.write('include "stdgates.inc";\n')
            file.write("\n")
            if self.results != {}:
                for i in self.results:
                    res = self.results[i]
                    file.write("// measurement result of qubit q" + str(i) + "\n")
                    file.write("bit c" + str(i) + " = " + str(res) + ";\n")
                    file.write("\n")
            for cmd in self.__seq:
                file.writelines(cmd_to_qasm3(cmd))

    def is_parameterized(self) -> bool:
        """
        Return `True` if there is at least one measurement angle that is not just an instance of `SupportsFloat`.

        A parameterized pattern is a pattern where at least one
        measurement angle is an expression that is not a number,
        typically an instance of `sympy.Expr` (but we don't force to
        choose `sympy` here).

        """
        return any(not isinstance(cmd.angle, SupportsFloat) for cmd in self if cmd.kind == command.CommandKind.M)

    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Pattern:
        """Return a copy of the pattern where all occurrences of the given variable in measurement angles are substituted by the given value."""
        result = self.copy()
        for cmd in result:
            if cmd.kind == command.CommandKind.M:
                cmd.angle = parameter.subs(cmd.angle, variable, substitute)
        return result

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> Pattern:
        """Return a copy of the pattern where all occurrences of the given keys in measurement angles are substituted by the given values in parallel."""
        result = self.copy()
        for cmd in result:
            if cmd.kind == command.CommandKind.M:
                cmd.angle = parameter.xreplace(cmd.angle, assignment)
        return result

    def copy(self) -> Pattern:
        """Return a copy of the pattern."""
        result = self.__new__(self.__class__)
        result.__seq = [copy.copy(cmd) for cmd in self.__seq]
        result.__input_nodes = self.__input_nodes.copy()
        result.__output_nodes = self.__output_nodes.copy()
        result.__n_node = self.__n_node
        result._pauli_preprocessed = self._pauli_preprocessed
        result.results = self.results.copy()
        return result

    def move_pauli_measurements_to_the_front(self, leave_nodes: set[int] | None = None) -> None:
        """Move all the Pauli measurements to the front of the sequence (except nodes in `leave_nodes`)."""
        if leave_nodes is None:
            leave_nodes = set()
        self.standardize()
        pauli_nodes = {}
        shift_domains: dict[int, set[int]] = {}

        def expand_domain(domain: set[int]) -> None:
            """Merge previously shifted domains into ``domain``.

            Parameters
            ----------
            domain : set[int]
                Domain to update with any accumulated shift information.
            """
            for node in domain & shift_domains.keys():
                domain ^= shift_domains[node]

        for cmd in self:
            # Use of == for mypy
            if cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
                expand_domain(cmd.domain)
            if cmd.kind == CommandKind.M:
                expand_domain(cmd.s_domain)
                expand_domain(cmd.t_domain)
                pm = PauliMeasurement.try_from(
                    cmd.plane, cmd.angle
                )  # None returned if the measurement is not in Pauli basis
                if pm is not None and cmd.node not in leave_nodes:
                    if pm.axis == Axis.X:
                        # M^X X^s Z^t = M^{XY,0} X^s Z^t
                        #             = M^{XY,(-1)^s·0+tπ}
                        #             = S^t M^X
                        # M^{-X} X^s Z^t = M^{XY,π} X^s Z^t
                        #                = M^{XY,(-1)^s·π+tπ}
                        #                = S^t M^{-X}
                        shift_domains[cmd.node] = cmd.t_domain
                    elif pm.axis == Axis.Y:
                        # M^Y X^s Z^t = M^{XY,π/2} X^s Z^t
                        #             = M^{XY,(-1)^s·π/2+tπ}
                        #             = M^{XY,π/2+(s+t)π}      (since -π/2 = π/2 - π ≡ π/2 + π (mod 2π))
                        #             = S^{s+t} M^Y
                        # M^{-Y} X^s Z^t = M^{XY,-π/2} X^s Z^t
                        #                = M^{XY,(-1)^s·(-π/2)+tπ}
                        #                = M^{XY,-π/2+(s+t)π}  (since π/2 = -π/2 + π)
                        #                = S^{s+t} M^{-Y}
                        shift_domains[cmd.node] = cmd.s_domain ^ cmd.t_domain
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
                        shift_domains[cmd.node] = cmd.s_domain
                    else:
                        assert_never(pm.axis)
                    cmd.s_domain = set()
                    cmd.t_domain = set()
                    pauli_nodes[cmd.node] = cmd

        # Create a new sequence with all Pauli nodes to the front
        new_seq: list[Command] = []
        pauli_nodes_inserted = False
        for cmd in self:
            if cmd.kind == CommandKind.M:
                if cmd.node not in pauli_nodes:
                    if not pauli_nodes_inserted:
                        new_seq.extend(pauli_nodes.values())
                        pauli_nodes_inserted = True
                    new_seq.append(cmd)
            else:
                new_seq.append(cmd)
        if not pauli_nodes_inserted:
            new_seq.extend(pauli_nodes.values())
        self.__seq = new_seq


def measure_pauli(pattern: Pattern, leave_input: bool, copy: bool = False) -> Pattern:
    """Perform Pauli measurement of a pattern by fast graph state simulator.

    Uses the decorated-graph method implemented in graphix.graphsim to perform
    the measurements in Pauli bases, and then sort remaining nodes back into
    pattern together with Clifford commands.

    TODO: non-XY plane measurements in original pattern

    Parameters
    ----------
    pattern : graphix.pattern.Pattern object
    leave_input : bool
        True: input nodes will not be removed
        False: all the nodes measured in Pauli bases will be removed
    copy : bool
        True: changes will be applied to new copied object and will be returned
        False: changes will be applied to the supplied Pattern object

    Returns
    -------
    new_pattern : graphix.Pattern object
        pattern with Pauli measurement removed.
        only returned if copy argument is True.


    .. seealso:: :class:`graphix.graphsim.GraphState`
    """
    if not pattern.is_standard():
        pattern.standardize()
    nodes, edges = pattern.get_graph()
    vop_init = pattern.get_vops(conj=False)
    graph_state = GraphState(nodes=nodes, edges=edges, vops=vop_init)
    results: dict[int, Outcome] = {}
    to_measure, non_pauli_meas = pauli_nodes(pattern, leave_input)
    if not leave_input and len(list(set(pattern.input_nodes) & {i[0].node for i in to_measure})) > 0:
        new_inputs = []
    else:
        new_inputs = pattern.input_nodes
    for cmd in to_measure:
        pattern_cmd = cmd[0]
        measurement_basis = cmd[1]
        # extract signals for adaptive angle.
        s_signal = 0
        t_signal = 0
        if measurement_basis.axis == Axis.X:  # X measurement is not affected by s_signal
            t_signal = sum(results[j] for j in pattern_cmd.t_domain)
        elif measurement_basis.axis == Axis.Y:
            s_signal = sum(results[j] for j in pattern_cmd.s_domain)
            t_signal = sum(results[j] for j in pattern_cmd.t_domain)
        elif measurement_basis.axis == Axis.Z:  # Z measurement is not affected by t_signal
            s_signal = sum(results[j] for j in pattern_cmd.s_domain)
        else:
            assert_never(measurement_basis.axis)

        if int(s_signal % 2) == 1:  # equivalent to X byproduct
            graph_state.h(pattern_cmd.node)
            graph_state.z(pattern_cmd.node)
            graph_state.h(pattern_cmd.node)
        if int(t_signal % 2) == 1:  # equivalent to Z byproduct
            graph_state.z(pattern_cmd.node)
        basis = measurement_basis
        if basis.axis == Axis.X:
            measure = graph_state.measure_x
        elif basis.axis == Axis.Y:
            measure = graph_state.measure_y
        elif basis.axis == Axis.Z:
            measure = graph_state.measure_z
        else:
            assert_never(basis.axis)
        if basis.sign == Sign.PLUS:
            results[pattern_cmd.node] = measure(pattern_cmd.node, choice=0)
        else:
            results[pattern_cmd.node] = 0 if measure(pattern_cmd.node, choice=1) else 1

    # measure (remove) isolated nodes. if they aren't Pauli measurements,
    # measuring one of the results with probability of 1 should not occur as was possible above for Pauli measurements,
    # which means we can just choose s=0. We should not remove output nodes even if isolated.
    isolates = graph_state.get_isolates()
    for node in non_pauli_meas:
        if (node in isolates) and (node not in pattern.output_nodes):
            graph_state.remove_node(node)
            results[node] = 0

    # update command sequence
    vops = graph_state.get_vops()
    new_seq: list[Command] = []
    new_seq.extend(command.N(node=index) for index in set(graph_state.nodes) - set(new_inputs))
    new_seq.extend(command.E(nodes=edge) for edge in graph_state.edges)
    new_seq.extend(
        cmd.clifford(Clifford(vops[cmd.node]))
        for cmd in pattern
        if cmd.kind == CommandKind.M and cmd.node in graph_state.nodes
    )
    new_seq.extend(
        command.C(node=index, clifford=Clifford(vops[index]))
        for index in pattern.output_nodes
        if vops[index] != Clifford.I
    )
    new_seq.extend(cmd for cmd in pattern if cmd.kind in {CommandKind.X, CommandKind.Z})

    pat = Pattern() if copy else pattern

    output_nodes = deepcopy(pattern.output_nodes)
    pat.replace(new_seq, input_nodes=new_inputs)
    pat.reorder_output_nodes(output_nodes)
    assert pat.n_node == len(graph_state.nodes)
    pat.results = results
    pat._pauli_preprocessed = True
    return pat


def pauli_nodes(pattern: Pattern, leave_input: bool) -> tuple[list[tuple[command.M, PauliMeasurement]], set[int]]:
    """Return the list of measurement commands that are in Pauli bases and that are not dependent on any non-Pauli measurements.

    Parameters
    ----------
    pattern : graphix.Pattern object
    leave_input : bool

    Returns
    -------
    pauli_node : list
        list of measures
    non_pauli_nodes : set[int]
    """
    if not pattern.is_standard():
        pattern.standardize()
    m_commands = pattern.get_measurement_commands()
    pauli_node: list[tuple[command.M, PauliMeasurement]] = []
    # Nodes that are non-Pauli measured, or pauli measured but depends on pauli measurement
    non_pauli_node: set[int] = set()
    for cmd in m_commands:
        pm = PauliMeasurement.try_from(cmd.plane, cmd.angle)  # None returned if the measurement is not in Pauli basis
        if pm is not None and (cmd.node not in pattern.input_nodes or not leave_input):
            # Pauli measurement to be removed
            if pm.axis == Axis.X:
                if cmd.t_domain & non_pauli_node:  # cmd depend on non-Pauli measurement
                    non_pauli_node.add(cmd.node)
                else:
                    pauli_node.append((cmd, pm))
            elif pm.axis == Axis.Y:
                if (cmd.s_domain | cmd.t_domain) & non_pauli_node:  # cmd depend on non-Pauli measurement
                    non_pauli_node.add(cmd.node)
                else:
                    pauli_node.append((cmd, pm))
            elif pm.axis == Axis.Z:
                if cmd.s_domain & non_pauli_node:  # cmd depend on non-Pauli measurement
                    non_pauli_node.add(cmd.node)
                else:
                    pauli_node.append((cmd, pm))
            else:
                raise ValueError("Unknown Pauli measurement basis")
        else:
            non_pauli_node.add(cmd.node)
    return pauli_node, non_pauli_node


def cmd_to_qasm3(cmd: Command) -> Iterator[str]:
    """Convert a command in the pattern into OpenQASM 3.0 statement.

    Parameter
    ---------
    cmd : list
        command [type:str, node:int, attr]

    Yields
    ------
    string
        translated pattern commands in OpenQASM 3.0 language

    """
    if cmd.kind == CommandKind.N:
        qubit = cmd.node
        yield "// prepare qubit q" + str(qubit) + "\n"
        yield "qubit q" + str(qubit) + ";\n"
        yield "h q" + str(qubit) + ";\n"
        yield "\n"

    elif cmd.kind == CommandKind.E:
        qubits = cmd.nodes
        yield "// entangle qubit q" + str(qubits[0]) + " and q" + str(qubits[1]) + "\n"
        yield "cz q" + str(qubits[0]) + ", q" + str(qubits[1]) + ";\n"
        yield "\n"

    elif cmd.kind == CommandKind.M:
        qubit = cmd.node
        plane = cmd.plane
        alpha = cmd.angle
        sdomain = cmd.s_domain
        tdomain = cmd.t_domain
        yield "// measure qubit q" + str(qubit) + "\n"
        yield "bit c" + str(qubit) + ";\n"
        yield "float theta" + str(qubit) + " = 0;\n"
        if plane == Plane.XY:
            if sdomain:
                yield "int s" + str(qubit) + " = 0;\n"
                for sid in sdomain:
                    yield "s" + str(qubit) + " += c" + str(sid) + ";\n"
                yield "theta" + str(qubit) + " += (-1)**(s" + str(qubit) + " % 2) * (" + str(alpha) + " * pi);\n"
            if tdomain:
                yield "int t" + str(qubit) + " = 0;\n"
                for tid in tdomain:
                    yield "t" + str(qubit) + " += c" + str(tid) + ";\n"
                yield "theta" + str(qubit) + " += t" + str(qubit) + " * pi;\n"
            yield "p(-theta" + str(qubit) + ") q" + str(qubit) + ";\n"
            yield "h q" + str(qubit) + ";\n"
            yield "c" + str(qubit) + " = measure q" + str(qubit) + ";\n"
            yield "h q" + str(qubit) + ";\n"
            yield "p(theta" + str(qubit) + ") q" + str(qubit) + ";\n"
            yield "\n"

    # Use of == for mypy
    elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
        qubit = cmd.node
        sdomain = cmd.domain
        yield "// byproduct correction on qubit q" + str(qubit) + "\n"
        yield "int s" + str(qubit) + " = 0;\n"
        for sid in sdomain:
            yield "s" + str(qubit) + " += c" + str(sid) + ";\n"
        yield "if(s" + str(qubit) + " % 2 == 1){\n"
        if cmd.kind == CommandKind.X:
            yield "\t x q" + str(qubit) + ";\n}\n"
        else:
            yield "\t z q" + str(qubit) + ";\n}\n"
        yield "\n"

    elif cmd.kind == CommandKind.C:
        qubit = cmd.node
        yield "// Clifford operations on qubit q" + str(qubit) + "\n"
        for op in cmd.clifford.qasm3:
            yield str(op) + " q" + str(qubit) + ";\n"
        yield "\n"

    else:
        raise ValueError(f"invalid command {cmd}")


def assert_permutation(original: list[int], user: list[int]) -> None:
    """Check that the provided `user` node list is a permutation from `original`."""
    node_set = set(user)
    if node_set != set(original):
        raise ValueError(f"{node_set} != {set(original)}")
    for node in user:
        if node in node_set:
            node_set.remove(node)
        else:
            raise ValueError(f"{node} appears twice")


@dataclass
class ExtractedSignal:
    """Return data structure for `extract_signal`."""

    s_domain: set[int]
    "New `s_domain` for the measure command."

    t_domain: set[int]
    "New `t_domain` for the measure command."

    signal: set[int]
    "Domain for the shift command."


def extract_signal(plane: Plane, s_domain: set[int], t_domain: set[int]) -> ExtractedSignal:
    """Extract signal from domains."""
    if plane == Plane.XY:
        return ExtractedSignal(s_domain=s_domain, t_domain=set(), signal=t_domain)
    if plane == Plane.XZ:
        return ExtractedSignal(s_domain=set(), t_domain=s_domain ^ t_domain, signal=s_domain)
    if plane == Plane.YZ:
        return ExtractedSignal(s_domain=set(), t_domain=t_domain, signal=s_domain)
    assert_never(plane)


def shift_outcomes(outcomes: dict[int, Outcome], signal_dict: dict[int, set[int]]) -> dict[int, Outcome]:
    """Update outcomes with shifted signals.

    Shifted signals (as returned by the method
    :func:`Pattern.shift_signals`) affect classical outputs
    (measurements) while leaving the quantum state invariant.

    This method updates the given `outcomes` by swapping the
    measurements affected by signals. This can be used either to
    transform the value of :data:`Pattern.results` into measurements
    observed in the unshifted pattern, or vice versa.

    Parameters
    ----------
    outcomes : dict[int, int]
        Classical outputs.
    signal_dict : dict[int, set[int]]
        For each node, the signal that has been shifted
        (as returned by :func:`Pattern.shift_signals`).

    Returns
    -------
    shifted_outcomes : dict[int, int]
        Classical outputs updated with shifted signals.

    """
    return {
        node: toggle_outcome(outcome) if sum(outcomes[i] for i in signal_dict.get(node, [])) % 2 == 1 else outcome
        for node, outcome in outcomes.items()
    }
