"""MBQC pattern according to Measurement Calculus
ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass

import networkx as nx
import numpy as np
import typing_extensions

import graphix.clifford
import graphix.pauli
from graphix import command
from graphix.clifford import CLIFFORD_CONJ, CLIFFORD_MEASURE, CLIFFORD_MUL, CLIFFORD_TO_QASM3
from graphix.command import CommandKind
from graphix.device_interface import PatternRunner
from graphix.gflow import find_flow, find_gflow, get_layers
from graphix.graphsim.graphstate import GraphState
from graphix.pauli import Plane
from graphix.simulator import PatternSimulator
from graphix.visualization import GraphVisualizer


class NodeAlreadyPrepared(Exception):
    def __init__(self, node: int):
        self.__node = node

    @property
    def node(self):
        return self.__node

    @property
    def __str__(self) -> str:
        return f"Node already prepared: {self.__node}"


class Pattern:
    """
    MBQC pattern class

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
    Nnode : int
        total number of nodes in the resource state
    """

    def __init__(self, input_nodes: list[int] | None = None) -> None:
        """
        :param input_nodes:  optional, list of input qubits
        """
        if input_nodes is None:
            input_nodes = []
        self.results = {}  # measurement results from the graph state simulator
        self.__input_nodes = list(input_nodes)  # input nodes (list() makes our own copy of the list)
        self.__Nnode = len(input_nodes)  # total number of nodes in the graph state
        self._pauli_preprocessed = False  # flag for `measure_pauli` preprocessing completion

        self.__seq: list[command.Command] = []
        # output nodes are initially input nodes, since none are measured yet
        self.__output_nodes = list(input_nodes)

    def add(self, cmd: command.Command):
        """add command to the end of the pattern.
        an MBQC command is specified by a list of [type, node, attr], where

            type : 'N', 'M', 'E', 'X', 'Z', 'S' or 'C'
            nodes : int for 'N', 'M', 'X', 'Z', 'S', 'C' commands
            nodes : tuple (i, j) for 'E' command
            attr for N (node preparation):
                none
            attr for E (entanglement):
                none
            attr for M (measurement):
                meas_plane : 'XY','YZ' or 'XZ'
                angle : float, in radian / pi
                s_domain : list
                t_domain : list
            attr for X:
                signal_domain : list
            attr for Z:
                signal_domain : list
            attr for S:
                signal_domain : list
            attr for C:
                clifford_index : int

        Parameters
        ----------
        cmd : list
            MBQC command.
        """
        if cmd.kind == command.CommandKind.N:
            if cmd.node in self.__output_nodes:
                raise NodeAlreadyPrepared(cmd.node)
            self.__Nnode += 1
            self.__output_nodes.append(cmd.node)
        elif cmd.kind == command.CommandKind.M:
            self.__output_nodes.remove(cmd.node)
        self.__seq.append(cmd)

    def extend(self, cmds: list[command.Command]):
        """Add a list of commands.

        :param cmds: list of commands
        """
        for cmd in cmds:
            self.add(cmd)

    def clear(self):
        """Clear the sequence of pattern commands."""
        self.__Nnode = len(self.__input_nodes)
        self.__seq = []
        self.__output_nodes = list(self.__input_nodes)

    def replace(self, cmds: list[command.Command], input_nodes=None):
        """Replace pattern with a given sequence of pattern commands.

        :param cmds: list of commands

        :param input_nodes:  optional, list of input qubits
        (by default, keep the same input nodes as before)
        """
        if input_nodes is not None:
            self.__input_nodes = list(input_nodes)
        self.clear()
        self.extend(cmds)

    @property
    def input_nodes(self):
        """list of input nodes"""
        return list(self.__input_nodes)  # copy for preventing modification

    @property
    def output_nodes(self):
        """list of all nodes that are either `input_nodes` or prepared with
        `N` commands and that have not been measured with an `M` command
        """
        return list(self.__output_nodes)  # copy for preventing modification

    def __len__(self):
        """length of command sequence"""
        return len(self.__seq)

    def __iter__(self):
        """iterate over commands"""
        return iter(self.__seq)

    def __getitem__(self, index):
        return self.__seq[index]

    @property
    def Nnode(self):
        """count of nodes that are either `input_nodes` or prepared with `N` commands"""
        return self.__Nnode

    def reorder_output_nodes(self, output_nodes: list[int]):
        """arrange the order of output_nodes.

        Parameters
        ----------
        output_nodes: list of int
            output nodes order determined by user. each index corresponds to that of logical qubits.
        """
        output_nodes = list(output_nodes)  # make our own copy (allow iterators to be passed)
        assert_permutation(self.__output_nodes, output_nodes)
        self.__output_nodes = output_nodes

    def reorder_input_nodes(self, input_nodes: list[int]):
        """arrange the order of input_nodes.

        Parameters
        ----------
        input_nodes: list of int
            input nodes order determined by user. each index corresponds to that of logical qubits.
        """
        assert_permutation(self.__input_nodes, input_nodes)
        self.__input_nodes = list(input_nodes)

    def __repr__(self):
        return (
            f"graphix.pattern.Pattern object with {len(self.__seq)} commands and {len(self.output_nodes)} output qubits"
        )

    def __eq__(self, other: Pattern) -> bool:
        return (
            self.__seq == other.__seq
            and self.input_nodes == other.input_nodes
            and self.output_nodes == other.output_nodes
        )

    def print_pattern(self, lim=40, target: list[command.CommandKind] | None = None) -> None:
        """print the pattern sequence (Pattern.seq).

        Parameters
        ----------
        lim: int, optional
            maximum number of commands to show
        target : list of command.CommandKind, optional
            show only specified commands, e.g. [CommandKind.M, CommandKind.X, CommandKind.Z]
        """
        if len(self.__seq) < lim:
            nmax = len(self.__seq)
        else:
            nmax = lim
        if target is None:
            target = [
                command.CommandKind.N,
                command.CommandKind.E,
                command.CommandKind.M,
                command.CommandKind.X,
                command.CommandKind.Z,
                command.CommandKind.C,
            ]
        count = 0
        i = -1
        while count < nmax:
            i = i + 1
            if i == len(self.__seq):
                break
            cmd = self.__seq[i]
            if cmd.kind == command.CommandKind.N and (command.CommandKind.N in target):
                count += 1
                print(f"N, node = {cmd.node}")
            elif cmd.kind == command.CommandKind.E and (command.CommandKind.E in target):
                count += 1
                print(f"E, nodes = {cmd.nodes}")
            elif cmd.kind == command.CommandKind.M and (command.CommandKind.M in target):
                count += 1
                print(
                    f"M, node = {cmd.node}, plane = {cmd.plane}, angle(pi) = {cmd.angle}, "
                    + f"s_domain = {cmd.s_domain}, t_domain = {cmd.t_domain}"
                )
            elif cmd.kind == command.CommandKind.X and (command.CommandKind.X in target):
                count += 1
                print(f"X byproduct, node = {cmd.node}, domain = {cmd.domain}")
            elif cmd.kind == command.CommandKind.Z and (command.CommandKind.Z in target):
                count += 1
                print(f"Z byproduct, node = {cmd.node}, domain = {cmd.domain}")
            elif cmd.kind == command.CommandKind.C and (command.CommandKind.C in target):
                count += 1
                print(f"Clifford, node = {cmd.node}, Clifford index = {cmd.cliff_index}")

        if len(self.__seq) > i + 1:
            print(f"{len(self.__seq)-lim} more commands truncated. Change lim argument of print_pattern() to show more")

    def get_local_pattern(self):
        """Get a local pattern transpiled from the pattern.

        Returns
        -------
        localpattern : LocalPattern
            transpiled local pattern.
        """
        standardized = self.is_standard()

        def fresh_node():
            return {
                "seq": [],
                "Mprop": [None, None, set(), set()],
                "Xsignal": set(),
                "Xsignals": [],
                "Zsignal": set(),
                "is_input": False,
                "is_output": False,
            }

        node_prop = {u: fresh_node() for u in self.__input_nodes}
        morder = []
        for cmd in self.__seq:
            kind = cmd.kind
            if kind == command.CommandKind.N:
                node_prop[cmd.node] = fresh_node()
            elif kind == command.CommandKind.E:
                node_prop[cmd.nodes[1]]["seq"].append(cmd.nodes[0])
                node_prop[cmd.nodes[0]]["seq"].append(cmd.nodes[1])
            elif kind == command.CommandKind.M:
                node_prop[cmd.node]["Mprop"] = [cmd.plane, cmd.angle, cmd.s_domain, cmd.t_domain]
                node_prop[cmd.node]["seq"].append(-1)
                morder.append(cmd.node)
            elif kind == command.CommandKind.X:
                if standardized:
                    node_prop[cmd.node]["Xsignal"] ^= cmd.domain
                    node_prop[cmd.node]["Xsignals"] += [cmd.domain]
                else:
                    node_prop[cmd.node]["Xsignals"].append(cmd.domain)
                node_prop[cmd.node]["seq"].append(-2)
            elif kind == command.CommandKind.Z:
                node_prop[cmd.node]["Zsignal"] ^= cmd.domain
                node_prop[cmd.node]["seq"].append(-3)
            elif kind == command.CommandKind.C:
                node_prop[cmd.node]["vop"] = cmd.cliff_index
                node_prop[cmd.node]["seq"].append(-4)
            elif kind == command.CommandKind.S:
                raise NotImplementedError()
            else:
                raise ValueError(f"command {cmd} is invalid!")
        nodes = dict()
        for index in node_prop.keys():
            if index in self.output_nodes:
                node_prop[index]["is_output"] = True
            if index in self.input_nodes:
                node_prop[index]["is_input"] = True
            node = CommandNode(index, **node_prop[index])
            nodes[index] = node
        return LocalPattern(nodes, self.input_nodes, self.output_nodes, morder)

    def standardize(self, method="direct"):
        """Executes standardization of the pattern.
        'standard' pattern is one where commands are sorted in the order of
        'N', 'E', 'M' and then byproduct commands ('X' and 'Z').

        Parameters
        ----------
        method : str, optional
            'global' corresponds to a conventional standardization executed on Pattern class.
            'local' standardization is executed on LocalPattern class. In all cases, local pattern standardization is significantly faster than conventional one.
            defaults to 'local'
        """
        if method == "direct":
            self.standardize_direct()
            return
        if method not in {"local", "global"}:
            raise ValueError("Invalid method")
        warnings.warn(
            f"Method `{method}` is deprecated for `standardize`. Please use the default `direct` method instead. See https://github.com/TeamGraphix/graphix/pull/190 for more informations.",
            stacklevel=1,
        )
        if method == "local":
            localpattern = self.get_local_pattern()
            localpattern.standardize()
            self.__seq = localpattern.get_pattern().__seq
        elif method == "global":
            self._move_N_to_left()
            self._move_byproduct_to_right()
            self._move_E_after_N()

    def standardize_direct(self) -> None:
        """
        This algorithm sort the commands in the following order:
        `N`, `E`, `M`, `C`, `Z`, `X`.
        """
        n_list = []
        e_list = []
        m_list = []
        c_dict = {}
        z_dict = {}
        x_dict = {}

        def add_correction_domain(
            domain_dict: dict[command.Node, command.Command], node: command.Node, domain: set[command.Node]
        ) -> None:
            if previous_domain := domain_dict.get(node):
                previous_domain ^= domain
            else:
                domain_dict[node] = domain.copy()

        for cmd in self:
            if cmd.kind == CommandKind.N:
                n_list.append(cmd)
            elif cmd.kind == CommandKind.E:
                for side in (0, 1):
                    if s_domain := x_dict.get(cmd.nodes[side], None):
                        add_correction_domain(z_dict, cmd.nodes[1 - side], s_domain)
                e_list.append(cmd)
            elif cmd.kind == CommandKind.M:
                if cliff_index := c_dict.pop(cmd.node, None):
                    cmd = cmd.clifford(graphix.clifford.TABLE[cliff_index])
                if t_domain := z_dict.pop(cmd.node, None):
                    cmd.t_domain ^= t_domain
                if s_domain := x_dict.pop(cmd.node, None):
                    cmd.s_domain ^= s_domain
                m_list.append(cmd)
            elif cmd.kind == CommandKind.Z:
                add_correction_domain(z_dict, cmd.node, cmd.domain)
            elif cmd.kind == CommandKind.X:
                add_correction_domain(x_dict, cmd.node, cmd.domain)
            elif cmd.kind == CommandKind.C:
                # If some `X^sZ^t` have been applied to the node, compute `X^s'Z^t'`
                # such that `CX^sZ^t = X^s'Z^t'C` since the Clifford command will
                # be applied first (i.e., in right-most position).
                t_domain = z_dict.pop(cmd.node, set())
                s_domain = x_dict.pop(cmd.node, set())
                domains = graphix.clifford.TABLE[cmd.cliff_index].conj.commute_domains(
                    graphix.clifford.Domains(s_domain, t_domain)
                )
                if domains.t_domain:
                    z_dict[cmd.node] = domains.t_domain
                if domains.s_domain:
                    x_dict[cmd.node] = domains.s_domain
                # Each pattern command is applied by left multiplication: if a clifford `C`
                # has been already applied to a node, applying a clifford `C'` to the same
                # node is equivalent to apply `C'C` to a fresh node.
                c_dict[cmd.node] = CLIFFORD_MUL[cmd.cliff_index][c_dict.get(cmd.node, 0)]
        self.__seq = [
            *n_list,
            *e_list,
            *m_list,
            *(command.C(node=node, cliff_index=cliff_index) for node, cliff_index in c_dict.items()),
            *(command.Z(node=node, domain=domain) for node, domain in z_dict.items()),
            *(command.X(node=node, domain=domain) for node, domain in x_dict.items()),
        ]

    def is_standard(self):
        """determines whether the command sequence is standard

        Returns
        -------
        is_standard : bool
            True if the pattern is standard
        """
        it = iter(self)
        try:
            kind = next(it).kind
            while kind == command.CommandKind.N:
                kind = next(it).kind
            while kind == command.CommandKind.E:
                kind = next(it).kind
            while kind == command.CommandKind.M:
                kind = next(it).kind
            xzc = {command.CommandKind.X, command.CommandKind.Z, command.CommandKind.C}
            while kind in xzc:
                kind = next(it).kind
            return False
        except StopIteration:
            return True

    def shift_signals(self, method="direct") -> dict[int, list[int]]:
        """Performs signal shifting procedure
        Extract the t-dependence of the measurement into 'S' commands
        and commute them to the end of the command sequence where it can be removed.
        This procedure simplifies the dependence structure of the pattern.

        Ref for the original 'global' method:
            V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)
        Ref for the 'local' method:
            S. Sunami and M. Fukushima, in preparation

        Parameters
        ----------
        method : str, optional
            'global' shift_signals is executed on a conventional Pattern sequence.
            'local' shift_signals is done on a LocalPattern class which is faster but results in equivalent pattern.
            defaults to 'local'

        Returns
        -------
        swapped_dict : dict[int, list[int]]
            for each node, the signal that have been shifted if the outcome is
            swapped by the shift.
        """
        if method == "direct":
            return self.shift_signals_direct()
        if method not in {"local", "global"}:
            raise ValueError("Invalid method")
        warnings.warn(
            f"Method `{method}` is deprecated for `shift_signals`. Please use the default `direct` method instead. See https://github.com/TeamGraphix/graphix/pull/190 for more informations.",
            stacklevel=1,
        )
        if method == "local":
            localpattern = self.get_local_pattern()
            swapped_dict = localpattern.shift_signals()
            self.__seq = localpattern.get_pattern().__seq
        elif method == "global":
            swapped_dict = self.extract_signals()
            target = self._find_op_to_be_moved(command.CommandKind.S, rev=True)
            while target is not None:
                if target == len(self.__seq) - 1:
                    self.__seq.pop(target)
                    target = self._find_op_to_be_moved(command.CommandKind.S, rev=True)
                    continue
                cmd = self.__seq[target + 1]
                kind = cmd.kind
                if kind == command.CommandKind.X:
                    self._commute_XS(target)
                elif kind == command.CommandKind.Z:
                    self._commute_ZS(target)
                elif kind == command.CommandKind.M:
                    self._commute_MS(target)
                elif kind == command.CommandKind.S:
                    self._commute_SS(target)
                else:
                    self._commute_with_following(target)
                target += 1
        return swapped_dict

    def shift_signals_direct(self) -> dict[int, set[int]]:
        signal_dict = {}

        def expand_domain(domain: set[command.Node]) -> None:
            for node in domain & signal_dict.keys():
                domain ^= signal_dict[node]

        for cmd in self:
            if cmd.kind == CommandKind.M:
                s_domain = cmd.s_domain
                t_domain = cmd.t_domain
                expand_domain(s_domain)
                expand_domain(t_domain)
                plane = cmd.plane
                if plane == Plane.XY:
                    # M^{XY,α} X^s Z^t = M^{XY,(-1)^s·α+tπ}
                    #                  = S^t M^{XY,(-1)^s·α}
                    #                  = S^t M^{XY,α} X^s
                    if t_domain:
                        cmd.t_domain = set()
                        signal_dict[cmd.node] = t_domain
                elif plane == Plane.XZ:
                    # M^{XZ,α} X^s Z^t = M^{XZ,(-1)^t((-1)^s·α+sπ)}
                    #                  = M^{XZ,(-1)^{s+t}·α+(-1)^t·sπ}
                    #                  = M^{XZ,(-1)^{s+t}·α+sπ         (since (-1)^t·π ≡ π (mod 2π))
                    #                  = S^s M^{XZ,(-1)^{s+t}·α}
                    #                  = S^s M^{XZ,α} Z^{s+t}
                    if s_domain:
                        cmd.s_domain = set()
                        t_domain ^= s_domain
                        signal_dict[cmd.node] = s_domain
                elif plane == Plane.YZ:
                    # M^{YZ,α} X^s Z^t = M^{YZ,(-1)^t·α+sπ)}
                    #                  = S^s M^{YZ,(-1)^t·α}
                    #                  = S^s M^{YZ,α} Z^t
                    if s_domain:
                        cmd.s_domain = set()
                        signal_dict[cmd.node] = s_domain
            elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:
                expand_domain(cmd.domain)
        return signal_dict

    def _find_op_to_be_moved(self, op: command.CommandKind, rev=False, skipnum=0):
        """Internal method for pattern modification.

        Parameters
        ----------
        op : command.CommandKind, N, E, M, X, Z, S
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

    def _commute_EX(self, target):
        """Internal method to perform the commutation of E and X.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a X command followed by E command
        """
        assert self.__seq[target].kind == command.CommandKind.X
        assert self.__seq[target + 1].kind == command.CommandKind.E
        X = self.__seq[target]
        E = self.__seq[target + 1]
        if E.nodes[0] == X.node:
            Z = command.Z(node=E.nodes[1], domain=X.domain)
            self.__seq.pop(target + 1)  # del E
            self.__seq.insert(target, Z)  # add Z in front of X
            self.__seq.insert(target, E)  # add E in front of Z
            return True
        elif E.nodes[1] == X.node:
            Z = command.Z(node=E.nodes[0], domain=X.domain)
            self.__seq.pop(target + 1)  # del E
            self.__seq.insert(target, Z)  # add Z in front of X
            self.__seq.insert(target, E)  # add E in front of Z
            return True
        else:
            self._commute_with_following(target)
            return False

    def _commute_MX(self, target):
        """Internal method to perform the commutation of M and X.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a X command followed by M command
        """
        assert self.__seq[target].kind == command.CommandKind.X
        assert self.__seq[target + 1].kind == command.CommandKind.M
        X = self.__seq[target]
        M = self.__seq[target + 1]
        if X.node == M.node:
            M.s_domain ^= X.domain
            self.__seq.pop(target)  # del X
            return True
        else:
            self._commute_with_following(target)
            return False

    def _commute_MZ(self, target):
        """Internal method to perform the commutation of M and Z.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a Z command followed by M command
        """
        assert self.__seq[target].kind == command.CommandKind.Z
        assert self.__seq[target + 1].kind == command.CommandKind.M
        Z = self.__seq[target]
        M = self.__seq[target + 1]
        if Z.node == M.node:
            M.t_domain ^= Z.domain
            self.__seq.pop(target)  # del Z
            return True
        else:
            self._commute_with_following(target)
            return False

    def _commute_XS(self, target):
        """Internal method to perform the commutation of X and S.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by X command
        """
        assert self.__seq[target].kind == command.CommandKind.S
        assert self.__seq[target + 1].kind == command.CommandKind.X
        S = self.__seq[target]
        X = self.__seq[target + 1]
        if S.node in X.domain:
            X.domain ^= S.domain
        self._commute_with_following(target)

    def _commute_ZS(self, target):
        """Internal method to perform the commutation of Z and S.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by Z command
        """
        assert self.__seq[target].kind == command.CommandKind.S
        assert self.__seq[target + 1].kind == command.CommandKind.Z
        S = self.__seq[target]
        Z = self.__seq[target + 1]
        if S.node in Z.domain:
            Z.domain ^= S.domain
        self._commute_with_following(target)

    def _commute_MS(self, target):
        """Internal method to perform the commutation of M and S.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by M command
        """
        assert self.__seq[target].kind == command.CommandKind.S
        assert self.__seq[target + 1].kind == command.CommandKind.M
        S = self.__seq[target]
        M = self.__seq[target + 1]
        if S.node in M.s_domain:
            M.s_domain ^= S.domain
        if S.node in M.t_domain:
            M.t_domain ^= S.domain
        self._commute_with_following(target)

    def _commute_SS(self, target):
        """Internal method to perform the commutation of two S commands.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by S command
        """
        assert self.__seq[target].kind == command.CommandKind.S
        assert self.__seq[target + 1].kind == command.CommandKind.S
        S1 = self.__seq[target]
        S2 = self.__seq[target + 1]
        if S1.node in S2.domain:
            S2.domain ^= S1.domain
        self._commute_with_following(target)

    def _commute_with_following(self, target):
        """Internal method to perform the commutation of
        two consecutive commands that commutes.
        commutes the target command with the following command.

        Parameters
        ----------
        target : int
            target command index
        """
        A = self.__seq[target + 1]
        self.__seq.pop(target + 1)
        self.__seq.insert(target, A)

    def _commute_with_preceding(self, target):
        """Internal method to perform the commutation of
        two consecutive commands that commutes.
        commutes the target command with the preceding command.

        Parameters
        ----------
        target : int
            target command index
        """
        A = self.__seq[target - 1]
        self.__seq.pop(target - 1)
        self.__seq.insert(target, A)

    def _move_N_to_left(self):
        """Internal method to move all 'N' commands to the start of the sequence.
        N can be moved to the start of sequence without the need of considering
        commutation relations.
        """
        new_seq = []
        Nlist = []
        for cmd in self.__seq:
            if cmd.kind == command.CommandKind.N:
                Nlist.append(cmd)
            else:
                new_seq.append(cmd)
        Nlist.sort(key=lambda N_cmd: N_cmd.node)
        self.__seq = Nlist + new_seq

    def _move_byproduct_to_right(self):
        """Internal method to move the byproduct commands to the end of sequence,
        using the commutation relations implemented in graphix.Pattern class
        """
        # First, we move all X commands to the end of sequence
        index = len(self.__seq) - 1
        X_limit = len(self.__seq) - 1
        while index > 0:
            if self.__seq[index].kind == command.CommandKind.X:
                index_X = index
                while index_X < X_limit:
                    cmd = self.__seq[index_X + 1]
                    kind = cmd.kind
                    if kind == command.CommandKind.E:
                        move = self._commute_EX(index_X)
                        if move:
                            X_limit += 1  # addition of extra Z means target must be increased
                            index_X += 1
                    elif kind == command.CommandKind.M:
                        search = self._commute_MX(index_X)
                        if search:
                            X_limit -= 1  # XM commutation rule removes X command
                            break
                    else:
                        self._commute_with_following(index_X)
                    index_X += 1
                else:
                    X_limit -= 1
            index -= 1
        # then, move Z to the end of sequence in front of X
        index = X_limit
        Z_limit = X_limit
        while index > 0:
            if self.__seq[index].kind == command.CommandKind.Z:
                index_Z = index
                while index_Z < Z_limit:
                    cmd = self.__seq[index_Z + 1]
                    if cmd.kind == command.CommandKind.M:
                        search = self._commute_MZ(index_Z)
                        if search:
                            Z_limit -= 1  # ZM commutation rule removes Z command
                            break
                    else:
                        self._commute_with_following(index_Z)
                    index_Z += 1
            index -= 1

    def _move_E_after_N(self):
        """Internal method to move all E commands to the start of sequence,
        before all N commands. assumes that _move_N_to_left() method was called.
        """
        moved_E = 0
        target = self._find_op_to_be_moved(command.CommandKind.E, skipnum=moved_E)
        while target is not None:
            if (target == 0) or (
                self.__seq[target - 1].kind == command.CommandKind.N
                or self.__seq[target - 1].kind == command.CommandKind.E
            ):
                moved_E += 1
                target = self._find_op_to_be_moved(command.CommandKind.E, skipnum=moved_E)
                continue
            self._commute_with_preceding(target)
            target -= 1

    def extract_signals(self) -> dict[int, list[int]]:
        """Extracts 't' domain of measurement commands, turn them into
        signal 'S' commands and add to the command sequence.
        This is used for shift_signals() method.
        """
        signal_dict = {}
        pos = 0
        while pos < len(self.__seq):
            if self.__seq[pos].kind == command.CommandKind.M:
                cmd: command.M = self.__seq[pos]
                extracted_signal = extract_signal(cmd.plane, cmd.s_domain, cmd.t_domain)
                if extracted_signal.signal:
                    self.__seq.insert(pos + 1, command.S(node=cmd.node, domain=extracted_signal.signal))
                    cmd.s_domain = extracted_signal.s_domain
                    cmd.t_domain = extracted_signal.t_domain
                    pos += 1
                signal_dict[cmd.node] = extracted_signal.signal
            pos += 1
        return signal_dict

    def _get_dependency(self):
        """Get dependency (byproduct correction & dependent measurement)
        structure of nodes in the graph (resource) state, according to the pattern.
        This is used to determine the optimum measurement order.

        Returns
        -------
        dependency : dict of set
            index is node number. all nodes in the each set must be measured before measuring
        """
        nodes, _ = self.get_graph()
        dependency = {i: set() for i in nodes}
        for cmd in self.__seq:
            if cmd.kind == command.CommandKind.M:
                dependency[cmd.node] = dependency[cmd.node] | cmd.s_domain | cmd.t_domain
            elif cmd.kind == command.CommandKind.X:
                dependency[cmd.node] = dependency[cmd.node] | cmd.domain
            elif cmd.kind == command.CommandKind.Z:
                dependency[cmd.node] = dependency[cmd.node] | cmd.domain
        return dependency

    def update_dependency(self, measured, dependency):
        """Remove measured nodes from the 'dependency'.

        Parameters
        ----------
        measured: set of int
            measured nodes.
        dependency: dict of set
            which is produced by `_get_dependency`

        Returns
        --------
        dependency: dict of set
            updated dependency information
        """
        for i in dependency.keys():
            dependency[i] -= measured
        return dependency

    def get_layers(self):
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
        dependency = self.update_dependency(measured, dependency)
        not_measured = set(self.__input_nodes)
        for cmd in self.__seq:
            if cmd.kind == command.CommandKind.N:
                if cmd.node not in self.output_nodes:
                    not_measured = not_measured | {cmd.node}
        depth = 0
        l_k = dict()
        k = 0
        while not_measured:
            l_k[k] = set()
            for i in not_measured:
                if not dependency[i]:
                    l_k[k] = l_k[k] | {i}
            dependency = self.update_dependency(l_k[k], dependency)
            not_measured -= l_k[k]
            k += 1
            depth = k
        return depth, l_k

    def _measurement_order_depth(self):
        """Obtain a measurement order which reduces the depth of a pattern.

        Returns
        -------
        meas_order: list of int
            optimal measurement order for parallel computing
        """
        d, l_k = self.get_layers()
        meas_order = []
        for i in range(d):
            meas_order.extend(l_k[i])
        return meas_order

    def connected_edges(self, node, edges):
        """Search not activated edges connected to the specified node

        Returns
        -------
        connected: set of tuple
                set of connected edges
        """

        connected = set()
        for edge in edges:
            if edge[0] == node:
                connected = connected | {edge}
            elif edge[1] == node:
                connected = connected | {edge}
        return connected

    def _measurement_order_space(self):
        """Determine measurement order that heuristically optimises the max_space of a pattern

        Returns
        -------
        meas_order: list of int
            sub-optimal measurement order for classical simulation
        """
        # NOTE calling get_graph
        nodes, edges = self.get_graph()
        nodes = set(nodes)
        edges = set(edges)
        not_measured = nodes - set(self.output_nodes)
        dependency = self._get_dependency()
        dependency = self.update_dependency(self.results.keys(), dependency)
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
            dependency = self.update_dependency({next_node}, dependency)
            not_measured -= {next_node}
            edges -= removable_edges
        return meas_order

    def get_measurement_order_from_flow(self):
        """Return a measurement order generated from flow. If a graph has flow, the minimum 'max_space' of a pattern is guaranteed to width+1.

        Returns
        -------
        meas_order: list of int
            measurement order
        """
        # NOTE calling get_graph
        nodes, edges = self.get_graph()
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        vin = set(self.input_nodes) if self.input_nodes is not None else set()
        vout = set(self.output_nodes)
        meas_planes = self.get_meas_plane()
        f, l_k = find_flow(G, vin, vout, meas_planes=meas_planes)
        if f is None:
            return None
        depth, layer = get_layers(l_k)
        meas_order = []
        for i in range(depth):
            k = depth - i
            nodes = layer[k]
            meas_order += nodes  # NOTE this is list concatenation
        return meas_order

    def get_measurement_order_from_gflow(self):
        """Returns a list containing the node indices,
        in the order of measurements which can be performed with minimum depth.

        Returns
        -------
        meas_order : list of int
            measurement order
        """
        # NOTE calling get_graph
        nodes, edges = self.get_graph()
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        isolated = list(nx.isolates(G))
        if isolated:
            raise ValueError("The input graph must be connected")
        vin = set(self.input_nodes) if self.input_nodes is not None else set()
        vout = set(self.output_nodes)
        meas_plane = self.get_meas_plane()
        g, l_k = find_gflow(G, vin, vout, meas_plane=meas_plane)
        if not g:
            raise ValueError("No gflow found")
        k, layers = get_layers(l_k)
        meas_order = []
        while k > 0:
            meas_order.extend(layers[k])
            k -= 1
        return meas_order

    def sort_measurement_commands(self, meas_order):
        """Convert measurement order to sequence of measurement commands

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
                if self.__seq[target].kind == command.CommandKind.M and (self.__seq[target].node == i):
                    meas_cmds.append(self.__seq[target])
                    break
                target += 1
        return meas_cmds

    def get_measurement_commands(self) -> list[command.M]:
        """Returns the list containing the measurement commands,
        in the order of measurements

        Returns
        -------
        meas_cmds : list
            list of measurement commands in the order of meaurements
        """
        if not self.is_standard():
            self.standardize()
        meas_cmds = []
        ind = self._find_op_to_be_moved(command.CommandKind.M)
        if ind is None:
            return []
        while True:
            try:
                cmd = self.__seq[ind]
            except IndexError:
                break
            if cmd.kind != command.CommandKind.M:
                break
            meas_cmds.append(cmd)
            ind += 1
        return meas_cmds

    def get_meas_plane(self):
        """get measurement plane from the pattern.

        Returns
        -------
        meas_plane: dict of graphix.pauli.Plane
            list of planes representing measurement plane for each node.
        """
        meas_plane = dict()
        for cmd in self.__seq:
            if cmd.kind == command.CommandKind.M:
                meas_plane[cmd.node] = cmd.plane
        return meas_plane

    def get_angles(self):
        """Get measurement angles of the pattern.

        Returns
        -------
        angles : dict
            measurement angles of the each node.
        """
        angles = {}
        for cmd in self.__seq:
            if cmd.kind == command.CommandKind.M:
                angles[cmd.node] = cmd.angle
        return angles

    def get_max_degree(self):
        """Get max degree of a pattern

        Returns
        -------
        max_degree : int
            max degree of a pattern
        """
        nodes, edges = self.get_graph()
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        degree = g.degree()
        max_degree = max([i for i in dict(degree).values()])
        return max_degree

    def get_graph(self):
        """returns the list of nodes and edges from the command sequence,
        extracted from 'N' and 'E' commands.

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
            if cmd.kind == command.CommandKind.N:
                assert cmd.node not in node_list
                node_list.append(cmd.node)
            elif cmd.kind == command.CommandKind.E:
                edge_list.append(cmd.nodes)
        return node_list, edge_list

    def get_isolated_nodes(self):
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
        isolated_nodes = node_set - connected_node_set
        return isolated_nodes

    def get_vops(self, conj=False, include_identity=False):
        """Get local-Clifford decorations from measurement or Clifford commands.

        Parameters
        ----------
            conj (False) : bool, optional
                Apply conjugations to all local Clifford operators.
            include_identity (False) : bool, optional
                Whether or not to include identity gates in the output

        Returns:
            vops : dict
        """
        vops = dict()
        for cmd in self.__seq:
            if cmd.kind == command.CommandKind.M:
                if include_identity:
                    vops[cmd.node] = cmd.vop
            elif cmd.kind == command.CommandKind.C:
                if cmd.cliff_index == 0:
                    if include_identity:
                        vops[cmd.node] = cmd.cliff_index
                else:
                    if conj:
                        vops[cmd.node] = CLIFFORD_CONJ[cmd.cliff_index]
                    else:
                        vops[cmd.node] = cmd.cliff_index
        for out in self.output_nodes:
            if out not in vops.keys():
                if include_identity:
                    vops[out] = 0
        return vops

    def connected_nodes(self, node, prepared=None):
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
        node_list = []
        ind = self._find_op_to_be_moved(command.CommandKind.E)
        if ind is not None:  # end -> 'node' is isolated
            cmd = self.__seq[ind]
            while cmd.kind == command.CommandKind.E:
                if cmd.nodes[0] == node:
                    if cmd.nodes[1] not in prepared:
                        node_list.append(cmd.nodes[1])
                elif cmd.nodes[1] == node:
                    if cmd.nodes[0] not in prepared:
                        node_list.append(cmd.nodes[0])
                ind += 1
                cmd = self.__seq[ind]
        return node_list

    def standardize_and_shift_signals(self, method="local"):
        """Executes standardization and signal shifting.

        Parameters
        ----------
        method : str, optional
            'global' corresponds to a conventional method executed on Pattern class.
            'local' standardization is executed on LocalPattern class.
            defaults to 'local'
        """
        warnings.warn(
            "`Pattern.standardize_and_shift_signals` is deprecated. Please use `Pattern.standardize` and `Pattern.shift_signals` in sequence instead. See https://github.com/TeamGraphix/graphix/pull/190 for more informations.",
            stacklevel=1,
        )
        if method == "local":
            localpattern = self.get_local_pattern()
            localpattern.standardize()
            localpattern.shift_signals()
            self.__seq = localpattern.get_pattern().__seq
        elif method == "global" or method == "direct":
            self.standardize(method)
            self.shift_signals(method)
        else:
            raise ValueError("Invalid method")

    def correction_commands(self):
        """Returns the list of byproduct correction commands"""
        assert self.is_standard()
        return [seqi for seqi in self.__seq if seqi.kind in (command.CommandKind.X, command.CommandKind.Z)]

    def parallelize_pattern(self):
        """Optimize the pattern to reduce the depth of the computation
        by gathering measurement commands that can be performed simultaneously.
        This optimized pattern runs efficiently on GPUs and quantum hardwares with
        depth (e.g. coherence time) limitations.
        """
        if not self.is_standard():
            self.standardize()
        meas_order = self._measurement_order_depth()
        self._reorder_pattern(self.sort_measurement_commands(meas_order))

    def minimize_space(self):
        """Optimize the pattern to minimize the max_space property of
        the pattern i.e. the optimized pattern has significantly
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

    def _reorder_pattern(self, meas_commands: list[command.M]):
        """internal method to reorder the command sequence

        Parameters
        ----------
        meas_commands : list of command
            list of measurement ('M') commands
        """
        prepared = set(self.input_nodes)
        measured = set()
        new = []
        c_list = []

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
            if cmd.kind == command.CommandKind.N and cmd.node not in prepared:
                new.append(command.N(node=cmd.node))
            elif cmd.kind == command.CommandKind.E and all(node in self.output_nodes for node in cmd.nodes):
                new.append(cmd)
            elif cmd.kind == command.CommandKind.C:  # Add Clifford nodes
                new.append(cmd)
            elif cmd.kind in {command.CommandKind.Z, command.CommandKind.X}:  # Add corrections
                c_list.append(cmd)

        # c_list = self.correction_commands()
        new.extend(c_list)
        self.__seq = new

    def max_space(self):
        """The maximum number of nodes that must be present in the graph (graph space) during the execution of the pattern.
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
            if cmd.kind == command.CommandKind.N:
                nodes += 1
            elif cmd.kind == command.CommandKind.M:
                nodes -= 1
            if nodes > max_nodes:
                max_nodes = nodes
        return max_nodes

    def space_list(self):
        """Returns the list of the number of nodes present in the graph (space)
        during each step of execution of the pattern (for N and M commands).

        Returns
        -------
        N_list : list
            time evolution of 'space' at each 'N' and 'M' commands of pattern.
        """
        nodes = 0
        N_list = []
        for cmd in self.__seq:
            if cmd.kind == command.CommandKind.N:
                nodes += 1
                N_list.append(nodes)
            elif cmd.kind == command.CommandKind.M:
                nodes -= 1
                N_list.append(nodes)
        return N_list

    def simulate_pattern(self, backend="statevector", **kwargs):
        """Simulate the execution of the pattern by using
        :class:`graphix.simulator.PatternSimulator`.

        Available backend: ['statevector', 'densitymatrix', 'tensornetwork']

        Parameters
        ----------
        backend : str
            optional parameter to select simulator backend.
        kwargs: keyword args for specified backend.

        Returns
        -------
        state :
            quantum state representation for the selected backend.

        .. seealso:: :class:`graphix.simulator.PatternSimulator`
        """
        sim = PatternSimulator(self, backend=backend, **kwargs)
        state = sim.run()
        return state

    def run_pattern(self, backend, **kwargs):
        """run the pattern on cloud-based quantum devices and their simulators.
        Available backend: ['ibmq']

        Parameters
        ----------
        backend : str
            parameter to select executor backend.
        kwargs: keyword args for specified backend.

        Returns
        -------
        result :
            the measurement result,
            in the representation depending on the backend used.
        """
        exe = PatternRunner(self, backend=backend, **kwargs)
        result = exe.run()
        return result

    def perform_pauli_measurements(self, leave_input=False, use_rustworkx=False):
        """Perform Pauli measurements in the pattern using
        efficient stabilizer simulator.

        .. seealso:: :func:`measure_pauli`

        """
        measure_pauli(self, leave_input, copy=False, use_rustworkx=use_rustworkx)

    def draw_graph(
        self,
        flow_from_pattern=True,
        show_pauli_measurement=True,
        show_local_clifford=False,
        show_measurement_planes=False,
        show_loop=True,
        node_distance=(1, 1),
        figsize=None,
        save=False,
        filename=None,
    ):
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
        g = nx.Graph()
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

    def to_qasm3(self, filename):
        """Export measurement pattern to OpenQASM 3.0 file

        Parameters
        ----------
        filename : str
            file name to export to. example: "filename.qasm"
        """
        with open(filename + ".qasm", "w") as file:
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
                for line in cmd_to_qasm3(cmd):
                    file.write(line)

    def copy(self) -> Pattern:
        result = self.__new__(self.__class__)
        result.__seq = [cmd.model_copy() for cmd in self.__seq]
        result.__input_nodes = self.__input_nodes.copy()
        result.__output_nodes = self.__output_nodes.copy()
        result.__Nnode = self.__Nnode
        result._pauli_preprocessed = self._pauli_preprocessed
        result.results = self.results.copy()
        return result


class CommandNode:
    """A node decorated with a distributed command sequence.

    Attributes
    ----------
    index : int
        node index
    seq : list
        command sequence. In this class, a command sequence follows the rules noted below.

        E: pair node's index(>=0)
        M: -1
        X: -2
        Z: -3
        C: -4
    Mprop : list
        attributes for a measurement command. consists of [meas_plane, angle, s_domain, t_domain]
    result : int
        measurement result of the node
    Xsignal : list
        signal domain
    Xsignals : list
        signal domain. Xsignals may contains lists. For standardization, this variable is used.
    Zsignal : list
        signal domain
    input : bool
        whether the node is an input or not
    output : bool
        whether the node is an output or not
    """

    def __init__(self, node_index, seq, Mprop, Zsignal, is_input, is_output, Xsignal=None, Xsignals=None):
        """
        Parameters
        ----------
        node_index : int
            node index

        seq : list
            distributed command sequence

        Mprop : list
            attributes for measurement command

        Xsignal : list
            signal domain for X byproduct correction

        Xsignals : list of list
            signal domains for X byproduct correction
            Xsignal or Xsignals must be specified

        Zsignal : list
            signal domain for Z byproduct correction

        is_input : bool
            whether the node is an input or not

        is_output : bool
            whether the node is an output or not
        """
        if Xsignals is None:
            Xsignals = []
        if Xsignal is None:
            Xsignal = set()
        self.index = node_index
        self.seq = seq  # composed of [E, M, X, Z, C]
        self.Mprop = Mprop
        self.result = None
        self.Xsignal = Xsignal
        self.Xsignals = Xsignals
        self.Zsignal = Zsignal  # appeared at most e + 1
        self.input = is_input
        self.output = is_output

    def is_standard(self):
        """Check whether the local command sequence is standardized.

        Returns
        -------
        standardized : Bool
            whether the local command sequence is standardized or not
        """
        order_dict = {
            -1: [-1, -2, -3, -4],
            -2: [-2, -3, -4],
            -3: [-2, -3, -4],
            -4: [-4],
        }
        standardized = True
        cmd_ref = 0
        for cmd in self.seq:
            if cmd_ref >= 0:
                pass
            else:
                standardized &= cmd in order_dict[cmd_ref]
            cmd_ref = cmd
        return standardized

    def commute_X(self):
        """Move all X correction commands to the back.

        Returns
        -------
        EXcommutated_nodes : dict
            when X commutes with E, Z correction is added on the pair node. This dict specifies target nodes where Zs will be added.
        """
        EXcommutated_nodes = dict()
        combined_Xsignal = set()
        for Xsignal in self.Xsignals:
            Xpos = self.seq.index(-2)
            for i in range(Xpos, len(self.seq)):
                if self.seq[i] >= 0:
                    try:
                        EXcommutated_nodes[self.seq[i]] ^= Xsignal
                    except KeyError:
                        EXcommutated_nodes[self.seq[i]] = Xsignal
            self.seq.remove(-2)
            combined_Xsignal ^= Xsignal
        if self.output:
            self.seq.append(-2)  # put X on the end of the pattern
            self.Xsignal = combined_Xsignal
            self.Xsignals = [combined_Xsignal]
        else:
            self.Mprop[2] ^= combined_Xsignal
            self.Xsignal = []
            self.Xsignals = []
        return EXcommutated_nodes

    def commute_Z(self):
        """Move all Zs to the back. EZ commutation produces no additional command unlike EX commutation."""
        z_in_seq = False
        while -3 in self.seq:
            z_in_seq = True
            self.seq.remove(-3)
        if self.output and z_in_seq:
            self.seq.append(-3)
        else:
            self.Mprop[3] ^= self.Zsignal
            self.Zsignal = []

    def _add_Z(self, pair, signal):
        """Add Z correction into the node.

        Parameters
        ----------
        pair : int
            a node index where the Z is produced. The additional Z will be inserted just behind the E(with pair) command
        signal : list
            signal domain for the additional Z correction
        """
        # caused by EX commutation.
        self.Zsignal ^= signal
        Epos = self.seq.index(pair)
        self.seq.insert(Epos + 1, -3)

    def print_pattern(self):
        """Print the local command sequence"""
        for cmd in self.seq:
            print(self.get_command(cmd))

    def get_command(self, cmd):
        """Get a command with full description. Patterns with more than one X or Z corrections are not supported.

        Parameters
        ----------
        cmd : int
            an integer corresponds to a command as described below.
            E: pair node's index(>=0)
            M: -1
            X: -2
            Z: -3
            C: -4

        Returns
        -------
        MBQC command : list
            a command for a global pattern
        """
        if cmd >= 0:
            return command.E(nodes=(self.index, cmd))
        elif cmd == -1:
            return command.M(
                node=self.index,
                plane=self.Mprop[0],
                angle=self.Mprop[1],
                s_domain=self.Mprop[2],
                t_domain=self.Mprop[3],
            )
        elif cmd == -2:
            if self.seq.count(-2) > 1:
                raise NotImplementedError("Patterns with more than one X corrections are not supported")
            return command.X(node=self.index, domain=self.Xsignal)
        elif cmd == -3:
            if self.seq.count(-3) > 1:
                raise NotImplementedError("Patterns with more than one Z corrections are not supported")
            return command.Z(node=self.index, domain=self.Zsignal)
        elif cmd == -4:
            return command.C(node=self.index, cliff_index=self.vop)

    def get_signal_destination(self):
        """get signal destination

        Returns
        -------
        signal_destination : set
            Counterpart of 'dependent nodes'. measurement results of each node propagate to the nodes specified by 'signal_distination'.
        """
        signal_destination = self.Mprop[2] | self.Mprop[3] | self.Xsignal | self.Zsignal
        return signal_destination

    def get_signal_destination_dict(self):
        """get signal destination. distinguish the kind of signals.

        Returns
        -------
        signal_destination_dict : dict
            Counterpart of 'dependent nodes'. Unlike 'get_signal_destination', types of domains are memorarized. measurement results of each node propagate to the nodes specified by 'signal_distination_dict'.
        """
        dependent_nodes_dict = dict()
        dependent_nodes_dict["Ms"] = self.Mprop[2]
        dependent_nodes_dict["Mt"] = self.Mprop[3]
        dependent_nodes_dict["X"] = self.Xsignal
        dependent_nodes_dict["Z"] = self.Zsignal
        return dependent_nodes_dict


class LocalPattern:
    """MBQC Local Pattern class

    Instead of storing commands as a 1D list as in Pattern class, here we distribute them to each node.
    This data structure is efficient for command operations such as commutation and signal propagation.
    This results in faster standardization and signal shifting.

    Attributes
    ----------
    nodes : set
        set of nodes with distributed command sequences

    input_nodes : list
        list of input node indices.

    output_nodes : list
        list of output node indices.

    morder : list
        list of node indices in a measurement order.

    signal_destination : dict
    stores the set of nodes where dependent feedforward operations are performed, from the result of measurement at each node.
    stored separately for each nodes, and for each kind of signal(Ms, Mt, X, Z).
    """

    def __init__(self, nodes=None, input_nodes=None, output_nodes=None, morder=None):
        """
        Parameters
        ----------
        nodes : dict
            dict of command decorated nodes. defaults to an empty dict.
        output_nodes : list, optional
            list of output node indices. defaults to [].
        morder : list, optional
            list of node indices in a measurement order. defaults to [].
        """
        if morder is None:
            morder = []
        if output_nodes is None:
            output_nodes = []
        if input_nodes is None:
            input_nodes = []
        if nodes is None:
            nodes = dict()
        self.nodes = nodes  # dict of Pattern.CommandNode
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.morder = morder
        self.signal_destination = {i: {"Ms": set(), "Mt": set(), "X": set(), "Z": set()} for i in self.nodes.keys()}

    def is_standard(self):
        """Check whether the local pattern is standardized or not

        Returns
        -------
        standardized : bool
            whether the local pattern is standardized or not
        """
        standardized = True
        for node in self.nodes.values():
            standardized &= node.is_standard()
        return standardized

    def Xshift(self):
        """Move X to the back of the pattern"""
        for index, node in self.nodes.items():
            EXcomutation = node.commute_X()
            for target_index, signal in EXcomutation.items():
                self.nodes[target_index]._add_Z(index, signal)

    def Zshift(self):
        """Move Z to the back of the pattern. This method can be executed separately"""
        for node in self.nodes.values():
            node.commute_Z()

    def standardize(self):
        """Standardize pattern. In this structure, it is enough to move all byproduct corrections to the back"""
        self.Xshift()
        self.Zshift()

    def collect_signal_destination(self):
        """Calculate signal destinations by considering dependencies of each node."""
        for index, node in self.nodes.items():
            dependent_node_dicts = node.get_signal_destination_dict()
            for dependent_node in dependent_node_dicts["Ms"]:
                self.signal_destination[dependent_node]["Ms"] |= {index}
            for dependent_node in dependent_node_dicts["Mt"]:
                self.signal_destination[dependent_node]["Mt"] |= {index}
            for dependent_node in dependent_node_dicts["X"]:
                self.signal_destination[dependent_node]["X"] |= {index}
            for dependent_node in dependent_node_dicts["Z"]:
                self.signal_destination[dependent_node]["Z"] |= {index}

    def shift_signals(self) -> dict[int, list[int]]:
        """Shift signals to the back based on signal destinations."""
        self.collect_signal_destination()
        signal_dict = {}
        for node_index in self.morder + self.output_nodes:
            node = self.nodes[node_index]
            if node.Mprop[0] is None:
                continue
            extracted_signal = extract_signal(node.Mprop[0], node.Mprop[2], node.Mprop[3])
            signal = extracted_signal.signal
            signal_dict[node_index] = signal
            self.nodes[node_index].Mprop[2] = extracted_signal.s_domain
            self.nodes[node_index].Mprop[3] = extracted_signal.t_domain
            for signal_label, destinated_nodes in self.signal_destination[node_index].items():
                for destinated_node in destinated_nodes:
                    node = self.nodes[destinated_node]
                    if signal_label == "Ms":
                        node.Mprop[2] ^= signal
                    elif signal_label == "Mt":
                        node.Mprop[3] ^= signal
                    elif signal_label == "X":
                        node.Xsignal ^= signal
                    elif signal_label == "Z":
                        node.Zsignal ^= signal
                    else:
                        raise ValueError(f"Invalid signal label: {signal_label}")
        return signal_dict

    def get_graph(self):
        """Get a graph from a local pattern

        Returns
        -------
        nodes : list
            list of node indices
        edges : list
            list of edges
        """
        nodes = []
        edges = []
        for index, node in self.nodes.items():
            nodes.append(index)
            for cmd in node.seq:
                if cmd >= 0:
                    if index > cmd:
                        edges.append((cmd, index))
        return nodes, edges

    def get_pattern(self):
        """Convert a local pattern into a corresponding global pattern. Currently, only standardized pattern is supported.

        Returns
        -------
        pattern : Pattern
            standardized global pattern
        """
        assert self.is_standard()
        pattern = Pattern(input_nodes=self.input_nodes)
        Nseq = [command.N(node=i) for i in self.nodes.keys() - self.input_nodes]
        Eseq = []
        Mseq = []
        Xseq = []
        Zseq = []
        Cseq = []
        for node_index in self.morder + self.output_nodes:
            node = self.nodes[node_index]
            for cmd in node.seq:
                if cmd >= 0:
                    Eseq.append(node.get_command(cmd))
                    self.nodes[cmd].seq.remove(node_index)
                elif cmd == -1:
                    Mseq.append(node.get_command(cmd))
                elif cmd == -2:
                    Xseq.append(node.get_command(cmd))
                elif cmd == -3:
                    Zseq.append(node.get_command(cmd))
                elif cmd == -4:
                    Cseq.append(node.get_command(cmd))
                else:
                    raise ValueError(f"command {cmd} is invalid!")
            if node.result is not None:
                pattern.results[node.index] = node.result
        pattern.replace(Nseq + Eseq + Mseq + Xseq + Zseq + Cseq)
        return pattern


def xor_combination_list(list1, list2):
    """Combine two lists according to XOR operation.

    Parameters
    ----------
    list1 : list
        list to be combined
    list2 : list
        list to be combined

    Returns
    -------
    result : list
        xor-combined list
    """
    result = list2
    for elem in list1:
        if elem in result:
            result.remove(elem)
        else:
            result.append(elem)
    return result


def measure_pauli(pattern, leave_input, copy=False, use_rustworkx=False):
    """Perform Pauli measurement of a pattern by fast graph state simulator
    uses the decorated-graph method implemented in graphix.graphsim to perform
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
    graph_state = GraphState(nodes=nodes, edges=edges, vops=vop_init, use_rustworkx=use_rustworkx)
    results = {}
    to_measure, non_pauli_meas = pauli_nodes(pattern, leave_input)
    if not leave_input and len(list(set(pattern.input_nodes) & set([i[0].node for i in to_measure]))) > 0:
        new_inputs = []
    else:
        new_inputs = pattern.input_nodes
    for cmd in to_measure:
        pattern_cmd: command.Command = cmd[0]
        measurement_basis: str = cmd[1]
        # extract signals for adaptive angle.
        s_signal = 0
        t_signal = 0
        if measurement_basis in [
            "+X",
            "-X",
        ]:  # X meaurement is not affected by s_signal
            t_signal = sum([results[j] for j in pattern_cmd.t_domain])
        elif measurement_basis in ["+Y", "-Y"]:
            s_signal = sum([results[j] for j in pattern_cmd.s_domain])
            t_signal = sum([results[j] for j in pattern_cmd.t_domain])
        elif measurement_basis in [
            "+Z",
            "-Z",
        ]:  # Z meaurement is not affected by t_signal
            s_signal = sum([results[j] for j in pattern_cmd.s_domain])
        else:
            raise ValueError("unknown Pauli measurement basis", measurement_basis)

        if int(s_signal % 2) == 1:  # equivalent to X byproduct
            graph_state.h(pattern_cmd.node)
            graph_state.z(pattern_cmd.node)
            graph_state.h(pattern_cmd.node)
        if int(t_signal % 2) == 1:  # equivalent to Z byproduct
            graph_state.z(pattern_cmd.node)
        basis = measurement_basis
        if basis == "+X":
            results[pattern_cmd.node] = graph_state.measure_x(pattern_cmd.node, choice=0)
        elif basis == "-X":
            results[pattern_cmd.node] = 1 - graph_state.measure_x(pattern_cmd.node, choice=1)
        elif basis == "+Y":
            results[pattern_cmd.node] = graph_state.measure_y(pattern_cmd.node, choice=0)
        elif basis == "-Y":
            results[pattern_cmd.node] = 1 - graph_state.measure_y(pattern_cmd.node, choice=1)
        elif basis == "+Z":
            results[pattern_cmd.node] = graph_state.measure_z(pattern_cmd.node, choice=0)
        elif basis == "-Z":
            results[pattern_cmd.node] = 1 - graph_state.measure_z(pattern_cmd.node, choice=1)
        else:
            raise ValueError("unknown Pauli measurement basis", measurement_basis)

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
    new_seq = []
    new_seq.extend(command.N(node=index) for index in set(graph_state.nodes) - set(new_inputs))
    new_seq.extend(command.E(nodes=edge) for edge in graph_state.edges)
    new_seq.extend(
        cmd.clifford(graphix.clifford.get(vops[cmd.node]))
        for cmd in pattern
        if cmd.kind == command.CommandKind.M and cmd.node in graph_state.nodes
    )
    new_seq.extend(command.C(node=index, cliff_index=vops[index]) for index in pattern.output_nodes if vops[index] != 0)
    new_seq.extend(cmd for cmd in pattern if cmd.kind in (command.CommandKind.X, command.CommandKind.Z))

    if copy:
        pat = Pattern()
    else:
        pat = pattern

    output_nodes = deepcopy(pattern.output_nodes)
    pat.replace(new_seq, input_nodes=new_inputs)
    pat.reorder_output_nodes(output_nodes)
    assert pat.Nnode == len(graph_state.nodes)
    pat.results = results
    pat._pauli_preprocessed = True
    return pat


def pauli_nodes(pattern: Pattern, leave_input: bool):
    """returns the list of measurement commands that are in Pauli bases
    and that are not dependent on any non-Pauli measurements

    Parameters
    ----------
    pattern : graphix.Pattern object
    leave_input : bool

    Returns
    -------
    pauli_node : list
        list of node indices
    """
    if not pattern.is_standard():
        pattern.standardize()
    m_commands = pattern.get_measurement_commands()
    pauli_node: list[tuple[command.M, str]] = []
    # Nodes that are non-Pauli measured, or pauli measured but depends on pauli measurement
    non_pauli_node: set[int] = set()
    for cmd in m_commands:
        pm = is_pauli_measurement(cmd, ignore_vop=True)
        if pm is not None and (cmd.node not in pattern.input_nodes or not leave_input):
            # Pauli measurement to be removed
            if pm in ["+X", "-X"]:
                if cmd.t_domain & non_pauli_node:  # cmd depend on non-Pauli measurement
                    non_pauli_node.add(cmd.node)
                else:
                    pauli_node.append((cmd, pm))
            elif pm in ["+Y", "-Y"]:
                if (cmd.s_domain | cmd.t_domain) & non_pauli_node:  # cmd depend on non-Pauli measurement
                    non_pauli_node.add(cmd.node)
                else:
                    pauli_node.append((cmd, pm))
            elif pm in ["+Z", "-Z"]:
                if cmd.s_domain & non_pauli_node:  # cmd depend on non-Pauli measurement
                    non_pauli_node.add(cmd.node)
                else:
                    pauli_node.append((cmd, pm))
            else:
                raise ValueError("Unknown Pauli measurement basis")
        else:
            non_pauli_node.add(cmd.node)
    return pauli_node, non_pauli_node


def is_pauli_measurement(cmd: command.Command, ignore_vop=True):
    """Determines whether or not the measurement command is a Pauli measurement,
    and if so returns the measurement basis.

    Parameters
    ----------
    cmd : list
        measurement command. list containing the information of the measurement,
        "M", node index, measurement plane, angle (in unit of pi), s-signal, t-signal, clifford index.

        e.g. `['M', 2, 'XY', 0.25, [], [], 6]`
        for measurement of node 2, in 4/pi angle in XY plane, with local Clifford index 6 (Hadamard).
    ignore_vop : bool
        whether or not to ignore local Clifford to detemrine the measurement basis.

    Returns
    -------
        str, one of '+X', '-X', '+Y', '-Y', '+Z', '-Z'
        if the measurement is not in Pauli basis, returns None.
    """
    assert cmd.kind == command.CommandKind.M
    basis_str = [("+X", "-X"), ("+Y", "-Y"), ("+Z", "-Z")]
    # first item: 0, 1 or 2. correspond to choice of X, Y and Z
    # second item: 0 or 1. correspond to sign (+, -)
    basis_index = (0, 0)
    if np.mod(cmd.angle, 2) == 0:
        if cmd.plane == graphix.pauli.Plane.XY:
            basis_index = (0, 0)
        elif cmd.plane == graphix.pauli.Plane.YZ:
            basis_index = (1, 0)
        elif cmd.plane == graphix.pauli.Plane.XZ:
            basis_index = (0, 0)
        else:
            raise ValueError("Unknown measurement plane")
    elif np.mod(cmd.angle, 2) == 1:
        if cmd.plane == graphix.pauli.Plane.XY:
            basis_index = (0, 1)
        elif cmd.plane == graphix.pauli.Plane.YZ:
            basis_index = (1, 1)
        elif cmd.plane == graphix.pauli.Plane.XZ:
            basis_index = (0, 1)
        else:
            raise ValueError("Unknown measurement plane")
    elif np.mod(cmd.angle, 2) == 0.5:
        if cmd.plane == graphix.pauli.Plane.XY:
            basis_index = (1, 0)
        elif cmd.plane == graphix.pauli.Plane.YZ:
            basis_index = (2, 0)
        elif cmd.plane == graphix.pauli.Plane.XZ:
            basis_index = (2, 0)
        else:
            raise ValueError("Unknown measurement plane")
    elif np.mod(cmd.angle, 2) == 1.5:
        if cmd.plane == graphix.pauli.Plane.XY:
            basis_index = (1, 1)
        elif cmd.plane == graphix.pauli.Plane.YZ:
            basis_index = (2, 1)
        elif cmd.plane == graphix.pauli.Plane.XZ:
            basis_index = (2, 1)
        else:
            raise ValueError("Unknown measurement plane")
    else:
        return None
    if not ignore_vop:
        basis_index = (
            CLIFFORD_MEASURE[cmd.vop][basis_index[0]][0],
            int(np.abs(basis_index[1] - CLIFFORD_MEASURE[cmd.vop][basis_index[0]][1])),
        )
    return basis_str[basis_index[0]][basis_index[1]]


def cmd_to_qasm3(cmd):
    """Converts a command in the pattern into OpenQASM 3.0 statement.

    Parameter
    ---------
    cmd : list
        command [type:str, node:int, attr]

    Yields
    ------
    string
        translated pattern commands in OpenQASM 3.0 language

    """
    name = cmd.name
    if name == "N":
        qubit = cmd.node
        yield "// prepare qubit q" + str(qubit) + "\n"
        yield "qubit q" + str(qubit) + ";\n"
        yield "h q" + str(qubit) + ";\n"
        yield "\n"

    elif name == "E":
        qubits = cmd.nodes
        yield "// entangle qubit q" + str(qubits[0]) + " and q" + str(qubits[1]) + "\n"
        yield "cz q" + str(qubits[0]) + ", q" + str(qubits[1]) + ";\n"
        yield "\n"

    elif name == "M":
        qubit = cmd.node
        plane = cmd.plane
        alpha = cmd.angle
        sdomain = cmd.s_domain
        tdomain = cmd.t_domain
        yield "// measure qubit q" + str(qubit) + "\n"
        yield "bit c" + str(qubit) + ";\n"
        yield "float theta" + str(qubit) + " = 0;\n"
        if plane == graphix.pauli.Plane.XY:
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

    elif (name == "X") or (name == "Z"):
        qubit = cmd.node
        sdomain = cmd.domain
        yield "// byproduct correction on qubit q" + str(qubit) + "\n"
        yield "int s" + str(qubit) + " = 0;\n"
        for sid in sdomain:
            yield "s" + str(qubit) + " += c" + str(sid) + ";\n"
        yield "if(s" + str(qubit) + " % 2 == 1){\n"
        if name == "X":
            yield "\t x q" + str(qubit) + ";\n}\n"
        else:
            yield "\t z q" + str(qubit) + ";\n}\n"
        yield "\n"

    elif name == "C":
        qubit = cmd.node
        cid = cmd.cliff_index
        yield "// Clifford operations on qubit q" + str(qubit) + "\n"
        for op in CLIFFORD_TO_QASM3[cid]:
            yield str(op) + " q" + str(qubit) + ";\n"
        yield "\n"

    else:
        raise ValueError(f"invalid command {name}")


def assert_permutation(original, user):
    node_set = set(user)
    assert node_set == set(original), f"{node_set} != {set(original)}"
    for node in user:
        if node in node_set:
            node_set.remove(node)
        else:
            raise ValueError(f"{node} appears twice")


@dataclass
class ExtractedSignal:
    """
    Return data structure for `extract_signal`.
    """

    s_domain: set[int]
    "New `s_domain` for the measure command."

    t_domain: set[int]
    "New `t_domain` for the measure command."

    signal: set[int]
    "Domain for the shift command."


def extract_signal(plane: Plane, s_domain: set[int], t_domain: set[int]) -> ExtractedSignal:
    if plane == Plane.XY:
        return ExtractedSignal(s_domain=s_domain, t_domain=set(), signal=t_domain)
    if plane == Plane.XZ:
        return ExtractedSignal(s_domain=set(), t_domain=s_domain ^ t_domain, signal=s_domain)
    if plane == Plane.YZ:
        return ExtractedSignal(s_domain=set(), t_domain=t_domain, signal=s_domain)
    typing_extensions.assert_never(plane)
