"""MBQC pattern according to Measurement Calculus
ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)
"""

from copy import deepcopy

import networkx as nx
import numpy as np

from graphix.clifford import CLIFFORD_CONJ, CLIFFORD_MEASURE, CLIFFORD_TO_QASM3
from graphix.device_interface import PatternRunner
from graphix.gflow import find_flow, find_gflow, get_layers
from graphix.graphsim.graphstate import GraphState
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

    def __init__(self, input_nodes=[]):
        """
        :param input_nodes:  optional, list of input qubits
        """
        self.results = {}  # measurement results from the graph state simulator
        self.__input_nodes = list(input_nodes)  # input nodes (list() makes our own copy of the list)
        self.__Nnode = len(input_nodes)  # total number of nodes in the graph state
        self._pauli_preprocessed = False  # flag for `measure_pauli` preprocessing completion

        self.__seq = []
        # output nodes are initially input nodes, since none are measured yet
        self.__output_nodes = list(input_nodes)

    def add(self, cmd):
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
        assert isinstance(cmd, list)
        assert cmd[0] in ["N", "E", "M", "X", "Z", "S", "C"]
        if cmd[0] == "N":
            if cmd[1] in self.__output_nodes:
                raise NodeAlreadyPrepared(cmd[1])
            self.__Nnode += 1
            self.__output_nodes.append(cmd[1])
        elif cmd[0] == "M":
            self.__output_nodes.remove(cmd[1])
        self.__seq.append(cmd)

    def extend(self, cmds):
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

    def replace(self, cmds, input_nodes=None):
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

    def reorder_output_nodes(self, output_nodes):
        """arrange the order of output_nodes.

        Parameters
        ----------
        output_nodes: list of int
            output nodes order determined by user. each index corresponds to that of logical qubits.
        """
        output_nodes = list(output_nodes)  # make our own copy (allow iterators to be passed)
        assert_permutation(self.__output_nodes, output_nodes)
        self.__output_nodes = output_nodes

    def reorder_input_nodes(self, input_nodes):
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

    def equal(self, other):
        return (
            self.__seq == other.__seq
            and self.input_nodes == other.input_nodes
            and self.output_nodes == other.output_nodes
        )

    def print_pattern(self, lim=40, filter=None):
        """print the pattern sequence (Pattern.seq).

        Parameters
        ----------
        lim: int, optional
            maximum number of commands to show
        filter : list of str, optional
            show only specified commands, e.g. ['M', 'X', 'Z']
        """
        if len(self.__seq) < lim:
            nmax = len(self.__seq)
        else:
            nmax = lim
        if filter is None:
            filter = ["N", "E", "M", "X", "Z", "C"]
        count = 0
        i = -1
        while count < nmax:
            i = i + 1
            if i == len(self.__seq):
                break
            if self.__seq[i][0] == "N" and ("N" in filter):
                count += 1
                print(f"N, node = {self.__seq[i][1]}")
            elif self.__seq[i][0] == "E" and ("E" in filter):
                count += 1
                print(f"E, nodes = {self.__seq[i][1]}")
            elif self.__seq[i][0] == "M" and ("M" in filter):
                count += 1
                if len(self.__seq[i]) == 6:
                    print(
                        f"M, node = {self.__seq[i][1]}, plane = {self.__seq[i][2]}, angle(pi) = {self.__seq[i][3]}, "
                        + f"s-domain = {self.__seq[i][4]}, t_domain = {self.__seq[i][5]}"
                    )
                elif len(self.__seq[i]) == 7:
                    print(
                        f"M, node = {self.__seq[i][1]}, plane = {self.__seq[i][2]}, angle(pi) = {self.__seq[i][3]}, "
                        + f"s-domain = {self.__seq[i][4]}, t_domain = {self.__seq[i][5]}, Clifford index = {self.__seq[i][6]}"
                    )
            elif self.__seq[i][0] == "X" and ("X" in filter):
                count += 1
                # remove duplicates
                _domain = np.array(self.__seq[i][2])
                uind = np.unique(_domain)
                unique_domain = []
                for ind in uind:
                    if np.mod(np.count_nonzero(_domain == ind), 2) == 1:
                        unique_domain.append(ind)
                print(f"X byproduct, node = {self.__seq[i][1]}, domain = {unique_domain}")
            elif self.__seq[i][0] == "Z" and ("Z" in filter):
                count += 1
                # remove duplicates
                _domain = np.array(self.__seq[i][2])
                uind = np.unique(_domain)
                unique_domain = []
                for ind in uind:
                    if np.mod(np.count_nonzero(_domain == ind), 2) == 1:
                        unique_domain.append(ind)
                print(f"Z byproduct, node = {self.__seq[i][1]}, domain = {unique_domain}")
            elif self.__seq[i][0] == "C" and ("C" in filter):
                count += 1
                print(f"Clifford, node = {self.__seq[i][1]}, Clifford index = {self.__seq[i][2]}")

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
                "Mprop": [None, None, [], [], 0],
                "Xsignal": [],
                "Xsignals": [],
                "Zsignal": [],
                "input": False,
                "output": False,
            }

        node_prop = {input: fresh_node() for input in self.__input_nodes}
        morder = []
        for cmd in self.__seq:
            if cmd[0] == "N":
                node_prop[cmd[1]] = fresh_node()
            elif cmd[0] == "E":
                node_prop[cmd[1][1]]["seq"].append(cmd[1][0])
                node_prop[cmd[1][0]]["seq"].append(cmd[1][1])
            elif cmd[0] == "M":
                node_prop[cmd[1]]["Mprop"] = cmd[2:]
                node_prop[cmd[1]]["seq"].append(-1)
                morder.append(cmd[1])
            elif cmd[0] == "X":
                if standardized:
                    node_prop[cmd[1]]["Xsignal"] += cmd[2]
                    node_prop[cmd[1]]["Xsignals"] += [cmd[2]]
                else:
                    node_prop[cmd[1]]["Xsignals"].append(cmd[2])
                node_prop[cmd[1]]["seq"].append(-2)
            elif cmd[0] == "Z":
                node_prop[cmd[1]]["Zsignal"] += cmd[2]
                node_prop[cmd[1]]["seq"].append(-3)
            elif cmd[0] == "C":
                node_prop[cmd[1]]["vop"] = cmd[2]
                node_prop[cmd[1]]["seq"].append(-4)
            elif cmd[0] == "S":
                raise NotImplementedError()
            else:
                raise ValueError(f"command {cmd} is invalid!")
        nodes = dict()
        for index in node_prop.keys():
            if index in self.output_nodes:
                node_prop[index]["output"] = True
            if index in self.input_nodes:
                node_prop[index]["input"] = True
            node = CommandNode(index, **node_prop[index])
            nodes[index] = node
        return LocalPattern(nodes, self.input_nodes, self.output_nodes, morder)

    def standardize(self, method="local"):
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
        if method == "local":
            localpattern = self.get_local_pattern()
            localpattern.standardize()
            self.__seq = localpattern.get_pattern().__seq
        elif method == "global":
            self._move_N_to_left()
            self._move_byproduct_to_right()
            self._move_E_after_N()
        else:
            raise ValueError("Invalid method")

    def is_standard(self):
        """determines whether the command sequence is standard

        Returns
        -------
        is_standard : bool
            True if the pattern is standard
        """
        order_dict = {
            "N": ["N", "E", "M", "X", "Z", "C"],
            "E": ["E", "M", "X", "Z", "C"],
            "M": ["M", "X", "Z", "C"],
            "X": ["X", "Z", "C"],
            "Z": ["X", "Z", "C"],
            "C": ["X", "Z", "C"],
        }
        result = True
        op_ref = "N"
        for cmd in self.__seq:
            op = cmd[0]
            result = result & (op in order_dict[op_ref])
            op_ref = op
        return result

    def shift_signals(self, method="local"):
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
        """
        if method == "local":
            localpattern = self.get_local_pattern()
            localpattern.shift_signals()
            self.__seq = localpattern.get_pattern().__seq
        elif method == "global":
            self.extract_signals()
            target = self._find_op_to_be_moved("S", rev=True)
            while target != "end":
                if target == len(self.__seq) - 1:
                    self.__seq.pop(target)
                    target = self._find_op_to_be_moved("S", rev=True)
                    continue
                if self.__seq[target + 1][0] == "X":
                    self._commute_XS(target)
                elif self.__seq[target + 1][0] == "Z":
                    self._commute_ZS(target)
                elif self.__seq[target + 1][0] == "M":
                    self._commute_MS(target)
                elif self.__seq[target + 1][0] == "S":
                    self._commute_SS(target)
                else:
                    self._commute_with_following(target)
                target += 1
        else:
            raise ValueError("Invalid method")

    def _find_op_to_be_moved(self, op, rev=False, skipnum=0):
        """Internal method for pattern modification.

        Parameters
        ----------
        op : str, 'N', 'E', 'M', 'X', 'Z', 'S'
            command types to be searched
        rev : bool
            search from the end (true) or start (false) of seq
        skipnum : int
            skip the detected command by specified times
        """
        if not rev:  # search from the start
            target = 0
            step = 1
        else:  # search from the back
            target = len(self.__seq) - 1
            step = -1
        ite = 0
        num_ops = 0
        while ite < len(self.__seq):
            if self.__seq[target][0] == op:
                num_ops += 1
            if num_ops == skipnum + 1:
                return target
            ite += 1
            target += step
        target = "end"
        return target

    def _commute_EX(self, target):
        """Internal method to perform the commutation of E and X.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a X command followed by E command
        """
        assert self.__seq[target][0] == "X"
        assert self.__seq[target + 1][0] == "E"
        X = self.__seq[target]
        E = self.__seq[target + 1]
        if E[1][0] == X[1]:
            Z = ["Z", E[1][1], X[2]]
            self.__seq.pop(target + 1)  # del E
            self.__seq.insert(target, Z)  # add Z in front of X
            self.__seq.insert(target, E)  # add E in front of Z
            return True
        elif E[1][1] == X[1]:
            Z = ["Z", E[1][0], X[2]]
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
        assert self.__seq[target][0] == "X"
        assert self.__seq[target + 1][0] == "M"
        X = self.__seq[target]
        M = self.__seq[target + 1]
        if X[1] == M[1]:  # s to s+r
            if len(M) == 7:
                vop = M[6]
            else:
                vop = 0
            if M[2] == "YZ" or vop == 6:
                M[5].extend(X[2])
            elif M[2] == "XY":
                M[4].extend(X[2])
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
        assert self.__seq[target][0] == "Z"
        assert self.__seq[target + 1][0] == "M"
        Z = self.__seq[target]
        M = self.__seq[target + 1]
        if Z[1] == M[1]:
            if len(M) == 7:
                vop = M[6]
            else:
                vop = 0
            if M[2] == "YZ" or vop == 6:
                M[4].extend(Z[2])
            elif M[2] == "XY":
                M[5].extend(Z[2])
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
        assert self.__seq[target][0] == "S"
        assert self.__seq[target + 1][0] == "X"
        S = self.__seq[target]
        X = self.__seq[target + 1]
        if np.mod(X[2].count(S[1]), 2):
            X[2].extend(S[2])
        self._commute_with_following(target)

    def _commute_ZS(self, target):
        """Internal method to perform the commutation of Z and S.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by Z command
        """
        assert self.__seq[target][0] == "S"
        assert self.__seq[target + 1][0] == "Z"
        S = self.__seq[target]
        Z = self.__seq[target + 1]
        if np.mod(Z[2].count(S[1]), 2):
            Z[2].extend(S[2])
        self._commute_with_following(target)

    def _commute_MS(self, target):
        """Internal method to perform the commutation of M and S.

        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by M command
        """
        assert self.__seq[target][0] == "S"
        assert self.__seq[target + 1][0] == "M"
        S = self.__seq[target]
        M = self.__seq[target + 1]
        if np.mod(M[4].count(S[1]), 2):
            M[4].extend(S[2])
        if np.mod(M[5].count(S[1]), 2):
            M[5].extend(S[2])
        self._commute_with_following(target)

    def _commute_SS(self, target):
        """Internal method to perform the commutation of two S commands.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by S command
        """
        assert self.__seq[target][0] == "S"
        assert self.__seq[target + 1][0] == "S"
        S1 = self.__seq[target]
        S2 = self.__seq[target + 1]
        if np.mod(S2[2].count(S1[1]), 2):
            S2[2].extend(S1[2])
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
        Nlist = []
        for cmd in self.__seq:
            if cmd[0] == "N":
                Nlist.append(cmd)
        Nlist.sort()
        for N in Nlist:
            self.__seq.remove(N)
        self.__seq = Nlist + self.__seq

    def _move_byproduct_to_right(self):
        """Internal method to move the byproduct commands to the end of sequence,
        using the commutation relations implemented in graphix.Pattern class
        """
        # First, we move all X commands to the end of sequence
        moved_X = 0  # number of moved X
        target = self._find_op_to_be_moved("X", rev=True, skipnum=moved_X)
        while target != "end":
            if (target == len(self.__seq) - 1) or (self.__seq[target + 1] == "X"):
                moved_X += 1
                target = self._find_op_to_be_moved("X", rev=True, skipnum=moved_X)
                continue
            if self.__seq[target + 1][0] == "E":
                move = self._commute_EX(target)
                if move:
                    target += 1  # addition of extra Z means target must be increased
            elif self.__seq[target + 1][0] == "M":
                search = self._commute_MX(target)
                if search:
                    target = self._find_op_to_be_moved("X", rev=True, skipnum=moved_X)
                    continue  # XM commutation rule removes X command
            else:
                self._commute_with_following(target)
            target += 1

        # then, move Z to the end of sequence in front of X
        moved_Z = 0  # number of moved Z
        target = self._find_op_to_be_moved("Z", rev=True, skipnum=moved_Z)
        while target != "end":
            if (target == len(self.__seq) - 1) or (self.__seq[target + 1][0] == ("X" or "Z")):
                moved_Z += 1
                target = self._find_op_to_be_moved("Z", rev=True, skipnum=moved_Z)
                continue
            if self.__seq[target + 1][0] == "M":
                search = self._commute_MZ(target)
                if search:
                    target = self._find_op_to_be_moved("Z", rev=True, skipnum=moved_Z)
                    continue  # ZM commutation rule removes Z command
            else:
                self._commute_with_following(target)
            target += 1

    def _move_E_after_N(self):
        """Internal method to move all E commands to the start of sequence,
        before all N commands. assumes that _move_N_to_left() method was called.
        """
        moved_E = 0
        target = self._find_op_to_be_moved("E", skipnum=moved_E)
        while target != "end":
            if (target == 0) or (self.__seq[target - 1][0] == ("N" or "E")):
                moved_E += 1
                target = self._find_op_to_be_moved("E", skipnum=moved_E)
                continue
            self._commute_with_preceding(target)
            target -= 1

    def extract_signals(self):
        """Extracts 't' domain of measurement commands, turn them into
        signal 'S' commands and add to the command sequence.
        This is used for shift_signals() method.
        """
        pos = 0
        while pos < len(self.__seq):
            cmd = self.__seq[pos]
            if cmd[0] == "M":
                if cmd[2] == "XY":
                    node = cmd[1]
                    if cmd[5]:
                        self.__seq.insert(pos + 1, ["S", node, cmd[5]])
                        cmd[5] = []
                        pos += 1
            pos += 1

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
            if cmd[0] == "M":
                dependency[cmd[1]] = dependency[cmd[1]] | set(cmd[4]) | set(cmd[5])
            elif cmd[0] == "X":
                dependency[cmd[1]] = dependency[cmd[1]] | set(cmd[2])
            elif cmd[0] == "Z":
                dependency[cmd[1]] = dependency[cmd[1]] | set(cmd[2])
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
            if cmd[0] == "N":
                if cmd[1] not in self.output_nodes:
                    not_measured = not_measured | {cmd[1]}
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
            for node in l_k[i]:
                meas_order.append(node)
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
            for node in layers[k]:
                meas_order.append(node)
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
                if (self.__seq[target][0] == "M") & (self.__seq[target][1] == i):
                    meas_cmds.append(self.__seq[target])
                    break
                target += 1
        return meas_cmds

    def get_measurement_commands(self):
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
        ind = self._find_op_to_be_moved("M")
        if ind == "end":
            return []
        while True:
            try:
                cmd = self.__seq[ind]
            except IndexError:
                break
            if cmd[0] != "M":
                break
            meas_cmds.append(cmd)
            ind += 1
        return meas_cmds

    def get_meas_plane(self):
        """get measurement plane from the pattern.

        Returns
        -------
        meas_plane: dict of str
            list of str representing measurement plane for each node.
        """
        meas_plane = dict()
        order = ["X", "Y", "Z"]
        for cmd in self.__seq:
            if cmd[0] == "M":
                mplane = cmd[2]
                if len(cmd) == 7:
                    converted_mplane = ""
                    clifford_measure = CLIFFORD_MEASURE[cmd[6]]
                    for axis in mplane:
                        converted = order[clifford_measure[order.index(axis)][0]]
                        converted_mplane += converted
                    mplane = "".join(sorted(converted_mplane))
                meas_plane[cmd[1]] = mplane
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
            if cmd[0] == "M":
                angles[cmd[1]] = cmd[3]
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
            if cmd[0] == "N":
                assert cmd[1] not in node_list
                node_list.append(cmd[1])
            elif cmd[0] == "E":
                edge_list.append(cmd[1])
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
            if cmd[0] == "M":
                if len(cmd) == 7:
                    if cmd[6] == 0:
                        if include_identity:
                            vops[cmd[1]] = cmd[6]
                    else:
                        if conj:
                            vops[cmd[1]] = CLIFFORD_CONJ[cmd[6]]
                        else:
                            vops[cmd[1]] = cmd[6]
                else:
                    if include_identity:
                        vops[cmd[1]] = 0
            elif cmd[0] == "C":
                if cmd[2] == 0:
                    if include_identity:
                        vops[cmd[1]] = cmd[2]
                else:
                    if conj:
                        vops[cmd[1]] = CLIFFORD_CONJ[cmd[2]]
                    else:
                        vops[cmd[1]] = cmd[2]
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
        ind = self._find_op_to_be_moved("E")
        if not ind == "end":  # end -> 'node' is isolated
            while self.__seq[ind][0] == "E":
                if self.__seq[ind][1][0] == node:
                    if self.__seq[ind][1][1] not in prepared:
                        node_list.append(self.__seq[ind][1][1])
                elif self.__seq[ind][1][1] == node:
                    if self.__seq[ind][1][0] not in prepared:
                        node_list.append(self.__seq[ind][1][0])
                ind += 1
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
        if method == "local":
            localpattern = self.get_local_pattern()
            localpattern.standardize()
            localpattern.shift_signals()
            self.__seq = localpattern.get_pattern().__seq
        elif method == "global":
            self.standardize()
            self.shift_signals()
        else:
            raise ValueError("Invalid method")

    def correction_commands(self):
        """Returns the list of byproduct correction commands"""
        assert self.is_standard()
        Clist = []
        for i in range(len(self.__seq)):
            if self.__seq[i][0] in ["X", "Z"]:
                Clist.append(self.__seq[i])
        return Clist

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

    def _reorder_pattern(self, meas_commands):
        """internal method to reorder the command sequence

        Parameters
        ----------
        meas_commands : list of command
            list of measurement ('M') commands
        """
        prepared = self.input_nodes
        measured = []
        new = []
        for cmd in meas_commands:
            node = cmd[1]
            if node not in prepared:
                new.append(["N", node])
                prepared.append(node)
            node_list = self.connected_nodes(node, measured)
            for add_node in node_list:
                if add_node not in prepared:
                    new.append(["N", add_node])
                    prepared.append(add_node)
                new.append(["E", (node, add_node)])
            new.append(cmd)
            measured.append(node)

        # add isolated nodes
        for cmd in self.__seq:
            if cmd[0] == "N":
                if cmd[1] not in prepared:
                    new.append(["N", cmd[1]])
        for cmd in self.__seq:
            if cmd[0] == "E":
                if cmd[1][0] in self.output_nodes:
                    if cmd[1][1] in self.output_nodes:
                        new.append(cmd)

        # add Clifford nodes
        for cmd in self.__seq:
            if cmd[0] == "C":
                new.append(cmd)

        # add corrections
        c_list = self.correction_commands()
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
            if cmd[0] == "N":
                nodes += 1
            elif cmd[0] == "M":
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
            if cmd[0] == "N":
                nodes += 1
                N_list.append(nodes)
            elif cmd[0] == "M":
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
                pattern=deepcopy(self),
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
                for id in self.results:
                    res = self.results[id]
                    file.write("// measurement result of qubit q" + str(id) + "\n")
                    file.write("bit c" + str(id) + " = " + str(res) + ";\n")
                    file.write("\n")
            for command in self.__seq:
                for line in cmd_to_qasm3(command):
                    file.write(line)


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
        attributes for a measurement command. consists of [meas_plane, angle, s_domain, t_domain, vop]
    result : int
        measurement result of the node
    Xsignal : list
        signal domain
    Xsignals : list
        signal domain. Xsignals may contains lists. For standardization, this variable is used.
    Zsignal : list
        signal domain
    vop : int
        value for clifford index
    input : bool
        whether the node is an input or not
    output : bool
        whether the node is an output or not
    """

    def __init__(self, node_index, seq, Mprop, Zsignal, input, output, Xsignal=[], Xsignals=[]):
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

        input : bool
            whether the node is an input or not

        output : bool
            whether the node is an output or not
        """
        self.index = node_index
        self.seq = seq  # composed of [E, M, X, Z, C]
        self.Mprop = Mprop
        self.result = None
        self.Xsignal = Xsignal
        self.Xsignals = Xsignals
        self.Zsignal = Zsignal  # appeared at most e + 1
        self.vop = Mprop[4] if len(Mprop) == 5 else 0
        self.input = input
        self.output = output

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
        combined_Xsignal = []
        for Xsignal in self.Xsignals:
            Xpos = self.seq.index(-2)
            for i in range(Xpos, len(self.seq)):
                if self.seq[i] >= 0:
                    try:
                        EXcommutated_nodes[self.seq[i]] += Xsignal
                    except KeyError:
                        EXcommutated_nodes[self.seq[i]] = Xsignal
            self.seq.remove(-2)
            combined_Xsignal += Xsignal
        if self.output:
            self.seq.append(-2)  # put X on the end of the pattern
            self.Xsignal = combined_Xsignal
            self.Xsignals = [combined_Xsignal]
        else:
            if self.Mprop[0] == "YZ" or self.vop == 6:
                self.Mprop[3] = xor_combination_list(combined_Xsignal, self.Mprop[3])
            elif self.Mprop[0] == "XY":
                self.Mprop[2] = xor_combination_list(combined_Xsignal, self.Mprop[2])
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
            if self.Mprop[0] == "YZ" or self.vop == 6:
                self.Mprop[2] = xor_combination_list(self.Zsignal, self.Mprop[2])
            elif self.Mprop[0] == "XY":
                self.Mprop[3] = xor_combination_list(self.Zsignal, self.Mprop[3])
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
        self.Zsignal = xor_combination_list(signal, self.Zsignal)
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
            return ["E", (self.index, cmd)]
        elif cmd == -1:
            return ["M", self.index] + self.Mprop
        elif cmd == -2:
            if self.seq.count(-2) > 1:
                raise NotImplementedError("Patterns with more than one X corrections are not supported")
            return ["X", self.index, self.Xsignal]
        elif cmd == -3:
            if self.seq.count(-3) > 1:
                raise NotImplementedError("Patterns with more than one Z corrections are not supported")
            return ["Z", self.index, self.Zsignal]
        elif cmd == -4:
            return ["C", self.index, self.vop]

    def get_signal_destination(self):
        """get signal destination

        Returns
        -------
        signal_destination : set
            Counterpart of 'dependent nodes'. measurement results of each node propagate to the nodes specified by 'signal_distination'.
        """
        signal_destination = set(self.Mprop[2]) | set(self.Mprop[3]) | set(self.Xsignal) | set(self.Zsignal)
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

    def __init__(self, nodes=dict(), input_nodes=[], output_nodes=[], morder=[]):
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

    def shift_signals(self):
        """Shift signals to the back based on signal destinations."""
        self.collect_signal_destination()
        for node_index in self.morder + self.output_nodes:
            signal = self.nodes[node_index].Mprop[3]
            self.nodes[node_index].Mprop[3] = []
            for signal_label, destinated_nodes in self.signal_destination[node_index].items():
                for destinated_node in destinated_nodes:
                    node = self.nodes[destinated_node]
                    if signal_label == "Ms":
                        node.Mprop[2] += signal
                    elif signal_label == "Mt":
                        node.Mprop[3] += signal
                    elif signal_label == "X":
                        node.Xsignal += signal
                    elif signal_label == "Z":
                        node.Zsignal += signal
                    else:
                        raise ValueError(f"Invalid signal label: {signal_label}")

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
        Nseq = [["N", i] for i in self.nodes.keys() - self.input_nodes]
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
    if not leave_input and len(list(set(pattern.input_nodes) & set([i[0][1] for i in to_measure]))) > 0:
        new_inputs = []
    else:
        new_inputs = pattern.input_nodes
    for cmd in to_measure:
        # extract signals for adaptive angle.
        if cmd[1] in ["+X", "-X"]:
            s_signal = 0  # X meaurement is not affected by s_signal
            t_signal = np.sum([results[j] for j in cmd[0][5]])
        elif cmd[1] in ["+Y", "-Y"]:
            s_signal = np.sum([results[j] for j in cmd[0][4]])
            t_signal = np.sum([results[j] for j in cmd[0][5]])
        elif cmd[1] in ["+Z", "-Z"]:
            s_signal = np.sum([results[j] for j in cmd[0][4]])
            t_signal = 0  # Z meaurement is not affected by t_signal
        else:
            raise ValueError("unknown Pauli measurement basis", cmd[1])

        if int(s_signal % 2) == 1:  # equivalent to X byproduct
            graph_state.h(cmd[0][1])
            graph_state.z(cmd[0][1])
            graph_state.h(cmd[0][1])
        if int(t_signal % 2) == 1:  # equivalent to Z byproduct
            graph_state.z(cmd[0][1])

        if cmd[1] == "+X":
            results[cmd[0][1]] = graph_state.measure_x(cmd[0][1], choice=0)
        elif cmd[1] == "-X":
            results[cmd[0][1]] = 1 - graph_state.measure_x(cmd[0][1], choice=1)
        elif cmd[1] == "+Y":
            results[cmd[0][1]] = graph_state.measure_y(cmd[0][1], choice=0)
        elif cmd[1] == "-Y":
            results[cmd[0][1]] = 1 - graph_state.measure_y(cmd[0][1], choice=1)
        elif cmd[1] == "+Z":
            results[cmd[0][1]] = graph_state.measure_z(cmd[0][1], choice=0)
        elif cmd[1] == "-Z":
            results[cmd[0][1]] = 1 - graph_state.measure_z(cmd[0][1], choice=1)

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
    # TO CHECK: why the order is relevant?
    for index in graph_state.nodes:
        if index not in new_inputs:
            new_seq.append(["N", index])
    for edge in graph_state.edges:
        new_seq.append(["E", edge])
    for cmd in pattern:
        if cmd[0] == "M":
            if cmd[1] in graph_state.nodes:
                cmd_new = deepcopy(cmd)
                new_clifford_ = vops[cmd[1]]
                if len(cmd_new) == 7:
                    cmd_new[6] = new_clifford_
                else:
                    cmd_new.append(new_clifford_)
                new_seq.append(cmd_new)
    for index in pattern.output_nodes:
        new_clifford_ = vops[index]
        if new_clifford_ != 0:
            new_seq.append(["C", index, new_clifford_])
    for cmd in pattern:
        if cmd[0] == "X" or cmd[0] == "Z":
            new_seq.append(cmd)

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


def pauli_nodes(pattern, leave_input):
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
    pauli_node = []
    # Nodes that are non-Pauli measured, or pauli measured but depends on pauli measurement
    non_pauli_node = []
    for cmd in m_commands:
        pm = is_pauli_measurement(cmd, ignore_vop=True)
        if pm is not None and (cmd[1] not in pattern.input_nodes or not leave_input):  # Pauli measurement to be removed
            if pm in ["+X", "-X"]:
                t_cond = np.any(np.isin(cmd[5], np.array(non_pauli_node)))
                if t_cond:  # cmd depend on non-Pauli measurement
                    non_pauli_node.append(cmd[1])
                else:
                    pauli_node.append((cmd, pm))
            elif pm in ["+Y", "-Y"]:
                s_cond = np.any(np.isin(cmd[4], np.array(non_pauli_node)))
                t_cond = np.any(np.isin(cmd[5], np.array(non_pauli_node)))
                if t_cond or s_cond:  # cmd depend on non-Pauli measurement
                    non_pauli_node.append(cmd[1])
                else:
                    pauli_node.append((cmd, pm))
            elif pm in ["+Z", "-Z"]:
                s_cond = np.any(np.isin(cmd[4], np.array(non_pauli_node)))
                if s_cond:  # cmd depend on non-Pauli measurement
                    non_pauli_node.append(cmd[1])
                else:
                    pauli_node.append((cmd, pm))
            else:
                raise ValueError("Unknown Pauli measurement basis")
        else:
            non_pauli_node.append(cmd[1])
    return pauli_node, non_pauli_node


def is_pauli_measurement(cmd, ignore_vop=True):
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
    assert cmd[0] == "M"
    basis_str = [("+X", "-X"), ("+Y", "-Y"), ("+Z", "-Z")]
    if len(cmd) == 7:
        vop = cmd[6]
    else:
        vop = 0
    # first item: 0, 1 or 2. correspond to choice of X, Y and Z
    # second item: 0 or 1. correspond to sign (+, -)
    basis_index = (0, 0)
    if np.mod(cmd[3], 2) == 0:
        if cmd[2] == "XY":
            basis_index = (0, 0)
        elif cmd[2] == "YZ":
            basis_index = (1, 0)
        elif cmd[2] == "XZ":
            basis_index = (0, 0)
        else:
            raise ValueError("Unknown measurement plane")
    elif np.mod(cmd[3], 2) == 1:
        if cmd[2] == "XY":
            basis_index = (0, 1)
        elif cmd[2] == "YZ":
            basis_index = (1, 1)
        elif cmd[2] == "XZ":
            basis_index = (0, 1)
        else:
            raise ValueError("Unknown measurement plane")
    elif np.mod(cmd[3], 2) == 0.5:
        if cmd[2] == "XY":
            basis_index = (1, 0)
        elif cmd[2] == "YZ":
            basis_index = (2, 0)
        elif cmd[2] == "XZ":
            basis_index = (2, 0)
        else:
            raise ValueError("Unknown measurement plane")
    elif np.mod(cmd[3], 2) == 1.5:
        if cmd[2] == "XY":
            basis_index = (1, 1)
        elif cmd[2] == "YZ":
            basis_index = (2, 1)
        elif cmd[2] == "XZ":
            basis_index = (2, 1)
        else:
            raise ValueError("Unknown measurement plane")
    else:
        return None
    if not ignore_vop:
        basis_index = (
            CLIFFORD_MEASURE[vop][basis_index[0]][0],
            int(np.abs(basis_index[1] - CLIFFORD_MEASURE[vop][basis_index[0]][1])),
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
    name = cmd[0]
    if name == "N":
        qubit = cmd[1]
        yield "// prepare qubit q" + str(qubit) + "\n"
        yield "qubit q" + str(qubit) + ";\n"
        yield "h q" + str(qubit) + ";\n"
        yield "\n"

    elif name == "E":
        qubits = cmd[1]
        yield "// entangle qubit q" + str(qubits[0]) + " and q" + str(qubits[1]) + "\n"
        yield "cz q" + str(qubits[0]) + ", q" + str(qubits[1]) + ";\n"
        yield "\n"

    elif name == "M":
        qubit = cmd[1]
        plane = cmd[2]
        alpha = cmd[3]
        sdomain = cmd[4]
        tdomain = cmd[5]
        yield "// measure qubit q" + str(qubit) + "\n"
        yield "bit c" + str(qubit) + ";\n"
        yield "float theta" + str(qubit) + " = 0;\n"
        if plane == "XY":
            if sdomain != []:
                yield "int s" + str(qubit) + " = 0;\n"
                for sid in sdomain:
                    yield "s" + str(qubit) + " += c" + str(sid) + ";\n"
                yield "theta" + str(qubit) + " += (-1)**(s" + str(qubit) + " % 2) * (" + str(alpha) + " * pi);\n"
            if tdomain != []:
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
        qubit = cmd[1]
        sdomain = cmd[2]
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
        qubit = cmd[1]
        cid = cmd[2]
        yield "// Clifford operations on qubit q" + str(qubit) + "\n"
        for op in CLIFFORD_TO_QASM3[cid]:
            yield str(op) + " q" + str(qubit) + ";\n"
        yield "\n"

    else:
        raise ValueError("invalid command {}".format(name))


def assert_permutation(original, user):
    node_set = set(user)
    assert node_set == set(original), f"{node_set} != {set(original)}"
    for node in user:
        if node in node_set:
            node_set.remove(node)
        else:
            assert False, f"{node} appears twice"
