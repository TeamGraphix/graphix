"""MBQC pattern according to Measurement Calculus
ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)
"""
import numpy as np
from graphix.simulator import PatternSimulator
from copy import deepcopy

class Pattern:
    """ MBQC pattern class
    Pattern holds a sequence of commands to operate the MBQC (Pattern.seq),
    and provide modification strategies to improve the structure and simulation
    efficiency of the pattern accoring to measurement calculus.

    ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)

    Attributes:
    -----------
    input_nodes : list of input node indices
    seq : list of commands
        each command is a list [type, nodes, attr] which will be
        applied in the order of list indices.
        type is one of {'N', 'M', 'E', 'X', 'Z', 'S'}
        nodes : int for {'N', 'M', 'X', 'Z', 'S'} commands
        nodes : tuple (i, j) for {'E'} command
        attr for N: none
        attr for M: meas_plane, angle, s_domain, t_domain
        attr for X: signal_domain
        attr for Z: signal_domain
        attr for S: signal_domain
    Nnode : int
        total number of nodes in the resource state
    results : dict
        stores measurement results from graph state simulator
    """

    def __init__(self, input_nodes):
        """Initialize pattern object
        Parameters
        ---------
        input_nodes : list of int
            input node indices
        """
        self.input_nodes = input_nodes
        self.seq = []
        self.node_index = []
        self.results = {}
        self.output_nodes = []
        self.output_sorted = False
        self.Nnode = len(input_nodes)

    def add_command(self, cmd):
        """add command to the end of the pattern.
        an MBQC command is specified by a list of [type, node, attr], where

            type : 'N', 'M', 'E', 'X', 'Z' or 'S'
            nodes : int for 'N', 'M', 'X', 'Z', 'S' commands
            nodes : tuple (i, j) for 'E' command
            attr for N (node preparation):
                none
            attr for E (entanglement):
                none
            attr for M (measurement):
                meas_plane : 'XY','YZ' or 'ZX'
                angle : float, in radian / pi
                s_domain : list
                t_domain : list
            attr for X:
                signal_domain : list
            attr for Z:
                signal_domain : list
            attr for S:
                signal_domain : list

        Parameters
        ----------
        cmd : list
            MBQC command.
        """
        assert type(cmd) == list
        assert cmd[0] in ['N', 'E', 'M', 'X', 'Z', 'S']
        if cmd[0] == 'N':
            self.Nnode += 1
        self.seq.append(cmd)

    def standardize(self):
        """Executes standardization of the pattern.
        'standard' pattern is one where commands are sorted in the order of
        'N', 'E', 'M' and then byproduct commands ('X' and 'Z').
        """
        self._move_N_to_left()
        self._move_byproduct_to_right()
        self._move_E_after_N()

    def is_standard(self):
        """ determines whether the command sequence is standard
        Returns
        -------
        is_standard : bool
        """
        order_dict = {'N': ['N', 'E', 'M', 'X', 'Z'], 'E': ['E', 'M', 'X', 'Z'], 'M': ['M', 'X', 'Z'], 'X': ['X', 'Z'], 'Z': ['X', 'Z']}
        result = True
        op_ref = 'N'
        for cmd in self.seq:
            op = cmd[0]
            result = result & (op in order_dict[op_ref])
            op_ref = op
        return result

    def shift_signals(self):
        """ Performs signal shifting procedure
        Extract the t-dependence of the measurement into 'S' commands
        and commute them towards the end of the command sequence,
        where it can be deleted.
        In many cases, this procedure simplifies the dependence structure
        of the pattern. For patterns transpiled from gate sequencees,
        this result in the removal of s- and t- domains on Pauli measurement commands.

        Ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)
        """
        if not self.is_standard():
            self.standardize()
        self.extract_signals()
        target = self._find_op_to_be_moved('S', rev = True)
        while target != 'end':
            if target == len(self.seq)-1:
                self.seq.pop(target)
                target = self._find_op_to_be_moved('S', rev = True)
                continue
            if self.seq[target + 1][0] == 'X':
                self._commute_XS(target)
            elif self.seq[target + 1][0] == 'Z':
                self._commute_ZS(target)
            elif self.seq[target + 1][0] == 'M':
                self._commute_MS(target)
            elif self.seq[target + 1][0] == 'S':
                self._commute_SS(target)
            else:
                self._commute_with_following(target)
            target += 1

    def sort_output(self):
        """Add teleportation commands to the sequence,
        to make sure the ordering of the output qubits is the
        same as input logical qubits.
        """
        assert not self.output_sorted
        sorted = deepcopy(self.output_nodes)
        sorted.sort()
        if self.output_nodes == sorted:
            pass
        else:
            new_output_nodes = []
            for i in range(len(self.output_nodes)):
                new_output_nodes.append(self.teleport_node(self.output_nodes[i]))
            self.output_nodes = new_output_nodes
        self.output_sorted = True

    def _find_op_to_be_moved(self, op, rev = False, skipnum = 0):
        """ Internal method for pattern modification.
        Parameters
        ----------
        op : str, 'N', 'E', 'M', 'X', 'Z', 'S'
            command types to be searched

        """
        if not rev: # search from front
            target = 0
            step = 1
        else: # search from back
            target = len(self.seq)-1
            step = -1
        ite = 0
        num_ops = 0
        while ite < len(self.seq):
            if self.seq[target][0] == op:
                num_ops += 1
            if num_ops == skipnum + 1:
                return target
            ite += 1
            target += step
        target = 'end'
        return target

    def _commute_EX(self, target):
        """ Internal method to perform the commutation of E and X.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a X command followed by E command
        """
        assert self.seq[target][0] == 'X'
        assert self.seq[target + 1][0] == 'E'
        X = self.seq[target]
        E = self.seq[target + 1]
        if E[1][0] == X[1]:
            Z = ['Z', E[1][1], X[2]]
            self.seq.pop(target + 1) # del E
            self.seq.insert(target, Z) # add Z in front of X
            self.seq.insert(target, E) # add E in front of Z
            return True
        elif E[1][1] == X[1]:
            Z = ['Z', E[1][0], X[2]]
            self.seq.pop(target + 1) # del E
            self.seq.insert(target, Z) # add Z in front of X
            self.seq.insert(target, E) # add E in front of Z
            return True
        else:
            self._commute_with_following(target)
            return False

    def _commute_MX(self, target):
        """ Internal method to perform the commutation of M and X.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a X command followed by M command
        """
        assert self.seq[target][0] == 'X'
        assert self.seq[target + 1][0] == 'M'
        X = self.seq[target]
        M = self.seq[target + 1]
        if X[1] == M[1]:  # s to s+r
            M[4].extend(X[2])
            self.seq.pop(target) # del X
            return True
        else:
            self._commute_with_following(target)
            return False

    def _commute_MZ(self, target):
        """ Internal method to perform the commutation of M and Z.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a Z command followed by M command
        """
        assert self.seq[target][0] == 'Z'
        assert self.seq[target + 1][0] == 'M'
        Z = self.seq[target]
        M = self.seq[target + 1]
        if Z[1] == M[1]:
            M[5].extend(Z[2])
            self.seq.pop(target) # del Z
            return True
        else:
            self._commute_with_following(target)
            return False

    def _commute_XS(self, target):
        """ Internal method to perform the commutation of X and S.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by X command
        """
        assert self.seq[target][0] == 'S'
        assert self.seq[target + 1][0] == 'X'
        S = self.seq[target]
        X = self.seq[target + 1]
        if np.mod(X[2].count(S[1]), 2):
            X[2].extend(S[2])
        self._commute_with_following(target)

    def _commute_ZS(self, target):
        """ Internal method to perform the commutation of Z and S.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by Z command
        """
        assert self.seq[target][0] == 'S'
        assert self.seq[target + 1][0] == 'Z'
        S = self.seq[target]
        Z = self.seq[target + 1]
        if np.mod(Z[2].count(S[1]), 2):
            Z[2].extend(S[2])
        self._commute_with_following(target)

    def _commute_MS(self, target):
        """ Internal method to perform the commutation of M and S.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by M command
        """
        assert self.seq[target][0] == 'S'
        assert self.seq[target + 1][0] == 'M'
        S = self.seq[target]
        M = self.seq[target + 1]
        if np.mod(M[4].count(S[1]), 2):
            M[4].extend(S[2])
        if np.mod(M[5].count(S[1]), 2):
            M[5].extend(S[2])
        self._commute_with_following(target)

    def _commute_SS(self, target):
        """ Internal method to perform the commutation of two S commands.
        Parameters
        ----------
        target : int
            target command index. this must point to
            a S command followed by S command
        """
        assert self.seq[target][0] == 'S'
        assert self.seq[target + 1][0] == 'S'
        S1 = self.seq[target]
        S2 = self.seq[target + 1]
        if np.mod(S2[2].count(S1[1]), 2):
            S2[2].extend(S1[2])
        self._commute_with_following(target)

    def _commute_with_following(self, target):
        """ Internal method to perform the commutation of
        two consecutive commands that commutes.
        commutes the target command with the following command.

        Parameters
        ----------
        target : int
            target command index
        """
        A = self.seq[target + 1]
        self.seq.pop(target + 1)
        self.seq.insert(target, A)

    def _commute_with_preceding(self, target):
        """ Internal method to perform the commutation of
        two consecutive commands that commutes.
        commutes the target command with the preceding command.

        Parameters
        ----------
        target : int
            target command index
        """
        A = self.seq[target - 1]
        self.seq.pop(target - 1)
        self.seq.insert(target, A)

    def _move_N_to_left(self):
        """Internal method to move all 'N' commands to the start of the sequence.
        N can be moved to the start of sequence without the need of considering
        commutation relations.
        """
        Nlist = []
        for cmd in self.seq:
            if cmd[0] == 'N':
                Nlist.append(cmd)
        Nlist.sort()
        for N in Nlist:
            self.seq.remove(N)
        self.seq = Nlist + self.seq

    def _move_byproduct_to_right(self):
        """Internal method to move the byproduct commands to the end of sequence,
        using the commutation relations implemented in graphix.Pattern class
        """
        # First, we move all X commands to the end of sequence
        moved_X = 0 # number of moved X
        target = self._find_op_to_be_moved('X',rev = True, skipnum = moved_X)
        while target != 'end':
            if (target == len(self.seq)-1) or (self.seq[target + 1] == 'X'):
                moved_X += 1
                target = self._find_op_to_be_moved('X',rev = True, skipnum = moved_X)
                continue
            if self.seq[target + 1][0] == 'E':
                move = self._commute_EX(target)
                if move:
                    target += 1 # addition of extra Z means target must be increased
            elif self.seq[target + 1][0] == 'M':
                search = self._commute_MX(target)
                if search:
                    target = self._find_op_to_be_moved('X', rev = True, skipnum = moved_X)
                    continue # XM commutation rule removes X command
            else:
                self._commute_with_following(target)
            target += 1

        # then, move Z to the end of sequence in front of X
        moved_Z = 0 # number of moved Z
        target = self._find_op_to_be_moved('Z', rev = True, skipnum = moved_Z)
        while target != 'end':
            if (target == len(self.seq) -1) or (self.seq[target + 1][0] == ('X' or 'Z')):
                moved_Z += 1
                target = self._find_op_to_be_moved('Z',rev = True, skipnum = moved_Z)
                continue
            if self.seq[target + 1][0] == 'M':
                search = self._commute_MZ(target)
                if search:
                    target = self._find_op_to_be_moved('Z',rev = True, skipnum = moved_Z)
                    continue # ZM commutation rule removes Z command
            else:
                self._commute_with_following(target)
            target += 1

    def _move_E_after_N(self):
        """Internal method to move all E commands to the start of sequence,
        before all N commands. assumes that _move_N_to_left() method was called.
        """
        moved_E = 0
        target = self._find_op_to_be_moved('E',skipnum = moved_E)
        while target != 'end':
            if (target == 0) or (self.seq[target - 1][0] == ('N' or 'E')):
                moved_E += 1
                target = self._find_op_to_be_moved('E',skipnum = moved_E)
                continue
            self._commute_with_preceding(target)
            target -= 1

    def extract_signals(self):
        """Extracts 't' domain of measurement commands, turn them into
        signal 'S' commands and add to the command sequence.
        This is used for signal_shifting() method.
        """
        pos = 0
        while pos < len(self.seq):
            cmd = self.seq[pos]
            if cmd[0] == 'M':
                node = cmd[1]
                if cmd[5]:
                    self.seq.insert(pos + 1, ['S', node, cmd[5]])
                    cmd[5] = []
                    pos += 1
            pos += 1

    def teleport_node(self, target):
        """Add command sequence for teleportation (Identity gate).
        This is primarily used to reorder the output nodes,
        so that they are sorted in the correct way at the end of statevector simulation.
        Parameters
        ----------
        target : int
            target node to be teleported.
            This must be an existing node in the command sequence
        """
        assert target in np.arange(self.Nnode)
        ancilla = [i for i in range(self.Nnode, self.Nnode + 2)]
        self.Nnode += 2
        for add_node in ancilla:
            self.seq.append(['N', add_node])
        self.seq.append(['E', (target, ancilla[0])])
        self.seq.append(['E', (ancilla[0], ancilla[1])])
        self.seq.append(['M', target, 'XY', 0, [], []])
        self.seq.append(['M', ancilla[0], 'XY', 0, [], []])
        self.seq.append(['X', ancilla[1], [ancilla[0]]])
        self.seq.append(['Z', ancilla[1], [target]])
        return ancilla[1]

    def get_measurement_order(self):
        """Returns the list containing the node indices,
        in the order of measurements
        Returns
        -------
        meas_order : list
            list of node indices in the order of meaurements
        """
        if not self.is_standard():
            self.standardize()
        meas_order = []
        ind = self._find_op_to_be_moved('M')
        if ind == 'end':
            return []
        while self.seq[ind][0] == 'M':
            meas_order.append(self.seq[ind])
            ind += 1
        return meas_order

    def connected_nodes(self, node, prepared = None):
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
            list of nodes that are entangled with speicifed node
        """
        if not self.is_standard():
            self.standardize()
        node_list = []
        ind = self._find_op_to_be_moved('E')
        if not ind == 'end': # end -> 'node' is isolated
            while self.seq[ind][0] == 'E':
                if self.seq[ind][1][0] == node:
                    if not self.seq[ind][1][1] in prepared:
                        node_list.append(self.seq[ind][1][1])
                elif self.seq[ind][1][1] == node:
                    if not self.seq[ind][1][0] in prepared:
                        node_list.append(self.seq[ind][1][0])
                ind += 1
        return node_list

    def correction_commands(self):
        """Returns the list of correction commands
        """
        assert self.is_standard()
        ind = self._find_op_to_be_moved('Z')
        if ind == 'end':
            ind = self._find_op_to_be_moved('X')
            if ind == 'end':
                return []
        Clist = self.seq[ind:]
        return Clist

    def optimize_pattern(self):
        """Optimize the pattern to minimize the max_space property of
        the pattern i.e. the optimized pattern has significantly reduced space requirement (memory space for classical simulation and maximum simultaneously prepared qubits for quantum hardwares).
        """
        if not self.is_standard():
            self.standardize()
        meas_flow = self.get_measurement_order()
        prepared = deepcopy(self.input_nodes)
        measured = []
        new = []
        for cmd in meas_flow:
            node = cmd[1]
            if not node in prepared:
                new.append(['N', node])
                prepared.append(node)
            node_list = self.connected_nodes(node, measured)
            for add_node in node_list:
                if not add_node in prepared:
                    new.append(['N', add_node])
                    prepared.append(add_node)
                new.append(['E', (node, add_node)])
            new.append(cmd)
            measured.append(node)

        # add corrections
        Clist = self.correction_commands()
        new.extend(Clist)

        # add isolated nodes
        for cmd in self.seq:
            if cmd[0] == 'N':
                if not cmd[1] in prepared:
                    new.append(['N', node])

        self.seq = new

    def max_space(self):
        """The maximum number of nodes that must be present in the graph (graph space) during the execution of the pattern.
        For statevector simulation, this is equivalent to the maximum memory
        needed for classical simulation.
        Returns
        -------
        n_nodes : int
            max number of nodes present in the graph during pattern execution.
        """
        max_nodes = 0
        nodes = len(self.input_nodes)
        for cmd in self.seq:
            if cmd[0] == 'N':
                nodes += 1
            elif cmd[0] == 'M':
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
        nodes = len(self.input_nodes)
        N_list = []
        for cmd in self.seq:
            if cmd[0] == 'N':
                nodes += 1
                N_list.append(nodes)
            elif cmd[0] == 'M':
                nodes -= 1
                N_list.append(nodes)
        return N_list

    def simulate_pattern(self, backend='statevector'):
        """Perform simulation of the pattern by using
        graphix.simulator.PatternSimulator class.
        Optional Parameters
        ----------
        backend : str
            simulator backend from {'statevector'}
        """
        sim = PatternSimulator(self, backend=backend)
        sim.run()
        return sim.sv
