"""MBQC pattern according to Measurement Calculus
ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)
"""
import numpy as np
from graphix.simulator import PatternSimulator
from graphix.graphsim import GraphState
from graphix.clifford import CLIFFORD_MEASURE, CLIFFORD_CONJ
from copy import deepcopy

class Pattern:
    """ MBQC pattern class
    Pattern holds a sequence of commands to operate the MBQC (Pattern.seq),
    and provide modification strategies to improve the structure and simulation
    efficiency of the pattern accoring to measurement calculus.

    ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)

    Attributes:
    -----------
    width : number of output qubits
    seq : list of commands
        each command is a list [type, nodes, attr] which will be
        applied in the order of list indices.
        type is one of {'N', 'M', 'E', 'X', 'Z', 'S', 'C'}
        nodes : int for {'N', 'M', 'X', 'Z', 'S', 'C'} commands
        nodes : tuple (i, j) for {'E'} command
        attr for N: none
        attr for M: meas_plane, angle, s_domain, t_domain
        attr for X: signal_domain
        attr for Z: signal_domain
        attr for S: signal_domain
        attr for C: clifford_index, as defined in graphix.clifford
    Nnode : int
        total number of nodes in the resource state
    results : dict
        stores measurement results from graph state simulator
    """

    def __init__(self, width):
        """Initialize pattern object
        Parameters
        ---------
        width : int
            number of output qubits
        """
        self.width = width
        self.seq = [['N', i] for i in range(width)]
        self.results = {}
        self.output_nodes = []
        self.Nnode = width

    def add_command(self, cmd):
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
        assert type(cmd) == list
        assert cmd[0] in ['N', 'E', 'M', 'X', 'Z', 'S', 'C']
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
        order_dict = {'N': ['N', 'E', 'M', 'X', 'Z', 'C'], \
                      'E': ['E', 'M', 'X', 'Z', 'C'],\
                      'M': ['M', 'X', 'Z', 'C'],\
                      'X': ['X', 'Z', 'C'],\
                      'Z': ['X', 'Z', 'C'], 'C': ['X', 'Z', 'C']}
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

    def _find_op_to_be_moved(self, op, rev = False, skipnum = 0):
        """ Internal method for pattern modification.
        Parameters
        ----------
        op : str, 'N', 'E', 'M', 'X', 'Z', 'S'
            command types to be searched
        rev : bool
            search from the end (true) or start (false) of seq
        skipnum : int
            skip the detected command by specified times
        """
        if not rev: # search from the start
            target = 0
            step = 1
        else: # search from the back
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
        target = self._find_op_to_be_moved('X', rev = True, skipnum = moved_X)
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
        This is used for shift_signals() method.
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

    def _get_dependency(self):
        """Get dependency (byproduct correction & dependent measurement)
        structure of nodes in the graph (resource) state, according to the pattern.
        This is used to determine the optimum measurement order.

        Returns
        -------
        dependency : dict of sets
            index is node number. all nodes in the each set must be measured before measuring
        """
        nodes, _ = self.get_graph()
        dependency = {i: set() for i in nodes}
        for cmd in self.seq:
            if cmd[0] == 'M':
                dependency[cmd[1]] = dependency[cmd[1]] | set(cmd[4]) | set(cmd[5])
            elif cmd[0] == 'X':
                dependency[cmd[1]] = dependency[cmd[1]] | set(cmd[2])
            elif cmd[0] =='Z':
                dependency[cmd[1]] = dependency[cmd[1]] | set(cmd[2])
        return dependency

    def update_dependency(self, measured, dependency):
        """Update dependency function. remove measured nodes from dependency.

        Parameters
        ---------
        measured: set
            measured nodes.
        dependency: dict of sets
            which is produced by `_get_dependency`

        Returns
        --------
        dependency: dict of sets
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
        depth: int
            depth of graph
        l_k: dict of sets
            nodes grouped by layer index(k)
        """
        dependency = self._get_dependency()
        measured = self.results.keys()
        dependency = self.update_dependency(measured, dependency)
        not_measured = set()
        for cmd in self.seq:
            if cmd[0] == 'N':
                if not cmd[1] in self.output_nodes:
                    not_measured = not_measured | {cmd[1]}
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
        """Obtain the measurement order which reduces the depth of
        pattern execution.

        Returns
        -------
        meas_order: list of ints
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
        connected: set of taples
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
        """Determine the optimal measurement order
         which minimize the `max_space` of the pattern.

        Returns
        -------
        meas_order: list of ints
            optimal measurement order for classical simulation
        """
        nodes, edges = self.get_graph()
        nodes = set(nodes)
        edges = set(edges)
        not_measured = nodes - set(self.output_nodes)
        dependency = self._get_dependency()
        dependency = self.update_dependency(self.results.keys(),dependency)
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
            assert next_node > -1
            meas_order.append(next_node)
            dependency = self.update_dependency({next_node}, dependency)
            not_measured -= {next_node}
            edges -= removable_edges
        return meas_order

    def sort_measurement_commands(self, meas_order):
        """Convert measurement order to sequence of measurement commands

        Parameters
        --------
        meas_order: list of ints
            optimal measurement order.

        Returns
        --------
        meas_flow: list of commands
            sorted measurement commands
        """
        meas_flow = []
        for i in meas_order:
            target = 0
            while True:
                if (self.seq[target][0] == 'M') &(self.seq[target][1] == i):
                    meas_flow.append(self.seq[target])
                    break
                target += 1
        return meas_flow

    def get_measurement_order(self):
        """Returns the list containing the node indices,
        in the order of measurements
        Returns
        -------
        meas_flow : list
            list of node indices in the order of meaurements
        """
        if not self.is_standard():
            self.standardize()
        meas_flow = []
        ind = self._find_op_to_be_moved('M')
        if ind == 'end':
            return []
        while self.seq[ind][0] == 'M':
            meas_flow.append(self.seq[ind])
            ind += 1
        return meas_flow

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
        node_list, edge_list = [], []
        for cmd in self.seq:
            if cmd[0] == 'N':
                node_list.append(cmd[1])
            elif cmd[0] == 'E':
                edge_list.append(cmd[1])
        return node_list, edge_list

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
        """Returns the list of byproduct correction commands
        """
        assert self.is_standard()
        ind = self._find_op_to_be_moved('Z')
        if ind == 'end':
            ind = self._find_op_to_be_moved('X')
            if ind == 'end':
                return []
        Clist = self.seq[ind:]
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
        meas_order = self._measurement_order_space()
        self._reorder_pattern(self.sort_measurement_commands(meas_order))

    def _reorder_pattern(self, meas_commands):
        """internal method to reorder the command sequence
        Parameters
        ----------
        meas_commands : list of commands
            list of measurement ('M') commands
        """
        prepared = []
        measured = []
        new = []
        for cmd in meas_commands:
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

        # add isolated nodes
        for cmd in self.seq:
            if cmd[0] == 'N':
                if not cmd[1] in prepared:
                    new.append(['N', cmd[1]])

        # add Clifford nodes
        for cmd in self.seq:
            if cmd[0] == 'C':
                new.append(cmd)

        # add corrections
        c_list = self.correction_commands()
        new.extend(c_list)

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
        nodes = 0
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
        nodes = 0
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

    def perform_pauli_measurements(self):
        """Perform Pauli measurements in the pattern using
        efficient stabilizer simulator """
        measure_pauli(self, copy=False)


def measure_pauli(pattern, copy=False):
    """Perform Pauli measurement of a pattern by fast graph state simulator
    uses the decorated-graph method implemented in graphix.graphsim to perform
    the measurements in Pauli bases, and then sort remaining nodes back into
    pattern together with Clifford commands.

    TODO: non-XY plane measurements in original pattern

    Parameters
    ----------
    pattern : grgaphix.Pattern object
    copy : bool
        True: changes will be applied to new copied object and will be returned
        False: changes will be applied to the supplied Pattern object

    Returns
    ------
    new_pattern : graphix.Pattern object
        pattern with Pauli measurement removed.
        only returned if copy argument is True.
    """
    if not pattern.is_standard():
        pattern.standardize()
    nodes, edges = pattern.get_graph()
    graph_state = GraphState(nodes=nodes, edges=edges)
    results = {}
    to_measure = pauli_nodes(pattern)
    for cmd in to_measure:
        # extract signals for adaptive angle
        if np.mod(cmd[3], 2) in [0, 1]: # \pm X Pauli measurement
            s_signal = 0 # X meaurement is not affected by s_signal
            t_signal = np.sum([results[j] for j in cmd[5]])
        elif np.mod(cmd[3], 2) in [0.5, 1.5]: # \pm Y Pauli measurement
            s_signal = np.sum([results[j] for j in cmd[4]])
            t_signal = np.sum([results[j] for j in cmd[5]])
        angle = cmd[3] * (-1)**s_signal + t_signal
        if np.mod(angle, 2) == 0:  # +x measurement
            results[cmd[1]] = graph_state.measure_x(cmd[1], choice=0)
        elif np.mod(angle, 2) == 1:  # -x measurement
            results[cmd[1]] = 1 - graph_state.measure_x(cmd[1], choice=1)
        elif np.mod(angle, 2) == 0.5:  # +y measurement
            results[cmd[1]] = graph_state.measure_y(cmd[1], choice=0)
        elif np.mod(angle, 2) == 1.5:  # -y measurement
            results[cmd[1]] = 1 - graph_state.measure_y(cmd[1], choice=1)

    # update command sequence
    pattern.Nnode = len(graph_state.nodes)
    vops = graph_state.get_vops()
    new_seq = []
    for index in iter(graph_state.nodes):
        new_seq.append(['N', index])
    for edge in iter(graph_state.edges):
        new_seq.append(['E', edge])
    for cmd in pattern.seq:
        if cmd[0] == 'M':
            if not cmd in to_measure:
                cmd_new = deepcopy(cmd)
                cmd_new.append(CLIFFORD_CONJ[vops[cmd[1]]])
                new_seq.append(cmd_new)
    for index in pattern.output_nodes:
        new_seq.append(['C', index, vops[index]])
    for cmd in pattern.seq:
        if cmd[0] == 'X' or cmd[0] == 'Z':
            new_seq.append(cmd)
    if copy:
        pat = deepcopy(pattern)
        pat.seq = new_seq
        pat.results = results
        return pat
    else:
        pattern.seq = new_seq
        pattern.results = results


def pauli_nodes(pattern):
    """returns the list of measurement commands that are in Pauli bases
    and that are not dependent on any non-Pauli measurements
    Parameters
    ----------
    pattern : graphix.Pattern object

    Returns
    -------
    pauli_node : list
        list of node indices
    """
    if not pattern.is_standard():
        pattern.standardize()
    m_commands = pattern.get_measurement_order()
    pauli_node = []
    non_pauli_node = []
    for cmd in m_commands:
        if cmd[3] in [-1, 0, 1]: # \pm X Pauli measurement
            t_cond = np.any(np.isin(cmd[5], non_pauli_node))
            if t_cond: # cmd depend on non-Pauli measurement
                non_pauli_node.append(cmd)
            else:# cmd do not depend on non-Pauli measurements
                # note: s_signal is irrelevant for X measurements
                # because change of sign will do nothing
                pauli_node.append(cmd)
        elif cmd[3] in [-0.5, 0.5]: # \pm Y Pauli measurement
            s_cond = np.any(np.isin(cmd[4], non_pauli_node))
            t_cond = np.any(np.isin(cmd[5], non_pauli_node))
            if s_cond or t_cond: # cmd depend on non-pauli measurement
                non_pauli_node.append(cmd)
            else:
                pauli_node.append(cmd)
        else:
            non_pauli_node.append(cmd)
    return pauli_node
