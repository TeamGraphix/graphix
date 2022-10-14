import numpy as np
from qiskit.quantum_info import Operator, Statevector, partial_trace
from graphix.ops import Ops
from graphix.simulator import Simulator
from graphix.clifford import CLIFFORD

class Patterns:
    """Pattern transpiler
    Convert command sequence to NEMC order or optimized order for simulation

    Attributes:
    -----------
    seq : list of commands
        seq = [[cmd1, nodes1], [cmd2, nodes2], ...]
        apply from left to right.
        cmd in {'N', 'M', 'E', 'X', 'Z', 'S'}
        nodes of {'N', 'M', 'X', 'Z', 'S'} is i: int
        nodes of {'E'} is (i, j): tuple
    sim : Simulator object from simulator
    sv : qiskit.quantum_info.Statevector
    node_index : list
        the order in list corresponds to the order of tensor product subspaces
    results : dict
        storage measurement results
    signal : dict
        storage signal used in signal shifting
    """

    def __init__(self, simulator):
        self.seq = []
        self.sim = simulator
        self.sv = Statevector([])
        self.node_index = []
        self.results = {}
        self.signal = {}

    ###########read Patterns from Circuit###########

    # search necessary edges to be prepared
    def search_edges(self, node, prepared_list):
        edges = []
        for edge in self.sim.edges:
            if (edge[0] == node) and not (edge[1] in prepared_list):
                edges.append(edge)
            elif (edge[1] == node) and not (edge[0] in prepared_list):
                edges.append(edge)
        return edges

    # read patterns from graph
    def read_patterns(self):
        # prepare all necessary nodes
        for node in range(self.sim.circ.width, self.sim.nodes):
            self.seq.append(['N', node])

        # append commands
        prepared_list = []
        for node in self.sim.measurement_order:
            edges = self.search_edges(node, prepared_list)
            for edge in edges:
                self.seq.append(['E', edge])
            if node in self.sim.byproductx:
                if self.sim.byproductx[node]:
                    self.seq.append(['X', node])
            if node in self.sim.byproductz:
                if self.sim.byproductz[node]:
                    self.seq.append(['Z', node])
            self.seq.append(['M', node])
            prepared_list.append(node)

        for node in self.sim.output_nodes:
            if node in self.sim.byproductx:
                if self.sim.byproductx[node]:
                    self.seq.append(['X', node])
            if node in self.sim.byproductz:
                if self.sim.byproductz[node]:
                    self.seq.append(['Z', node])



    ###########commutation function###########


    # exchange EX = XZE
    def commute_EX(self, target):
        X = self.seq[target]
        E = self.seq[target + 1]
        if E[1][0] == X[1]:
            Z = ['Z', E[1][1]]
            if X[1] in self.sim.byproductx.keys():
                if E[1][1] in self.sim.byproductz.keys(): # E_ijX_i^s = X_i^sZ_j^sE_ij
                    self.sim.byproductz[E[1][1]].extend(self.sim.byproductx[X[1]])
                else:
                    self.sim.byproductz[E[1][1]] = self.sim.byproductx[X[1]]
            self.seq.pop(target + 1) # del E
            self.seq.insert(target, Z) # add Z in front of X
            self.seq.insert(target, E) # add E in front of Z
        elif E[1][1] == X[1]:
            Z = ['Z', E[1][0]]
            if X[1] in self.sim.byproductx.keys():
                if E[1][0] in self.sim.byproductz.keys(): # E_ijX_i^s = X_i^sZ_j^sE_ij
                    self.sim.byproductz[E[1][0]].extend(self.sim.byproductx[X[1]])
                else:
                    self.sim.byproductz[E[1][0]] = self.sim.byproductx[X[1]]
            self.seq.pop(target + 1) # del E
            self.seq.insert(target, Z) # add Z in front of X
            self.seq.insert(target, E) # add E in front of Z
        else:
            self.free_commute_R(target)

    # exchange MX^s = M^s
    def commute_MX(self, target):
        X = self.seq[target]
        M = self.seq[target + 1]
        if X[1] == M[1]:  # s to s+r
            if X[1] in self.sim.byproductx.keys():
                if M[1] in self.sim.domains.keys():
                    self.sim.domains[M[1]][0].extend(self.sim.byproductx[X[1]])
                else:
                    self.sim.domains[M[1]][0] = self.sim.byproductx[X[1]]

            self.seq.pop(target) # del X
            self.sim.byproductx[X[1]] = []
        else:
            self.free_commute_R(target)

    # exchange MZ^r = ^rM
    def commute_MZ(self, target):
        Z = self.seq[target]
        M = self.seq[target + 1]
        if Z[1] == M[1]:
            if Z[1] in self.sim.byproductz.keys():
                if M[1] in self.sim.domains.keys():
                    self.sim.domains[M[1]][1].extend(self.sim.byproductz[Z[1]])
                else:
                    self.sim.domains[M[1]] = [self.sim.byproductz[Z[1]], []]
            self.seq.pop(target) # del Z
            self.sim.byproductz[Z[1]] = []
        else:
            self.free_commute_R(target)

    # exchange XS
    def commute_XS(self, target):
        S = self.seq[target]
        X = self.seq[target + 1]
        if S[1] in self.sim.byproductx[X[1]]:
            self.sim.byproductx[X[1]].extend(self.signal[S[1]])
        self.free_commute_R(target)

    # exchange ZS
    def commute_ZS(self, target):
        S = self.seq[target]
        Z = self.seq[target + 1]
        if S[1] in self.sim.byproductz[Z[1]]:
            self.sim.byproductz[Z[1]].extend(self.signal[S[1]])
        self.free_commute_R(target)

    # exchange MS
    def commute_MS(self, target):
        S = self.seq[target]
        M = self.seq[target + 1]
        if S[1] in self.sim.domains[M[1]][0]:
            self.sim.domains[M[1]][0].extend(self.signal[S[1]])
        if S[1] in self.sim.domains[M[1]][1]:
            self.sim.domains[M[1]][1].extend(self.signal[S[1]])
        self.free_commute_R(target)

    # exchange SS
    def commute_SS(self, target):
        S1 = self.seq[target]
        S2 = self.seq[target + 1]
        if S1[1] in self.signal[S2[1]]:
            self.signal[S2[1]].extend(self.signal[S1[1]])
        self.free_commute_R(target)

    # free commutation
    def free_commute_R(self, target):
        A = self.seq[target + 1]
        self.seq.pop(target + 1)
        self.seq.insert(target, A)

    def free_commute_L(self, target):
        A = self.seq[target - 1]
        self.seq.pop(target - 1)
        self.seq.insert(target, A)

    ###########Standardization###########
    def find_op_to_be_moved(self, op, rev = False, skipnum = 0):
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

    # Algorithm-1 move Preparations
    def NtoLeft(self):
        # commutation rule of AN(for any A) is free commutation. so drop all Ns and add in the beginning of sequence
        Nlist = []
        for cmd in self.seq:
            if cmd[0] == 'N':
                Nlist.append(cmd)
        Nlist.sort()
        for N in Nlist:
            self.seq.remove(N)
        self.seq = Nlist + self.seq

    # Algorithm-2 move Corrections(X,Z)
    def CtoRight(self):
        # move X first
        moved_X = 0 # storage number of moved X
        target = self.find_op_to_be_moved('X',rev = True, skipnum = moved_X)
        while target != 'end':
            if (target == len(self.seq)-1) or (self.seq[target + 1] == 'X'):
                moved_X += 1
                target = self.find_op_to_be_moved('X',rev = True, skipnum = moved_X)
                continue
            if self.seq[target + 1][0] == 'E':
                self.commute_EX(target)
            elif self.seq[target + 1][0] == 'M':
                self.commute_MX(target)
                target = self.find_op_to_be_moved('X', rev = True, skipnum = moved_X)
                continue # when XM commutation rule applied, X will be removed
            else:
                self.free_commute_R(target)
            # update target
            target += 1
        # move Z in front of X
        moved_Z = 0 # storage number of moved Z
        target = self.find_op_to_be_moved('Z', rev = True, skipnum = moved_Z)
        while target != 'end':
            if (target == len(self.seq) -1) or (self.seq[target + 1][0] == ('X' or 'Z')):
                moved_Z += 1
                target = self.find_op_to_be_moved('Z',rev = True, skipnum = moved_Z)
                continue
            if self.seq[target + 1][0] == 'M':
                self.commute_MZ(target)
                target = self.find_op_to_be_moved('Z',rev = True, skipnum = moved_Z)
                continue # when ZM commutation rule applied, Z will be removed
            else:
                self.free_commute_R(target)
            # update target
            target += 1
    # Algorithm-3 move Entanglement
    def EtoNextN(self):
        moved_E = 0
        target = self.find_op_to_be_moved('E',skipnum = moved_E)
        while target != 'end':
            if (target == 0) or (self.seq[target - 1][0] == ('N' or 'E')):
                moved_E += 1
                target = self.find_op_to_be_moved('E',skipnum = moved_E)
                continue
            self.free_commute_L(target)
            target -= 1

    # extract signal from measurement operator M
    def extract_S(self):
        pos = 0
        while pos < len(self.seq):
            cmd = self.seq[pos]
            if cmd[0] == 'M':
                node = cmd[1]
                if self.sim.domains[node][1]:
                    self.signal[node] = self.sim.domains[node][1]
                    self.seq.insert(pos + 1, ['S', node])
                    self.sim.domains[node][1] = [] # delete signal from 'M' op.
                    pos += 1
            pos += 1

    # signal shifting
    def signal_shifting(self):
        if not self.is_NEMC():
            self.Standardization()
        self.extract_S()
        target = self.find_op_to_be_moved('S', rev = True)
        while target != 'end':
            if target == len(self.seq)-1:
                self.seq.pop(target)
                target = self.find_op_to_be_moved('S', rev = True)
                continue
            if self.seq[target + 1][0] == 'X':
                self.commute_XS(target)
            elif self.seq[target + 1][0] == 'Z':
                self.commute_ZS(target)
            elif self.seq[target + 1][0] == 'M':
                self.commute_MS(target)
            elif self.seq[target + 1][0] == 'S':
                self.commute_SS(target)
            else:
                self.free_commute_R(target)
            target += 1

    # execute Standardization
    def Standardization(self):
        self.NtoLeft()
        self.CtoRight()
        self.EtoNextN()

    # check whether seq is NEMC
    def is_NEMC(self):
        order_dict = {'N': ['N', 'E', 'M', 'X', 'Z'], 'E': ['E', 'M', 'X', 'Z'], 'M': ['M', 'X', 'Z'], 'X': ['X', 'Z'], 'Z': ['X', 'Z']}
        result = True
        op_ref = 'N'
        for cmd in self.seq:
            op = cmd[0]
            result = result & (op in order_dict[op_ref])
            op_ref = op
        return result



    # ###########Optimization###########
    # free commutation rule is applied to N, E, M
    # get measurement order list
    def get_meas_flow(self): # this function can be replaced by more efficient algorithm
        if not self.is_NEMC():
            self.Standardization()
        meas_flow = []
        ind = self.find_op_to_be_moved('M')
        if ind == 'end':
            return []
        while self.seq[ind][0] == 'M':
            meas_flow.append(self.seq[ind])
            ind += 1
        return meas_flow

    # get necessary nodes list for measure the target node
    def search_must_prepared(self, node, prepared_list = None):
        if not self.is_NEMC():
            self.Standardization()
        node_list = []
        ind = self.find_op_to_be_moved('E')
        while self.seq[ind ][0] == 'E':
            if self.seq[ind][1][0] == node:
                if not self.seq[ind][1][1] in prepared_list:
                    node_list.append(self.seq[ind][1][1])
            elif self.seq[ind][1][1] == node:
                if not self.seq[ind][1][0] in prepared_list:
                    node_list.append(self.seq[ind][1][0])
            ind += 1
        return node_list

    # get byproduct correction sequence
    def getC(self):
        ind = self.find_op_to_be_moved('Z')
        if ind == 'end':
            ind = self.find_op_to_be_moved('X')
            if ind == 'end':
                return []
        Clist = self.seq[ind:]
        return Clist


    # execute Optimization
    def Optimization(self):
        if not self.is_NEMC():
            self.Standardization()
        meas_flow = self.get_meas_flow()
        prepared = [i for i in range(self.sim.circ.width)]
        measured = []
        new = []
        for cmd in meas_flow:
            node = cmd[1]
            if not node in prepared:
                new.append(['N', node])
                prepared.append(node)
            node_list = self.search_must_prepared(node, measured)
            for add_node in node_list:
                if not add_node in prepared:
                    new.append(['N', add_node])
                    prepared.append(add_node)
                new.append(['E', (node, add_node)])
            new.append(cmd)
            measured.append(node)

        # add corrections
        Clist = self.getC()
        new.extend(Clist)

        # add not-prepared nodes
        for node in range(self.sim.nodes):
            if not node in prepared:
                new.append(['N', node])

        self.seq = new


    ##### functions for investigation of pattern's properties ####
    # count max necessary nodes
    def minQR(self):
        max_nodes = 0
        nodes = self.sim.circ.width
        for cmd in self.seq:
            if cmd[0] == 'N':
                nodes += 1
            elif cmd[0] == 'M':
                nodes -= 1
            if nodes > max_nodes:
                max_nodes = nodes
        return max_nodes

    # return the number of prepared nodes at each N or M cmd
    def Nnodes_list(self):
        nodes = self.sim.circ.width
        N_list = []
        for cmd in self.seq:
            if cmd[0] == 'N':
                nodes += 1
                N_list.append(nodes)
            elif cmd[0] == 'M':
                nodes -= 1
                N_list.append(nodes)
        return N_list




    #### execute commands ####
    # return qubit number
    def count_qubit(self):
        return len(self.sv.dims())

    # normalize statevector
    def Normalization(self):
        self.sv = self.sv/self.sv.trace()**0.5

    # set input statevector
    def set_state(self, statevector):
        self.sv = statevector
        self.node_index.extend([i for i in range(self.count_qubit())])

    def set_initial(self):
        n = self.sim.circ.width
        self.sv = Statevector([1 for i in range(2**n)])
        self.Normalization()
        self.node_index.extend([i for i in range(n)])

    # add |+>^n state nodes
    def add_nodes(self, nodes):
        n = len(nodes)
        add_vector = Statevector([1 for i in range(2**n)])
        add_vector = add_vector/add_vector.trace()**0.5
        self.sv = self.sv.expand(add_vector)
        self.node_index.extend(nodes)

    # make entanglement
    def make_entangle(self,edge):
        target = self.node_index.index(edge[0])
        control = self.node_index.index(edge[1])
        self.sv = self.sv.evolve(Ops.cz, [control, target])

    # measure and remove measured subsystem
    def measure(self, node, angle):
        result = np.random.choice([0, 1])
        self.results[node] = result
        s_signal, t_signal = self.sim.extract_signal(node, self.results)
        angle = self.sim.angles[node] * np.pi * (-1)**s_signal + np.pi * t_signal

        meas_op = self.sim.meas_op(angle, self.sim.vop[node], choice = result)
        loc = self.node_index.index(node)
        self.sv = self.sv.evolve(meas_op, [loc])

        # trace out
        self.Normalization()
        state_dm = partial_trace(self.sv, [loc])
        self.sv = state_dm.to_statevector()

        # update node_index
        self.node_index.remove(node)

    # Feedforward
    def XFeedForward(self, node):
        if node in self.sim.byproductx.keys():
            if np.mod(np.sum([self.results[j] for j in self.sim.byproductx[node]]), 2):
                loc = self.node_index.index(node)
                self.sv = self.sv.evolve(Ops.x, [loc])

    def ZFeedForward(self, node):
        if node in self.sim.byproductz.keys():
            if np.mod(np.sum([self.results[j] for j in self.sim.byproductz[node]]), 2):
                loc = self.node_index.index(node)
                self.sv = self.sv.evolve(Ops.z, [loc])

    # simulate the pattern
    def execute_ptn(self):
        self.set_initial()
        for cmd in self.seq:
            if cmd[0] == 'N':
                self.add_nodes([cmd[1]])
            elif cmd[0] == 'E':
                self.make_entangle(cmd[1])
            elif cmd[0] == 'M':
                self.measure(cmd[1], self.sim.angles[cmd[1]])
            elif cmd[0] == 'X':
                self.XFeedForward(cmd[1])
            elif cmd[0] == 'Z':
                self.ZFeedForward(cmd[1])
            else:
                raise ValueError("invalid commands")
        # apply VOP to output vertices
        for i, j in enumerate(self.sim.output_nodes):
            self.sv = self.sv.evolve(Operator(CLIFFORD[self.sim.vop[j]]), [i])

