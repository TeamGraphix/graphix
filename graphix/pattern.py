from unittest import skip
import numpy as np
from simulator import Simulator
from transpiler import Circuit

class Patterns:
    """Pattern transpiler
    Convert command sequence to NEMC order or optimized order for simulation

    Attributes:
    -----------
    seq : list of commands
        seq = [[cmd1, nodes1], [cmd2, nodes2], ...]
        apply from left to right.
        cmd in {'N', 'M', 'E', 'X', 'Z'}
        nodes of {'N', 'M', 'X', 'Z'} is i: int
        nodes of {'E'} is (i, j)
    sim : Simulator object from simulator
    """

    def __init__(self, simulator):
        self.seq = []
        self.sim = simulator

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
            self.sim.byproductz[E[1][1]].extend(self.sim.byproductx[X[1]]) # E_ijX_i^s = X_i^sZ_j^sE_ij
            self.seq.pop(target + 1) # del E
            self.seq.insert(target, Z) # add Z in front of X
            self.seq.insert(target, E) # add E in front of Z
        elif E[1][1] == X[1]:
            Z = ['Z', E[1][0]]
            self.sim.byproductz[E[1][0]].extend(self.sim.byproductx[X[1]])
            self.seq.pop(target + 1) # del E
            self.seq.insert(target, Z) # add Z in front of X
            self.seq.insert(target, E) # add E in front of Z
        else:
            self.free_commute_R(target)

    # exchange MX^s = M^s
    def commute_MX(self, target):
        X = self.seq[target]
        M = self.seq[target + 1]
        if X[1] == M[1]:
            self.sim.domains[M[1]][0].extend(self.sim.byproductx[X[1]]) # s to s+r
            self.seq.pop(target) # del X
            self.sim.byproductx[X[1]] = []
        else:
            self.free_commute_R(target)

    # exchange MZ^r = ^rM
    def commute_MZ(self, target):
        Z = self.seq[target]
        M = self.seq[target + 1]
        if Z[1] == M[1]:
            self.sim.domains[M[1]][1].extend(self.sim.byproductz[Z[1]])
            self.seq.pop(target) # del Z
            self.sim.byproductz[Z[1]] = []
        else:
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



    ###########Optimization###########
