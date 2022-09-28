"""Gate-to-MBQC transpiler

accepts desired gate operations and transpile into MBQC measurement patterns.

"""
import numpy as np
from graphix.ops import States, Ops
from copy import deepcopy


class Circuit():
    """Gate-to-MBQC transpiler.

    Holds gate operations and translates into MBQC measurement patterns.
    Each nodes lie on 'rails' associated with logical qubits (in gate network).
    For example, applying CNOT gates to two-qubit state results in the graph below

    0 - - - 4 - 5
            |
    1 - 2 - 3

    Note that we use identity gate together with CNOT to avoid
    circular reference of byproduct (in this simple implementation).

    Attributes
    ----------
    width : int
        Number of logical qubits (for gate network)
    instruction : list
        List of gates applied
    edges : list
        Edges of the graph state.
    Nnode : int
        Total number of nodes required for the MBQC pattern
    lengths : list
        Total number of nodes associated with each logical qubit
    angles : dict
        Meausrement angles for each node
    measurement_order : list
        order of qubits to be measured to ensure determinisum
    byproductx : dict
        X Byproduct domains which apply to each node
    byproductx : dict
        Z Byproduct domains which apply to each node
    domains :
        Adaptive measurement domains (s and t in Measurement Calculus language)
    next_node : dict
        Next node in the 'rail' of logical qubit
    out : array
        Output node indices for each logical qubits
    pos : dict
        positions of nodes for graph drawing

    TODO: add LC instead of CLifford graphs
    """

    def __init__(self, width):
        """
        Parameters
        ----------
        width : int
            number of logical qubits for the gate network
        """
        self.width = width
        self.instruction = []

        self.Nnode = width
        self.edges = []
        self.lengths = [1 for j in range(width)]
        self.out = [j for j in range(width)]
        self.angles = dict()
        self.measurement_order = []
        self.byproductz = {j: [] for j in range(width)}
        self.byproductx = {j: [] for j in range(width)}
        self.next_node = dict()
        self.domains = dict()
        self.pos = {j: [0, -j] for j in range(width)}

    def cnot(self, control, target):
        """CNOT + identity gate
        Identity is necessary to simplify the measurement order.
        edges:
        t   -    a2 - a3
                  |
        c - a0 - a1


        Prameters
        ---------
        control : int
            control qubit
        target : int
            target qubit
        """
        assert control in np.arange(self.width)
        assert target in np.arange(self.width)
        assert control != target

        # assign new qubit labels
        ancilla = self.Nnode + np.arange(4, dtype=np.int8)
        self.Nnode += 4

        self.pos[ancilla[2]] = [self.lengths[target], -target]
        self.pos[ancilla[3]] = [self.lengths[target] + 1, -target]
        self.pos[ancilla[0]] = [self.lengths[control], -control]
        self.pos[ancilla[1]] = [self.lengths[control] + 1, -control]

        self.edges.append((self.out[target], ancilla[2]))
        self.edges.append((self.out[control], ancilla[0]))
        if self.byproductx[self.out[control]]:  # E_ij X_i = X_i Z_j E_ij
            self.byproductz[ancilla[0]] = self.byproductx[self.out[control]]

        self.edges.append((ancilla[0], ancilla[1]))
        self.edges.append((ancilla[1], ancilla[2]))
        self.edges.append((ancilla[2], ancilla[3]))

        self.byproductx[ancilla[3]] = [ancilla[2]]
        self.byproductz[ancilla[3]] = [self.out[target]]
        self.byproductx[ancilla[0]] = [self.out[control]]
        self.byproductx[ancilla[1]] = [ancilla[0]]
        self.byproductz[ancilla[1]] = [self.out[target], self.out[control]]
        self.byproductz[ancilla[2]] = [ancilla[0]]
        if self.byproductx[self.out[target]]:  # E_ij X_i = X_i Z_j E_ij
            for j in self.byproductx[self.out[target]]:
                self.byproductz[ancilla[2]].append(j)

        self.domains[self.out[target]] = [[], []]
        self.domains[self.out[control]] = [[], []]
        self.domains[ancilla[0]] = [[self.out[control]], []]
        self.domains[ancilla[2]] = [[], []]

        self.angles[self.out[target]] = 0
        self.angles[self.out[control]] = 0
        self.angles[ancilla[0]] = 0
        self.angles[ancilla[2]] = 0

        self.next_node[self.out[target]] = ancilla[2]
        self.next_node[ancilla[2]] = ancilla[3]
        self.next_node[self.out[control]] = ancilla[0]
        self.next_node[ancilla[0]] = ancilla[1]

        self.measurement_order.append(self.out[control])
        self.measurement_order.append(self.out[target])
        self.measurement_order.append(ancilla[0])
        self.measurement_order.append(ancilla[2])

        self.lengths[control] += 2
        self.lengths[target] += 2
        self.out[control] = ancilla[1]
        self.out[target] = ancilla[3]
        self.instruction.append(['CNOT', [target, control]])

    def h(self, qubit):
        """Hadamard gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        ancilla = self.Nnode
        self.Nnode += 1

        self.pos[ancilla] = [self.lengths[qubit], -qubit]
        self.edges.append((self.out[qubit], ancilla))
        if self.byproductx[self.out[qubit]]:  # E_ij X_i = X_i Z_j E_ij
            self.byproductz[ancilla] = self.byproductx[self.out[qubit]]
        else:
            self.byproductz[ancilla] = []
        self.byproductx[ancilla] = [self.out[qubit]]
        self.next_node[self.out[qubit]] = ancilla
        self.domains[self.out[qubit]] = [[], []]
        self.angles[self.out[qubit]] = 0
        self.measurement_order.append(self.out[qubit])

        self.lengths[qubit] += 1
        self.out[qubit] = ancilla
        self.instruction.append(['H', [qubit]])

    def s(self, qubit):
        """S gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)

        # assign new qubit labels
        ancilla1, ancilla2 = self.Nnode, self.Nnode + 1
        self.Nnode += 2

        self.pos[ancilla1] = [self.lengths[qubit], -qubit]
        self.pos[ancilla2] = [self.lengths[qubit] + 1, -qubit]

        self.edges.append((self.out[qubit], ancilla1))
        if self.byproductx[self.out[qubit]]:  # E_ij X_i = X_i Z_j E_ij
            self.byproductz[ancilla1] = self.byproductx[self.out[qubit]]

        self.edges.append((ancilla1, ancilla2))
        self.next_node[self.out[qubit]] = ancilla1
        self.next_node[ancilla1] = ancilla2

        self.byproductx[ancilla2] = [ancilla1]
        self.byproductz[ancilla2] = [self.out[qubit]]

        self.domains[self.out[qubit]] = [[], []]
        self.domains[ancilla1] = [[], []]

        self.angles[self.out[qubit]] = -0.5
        self.angles[ancilla1] = 0
        self.measurement_order.append(self.out[qubit])
        self.measurement_order.append(ancilla1)

        self.lengths[qubit] += 2
        self.out[qubit] = ancilla2
        self.instruction.append(['S', [qubit]])

    def x(self, qubit):
        """Pauli X gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)

        # assign new qubit labels
        ancilla1, ancilla2 = self.Nnode, self.Nnode + 1
        self.Nnode += 2

        self.pos[ancilla1] = [self.lengths[qubit], -qubit]
        self.pos[ancilla2] = [self.lengths[qubit] + 1, -qubit]

        self.edges.append((self.out[qubit], ancilla1))
        if self.byproductx[self.out[qubit]]:  # E_ij X_i = X_i Z_j E_ij
            self.byproductz[ancilla1] = self.byproductx[self.out[qubit]]

        self.edges.append((ancilla1, ancilla2))
        self.next_node[self.out[qubit]] = ancilla1
        self.next_node[ancilla1] = ancilla2

        self.byproductx[ancilla2] = [ancilla1]
        self.byproductz[ancilla2] = [self.out[qubit]]

        self.domains[self.out[qubit]] = [[], []]
        self.domains[ancilla1] = [[], []]

        self.angles[self.out[qubit]] = 0
        self.angles[ancilla1] = -1
        self.measurement_order.append(self.out[qubit])
        self.measurement_order.append(ancilla1)

        self.lengths[qubit] += 2
        self.out[qubit] = ancilla2
        self.instruction.append(['X', [qubit]])

    def y(self, qubit):
        """Pauli Y gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)

        # assign new qubit labels
        ancilla = self.Nnode + np.arange(4, dtype=np.int8)
        self.Nnode += 4

        # straight graph
        for i in range(4):
            self.pos[ancilla[i]] = [self.lengths[qubit] + i, -qubit]

        self.edges.append((self.out[qubit], ancilla[0]))
        self.next_node[self.out[qubit]] = ancilla[0]
        for i in range(3):
            self.edges.append((ancilla[i], ancilla[i + 1]))
            self.next_node[ancilla[i]] = ancilla[i + 1]
        if self.byproductx[self.out[qubit]]:  # E_ij X_i = X_i Z_j E_ij
            self.byproductz[ancilla[0]] = self.byproductx[self.out[qubit]]

        self.byproductx[ancilla[3]] = [ancilla[2]]
        self.byproductz[ancilla[3]] = [ancilla[1]]
        self.domains[self.out[qubit]] = [[], []]
        self.domains[ancilla[0]] = [[], []]
        self.domains[ancilla[1]] = [[ancilla[0]], [self.out[qubit]]]
        self.domains[ancilla[2]] = [[], [ancilla[0]]]

        self.angles[self.out[qubit]] = 0
        self.angles[ancilla[0]] = -1
        self.angles[ancilla[1]] = -1
        self.angles[ancilla[2]] = 0
        self.measurement_order.append(self.out[qubit])
        self.measurement_order.append(ancilla[0])
        self.measurement_order.append(ancilla[1])
        self.measurement_order.append(ancilla[2])

        self.lengths[qubit] += 4
        self.out[qubit] = ancilla[3]
        self.instruction.append(['Y', [qubit]])

    def z(self, qubit):
        """Pauli Z gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)

        # assign new qubit labels
        ancilla1, ancilla2 = self.Nnode, self.Nnode + 1
        self.Nnode += 2

        self.pos[ancilla1] = [self.lengths[qubit], -qubit]
        self.pos[ancilla2] = [self.lengths[qubit] + 1, -qubit]

        self.edges.append((self.out[qubit], ancilla1))
        if self.byproductx[self.out[qubit]]:  # E_ij X_i = X_i Z_j E_ij
            self.byproductz[ancilla1] = self.byproductx[self.out[qubit]]

        self.edges.append((ancilla1, ancilla2))
        self.next_node[self.out[qubit]] = ancilla1
        self.next_node[ancilla1] = ancilla2

        self.byproductx[ancilla2] = [ancilla1]
        self.byproductz[ancilla2] = [self.out[qubit]]

        self.domains[self.out[qubit]] = [[], []]
        self.domains[ancilla1] = [[], []]

        self.angles[self.out[qubit]] = -1
        self.angles[ancilla1] = 0
        self.measurement_order.append(self.out[qubit])
        self.measurement_order.append(ancilla1)

        self.lengths[qubit] += 2
        self.out[qubit] = ancilla2
        self.instruction.append(['Z', [qubit]])

    def rx(self, qubit, angle):
        """X rotation gate
        Prameters
        ---------
        qubit : int
            target qubit
        angle : float
            rotation angle in radian
        """
        assert qubit in np.arange(self.width)

        # assign new qubit labels
        ancilla1, ancilla2 = self.Nnode, self.Nnode + 1
        self.Nnode += 2

        self.pos[ancilla1] = [self.lengths[qubit], -qubit]
        self.pos[ancilla2] = [self.lengths[qubit] + 1, -qubit]

        self.edges.append((self.out[qubit], ancilla1))
        if self.byproductx[self.out[qubit]]:  # E_ij X_i = X_i Z_j E_ij
            self.byproductz[ancilla1] = self.byproductx[self.out[qubit]]

        self.edges.append((ancilla1, ancilla2))
        self.next_node[self.out[qubit]] = ancilla1
        self.next_node[ancilla1] = ancilla2

        self.byproductx[ancilla2] = [ancilla1]
        self.byproductz[ancilla2] = [self.out[qubit]]
        self.domains[self.out[qubit]] = [[], []]
        self.domains[ancilla1] = [[self.out[qubit]], []]

        self.angles[self.out[qubit]] = 0
        self.angles[ancilla1] = -1 * angle / np.pi
        self.measurement_order.append(self.out[qubit])
        self.measurement_order.append(ancilla1)

        self.lengths[qubit] += 2
        self.out[qubit] = ancilla2
        self.instruction.append(['Rx', [qubit], [angle]])

    def ry(self, qubit, angle):
        """Y rotation gate
        Prameters
        ---------
        qubit : int
            target qubit
        angle : float
            angle in radian
        """
        assert qubit in np.arange(self.width)

        # assign new qubit labels
        ancilla = self.Nnode + np.arange(4, dtype=np.int8)
        self.Nnode += 4

        # straight graph
        for i in range(4):
            self.pos[ancilla[i]] = [self.lengths[qubit] + i, -qubit]

        self.edges.append((self.out[qubit], ancilla[0]))
        if self.byproductx[self.out[qubit]]:  # E_ij X_i = X_i Z_j E_ij
            self.byproductz[ancilla[0]] = self.byproductx[self.out[qubit]]
        self.next_node[self.out[qubit]] = ancilla[0]
        for i in range(3):
            self.edges.append((ancilla[i], ancilla[i + 1]))
            self.next_node[ancilla[i]] = ancilla[i + 1]

        self.byproductx[ancilla[3]] = [ancilla[2]]
        self.byproductz[ancilla[3]] = [ancilla[1]]
        self.domains[self.out[qubit]] = [[], []]
        self.domains[ancilla[0]] = [[self.out[qubit]], []]
        self.domains[ancilla[1]] = [[], [self.out[qubit], ancilla[0]]]
        self.domains[ancilla[2]] = [[], [ancilla[0]]]

        self.angles[self.out[qubit]] = 0.5
        self.angles[ancilla[0]] = -1 * angle / np.pi
        self.angles[ancilla[1]] = -0.5
        self.angles[ancilla[2]] = 0
        self.measurement_order.append(self.out[qubit])
        self.measurement_order.append(ancilla[0])
        self.measurement_order.append(ancilla[1])
        self.measurement_order.append(ancilla[2])

        self.lengths[qubit] += 4
        self.out[qubit] = ancilla[3]
        self.instruction.append(['Ry', [qubit], [angle]])

    def rz(self, qubit, angle):
        """Z rotation gate
        Prameters
        ---------
        qubit : int
            target qubit
        angle : float
            rotation angle in radian
        """
        assert qubit in np.arange(self.width)

        # assign new qubit labels
        ancilla1, ancilla2 = self.Nnode, self.Nnode + 1
        self.Nnode += 2

        self.pos[ancilla1] = [self.lengths[qubit], -qubit]
        self.pos[ancilla2] = [self.lengths[qubit] + 1, -qubit]

        self.edges.append((self.out[qubit], ancilla1))
        if self.byproductx[self.out[qubit]]:  # E_ij X_i = X_i Z_j E_ij
            self.byproductz[ancilla1] = self.byproductx[self.out[qubit]]
        self.edges.append((ancilla1, ancilla2))
        self.next_node[self.out[qubit]] = ancilla1
        self.next_node[ancilla1] = ancilla2

        self.byproductx[ancilla2] = [ancilla1]
        self.byproductz[ancilla2] = [self.out[qubit]]

        self.domains[self.out[qubit]] = [[], []]
        self.domains[ancilla1] = [[self.out[qubit]], []]

        self.angles[self.out[qubit]] = - 1 * angle / np.pi
        self.angles[ancilla1] = 0
        self.measurement_order.append(self.out[qubit])
        self.measurement_order.append(ancilla1)

        self.lengths[qubit] += 2
        self.out[qubit] = ancilla2
        self.instruction.append(['Rz', [qubit], [angle]])

    def i(self, qubit):
        """identity (teleportation) gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)

        # assign new qubit labels
        ancilla1, ancilla2 = self.Nnode, self.Nnode + 1
        self.Nnode += 2

        self.pos[ancilla1] = [self.lengths[qubit], -qubit]
        self.pos[ancilla2] = [self.lengths[qubit] + 1, -qubit]

        self.edges.append((self.out[qubit], ancilla1))
        if self.byproductx[self.out[qubit]]:  # E_ij X_i = X_i Z_j E_ij
            self.byproductz[ancilla1] = self.byproductx[self.out[qubit]]

        self.edges.append((ancilla1, ancilla2))
        self.next_node[self.out[qubit]] = ancilla1
        self.next_node[ancilla1] = ancilla2

        self.byproductx[ancilla2] = [ancilla1]
        self.byproductz[ancilla2] = [self.out[qubit]]
        self.domains[self.out[qubit]] = [[], []]
        self.domains[ancilla1] = [[self.out[qubit]], []]

        self.angles[self.out[qubit]] = 0
        self.angles[ancilla1] = 0
        self.measurement_order.append(self.out[qubit])
        self.measurement_order.append(ancilla1)

        self.lengths[qubit] += 2
        self.out[qubit] = ancilla2
        self.instruction.append(['I', [qubit]])

    def reorder_qubits(self):
        for i in range(self.width):
            self.i(i)

    def sort_outputs(self):
        old_out = deepcopy(self.out)
        self.out.sort()

        # change indices from old_out to sorted one
        new_edges = []
        for i, j in self.edges:
            if i in old_out:
                i = self.out[old_out.index(i)]
            if j in old_out:
                j = self.out[old_out.index(j)]
            new_edges.append((i, j))
        bpx = dict()
        bpz = dict()
        for i in iter(self.byproductx.keys()):
            if i in old_out:
                bpx[self.out[old_out.index(i)]] = self.byproductx[i]
            else:
                bpx[i] = self.byproductx[i]
        for i in iter(self.byproductz.keys()):
            if i in old_out:
                bpz[self.out[old_out.index(i)]] = self.byproductz[i]
            else:
                bpz[i] = self.byproductz[i]
        next_node = dict()
        for i, j in iter(self.next_node.items()):
            if j in old_out:
                next_node[i] = self.out[old_out.index(j)]
            else:
                next_node[i] = j
        new_pos = dict()
        for i in iter(self.pos.keys()):
            if i in old_out:
                new_pos[self.out[old_out.index(i)]] = self.pos[i]
            else:
                new_pos[i] = self.pos[i]

        self.edges = new_edges
        self.byproductx = bpx
        self.byproductz = bpz
        self.next_node = next_node
        self.pos = new_pos

    def simulate_statevector(self, input_state=None):

        if input_state is None:
            state = States.xplus_state
            for i in range(self.width - 1):
                state = state.expand(States.xplus_state)
        else:
            state = input_state.copy()

        for i in range(len(self.instruction)):
            if self.instruction[i][0] == 'CNOT':
                state = state.evolve(Ops.cnot, self.instruction[i][1])
            elif self.instruction[i][0] == 'I':
                pass
            elif self.instruction[i][0] == 'S':
                state = state.evolve(Ops.s, self.instruction[i][1])
            elif self.instruction[i][0] == 'H':
                state = state.evolve(Ops.h, self.instruction[i][1])
            elif self.instruction[i][0] == 'X':
                state = state.evolve(Ops.x, self.instruction[i][1])
            elif self.instruction[i][0] == 'Y':
                state = state.evolve(Ops.y, self.instruction[i][1])
            elif self.instruction[i][0] == 'Z':
                state = state.evolve(Ops.z, self.instruction[i][1])
            elif self.instruction[i][0] == 'Rx':
                state = state.evolve(Ops.Rx(self.instruction[i][2][0]), self.instruction[i][1])
            elif self.instruction[i][0] == 'Ry':
                state = state.evolve(Ops.Ry(self.instruction[i][2][0]), self.instruction[i][1])
            elif self.instruction[i][0] == 'Rz':
                state = state.evolve(Ops.Rz(self.instruction[i][2][0]), self.instruction[i][1])

        return state
