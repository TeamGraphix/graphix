"""Gate-to-MBQC transpiler

accepts desired gate operations and transpile into MBQC measurement patterns.

"""
import numpy as np
from graphix.ops import States, Ops
from copy import deepcopy
from graphix.pattern import Pattern

class Circuit():
    """Gate-to-MBQC transpiler.

    Holds gate operations and translates into MBQC measurement patterns.

    Attributes
    ----------
    width : int
        Number of logical qubits (for gate network)
    instruction : list
        List of gates applied

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

    def cnot(self, control, target):
        """CNOT

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
        self.instruction.append(['CNOT', [target, control]])

    def h(self, qubit):
        """Hadamard gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(['H', qubit])

    def s(self, qubit):
        """S gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(['S', qubit])

    def x(self, qubit):
        """Pauli X gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(['X', qubit])

    def y(self, qubit):
        """Pauli Y gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(['Y', qubit])

    def z(self, qubit):
        """Pauli Z gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(['Z', qubit])

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
        self.instruction.append(['Rx', qubit, angle])


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
        self.instruction.append(['Ry', qubit, angle])

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
        self.instruction.append(['Rz', qubit, angle])

    def i(self, qubit):
        """identity (teleportation) gate
        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(['I', qubit])

    def transpile(self):
        """ gate-to-MBQC transpile function.
        Returns
        --------
        pattern : graphix.Pattern object
        """
        Nnode = self.width
        out = [j for j in range(self.width)]
        pattern = Pattern(self.width)
        for instr in self.instruction:
            if instr[0] == 'CNOT':
                ancilla = [Nnode, Nnode+1]
                out[instr[1][1]], out[instr[1][0]], seq =\
                    self._cnot_command(out[instr[1][1]], out[instr[1][0]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 2
            elif instr[0] == 'I':
                pass
            elif instr[0] == 'H':
                ancilla = Nnode
                out[instr[1]], seq = self._h_command(out[instr[1]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 1
            elif instr[0] == 'S':
                ancilla = [Nnode, Nnode+1]
                out[instr[1]], seq = self._s_command(out[instr[1]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 2
            elif instr[0] == 'X':
                ancilla = [Nnode, Nnode+1]
                out[instr[1]], seq = self._x_command(out[instr[1]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 2
            elif instr[0] == 'Y':
                ancilla = [Nnode, Nnode+1, Nnode+2, Nnode+3]
                out[instr[1]], seq = self._y_command(out[instr[1]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 4
            elif instr[0] == 'Z':
                ancilla = [Nnode, Nnode+1]
                out[instr[1]], seq = self._z_command(out[instr[1]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 2
            elif instr[0] == 'Rx':
                ancilla =[Nnode, Nnode+1]
                out[instr[1]], seq = self._rx_command(out[instr[1]], ancilla, instr[2])
                pattern.seq.extend(seq)
                Nnode += 2
            elif instr[0] == 'Ry':
                ancilla = [Nnode, Nnode+1, Nnode+2, Nnode+3]
                out[instr[1]], seq = self._ry_command(out[instr[1]], ancilla, instr[2])
                pattern.seq.extend(seq)
                Nnode += 4
            elif instr[0] == 'Rz':
                ancilla = [Nnode, Nnode+1]
                out[instr[1]], seq = self._rz_command(out[instr[1]], ancilla, instr[2])
                pattern.seq.extend(seq)
                Nnode += 2
            else:
                raise ValueError('Unknown instruction, commands not added')
        # self._sort_outputs(pattern, out)
        pattern.output_nodes = out
        pattern.Nnode = Nnode
        return pattern

    @classmethod
    def _cnot_command(self, control_node, target_node, ancilla):
        """ MBQC commands for CNOT gate
        Parameters
        ---------
        control_node : int
            control node on graph
        target : int
            target node on graph
        ancilla : list of two ints
            ancilla node indices to be added to graph

        Returns
        ---------
        control_out : int
            control node on graph after the gate
        target_out : int
            target node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [['N',ancilla[0]], ['N', ancilla[1]]]
        seq.append(['E', (target_node, ancilla[0])])
        seq.append(['E', (control_node, ancilla[0])])
        seq.append(['E', (ancilla[0], ancilla[1])])
        seq.append(['M', target_node, 'XY', 0, [], []])
        seq.append(['M', ancilla[0], 'XY', 0, [], []])
        seq.append(['X', ancilla[1], [ancilla[0]]])
        seq.append(['Z', ancilla[1], [target_node]])
        seq.append(['Z', control_node, [target_node]])
        return control_node, ancilla[1], seq

    @classmethod
    def _h_command(self, input_node, ancilla):
        """MBQC commands for Hadamard gate
        Parameters
        ---------
        input_node : int
            target node on graph
        ancilla : int
            ancilla node index to be added

        Returns
        ---------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        seq = [['N', ancilla]]
        seq.append(['E', (input_node, ancilla)])
        seq.append(['M', input_node, 'XY', 0, [], []])
        seq.append(['X', ancilla, [input_node]])
        return ancilla, seq

    @classmethod
    def _s_command(self, input_node, ancilla):
        """MBQC commands for S gate
        Parameters
        ---------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph

        Returns
        ---------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [['N',ancilla[0]], ['N', ancilla[1]]]# assign new qubit labels
        seq.append(['E', (input_node, ancilla[0])])
        seq.append(['E', (ancilla[0], ancilla[1])])
        seq.append(['M', input_node, 'XY', -0.5, [], []])
        seq.append(['M', ancilla[0], 'XY', 0, [], []])
        seq.append(['X', ancilla[1], [ancilla[0]]])
        seq.append(['Z', ancilla[1], [input_node]])
        return ancilla[1], seq

    @classmethod
    def _x_command(self, input_node, ancilla):
        """MBQC commands for Pauli X gate
        Parameters
        ---------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph

        Returns
        ---------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [['N',ancilla[0]], ['N', ancilla[1]]]# assign new qubit labels
        seq.append(['E', (input_node, ancilla[0])])
        seq.append(['E', (ancilla[0], ancilla[1])])
        seq.append(['M', input_node, 'XY', 0, [], []])
        seq.append(['M', ancilla[0], 'XY', -1, [], []])
        seq.append(['X', ancilla[1], [ancilla[0]]])
        seq.append(['Z', ancilla[1], [input_node]])
        return ancilla[1], seq

    @classmethod
    def _y_command(self, input_node, ancilla):
        """MBQC commands for Pauli Y gate
        Parameters
        ---------
        input_node : int
            input node index
        ancilla : list of four ints
            ancilla node indices to be added to graph

        Returns
        ---------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 4
        seq = [['N',ancilla[0]], ['N', ancilla[1]]]# assign new qubit labels
        seq.extend([['N',ancilla[2]], ['N', ancilla[3]]])
        seq.append(['E', (input_node, ancilla[0])])
        seq.append(['E', (ancilla[0], ancilla[1])])
        seq.append(['E', (ancilla[1], ancilla[2])])
        seq.append(['E', (ancilla[2], ancilla[3])])
        seq.append(['M', input_node, 'XY', 0.5, [], []])
        seq.append(['M', ancilla[0], 'XY', 1.0, [input_node], []])
        seq.append(['M', ancilla[1], 'XY', -0.5, [input_node], [ancilla[0]]])
        seq.append(['M', ancilla[2], 'XY', 0, [], [ancilla[0]]])
        seq.append(['X', ancilla[3], [ancilla[2]]])
        seq.append(['Z', ancilla[3], [ancilla[1]]])
        return ancilla[3], seq

    @classmethod
    def _z_command(self, input_node, ancilla):
        """MBQC commands for Pauli Z gate
        Parameters
        ---------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph

        Returns
        ---------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [['N',ancilla[0]], ['N', ancilla[1]]]# assign new qubit labels
        seq.append(['E', (input_node, ancilla[0])])
        seq.append(['E', (ancilla[0], ancilla[1])])
        seq.append(['M', input_node, 'XY', -1, [], []])
        seq.append(['M', ancilla[0], 'XY', 0, [], []])
        seq.append(['X', ancilla[1], [ancilla[0]]])
        seq.append(['Z', ancilla[1], [input_node]])
        return ancilla[1], seq

    @classmethod
    def _rx_command(self, input_node, ancilla, angle):
        """MBQC commands for X rotation gate
        Parameters
        ---------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph
        angle : float
            measurement angle in radian

        Returns
        ---------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [['N',ancilla[0]], ['N', ancilla[1]]]# assign new qubit labels
        seq.append(['E', (input_node, ancilla[0])])
        seq.append(['E', (ancilla[0], ancilla[1])])
        seq.append(['M', input_node, 'XY', 0, [], []])
        seq.append(['M', ancilla[0], 'XY', -1 * angle / np.pi, [input_node], []])
        seq.append(['X', ancilla[1], [ancilla[0]]])
        seq.append(['Z', ancilla[1], [input_node]])
        return ancilla[1], seq

    @classmethod
    def _ry_command(self, input_node, ancilla, angle):
        """MBQC commands for Y rotation gate
        Parameters
        ---------
        input_node : int
            input node index
        ancilla : list of four ints
            ancilla node indices to be added to graph
        angle : float
            rotation angle in radian

        Returns
        ---------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 4
        seq = [['N',ancilla[0]], ['N', ancilla[1]]]# assign new qubit labels
        seq.extend([['N',ancilla[2]], ['N', ancilla[3]]])
        seq.append(['E', (input_node, ancilla[0])])
        seq.append(['E', (ancilla[0], ancilla[1])])
        seq.append(['E', (ancilla[1], ancilla[2])])
        seq.append(['E', (ancilla[2], ancilla[3])])
        seq.append(['M', input_node, 'XY', 0.5, [], []])
        seq.append(['M', ancilla[0], 'XY', -1 * angle / np.pi, [input_node], []])
        seq.append(['M', ancilla[1], 'XY', -0.5, [input_node], [ancilla[0]]])
        seq.append(['M', ancilla[2], 'XY', 0, [], [ancilla[0]]])
        seq.append(['X', ancilla[3], [ancilla[2]]])
        seq.append(['Z', ancilla[3], [ancilla[1]]])
        return ancilla[3], seq

    @classmethod
    def _rz_command(self, input_node, ancilla, angle):
        """MBQC commands for Z rotation gate
        Parameters
        ---------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph
        angle : float
            measurement angle in radian

        Returns
        ---------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [['N',ancilla[0]], ['N', ancilla[1]]]# assign new qubit labels
        seq.append(['E', (input_node, ancilla[0])])
        seq.append(['E', (ancilla[0], ancilla[1])])
        seq.append(['M', input_node, 'XY', -1 * angle / np.pi, [], []])
        seq.append(['M', ancilla[0], 'XY', 0, [], []])
        seq.append(['X', ancilla[1], [ancilla[0]]])
        seq.append(['Z', ancilla[1], [input_node]])
        return ancilla[1], seq

    @classmethod
    def _sort_outputs(self, pattern, output_nodes):
        """Sort the node indices of ouput qubits.

        Parameters
        ---------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph
        angle : float
            measurement angle in radian

        Returns
        ---------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        old_out = deepcopy(output_nodes)
        output_nodes.sort()
        # check all commands and swap node indices
        for i in range(len(pattern.seq)):
            if pattern.seq[i][0] == 'E':
                j, k = pattern.seq[i][1]
                if j in old_out:
                    j = output_nodes[old_out.index(j)]
                if k in old_out:
                    k = output_nodes[old_out.index(k)]
                pattern.seq[i][1] = (j, k)
            elif pattern.seq[i][1] in old_out:
                pattern.seq[i][1] = output_nodes[old_out.index(pattern.seq[i][1])]

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
                state = state.evolve(Ops.s, [self.instruction[i][1]])
            elif self.instruction[i][0] == 'H':
                state = state.evolve(Ops.h, [self.instruction[i][1]])
            elif self.instruction[i][0] == 'X':
                state = state.evolve(Ops.x, [self.instruction[i][1]])
            elif self.instruction[i][0] == 'Y':
                state = state.evolve(Ops.y, [self.instruction[i][1]])
            elif self.instruction[i][0] == 'Z':
                state = state.evolve(Ops.z, [self.instruction[i][1]])
            elif self.instruction[i][0] == 'Rx':
                state = state.evolve(Ops.Rx(self.instruction[i][2]), [self.instruction[i][1]])
            elif self.instruction[i][0] == 'Ry':
                state = state.evolve(Ops.Ry(self.instruction[i][2]), [self.instruction[i][1]])
            elif self.instruction[i][0] == 'Rz':
                state = state.evolve(Ops.Rz(self.instruction[i][2]), [self.instruction[i][1]])

        return state
