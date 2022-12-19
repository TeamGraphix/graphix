"""Gate-to-MBQC transpiler

accepts desired gate operations and transpile into MBQC measurement patterns.

"""
import numpy as np
from graphix.ops import Ops
from copy import deepcopy
from graphix.pattern import Pattern
from graphix.sim.statevec import Statevec


class Circuit:
    """Gate-to-MBQC transpiler.

    Holds gate operations and translates into MBQC measurement patterns.

    Attributes
    ----------
    width : int
        Number of logical qubits (for gate network)
    instruction : list
        List containing the gate sequence applied.
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
        """CNOT gate

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
        self.instruction.append(["CNOT", [control, target]])

    def h(self, qubit):
        """Hadamard gate

        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(["H", qubit])

    def s(self, qubit):
        """S gate

        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(["S", qubit])

    def x(self, qubit):
        """Pauli X gate

        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(["X", qubit])

    def y(self, qubit):
        """Pauli Y gate

        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(["Y", qubit])

    def z(self, qubit):
        """Pauli Z gate

        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(["Z", qubit])

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
        self.instruction.append(["Rx", qubit, angle])

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
        self.instruction.append(["Ry", qubit, angle])

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
        self.instruction.append(["Rz", qubit, angle])

    def i(self, qubit):
        """identity (teleportation) gate

        Prameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in np.arange(self.width)
        self.instruction.append(["I", qubit])

    def transpile(self):
        """gate-to-MBQC transpile function.

        Returns
        --------
        pattern : :class:`graphix.pattern.Pattern` object
        """
        Nnode = self.width
        out = [j for j in range(self.width)]
        pattern = Pattern(self.width)
        for instr in self.instruction:
            if instr[0] == "CNOT":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1][0]], out[instr[1][1]], seq = self._cnot_command(
                    out[instr[1][0]], out[instr[1][1]], ancilla
                )
                pattern.seq.extend(seq)
                Nnode += 2
            elif instr[0] == "I":
                pass
            elif instr[0] == "H":
                ancilla = Nnode
                out[instr[1]], seq = self._h_command(out[instr[1]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 1
            elif instr[0] == "S":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1]], seq = self._s_command(out[instr[1]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 2
            elif instr[0] == "X":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1]], seq = self._x_command(out[instr[1]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 2
            elif instr[0] == "Y":
                ancilla = [Nnode, Nnode + 1, Nnode + 2, Nnode + 3]
                out[instr[1]], seq = self._y_command(out[instr[1]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 4
            elif instr[0] == "Z":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1]], seq = self._z_command(out[instr[1]], ancilla)
                pattern.seq.extend(seq)
                Nnode += 2
            elif instr[0] == "Rx":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1]], seq = self._rx_command(out[instr[1]], ancilla, instr[2])
                pattern.seq.extend(seq)
                Nnode += 2
            elif instr[0] == "Ry":
                ancilla = [Nnode, Nnode + 1, Nnode + 2, Nnode + 3]
                out[instr[1]], seq = self._ry_command(out[instr[1]], ancilla, instr[2])
                pattern.seq.extend(seq)
                Nnode += 4
            elif instr[0] == "Rz":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1]], seq = self._rz_command(out[instr[1]], ancilla, instr[2])
                pattern.seq.extend(seq)
                Nnode += 2
            else:
                raise ValueError("Unknown instruction, commands not added")
        pattern.output_nodes = out
        pattern.Nnode = Nnode
        return pattern

    def standardize_and_transpile(self):
        """gate-to-MBQC transpile function.
        Commutes all byproduct through gates, instead of through measurement
        commands, to generate standardized measurement pattern.

        Returns
        --------
        pattern : :class:`graphix.pattern.Pattern` object
        """
        self._N = []
        for i in range(self.width):
            self._N.append(["N", i])
        self._M = []
        self._E = []
        self._instr = []
        Nnode = self.width
        out = [j for j in range(self.width)]
        for instr in self.instruction:
            if instr[0] == "CNOT":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1][0]], out[instr[1][1]], seq = self._cnot_command(
                    out[instr[1][0]], out[instr[1][1]], ancilla
                )
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:5])
                self._M.extend(seq[5:7])
                Nnode += 2
                self._instr.append(instr)
                self._instr.append(["XC", instr[1][1], seq[7][2]])
                self._instr.append(["ZC", instr[1][1], seq[8][2]])
                self._instr.append(["ZC", instr[1][0], seq[9][2]])
            elif instr[0] == "I":
                pass
            elif instr[0] == "H":
                ancilla = Nnode
                out[instr[1]], seq = self._h_command(out[instr[1]], ancilla)
                self._N.append(seq[0])
                self._E.append(seq[1])
                self._M.append(seq[2])
                self._instr.append(instr)
                self._instr.append(["XC", instr[1], seq[3][2]])
                Nnode += 1
            elif instr[0] == "S":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1]], seq = self._s_command(out[instr[1]], ancilla)
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:4])
                self._M.extend(seq[4:6])
                self._instr.append(instr)
                self._instr.append(["XC", instr[1], seq[6][2]])
                self._instr.append(["ZC", instr[1], seq[7][2]])
                Nnode += 2
            elif instr[0] == "X":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1]], seq = self._x_command(out[instr[1]], ancilla)
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:4])
                self._M.extend(seq[4:6])
                self._instr.append(instr)
                self._instr.append(["XC", instr[1], seq[6][2]])
                self._instr.append(["ZC", instr[1], seq[7][2]])
                Nnode += 2
            elif instr[0] == "Y":
                ancilla = [Nnode, Nnode + 1, Nnode + 2, Nnode + 3]
                out[instr[1]], seq = self._y_command(out[instr[1]], ancilla)
                self._N.extend(seq[0:4])
                self._E.extend(seq[4:8])
                self._M.extend(seq[8:12])
                self._instr.append(instr)
                self._instr.append(["XC", instr[1], seq[12][2]])
                self._instr.append(["ZC", instr[1], seq[13][2]])
                Nnode += 4
            elif instr[0] == "Z":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1]], seq = self._z_command(out[instr[1]], ancilla)
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:4])
                self._M.extend(seq[4:6])
                self._instr.append(instr)
                self._instr.append(["XC", instr[1], seq[6][2]])
                self._instr.append(["ZC", instr[1], seq[7][2]])
                Nnode += 2
            elif instr[0] == "Rx":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1]], seq = self._rx_command(out[instr[1]], ancilla, instr[2])
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:4])
                self._M.extend(seq[4:6])
                instr_ = deepcopy(instr)
                instr_.append(len(self._M) - 1)  # index of arb angle measurement command
                self._instr.append(instr_)
                self._instr.append(["XC", instr[1], seq[6][2]])
                self._instr.append(["ZC", instr[1], seq[7][2]])
                Nnode += 2
            elif instr[0] == "Ry":
                ancilla = [Nnode, Nnode + 1, Nnode + 2, Nnode + 3]
                out[instr[1]], seq = self._ry_command(out[instr[1]], ancilla, instr[2])
                self._N.extend(seq[0:4])
                self._E.extend(seq[4:8])
                self._M.extend(seq[8:12])
                instr_ = deepcopy(instr)
                instr_.append(len(self._M) - 3)  # index of arb angle measurement command
                self._instr.append(instr_)
                self._instr.append(["XC", instr[1], seq[12][2]])
                self._instr.append(["ZC", instr[1], seq[13][2]])
                Nnode += 4
            elif instr[0] == "Rz":
                ancilla = [Nnode, Nnode + 1]
                out[instr[1]], seq = self._rz_command(out[instr[1]], ancilla, instr[2])
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:4])
                self._M.extend(seq[4:6])
                instr_ = deepcopy(instr)
                instr_.append(len(self._M) - 2)  # index of arb angle measurement command
                self._instr.append(instr_)
                self._instr.append(["XC", instr[1], seq[6][2]])
                self._instr.append(["ZC", instr[1], seq[7][2]])
                Nnode += 2
            else:
                raise ValueError("Unknown instruction, commands not added")

        # move xc, zc to the end of the self._instr, so they will be applied last
        self._move_byproduct_to_right()

        # create command sequence
        command_seq = []
        for cmd in self._N:
            command_seq.append(cmd)
        for cmd in reversed(self._E):
            command_seq.append(cmd)
        for cmd in self._M:
            command_seq.append(cmd)
        bpx_added = dict()
        bpz_added = dict()
        # byproduct command buffer
        z_cmds = []
        x_cmds = []
        for i in range(len(self._instr)):
            instr = self._instr[i]
            if instr[0] == "XC":
                if instr[1] in bpx_added.keys():
                    x_cmds[bpx_added[instr[1]]][2].extend(instr[2])
                else:
                    bpx_added[instr[1]] = len(x_cmds)
                    x_cmds.append(["X", out[instr[1]], deepcopy(instr[2])])
            elif instr[0] == "ZC":
                if instr[1] in bpz_added.keys():
                    z_cmds[bpz_added[instr[1]]][2].extend(instr[2])
                else:
                    bpz_added[instr[1]] = len(z_cmds)
                    z_cmds.append(["Z", out[instr[1]], deepcopy(instr[2])])
        # append z commands first (X and Z commute up to global phase)
        for cmd in z_cmds:
            command_seq.append(cmd)
        for cmd in x_cmds:
            command_seq.append(cmd)
        pattern = Pattern(self.width)
        pattern.output_nodes = out
        pattern.Nnode = Nnode
        pattern.seq = command_seq
        return pattern

    def _commute_with_cnot(self, target):
        assert self._instr[target][0] in ["XC", "ZC"]
        assert self._instr[target + 1][0] == "CNOT"
        if self._instr[target][0] == "XC" and self._instr[target][1] == self._instr[target + 1][1][0]:  # control
            new_cmd = ["XC", self._instr[target + 1][1][1], self._instr[target][2]]
            self._commute_with_following(target)
            self._instr.insert(target + 1, new_cmd)
            return target + 1
        elif self._instr[target][0] == "ZC" and self._instr[target][1] == self._instr[target + 1][1][1]:  # target
            new_cmd = ["ZC", self._instr[target + 1][1][0], self._instr[target][2]]
            self._commute_with_following(target)
            self._instr.insert(target + 1, new_cmd)
            return target + 1
        else:
            self._commute_with_following(target)
        return target

    def _commute_with_H(self, target):
        assert self._instr[target][0] in ["XC", "ZC"]
        assert self._instr[target + 1][0] == "H"
        if self._instr[target][1] == self._instr[target + 1][1]:
            if self._instr[target][0] == "XC":
                self._instr[target][0] = "ZC"  # byproduct changes to Z
                self._commute_with_following(target)
            else:
                self._instr[target][0] = "XC"  # byproduct changes to X
                self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_S(self, target):
        assert self._instr[target][0] in ["XC", "ZC"]
        assert self._instr[target + 1][0] == "S"
        if self._instr[target][1] == self._instr[target + 1][1]:
            if self._instr[target][0] == "XC":
                self._commute_with_following(target)
                # changes to Y = XZ
                self._instr.insert(target + 1, ["ZC", self._instr[target + 1][1], self._instr[target + 1][2]])
                return target + 1
        self._commute_with_following(target)
        return target

    def _commute_with_Rx(self, target):
        assert self._instr[target][0] in ["XC", "ZC"]
        assert self._instr[target + 1][0] == "Rx"
        if self._instr[target][1] == self._instr[target + 1][1]:
            if self._instr[target][0] == "ZC":
                # add to the s-domain
                self._M[self._instr[target + 1][3]][4].extend(self._instr[target][2])
                self._commute_with_following(target)
            else:
                self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_Ry(self, target):
        assert self._instr[target][0] in ["XC", "ZC"]
        assert self._instr[target + 1][0] == "Ry"
        if self._instr[target][1] == self._instr[target + 1][1]:
            # add to the s-domain
            self._M[self._instr[target + 1][3]][4].extend(self._instr[target][2])
            self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_Rz(self, target):
        assert self._instr[target][0] in ["XC", "ZC"]
        assert self._instr[target + 1][0] == "Rz"
        if self._instr[target][1] == self._instr[target + 1][1]:
            if self._instr[target][0] == "XC":
                # add to the s-domain
                self._M[self._instr[target + 1][3]][4].extend(self._instr[target][2])
                self._commute_with_following(target)
            else:
                self._commute_with_following(target)
        else:
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
        A = self._instr[target + 1]
        self._instr.pop(target + 1)
        self._instr.insert(target, A)

    def _find_byproduct_to_move(self, rev=False, skipnum=0):
        """Internal method for reordering commands
        Parameters
        ----------
        rev : bool
            search from the end (true) or start (false) of seq
        skipnum : int
            skip the detected command by specified times
        """
        if not rev:  # search from the start
            target = 0
            step = 1
        else:  # search from the back
            target = len(self._instr) - 1
            step = -1
        ite = 0
        num_ops = 0
        while ite < len(self._instr):
            if self._instr[target][0] in ["ZC", "XC"]:
                num_ops += 1
            if num_ops == skipnum + 1:
                return target
            ite += 1
            target += step
        target = "end"
        return target

    def _move_byproduct_to_right(self):
        """Internal method to move the byproduct 'gate' to the end of sequence, using the commutation relations"""
        moved = 0  # number of moved op
        target = self._find_byproduct_to_move(rev=True, skipnum=moved)
        while target != "end":
            if (target == len(self._instr) - 1) or (self._instr[target + 1][0] in ["XC", "ZC"]):
                moved += 1
                target = self._find_byproduct_to_move(rev=True, skipnum=moved)
                continue
            if self._instr[target + 1][0] == "CNOT":
                target = self._commute_with_cnot(target)
            elif self._instr[target + 1][0] == "H":
                self._commute_with_H(target)
            elif self._instr[target + 1][0] == "S":
                target = self._commute_with_S(target)
            elif self._instr[target + 1][0] == "Rx":
                self._commute_with_Rx(target)
            elif self._instr[target + 1][0] == "Ry":
                self._commute_with_Ry(target)
            elif self._instr[target + 1][0] == "Rz":
                self._commute_with_Rz(target)
            else:
                # Pauli gates commute up to global phase.
                self._commute_with_following(target)
            target += 1

    @classmethod
    def _cnot_command(self, control_node, target_node, ancilla):
        """MBQC commands for CNOT gate

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
        seq = [["N", ancilla[0]], ["N", ancilla[1]]]
        seq.append(["E", (target_node, ancilla[0])])
        seq.append(["E", (control_node, ancilla[0])])
        seq.append(["E", (ancilla[0], ancilla[1])])
        seq.append(["M", target_node, "XY", 0, [], []])
        seq.append(["M", ancilla[0], "XY", 0, [], []])
        seq.append(["X", ancilla[1], [ancilla[0]]])
        seq.append(["Z", ancilla[1], [target_node]])
        seq.append(["Z", control_node, [target_node]])
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
        seq = [["N", ancilla]]
        seq.append(["E", (input_node, ancilla)])
        seq.append(["M", input_node, "XY", 0, [], []])
        seq.append(["X", ancilla, [input_node]])
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
        seq = [["N", ancilla[0]], ["N", ancilla[1]]]
        seq.append(["E", (input_node, ancilla[0])])
        seq.append(["E", (ancilla[0], ancilla[1])])
        seq.append(["M", input_node, "XY", -0.5, [], []])
        seq.append(["M", ancilla[0], "XY", 0, [], []])
        seq.append(["X", ancilla[1], [ancilla[0]]])
        seq.append(["Z", ancilla[1], [input_node]])
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
        seq = [["N", ancilla[0]], ["N", ancilla[1]]]
        seq.append(["E", (input_node, ancilla[0])])
        seq.append(["E", (ancilla[0], ancilla[1])])
        seq.append(["M", input_node, "XY", 0, [], []])
        seq.append(["M", ancilla[0], "XY", -1, [], []])
        seq.append(["X", ancilla[1], [ancilla[0]]])
        seq.append(["Z", ancilla[1], [input_node]])
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
        seq = [["N", ancilla[0]], ["N", ancilla[1]]]  # assign new qubit labels
        seq.extend([["N", ancilla[2]], ["N", ancilla[3]]])
        seq.append(["E", (input_node, ancilla[0])])
        seq.append(["E", (ancilla[0], ancilla[1])])
        seq.append(["E", (ancilla[1], ancilla[2])])
        seq.append(["E", (ancilla[2], ancilla[3])])
        seq.append(["M", input_node, "XY", 0.5, [], []])
        seq.append(["M", ancilla[0], "XY", 1.0, [input_node], []])
        seq.append(["M", ancilla[1], "XY", -0.5, [input_node], []])
        seq.append(["M", ancilla[2], "XY", 0, [], []])
        seq.append(["X", ancilla[3], [ancilla[0], ancilla[2]]])
        seq.append(["Z", ancilla[3], [ancilla[0], ancilla[1]]])
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
        seq = [["N", ancilla[0]], ["N", ancilla[1]]]  # assign new qubit labels
        seq.append(["E", (input_node, ancilla[0])])
        seq.append(["E", (ancilla[0], ancilla[1])])
        seq.append(["M", input_node, "XY", -1, [], []])
        seq.append(["M", ancilla[0], "XY", 0, [], []])
        seq.append(["X", ancilla[1], [ancilla[0]]])
        seq.append(["Z", ancilla[1], [input_node]])
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
        seq = [["N", ancilla[0]], ["N", ancilla[1]]]  # assign new qubit labels
        seq.append(["E", (input_node, ancilla[0])])
        seq.append(["E", (ancilla[0], ancilla[1])])
        seq.append(["M", input_node, "XY", 0, [], []])
        seq.append(["M", ancilla[0], "XY", -1 * angle / np.pi, [input_node], []])
        seq.append(["X", ancilla[1], [ancilla[0]]])
        seq.append(["Z", ancilla[1], [input_node]])
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
        seq = [["N", ancilla[0]], ["N", ancilla[1]]]  # assign new qubit labels
        seq.extend([["N", ancilla[2]], ["N", ancilla[3]]])
        seq.append(["E", (input_node, ancilla[0])])
        seq.append(["E", (ancilla[0], ancilla[1])])
        seq.append(["E", (ancilla[1], ancilla[2])])
        seq.append(["E", (ancilla[2], ancilla[3])])
        seq.append(["M", input_node, "XY", 0.5, [], []])
        seq.append(["M", ancilla[0], "XY", -1 * angle / np.pi, [input_node], []])
        seq.append(["M", ancilla[1], "XY", -0.5, [input_node], []])
        seq.append(["M", ancilla[2], "XY", 0, [], []])
        seq.append(["X", ancilla[3], [ancilla[0], ancilla[2]]])
        seq.append(["Z", ancilla[3], [ancilla[0], ancilla[1]]])
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
        seq = [["N", ancilla[0]], ["N", ancilla[1]]]  # assign new qubit labels
        seq.append(["E", (input_node, ancilla[0])])
        seq.append(["E", (ancilla[0], ancilla[1])])
        seq.append(["M", input_node, "XY", -1 * angle / np.pi, [], []])
        seq.append(["M", ancilla[0], "XY", 0, [], []])
        seq.append(["X", ancilla[1], [ancilla[0]]])
        seq.append(["Z", ancilla[1], [input_node]])
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
            if pattern.seq[i][0] == "E":
                j, k = pattern.seq[i][1]
                if j in old_out:
                    j = output_nodes[old_out.index(j)]
                if k in old_out:
                    k = output_nodes[old_out.index(k)]
                pattern.seq[i][1] = (j, k)
            elif pattern.seq[i][1] in old_out:
                pattern.seq[i][1] = output_nodes[old_out.index(pattern.seq[i][1])]

    def simulate_statevector(self, input_state=None):
        """Run statevector simultion of the gate sequence, using graphix.Statevec

        Returns
        -------
        stete : graphix.Statevec
            output state of the statevector simulation.
        """

        if input_state is None:
            state = Statevec(nqubit=self.width)
        else:
            state = input_state

        for i in range(len(self.instruction)):
            if self.instruction[i][0] == "CNOT":
                state.CNOT((self.instruction[i][1][0], self.instruction[i][1][1]))
            elif self.instruction[i][0] == "I":
                pass
            elif self.instruction[i][0] == "S":
                state.evolve_single(Ops.s, self.instruction[i][1])
            elif self.instruction[i][0] == "H":
                state.evolve_single(Ops.h, self.instruction[i][1])
            elif self.instruction[i][0] == "X":
                state.evolve_single(Ops.x, self.instruction[i][1])
            elif self.instruction[i][0] == "Y":
                state.evolve_single(Ops.y, self.instruction[i][1])
            elif self.instruction[i][0] == "Z":
                state.evolve_single(Ops.z, self.instruction[i][1])
            elif self.instruction[i][0] == "Rx":
                state.evolve_single(Ops.Rx(self.instruction[i][2]), self.instruction[i][1])
            elif self.instruction[i][0] == "Ry":
                state.evolve_single(Ops.Ry(self.instruction[i][2]), self.instruction[i][1])
            elif self.instruction[i][0] == "Rz":
                state.evolve_single(Ops.Rz(self.instruction[i][2]), self.instruction[i][1])

        return state
