"""Gate-to-MBQC transpiler

accepts desired gate operations and transpile into MBQC measurement patterns.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from copy import deepcopy

import numpy as np

import graphix.parameter
import graphix.pauli
import graphix.sim.base_backend
import graphix.sim.statevec
from graphix import command, instruction
from graphix.command import CommandKind, E, M, N, X, Z
from graphix.ops import Ops
from graphix.pattern import Pattern


@dataclasses.dataclass
class TranspileResult:
    """
    The result of a transpilation.

    pattern : :class:`graphix.pattern.Pattern` object
    classical_outputs : tuple[int,...], index of nodes measured with `M` gates
    """

    pattern: Pattern
    classical_outputs: tuple[int, ...]


@dataclasses.dataclass
class SimulateResult:
    """
    The result of a simulation.

    statevec : :class:`graphix.sim.statevec.Statevec` object
    classical_measures : tuple[int,...], classical measures
    """

    statevec: graphix.sim.statevec.Statevec
    classical_measures: tuple[int, ...]


Angle = float | graphix.parameter.Expression


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

    def __init__(self, width: int):
        """
        Parameters
        ----------
        width : int
            number of logical qubits for the gate network
        """
        self.width = width
        self.instruction: list[instruction.Instruction] = []
        self.active_qubits = set(range(width))

    def cnot(self, control: int, target: int):
        """CNOT gate

        Parameters
        ---------
        control : int
            control qubit
        target : int
            target qubit
        """
        assert control in self.active_qubits
        assert target in self.active_qubits
        assert control != target
        self.instruction.append(instruction.CNOT(control=control, target=target))

    def swap(self, qubit1: int, qubit2: int):
        """SWAP gate

        Parameters
        ---------
        qubit1 : int
            first qubit to be swapped
        qubit2 : int
            second qubit to be swapped
        """
        assert qubit1 in self.active_qubits
        assert qubit2 in self.active_qubits
        assert qubit1 != qubit2
        self.instruction.append(instruction.SWAP(targets=(qubit1, qubit2)))

    def h(self, qubit: int):
        """Hadamard gate

        Parameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.H(target=qubit))

    def s(self, qubit: int):
        """S gate

        Parameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.S(target=qubit))

    def x(self, qubit):
        """Pauli X gate

        Parameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.X(target=qubit))

    def y(self, qubit: int):
        """Pauli Y gate

        Parameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.Y(target=qubit))

    def z(self, qubit: int):
        """Pauli Z gate

        Parameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.Z(target=qubit))

    def rx(self, qubit: int, angle: Angle):
        """X rotation gate

        Parameters
        ---------
        qubit : int
            target qubit
        angle : Angle
            rotation angle in radian
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RX(target=qubit, angle=angle))

    def ry(self, qubit: int, angle: Angle):
        """Y rotation gate

        Parameters
        ---------
        qubit : int
            target qubit
        angle : Angle
            angle in radian
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RY(target=qubit, angle=angle))

    def rz(self, qubit: int, angle: Angle):
        """Z rotation gate

        Parameters
        ---------
        qubit : int
            target qubit
        angle : Angle
            rotation angle in radian
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RZ(target=qubit, angle=angle))

    def rzz(self, control: int, target: int, angle: Angle):
        r"""ZZ-rotation gate.
        Equivalent to the sequence
        CNOT(control, target),
        Rz(target, angle),
        CNOT(control, target)

        and realizes rotation expressed by
        :math:`e^{-i \frac{\theta}{2} Z_c Z_t}`.

        Parameters
        ---------
        control : int
            control qubit
        target : int
            target qubit
        angle : Angle
            rotation angle in radian
        """
        assert control in self.active_qubits
        assert target in self.active_qubits
        self.instruction.append(instruction.RZZ(control=control, target=target, angle=angle))

    def ccx(self, control1: int, control2: int, target: int):
        r"""CCX (Toffoli) gate.

        Prameters
        ---------
        control1 : int
            first control qubit
        control2 : int
            second control qubit
        target : int
            target qubit
        """
        assert control1 in self.active_qubits
        assert control2 in self.active_qubits
        assert target in self.active_qubits
        assert control1 != control2 and control1 != target and control2 != target
        self.instruction.append(instruction.CCX(controls=(control1, control2), target=target))

    def i(self, qubit: int):
        """identity (teleportation) gate

        Parameters
        ---------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.I(target=qubit))

    def m(self, qubit: int, plane: graphix.pauli.Plane, angle: Angle):
        """measure a quantum qubit

        The measured qubit cannot be used afterwards.

        Parameters
        ---------
        qubit : int
            target qubit
        plane : graphix.pauli.Plane
        angle : Angle
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.M(target=qubit, plane=plane, angle=angle))
        self.active_qubits.remove(qubit)

    def transpile(self, opt: bool = False) -> TranspileResult:
        """gate-to-MBQC transpile function.

        Parameters
        ----------
        opt : bool
            Whether or not to use pre-optimized gateset with local-Clifford decoration.

        Returns
        --------
        result : :class:`TranspileResult` object
        """
        Nnode = self.width
        out = [j for j in range(self.width)]
        pattern = Pattern(input_nodes=[j for j in range(self.width)])
        classical_outputs = []
        for instr in self.instruction:
            kind = instr.kind
            if kind == instruction.InstructionKind.CNOT:
                ancilla = [Nnode, Nnode + 1]
                assert out[instr.control] is not None
                assert out[instr.target] is not None
                out[instr.control], out[instr.target], seq = self._cnot_command(
                    out[instr.control], out[instr.target], ancilla
                )
                pattern.extend(seq)
                Nnode += 2
            elif kind == instruction.InstructionKind.SWAP:
                out[instr.targets[0]], out[instr.targets[1]] = (
                    out[instr.targets[1]],
                    out[instr.targets[0]],
                )
            elif kind == instruction.InstructionKind.I:
                pass
            elif kind == instruction.InstructionKind.H:
                ancilla = Nnode
                out[instr.target], seq = self._h_command(out[instr.target], ancilla)
                pattern.extend(seq)
                Nnode += 1
            elif kind == instruction.InstructionKind.S:
                ancilla = [Nnode, Nnode + 1]
                out[instr.target], seq = self._s_command(out[instr.target], ancilla)
                pattern.extend(seq)
                Nnode += 2
            elif kind == instruction.InstructionKind.X:
                ancilla = [Nnode, Nnode + 1]
                out[instr.target], seq = self._x_command(out[instr.target], ancilla)
                pattern.extend(seq)
                Nnode += 2
            elif kind == instruction.InstructionKind.Y:
                ancilla = [Nnode, Nnode + 1, Nnode + 2, Nnode + 3]
                out[instr.target], seq = self._y_command(out[instr.target], ancilla)
                pattern.extend(seq)
                Nnode += 4
            elif kind == instruction.InstructionKind.Z:
                ancilla = [Nnode, Nnode + 1]
                out[instr.target], seq = self._z_command(out[instr.target], ancilla)
                pattern.extend(seq)
                Nnode += 2
            elif kind == instruction.InstructionKind.RX:
                ancilla = [Nnode, Nnode + 1]
                out[instr.target], seq = self._rx_command(out[instr.target], ancilla, instr.angle)
                pattern.extend(seq)
                Nnode += 2
            elif kind == instruction.InstructionKind.RY:
                ancilla = [Nnode, Nnode + 1, Nnode + 2, Nnode + 3]
                out[instr.target], seq = self._ry_command(out[instr.target], ancilla, instr.angle)
                pattern.extend(seq)
                Nnode += 4
            elif kind == instruction.InstructionKind.RZ:
                if opt:
                    ancilla = Nnode
                    out[instr.target], seq = self._rz_command_opt(out[instr.target], ancilla, instr.angle)
                    pattern.extend(seq)
                    Nnode += 1
                else:
                    ancilla = [Nnode, Nnode + 1]
                    out[instr.target], seq = self._rz_command(out[instr.target], ancilla, instr.angle)
                    pattern.extend(seq)
                    Nnode += 2
            elif kind == instruction.InstructionKind.RZZ:
                if opt:
                    ancilla = Nnode
                    (
                        out[instr.control],
                        out[instr.target],
                        seq,
                    ) = self._rzz_command_opt(out[instr.control], out[instr.target], ancilla, instr.angle)
                    pattern.extend(seq)
                    Nnode += 1
                else:
                    raise NotImplementedError(
                        "YZ-plane measurements not accepted and Rzz gate\
                        cannot be directly transpiled"
                    )
            elif kind == instruction.InstructionKind.CCX:
                if opt:
                    ancilla = [Nnode + i for i in range(11)]
                    (
                        out[instr.controls[0]],
                        out[instr.controls[1]],
                        out[instr.target],
                        seq,
                    ) = self._ccx_command_opt(
                        out[instr.controls[0]],
                        out[instr.controls[1]],
                        out[instr.target],
                        ancilla,
                    )
                    pattern.extend(seq)
                    Nnode += 11
                else:
                    ancilla = [Nnode + i for i in range(18)]
                    (
                        out[instr.controls[0]],
                        out[instr.controls[1]],
                        out[instr.target],
                        seq,
                    ) = self._ccx_command(
                        out[instr.controls[0]],
                        out[instr.controls[1]],
                        out[instr.target],
                        ancilla,
                    )
                    pattern.extend(seq)
                    Nnode += 18
            elif kind == instruction.InstructionKind.M:
                node_index = out[instr.target]
                seq = self._m_command(instr.target, instr.plane, instr.angle)
                pattern.extend(seq)
                classical_outputs.append(node_index)
                out[instr.target] = None
            else:
                raise ValueError("Unknown instruction, commands not added")
        out = filter(lambda node: node is not None, out)
        pattern.reorder_output_nodes(out)
        return TranspileResult(pattern, tuple(classical_outputs))

    def standardize_and_transpile(self, opt: bool = True) -> TranspileResult:
        """gate-to-MBQC transpile function.
        Commutes all byproduct through gates, instead of through measurement
        commands, to generate standardized measurement pattern.

        Parameters
        ----------
        opt : bool
            Whether or not to use pre-optimized gateset with local-Clifford decoration.

        Returns
        --------
        pattern : :class:`graphix.pattern.Pattern` object
        """
        self._N: list[N] = []
        # for i in range(self.width):
        #    self._N.append(["N", i])
        self._M: list[M] = []
        self._E: list[E] = []
        self._instr: list[instruction.Instruction] = []
        Nnode = self.width
        inputs = [j for j in range(self.width)]
        out = [j for j in range(self.width)]
        classical_outputs = []
        for instr in self.instruction:
            kind = instr.kind
            if kind == instruction.InstructionKind.CNOT:
                ancilla = [Nnode, Nnode + 1]
                assert out[instr.control] is not None
                assert out[instr.target] is not None
                out[instr.control], out[instr.target], seq = self._cnot_command(
                    out[instr.control], out[instr.target], ancilla
                )
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:5])
                self._M.extend(seq[5:7])
                Nnode += 2
                self._instr.append(instr)
                self._instr.append(
                    instruction.XC(
                        target=instr.target,
                        domain=seq[7].domain,
                    )
                )
                self._instr.append(
                    instruction.ZC(
                        target=instr.target,
                        domain=seq[8].domain,
                    )
                )
                self._instr.append(
                    instruction.ZC(
                        target=instr.control,
                        domain=seq[9].domain,
                    )
                )
            elif kind == instruction.InstructionKind.SWAP:
                out[instr.targets[0]], out[instr.targets[1]] = (
                    out[instr.targets[1]],
                    out[instr.targets[0]],
                )
                self._instr.append(instr)
            elif kind == instruction.InstructionKind.I:
                pass
            elif kind == instruction.InstructionKind.H:
                ancilla = Nnode
                out[instr.target], seq = self._h_command(out[instr.target], ancilla)
                self._N.append(seq[0])
                self._E.append(seq[1])
                self._M.append(seq[2])
                self._instr.append(instr)
                self._instr.append(
                    instruction.XC(
                        target=instr.target,
                        domain=seq[3].domain,
                    )
                )
                Nnode += 1
            elif kind == instruction.InstructionKind.S:
                ancilla = [Nnode, Nnode + 1]
                out[instr.target], seq = self._s_command(out[instr.target], ancilla)
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:4])
                self._M.extend(seq[4:6])
                self._instr.append(instr)
                self._instr.append(
                    instruction.XC(
                        target=instr.target,
                        domain=seq[6].domain,
                    )
                )
                self._instr.append(
                    instruction.ZC(
                        target=instr.target,
                        domain=seq[7].domain,
                    )
                )
                Nnode += 2
            elif kind == instruction.InstructionKind.X:
                ancilla = [Nnode, Nnode + 1]
                out[instr.target], seq = self._x_command(out[instr.target], ancilla)
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:4])
                self._M.extend(seq[4:6])
                self._instr.append(instr)
                self._instr.append(
                    instruction.XC(
                        target=instr.target,
                        domain=seq[6].domain,
                    )
                )
                self._instr.append(
                    instruction.ZC(
                        target=instr.target,
                        domain=seq[7].domain,
                    )
                )
                Nnode += 2
            elif kind == instruction.InstructionKind.Y:
                ancilla = [Nnode, Nnode + 1, Nnode + 2, Nnode + 3]
                out[instr.target], seq = self._y_command(out[instr.target], ancilla)
                self._N.extend(seq[0:4])
                self._E.extend(seq[4:8])
                self._M.extend(seq[8:12])
                self._instr.append(instr)
                self._instr.append(
                    instruction.XC(
                        target=instr.target,
                        domain=seq[12].domain,
                    )
                )
                self._instr.append(
                    instruction.ZC(
                        target=instr.target,
                        domain=seq[13].domain,
                    )
                )
                Nnode += 4
            elif kind == instruction.InstructionKind.Z:
                ancilla = [Nnode, Nnode + 1]
                out[instr.target], seq = self._z_command(out[instr.target], ancilla)
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:4])
                self._M.extend(seq[4:6])
                self._instr.append(instr)
                self._instr.append(
                    instruction.XC(
                        target=instr.target,
                        domain=seq[6].domain,
                    )
                )
                self._instr.append(
                    instruction.ZC(
                        target=instr.target,
                        domain=seq[7].domain,
                    )
                )
                Nnode += 2
            elif kind == instruction.InstructionKind.RX:
                ancilla = [Nnode, Nnode + 1]
                out[instr.target], seq = self._rx_command(out[instr.target], ancilla, instr.angle)
                self._N.extend(seq[0:2])
                self._E.extend(seq[2:4])
                self._M.extend(seq[4:6])
                instr_ = deepcopy(instr)
                instr_.meas_index = len(self._M) - 1  # index of arb angle measurement command
                self._instr.append(instr_)
                self._instr.append(
                    instruction.XC(
                        target=instr.target,
                        domain=seq[6].domain,
                    )
                )
                self._instr.append(
                    instruction.ZC(
                        target=instr.target,
                        domain=seq[7].domain,
                    )
                )
                Nnode += 2
            elif kind == instruction.InstructionKind.RY:
                ancilla = [Nnode, Nnode + 1, Nnode + 2, Nnode + 3]
                out[instr.target], seq = self._ry_command(out[instr.target], ancilla, instr.angle)
                self._N.extend(seq[0:4])
                self._E.extend(seq[4:8])
                self._M.extend(seq[8:12])
                instr_ = deepcopy(instr)
                instr_.meas_index = len(self._M) - 3  # index of arb angle measurement command
                self._instr.append(instr_)
                self._instr.append(
                    instruction.XC(
                        target=instr.target,
                        domain=seq[12].domain,
                    )
                )
                self._instr.append(
                    instruction.ZC(
                        target=instr.target,
                        domain=seq[13].domain,
                    )
                )
                Nnode += 4
            elif kind == instruction.InstructionKind.RZ:
                if opt:
                    ancilla = Nnode
                    out[instr.target], seq = self._rz_command_opt(out[instr.target], ancilla, instr.angle)
                    self._N.append(seq[0])
                    self._E.append(seq[1])
                    self._M.append(seq[2])
                    instr_ = deepcopy(instr)
                    instr_.meas_index = len(self._M) - 1  # index of arb angle measurement command
                    self._instr.append(instr_)
                    self._instr.append(
                        instruction.ZC(
                            target=instr.target,
                            domain=seq[3].domain,
                        )
                    )
                    Nnode += 1
                else:
                    ancilla = [Nnode, Nnode + 1]
                    out[instr.target], seq = self._rz_command(out[instr.target], ancilla, instr.angle)
                    self._N.extend(seq[0:2])
                    self._E.extend(seq[2:4])
                    self._M.extend(seq[4:6])
                    instr_ = deepcopy(instr)
                    instr_.meas_index = len(self._M) - 2  # index of arb angle measurement command
                    self._instr.append(instr_)
                    self._instr.append(
                        instruction.XC(
                            target=instr.target,
                            domain=seq[6].domain,
                        )
                    )
                    self._instr.append(
                        instruction.ZC(
                            target=instr.target,
                            domain=seq[7].domain,
                        )
                    )
                    Nnode += 2
            elif kind == instruction.InstructionKind.RZZ:
                ancilla = Nnode
                out[instr.control], out[instr.target], seq = self._rzz_command_opt(
                    out[instr.control], out[instr.target], ancilla, instr.angle
                )
                self._N.append(seq[0])
                self._E.extend(seq[1:3])
                self._M.append(seq[3])
                Nnode += 1
                instr_ = deepcopy(instr)
                instr_.meas_index = len(self._M) - 1  # index of arb angle measurement command
                self._instr.append(instr_)
                self._instr.append(
                    instruction.ZC(
                        target=instr.target,
                        domain=seq[4].domain,
                    )
                )
                self._instr.append(
                    instruction.ZC(
                        target=instr.control,
                        domain=seq[5].domain,
                    )
                )
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
        z_cmds: list[command.Z] = []
        x_cmds: list[command.X] = []
        for i in range(len(self._instr)):
            instr = self._instr[i]
            if instr.kind == instruction.InstructionKind.XC:
                if instr.target in bpx_added.keys():
                    x_cmds[bpx_added[instr.target]].domain.extend(instr.domain)
                else:
                    bpx_added[instr.target] = len(x_cmds)
                    x_cmds.append(X(node=out[instr.target], domain=deepcopy(instr.domain)))
            elif instr.kind == instruction.InstructionKind.ZC:
                if instr.target in bpz_added.keys():
                    z_cmds[bpz_added[instr.target]].domain.extend(instr.domain)
                else:
                    bpz_added[instr.target] = len(z_cmds)
                    z_cmds.append(Z(node=out[instr.target], domain=deepcopy(instr.domain)))
        # append z commands first (X and Z commute up to global phase)
        for cmd in z_cmds:
            command_seq.append(cmd)
        for cmd in x_cmds:
            command_seq.append(cmd)
        pattern = Pattern(input_nodes=inputs)
        pattern.extend(command_seq)
        out = filter(lambda node: node is not None, out)
        pattern.reorder_output_nodes(out)
        return TranspileResult(pattern, classical_outputs)

    def _commute_with_swap(self, target: int):
        correction_instr = self._instr[target]
        swap_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind.XC
            or correction_instr.kind == instruction.InstructionKind.ZC
        )
        assert swap_instr.kind == instruction.InstructionKind.SWAP
        if correction_instr.target == swap_instr.targets[0]:
            correction_instr.target = swap_instr.targets[1]
            self._commute_with_following(target)
        elif correction_instr.target == swap_instr.targets[1]:
            correction_instr.target = swap_instr.targets[0]
            self._commute_with_following(target)
        else:
            self._commute_with_following(target)
        return target

    def _commute_with_cnot(self, target: int):
        correction_instr = self._instr[target]
        cnot_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind.XC
            or correction_instr.kind == instruction.InstructionKind.ZC
        )
        assert cnot_instr.kind == instruction.InstructionKind.CNOT
        if (
            correction_instr.kind == instruction.InstructionKind.XC and correction_instr.target == cnot_instr.control
        ):  # control
            new_cmd = instruction.XC(
                target=cnot_instr.target,
                domain=correction_instr.domain,
            )
            self._commute_with_following(target)
            self._instr.insert(target + 1, new_cmd)
            return target + 1
        elif (
            correction_instr.kind == instruction.InstructionKind.ZC and correction_instr.target == cnot_instr.target
        ):  # target
            new_cmd = instruction.ZC(
                target=cnot_instr.control,
                domain=correction_instr.domain,
            )
            self._commute_with_following(target)
            self._instr.insert(target + 1, new_cmd)
            return target + 1
        else:
            self._commute_with_following(target)
        return target

    def _commute_with_H(self, target: int):
        correction_instr = self._instr[target]
        h_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind.XC
            or correction_instr.kind == instruction.InstructionKind.ZC
        )
        assert h_instr.kind == instruction.InstructionKind.H
        if correction_instr.target == h_instr.target:
            if correction_instr.kind == instruction.InstructionKind.XC:
                self._instr[target] = instruction.ZC(
                    target=correction_instr.target, domain=correction_instr.domain
                )  # byproduct changes to Z
                self._commute_with_following(target)
            else:
                self._instr[target] = instruction.XC(
                    target=correction_instr.target, domain=correction_instr.domain
                )  # byproduct changes to X
                self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_S(self, target: int):
        correction_instr = self._instr[target]
        s_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind.XC
            or correction_instr.kind == instruction.InstructionKind.ZC
        )
        assert s_instr.kind == instruction.InstructionKind.S
        if correction_instr.target == s_instr.target:
            if correction_instr.kind == instruction.InstructionKind.XC:
                self._commute_with_following(target)
                # changes to Y = XZ
                self._instr.insert(
                    target + 1,
                    instruction.ZC(
                        target=correction_instr.target,
                        domain=correction_instr.domain,
                    ),
                )
                return target + 1
        self._commute_with_following(target)
        return target

    def _commute_with_Rx(self, target: int):
        correction_instr = self._instr[target]
        rx_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind.XC
            or correction_instr.kind == instruction.InstructionKind.ZC
        )
        assert rx_instr.kind == instruction.InstructionKind.RX
        if correction_instr.target == rx_instr.target:
            if correction_instr.kind == instruction.InstructionKind.ZC:
                # add to the s-domain
                self._M[rx_instr.meas_index].s_domain.extend(correction_instr.domain)
                self._commute_with_following(target)
            else:
                self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_Ry(self, target: int):
        correction_instr = self._instr[target]
        ry_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind.XC
            or correction_instr.kind == instruction.InstructionKind.ZC
        )
        assert ry_instr.kind == instruction.InstructionKind.RY
        if correction_instr.target == ry_instr.target:
            # add to the s-domain
            self._M[ry_instr.meas_index].s_domain.extend(correction_instr.domain)
            self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_Rz(self, target: int):
        correction_instr = self._instr[target]
        rz_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind.XC
            or correction_instr.kind == instruction.InstructionKind.ZC
        )
        assert rz_instr.kind == instruction.InstructionKind.RZ
        if correction_instr.target == rz_instr.target:
            if correction_instr.kind == instruction.InstructionKind.XC:
                # add to the s-domain
                self._M[rz_instr.meas_index].s_domain.extend(correction_instr.domain)
                self._commute_with_following(target)
            else:
                self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_Rzz(self, target: int):
        correction_instr = self._instr[target]
        rzz_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind.XC
            or correction_instr.kind == instruction.InstructionKind.ZC
        )
        assert rzz_instr.kind == instruction.InstructionKind.RZZ
        if correction_instr.kind == instruction.InstructionKind.XC:
            cond = correction_instr.target == rzz_instr.control
            cond2 = correction_instr.target == rzz_instr.target
            if cond or cond2:
                # add to the s-domain
                self._M[rzz_instr.meas_index].s_domain.extend(correction_instr.domain)
        self._commute_with_following(target)

    def _commute_with_following(self, target: int):
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

    def _find_byproduct_to_move(self, rev: bool = False, skipnum: int = 0):
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
            if (
                self._instr[target].kind == instruction.InstructionKind.ZC
                or self._instr[target].kind == instruction.InstructionKind.XC
            ):
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
            if (target == len(self._instr) - 1) or (
                self._instr[target + 1].kind == instruction.InstructionKind.XC
                or self._instr[target + 1].kind == instruction.InstructionKind.ZC
            ):
                moved += 1
                target = self._find_byproduct_to_move(rev=True, skipnum=moved)
                continue
            next_instr = self._instr[target + 1]
            kind = next_instr.kind
            if kind == instruction.InstructionKind.CNOT:
                target = self._commute_with_cnot(target)
            elif kind == instruction.InstructionKind.SWAP:
                target = self._commute_with_swap(target)
            elif kind == instruction.InstructionKind.H:
                self._commute_with_H(target)
            elif kind == instruction.InstructionKind.S:
                target = self._commute_with_S(target)
            elif kind == instruction.InstructionKind.RX:
                self._commute_with_Rx(target)
            elif kind == instruction.InstructionKind.RY:
                self._commute_with_Ry(target)
            elif kind == instruction.InstructionKind.RZ:
                self._commute_with_Rz(target)
            elif kind == instruction.InstructionKind.RZZ:
                self._commute_with_Rzz(target)
            else:
                # Pauli gates commute up to global phase.
                self._commute_with_following(target)
            target += 1

    @classmethod
    def _cnot_command(
        self, control_node: int, target_node: int, ancilla: Sequence[int]
    ) -> tuple[int, int, list[command.Command]]:
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
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.append(E(nodes=(target_node, ancilla[0])))
        seq.append(E(nodes=(control_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=target_node))
        seq.append(M(node=ancilla[0]))
        seq.append(X(node=ancilla[1], domain=[ancilla[0]]))
        seq.append(Z(node=ancilla[1], domain=[target_node]))
        seq.append(Z(node=control_node, domain=[target_node]))
        return control_node, ancilla[1], seq

    @classmethod
    def _m_command(self, input_node: int, plane: graphix.pauli.Plane, angle: float):
        """MBQC commands for measuring qubit

        Parameters
        ---------
        input_node : int
            target node on graph
        plane : graphix.pauli.Plane
            plane of the measure
        angle : float
            angle of the measure (unit: pi radian)

        Returns
        ---------
        commands : list
            list of MBQC commands
        """
        seq = [M(node=input_node, plane=plane, angle=angle)]
        return seq

    @classmethod
    def _h_command(self, input_node: int, ancilla: int):
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
        seq = [N(node=ancilla)]
        seq.append(E(nodes=(input_node, ancilla)))
        seq.append(M(node=input_node))
        seq.append(X(node=ancilla, domain=[input_node]))
        return ancilla, seq

    @classmethod
    def _s_command(self, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
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
        seq = [N(node=ancilla[0]), command.N(node=ancilla[1])]
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=input_node, angle=-0.5))
        seq.append(M(node=ancilla[0]))
        seq.append(X(node=ancilla[1], domain=[ancilla[0]]))
        seq.append(Z(node=ancilla[1], domain=[input_node]))
        return ancilla[1], seq

    @classmethod
    def _x_command(self, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
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
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=input_node))
        seq.append(M(node=ancilla[0], angle=-1))
        seq.append(X(node=ancilla[1], domain=[ancilla[0]]))
        seq.append(Z(node=ancilla[1], domain=[input_node]))
        return ancilla[1], seq

    @classmethod
    def _y_command(self, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
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
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend([N(node=ancilla[2]), N(node=ancilla[3])])
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(E(nodes=(ancilla[1], ancilla[2])))
        seq.append(E(nodes=(ancilla[2], ancilla[3])))
        seq.append(M(node=input_node, angle=0.5))
        seq.append(M(node=ancilla[0], angle=1.0, s_domain=[input_node]))
        seq.append(M(node=ancilla[1], angle=-0.5, s_domain=[input_node]))
        seq.append(M(node=ancilla[2]))
        seq.append(X(node=ancilla[3], domain=[ancilla[0], ancilla[2]]))
        seq.append(Z(node=ancilla[3], domain=[ancilla[0], ancilla[1]]))
        return ancilla[3], seq

    @classmethod
    def _z_command(self, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
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
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=input_node, angle=-1))
        seq.append(M(node=ancilla[0]))
        seq.append(X(node=ancilla[1], domain=[ancilla[0]]))
        seq.append(Z(node=ancilla[1], domain=[input_node]))
        return ancilla[1], seq

    @classmethod
    def _rx_command(self, input_node: int, ancilla: Sequence[int], angle: float) -> tuple[int, list[command.Command]]:
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
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=input_node))
        seq.append(M(node=ancilla[0], angle=-angle / np.pi, s_domain=[input_node]))
        seq.append(X(node=ancilla[1], domain=[ancilla[0]]))
        seq.append(Z(node=ancilla[1], domain=[input_node]))
        return ancilla[1], seq

    @classmethod
    def _ry_command(self, input_node: int, ancilla: Sequence[int], angle: float) -> tuple[int, list[command.Command]]:
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
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend([N(node=ancilla[2]), N(node=ancilla[3])])
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(E(nodes=(ancilla[1], ancilla[2])))
        seq.append(E(nodes=(ancilla[2], ancilla[3])))
        seq.append(M(node=input_node, angle=0.5))
        seq.append(M(node=ancilla[0], angle=-angle / np.pi, s_domain=[input_node]))
        seq.append(M(node=ancilla[1], angle=-0.5, s_domain=[input_node]))
        seq.append(M(node=ancilla[2]))
        seq.append(X(node=ancilla[3], domain=[ancilla[0], ancilla[2]]))
        seq.append(Z(node=ancilla[3], domain=[ancilla[0], ancilla[1]]))
        return ancilla[3], seq

    @classmethod
    def _rz_command(self, input_node: int, ancilla: Sequence[int], angle: float) -> tuple[int, list[command.Command]]:
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
            node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]  # assign new qubit labels
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=input_node, angle=-angle / np.pi))
        seq.append(M(node=ancilla[0]))
        seq.append(X(node=ancilla[1], domain=[ancilla[0]]))
        seq.append(Z(node=ancilla[1], domain=[input_node]))
        return ancilla[1], seq

    @classmethod
    def _rz_command_opt(self, input_node: int, ancilla: int, angle: float) -> tuple[int, list[command.Command]]:
        """optimized MBQC commands for Z rotation gate

        Parameters
        ---------
        input_node : int
            input node index
        ancilla : int
            ancilla node index to be added to graph
        angle : float
            measurement angle in radian

        Returns
        ---------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        seq = [N(node=ancilla)]
        seq.append(E(nodes=(input_node, ancilla)))
        seq.append(M(node=ancilla, angle=-angle / np.pi, vop=6))
        seq.append(Z(node=input_node, domain=[ancilla]))
        return input_node, seq

    @classmethod
    def _rzz_command_opt(
        self, control_node: int, target_node: int, ancilla: int, angle: float
    ) -> tuple[int, int, list[command.Command]]:
        """Optimized MBQC commands for ZZ-rotation gate

        Parameters
        ---------
        input_node : int
            input node index
        ancilla : int
            ancilla node index
        angle : float
            measurement angle in radian

        Returns
        ---------
        out_node_control : int
            control node on graph after the gate
        out_node_target : int
            target node on graph after the gate
        commands : list
            list of MBQC commands
        """
        seq = [N(node=ancilla)]
        seq.append(E(nodes=(control_node, ancilla)))
        seq.append(E(nodes=(target_node, ancilla)))
        seq.append(M(node=ancilla, angle=-angle / np.pi, vop=6))
        seq.append(Z(node=control_node, domain=[ancilla]))
        seq.append(Z(node=target_node, domain=[ancilla]))
        return control_node, target_node, seq

    @classmethod
    def _ccx_command(
        self,
        control_node1: int,
        control_node2: int,
        target_node: int,
        ancilla: Sequence[int],
    ) -> tuple[int, int, int, list[command.Command]]:
        """MBQC commands for CCX gate

        Parameters
        ---------
        control_node1 : int
            first control node on graph
        control_node2 : int
            second control node on graph
        target_node : int
            target node on graph
        ancilla : list of int
            ancilla node indices to be added to graph

        Returns
        ---------
        control_out1 : int
            first control node on graph after the gate
        control_out2 : int
            second control node on graph after the gate
        target_out : int
            target node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 18
        seq = [N(node=ancilla[i]) for i in range(18)]  # assign new qubit labels
        seq.append(E(nodes=(target_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(E(nodes=(ancilla[1], ancilla[2])))
        seq.append(E(nodes=(ancilla[1], control_node2)))
        seq.append(E(nodes=(control_node1, ancilla[14])))
        seq.append(E(nodes=(ancilla[2], ancilla[3])))
        seq.append(E(nodes=(ancilla[14], ancilla[4])))
        seq.append(E(nodes=(ancilla[3], ancilla[5])))
        seq.append(E(nodes=(ancilla[3], ancilla[4])))
        seq.append(E(nodes=(ancilla[5], ancilla[6])))
        seq.append(E(nodes=(control_node2, ancilla[6])))
        seq.append(E(nodes=(control_node2, ancilla[9])))
        seq.append(E(nodes=(ancilla[6], ancilla[7])))
        seq.append(E(nodes=(ancilla[9], ancilla[4])))
        seq.append(E(nodes=(ancilla[9], ancilla[10])))
        seq.append(E(nodes=(ancilla[7], ancilla[8])))
        seq.append(E(nodes=(ancilla[10], ancilla[11])))
        seq.append(E(nodes=(ancilla[4], ancilla[8])))
        seq.append(E(nodes=(ancilla[4], ancilla[11])))
        seq.append(E(nodes=(ancilla[4], ancilla[16])))
        seq.append(E(nodes=(ancilla[8], ancilla[12])))
        seq.append(E(nodes=(ancilla[11], ancilla[15])))
        seq.append(E(nodes=(ancilla[12], ancilla[13])))
        seq.append(E(nodes=(ancilla[16], ancilla[17])))
        seq.append(M(node=target_node))
        seq.append(M(node=ancilla[0], s_domain=[target_node]))
        seq.append(M(node=ancilla[1], s_domain=[ancilla[0]]))
        seq.append(M(node=control_node1))
        seq.append(M(node=ancilla[2], angle=-1.75, s_domain=[ancilla[1], target_node]))
        seq.append(M(node=ancilla[14], s_domain=[control_node1]))
        seq.append(M(node=ancilla[3], s_domain=[ancilla[2], ancilla[0]]))
        seq.append(
            M(
                node=ancilla[5],
                angle=-0.25,
                s_domain=[ancilla[3], ancilla[1], ancilla[14], target_node],
            )
        )
        seq.append(M(node=control_node2, angle=-0.25))
        seq.append(M(node=ancilla[6], s_domain=[ancilla[5], ancilla[2], ancilla[0]]))
        seq.append(
            M(
                node=ancilla[9],
                s_domain=[
                    control_node2,
                    ancilla[0],
                    ancilla[5],
                    ancilla[2],
                    ancilla[0],
                ],
            )
        )
        seq.append(
            M(
                node=ancilla[7],
                angle=-1.75,
                s_domain=[ancilla[6], ancilla[3], ancilla[1], ancilla[14], target_node],
            )
        )
        seq.append(M(node=ancilla[10], angle=-1.75, s_domain=[ancilla[9], ancilla[14]]))
        seq.append(M(node=ancilla[4], angle=-0.25, s_domain=[ancilla[14]]))
        seq.append(
            M(
                node=ancilla[8],
                s_domain=[ancilla[7], ancilla[5], ancilla[2], ancilla[0]],
            )
        )
        seq.append(
            M(
                node=ancilla[11],
                s_domain=[
                    ancilla[10],
                    control_node2,
                    ancilla[0],
                    ancilla[5],
                    ancilla[2],
                    ancilla[0],
                ],
            )
        )
        seq.append(
            M(
                node=ancilla[12],
                angle=-0.25,
                s_domain=[
                    ancilla[8],
                    ancilla[14],
                    ancilla[6],
                    ancilla[3],
                    ancilla[1],
                    ancilla[14],
                    target_node,
                ],
            )
        )
        seq.append(
            M(
                node=ancilla[16],
                s_domain=[
                    ancilla[4],
                    control_node1,
                    ancilla[2],
                    control_node2,
                    ancilla[7],
                    ancilla[10],
                    ancilla[0],
                    ancilla[0],
                    ancilla[5],
                    ancilla[2],
                    ancilla[0],
                    ancilla[5],
                    ancilla[2],
                    ancilla[0],
                    control_node2,
                    ancilla[0],
                    ancilla[5],
                    ancilla[2],
                    ancilla[0],
                ],
            )
        )
        seq.append(X(node=ancilla[17], domain=[ancilla[14], ancilla[16]]))
        seq.append(X(node=ancilla[15], domain=[ancilla[9], ancilla[11]]))
        seq.append(
            X(
                node=ancilla[13],
                domain=[ancilla[0], ancilla[2], ancilla[5], ancilla[7], ancilla[12]],
            )
        )
        seq.append(
            Z(
                node=ancilla[17],
                domain=[ancilla[4], ancilla[5], ancilla[7], ancilla[10], control_node1],
            )
        )
        seq.append(
            Z(
                node=ancilla[15],
                domain=[control_node2, ancilla[2], ancilla[5], ancilla[10]],
            )
        )
        seq.append(
            Z(
                node=ancilla[13],
                domain=[ancilla[1], ancilla[3], ancilla[6], ancilla[8], target_node],
            )
        )
        return ancilla[17], ancilla[15], ancilla[13], seq

    @classmethod
    def _ccx_command_opt(
        self,
        control_node1: int,
        control_node2: int,
        target_node: int,
        ancilla: Sequence[int],
    ) -> tuple[int, int, int, list[command.Command]]:
        """Optimized MBQC commands for CCX gate

        Parameters
        ---------
        control_node1 : int
            first control node on graph
        control_node2 : int
            second control node on graph
        target_node : int
            target node on graph
        ancilla : list of int
            ancilla node indices to be added to graph

        Returns
        ---------
        control_out1 : int
            first control node on graph after the gate
        control_out2 : int
            second control node on graph after the gate
        target_out : int
            target node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 11
        seq = [N(node=ancilla[i]) for i in range(11)]
        seq.append(E(nodes=(control_node1, ancilla[8])))
        seq.append(E(nodes=(control_node2, ancilla[4])))
        seq.append(E(nodes=(control_node2, ancilla[5])))
        seq.append(E(nodes=(control_node2, ancilla[2])))
        seq.append(E(nodes=(control_node2, ancilla[0])))
        seq.append(E(nodes=(target_node, ancilla[6])))
        seq.append(E(nodes=(ancilla[0], ancilla[6])))
        seq.append(E(nodes=(ancilla[1], ancilla[10])))
        seq.append(E(nodes=(ancilla[2], ancilla[10])))
        seq.append(E(nodes=(ancilla[2], ancilla[6])))
        seq.append(E(nodes=(ancilla[3], ancilla[6])))
        seq.append(E(nodes=(ancilla[3], ancilla[10])))
        seq.append(E(nodes=(ancilla[4], ancilla[10])))
        seq.append(E(nodes=(ancilla[5], ancilla[9])))
        seq.append(E(nodes=(ancilla[6], ancilla[7])))
        seq.append(E(nodes=(ancilla[8], ancilla[10])))
        seq.append(M(node=target_node))
        seq.append(M(node=control_node1))
        seq.append(M(node=ancilla[0], angle=-1.75, s_domain=[target_node], vop=6))
        seq.append(M(node=ancilla[8], s_domain=[control_node1]))
        seq.append(M(node=ancilla[2], angle=-0.25, s_domain=[target_node, ancilla[8]], vop=6))
        seq.append(M(node=control_node2, angle=-0.25))
        seq.append(M(node=ancilla[3], angle=-1.75, s_domain=[ancilla[8], target_node], vop=6))
        seq.append(M(node=ancilla[4], angle=-1.75, s_domain=[ancilla[8]], vop=6))
        seq.append(M(node=ancilla[1], angle=-0.25, s_domain=[ancilla[8]], vop=6))
        seq.append(
            M(
                node=ancilla[5],
                s_domain=[control_node2, ancilla[0], ancilla[2], ancilla[4]],
            )
        )
        seq.append(M(node=ancilla[6], angle=-0.25, s_domain=[target_node]))
        seq.append(X(node=ancilla[10], domain=[ancilla[8]]))
        seq.append(X(node=ancilla[9], domain=[ancilla[5]]))
        seq.append(X(node=ancilla[7], domain=[ancilla[0], ancilla[2], ancilla[3], ancilla[6]]))
        seq.append(
            Z(
                node=ancilla[10],
                domain=[control_node1, ancilla[1], ancilla[2], ancilla[3], ancilla[4]],
            )
        )
        seq.append(
            Z(
                node=ancilla[9],
                domain=[control_node2, ancilla[0], ancilla[2], ancilla[4]],
            )
        )
        seq.append(Z(node=ancilla[7], domain=[target_node]))

        return ancilla[10], ancilla[9], ancilla[7], seq

    @classmethod
    def _sort_outputs(self, pattern: Pattern, output_nodes: Sequence[int]):
        """Sort the node indices of ouput qubits.

        Parameters
        ---------
        pattern : :meth:`~graphix.pattern.Pattern`
            pattern object
        output_nodes : list of int
            output node indices

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
        for cmd in pattern:
            if cmd.kind == CommandKind.E:
                j, k = cmd.nodes
                if j in old_out:
                    j = output_nodes[old_out.index(j)]
                if k in old_out:
                    k = output_nodes[old_out.index(k)]
                cmd.nodes = (j, k)
            elif cmd.nodes in old_out:
                cmd.nodes = output_nodes[old_out.index(cmd.nodes)]

    def simulate_statevector(self, input_state: graphix.sim.statevec.Data | None = None) -> SimulateResult:
        """Run statevector simulation of the gate sequence, using graphix.Statevec

        Parameters
        ----------
        input_state : :class:`graphix.Statevec`

        Returns
        -------
        result : :class:`SimulateResult`
            output state of the statevector simulation and results of classical measures.
        """

        if input_state is None:
            state = graphix.sim.statevec.Statevec(nqubit=self.width)
        else:
            state = graphix.sim.statevec.Statevec(nqubit=self.width, data=input_state)

        classical_measures = []

        for i in range(len(self.instruction)):
            instr = self.instruction[i]
            kind = instr.kind
            if kind == instruction.InstructionKind.CNOT:
                state.CNOT((instr.control, instr.target))
            elif kind == instruction.InstructionKind.SWAP:
                state.swap(instr.targets)
            elif kind == instruction.InstructionKind.I:
                pass
            elif kind == instruction.InstructionKind.S:
                state.evolve_single(Ops.s, instr.target)
            elif kind == instruction.InstructionKind.H:
                state.evolve_single(Ops.h, instr.target)
            elif kind == instruction.InstructionKind.X:
                state.evolve_single(Ops.x, instr.target)
            elif kind == instruction.InstructionKind.Y:
                state.evolve_single(Ops.y, instr.target)
            elif kind == instruction.InstructionKind.Z:
                state.evolve_single(Ops.z, instr.target)
            elif kind == instruction.InstructionKind.RX:
                state.evolve_single(Ops.Rx(instr.angle), instr.target)
            elif kind == instruction.InstructionKind.RY:
                state.evolve_single(Ops.Ry(instr.angle), instr.target)
            elif kind == instruction.InstructionKind.RZ:
                state.evolve_single(Ops.Rz(instr.angle), instr.target)
            elif kind == instruction.InstructionKind.RZZ:
                state.evolve(Ops.Rzz(instr.angle), [instr.control, instr.target])
            elif kind == instruction.InstructionKind.CCX:
                state.evolve(Ops.ccx, [instr.controls[0], instr.controls[1], instr.target])
            elif kind == instruction.InstructionKind.M:
                result = graphix.sim.base_backend.perform_measure(
                    instr.target, instr.plane, instr.angle * np.pi, state, np.random
                )
                classical_measures.append(result)
            else:
                raise ValueError(f"Unknown instruction: {instr}")

        return SimulateResult(state, classical_measures)

    def subs(self, variable, substitute) -> Circuit:
        result = Circuit(self.width)
        for instr in self.instruction:
            angle = getattr(instr, "angle", None)
            if angle is None:
                result.instruction.append(instr)
            else:
                new_instr = instr.model_copy(update={"angle": graphix.parameter.subs(angle, variable, substitute)})
                result.instruction.append(new_instr)
        return result
