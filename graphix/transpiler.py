"""Gate-to-MBQC transpiler.

accepts desired gate operations and transpile into MBQC measurement patterns.

"""

from __future__ import annotations

import dataclasses
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from graphix import command, instruction
from graphix.clifford import Clifford
from graphix.command import CommandKind, E, M, N, X, Z
from graphix.fundamentals import Plane
from graphix.ops import Ops
from graphix.pattern import Pattern
from graphix.sim import base_backend
from graphix.sim.statevec import Data, Statevec

if TYPE_CHECKING:
    from collections.abc import Sequence


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

    statevec: Statevec
    classical_measures: tuple[int, ...]


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
        Construct a circuit.

        Parameters
        ----------
        width : int
            number of logical qubits for the gate network
        """
        self.width = width
        self.instruction: list[instruction.Instruction] = []
        self.active_qubits = set(range(width))

    def cnot(self, control: int, target: int):
        """Apply a CNOT gate.

        Parameters
        ----------
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
        """Apply a SWAP gate.

        Parameters
        ----------
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
        """Apply a Hadamard gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.H(target=qubit))

    def s(self, qubit: int):
        """Apply an S gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.S(target=qubit))

    def x(self, qubit):
        """Apply a Pauli X gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.X(target=qubit))

    def y(self, qubit: int):
        """Apply a Pauli Y gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.Y(target=qubit))

    def z(self, qubit: int):
        """Apply a Pauli Z gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.Z(target=qubit))

    def rx(self, qubit: int, angle: float):
        """Apply an X rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : float
            rotation angle in radian
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RX(target=qubit, angle=angle))

    def ry(self, qubit: int, angle: float):
        """Apply a Y rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : float
            angle in radian
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RY(target=qubit, angle=angle))

    def rz(self, qubit: int, angle: float):
        """Apply a Z rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : float
            rotation angle in radian
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RZ(target=qubit, angle=angle))

    def rzz(self, control: int, target: int, angle: float):
        r"""Apply a ZZ-rotation gate.

        Equivalent to the sequence
        CNOT(control, target),
        Rz(target, angle),
        CNOT(control, target)

        and realizes rotation expressed by
        :math:`e^{-i \frac{\theta}{2} Z_c Z_t}`.

        Parameters
        ----------
        control : int
            control qubit
        target : int
            target qubit
        angle : float
            rotation angle in radian
        """
        assert control in self.active_qubits
        assert target in self.active_qubits
        self.instruction.append(instruction.RZZ(control=control, target=target, angle=angle))

    def ccx(self, control1: int, control2: int, target: int):
        r"""Apply a CCX (Toffoli) gate.

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
        assert control1 != control2
        assert control1 != target
        assert control2 != target
        self.instruction.append(instruction.CCX(controls=(control1, control2), target=target))

    def i(self, qubit: int):
        """Apply an identity (teleportation) gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.I(target=qubit))

    def m(self, qubit: int, plane: Plane, angle: float):
        """Measure a quantum qubit.

        The measured qubit cannot be used afterwards.

        Parameters
        ----------
        qubit : int
            target qubit
        plane : Plane
        angle : float
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.M(target=qubit, plane=plane, angle=angle))
        self.active_qubits.remove(qubit)

    def transpile(self, opt: bool = False) -> TranspileResult:
        """Transpile the circuit to a pattern.

        Parameters
        ----------
        opt : bool
            Whether or not to use pre-optimized gateset with local-Clifford decoration.

        Returns
        -------
        result : :class:`TranspileResult` object
        """
        n_node = self.width
        out = [j for j in range(self.width)]
        pattern = Pattern(input_nodes=[j for j in range(self.width)])
        classical_outputs = []
        for instr in self.instruction:
            kind = instr.kind
            if kind == instruction.InstructionKind.CNOT:
                ancilla = [n_node, n_node + 1]
                assert out[instr.control] is not None
                assert out[instr.target] is not None
                out[instr.control], out[instr.target], seq = self._cnot_command(
                    out[instr.control], out[instr.target], ancilla
                )
                pattern.extend(seq)
                n_node += 2
            elif kind == instruction.InstructionKind.SWAP:
                out[instr.targets[0]], out[instr.targets[1]] = (
                    out[instr.targets[1]],
                    out[instr.targets[0]],
                )
            elif kind == instruction.InstructionKind.I:
                pass
            elif kind == instruction.InstructionKind.H:
                ancilla = n_node
                out[instr.target], seq = self._h_command(out[instr.target], ancilla)
                pattern.extend(seq)
                n_node += 1
            elif kind == instruction.InstructionKind.S:
                ancilla = [n_node, n_node + 1]
                out[instr.target], seq = self._s_command(out[instr.target], ancilla)
                pattern.extend(seq)
                n_node += 2
            elif kind == instruction.InstructionKind.X:
                ancilla = [n_node, n_node + 1]
                out[instr.target], seq = self._x_command(out[instr.target], ancilla)
                pattern.extend(seq)
                n_node += 2
            elif kind == instruction.InstructionKind.Y:
                ancilla = [n_node, n_node + 1, n_node + 2, n_node + 3]
                out[instr.target], seq = self._y_command(out[instr.target], ancilla)
                pattern.extend(seq)
                n_node += 4
            elif kind == instruction.InstructionKind.Z:
                ancilla = [n_node, n_node + 1]
                out[instr.target], seq = self._z_command(out[instr.target], ancilla)
                pattern.extend(seq)
                n_node += 2
            elif kind == instruction.InstructionKind.RX:
                ancilla = [n_node, n_node + 1]
                out[instr.target], seq = self._rx_command(out[instr.target], ancilla, instr.angle)
                pattern.extend(seq)
                n_node += 2
            elif kind == instruction.InstructionKind.RY:
                ancilla = [n_node, n_node + 1, n_node + 2, n_node + 3]
                out[instr.target], seq = self._ry_command(out[instr.target], ancilla, instr.angle)
                pattern.extend(seq)
                n_node += 4
            elif kind == instruction.InstructionKind.RZ:
                if opt:
                    ancilla = n_node
                    out[instr.target], seq = self._rz_command_opt(out[instr.target], ancilla, instr.angle)
                    pattern.extend(seq)
                    n_node += 1
                else:
                    ancilla = [n_node, n_node + 1]
                    out[instr.target], seq = self._rz_command(out[instr.target], ancilla, instr.angle)
                    pattern.extend(seq)
                    n_node += 2
            elif kind == instruction.InstructionKind.RZZ:
                if opt:
                    ancilla = n_node
                    (
                        out[instr.control],
                        out[instr.target],
                        seq,
                    ) = self._rzz_command_opt(out[instr.control], out[instr.target], ancilla, instr.angle)
                    pattern.extend(seq)
                    n_node += 1
                else:
                    raise NotImplementedError(
                        "YZ-plane measurements not accepted and Rzz gate\
                        cannot be directly transpiled"
                    )
            elif kind == instruction.InstructionKind.CCX:
                if opt:
                    ancilla = [n_node + i for i in range(11)]
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
                    n_node += 11
                else:
                    ancilla = [n_node + i for i in range(18)]
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
                    n_node += 18
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
        """Transpile the circuit to a standardized pattern.

        Commutes all byproduct through gates, instead of through measurement
        commands, to generate standardized measurement pattern.

        Parameters
        ----------
        opt : bool
            Whether or not to use pre-optimized gateset with local-Clifford decoration.

        Returns
        -------
        pattern : :class:`graphix.pattern.Pattern` object
        """
        warnings.warn(
            "`Circuit.standardize_and_transpile` is deprecated. Please use `Circuit.transpile` and `Pattern.standardize` in sequence instead. See https://github.com/TeamGraphix/graphix/pull/190 for more informations.",
            stacklevel=1,
        )
        self._n: list[N] = []
        # for i in range(self.width):
        #    self._n.append(["N", i])
        self._m: list[M] = []
        self._e: list[E] = []
        self._instr: list[instruction.Instruction] = []
        n_node = self.width
        inputs = [j for j in range(self.width)]
        out = [j for j in range(self.width)]
        classical_outputs = []
        for instr in self.instruction:
            kind = instr.kind
            if kind == instruction.InstructionKind.CNOT:
                ancilla = [n_node, n_node + 1]
                assert out[instr.control] is not None
                assert out[instr.target] is not None
                out[instr.control], out[instr.target], seq = self._cnot_command(
                    out[instr.control], out[instr.target], ancilla
                )
                self._n.extend(seq[0:2])
                self._e.extend(seq[2:5])
                self._m.extend(seq[5:7])
                n_node += 2
                self._instr.append(instr)
                self._instr.append(
                    instruction._XC(
                        target=instr.target,
                        domain=seq[7].domain,
                    )
                )
                self._instr.append(
                    instruction._ZC(
                        target=instr.target,
                        domain=seq[8].domain,
                    )
                )
                self._instr.append(
                    instruction._ZC(
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
                ancilla = n_node
                out[instr.target], seq = self._h_command(out[instr.target], ancilla)
                self._n.append(seq[0])
                self._e.append(seq[1])
                self._m.append(seq[2])
                self._instr.append(instr)
                self._instr.append(
                    instruction._XC(
                        target=instr.target,
                        domain=seq[3].domain,
                    )
                )
                n_node += 1
            elif kind == instruction.InstructionKind.S:
                ancilla = [n_node, n_node + 1]
                out[instr.target], seq = self._s_command(out[instr.target], ancilla)
                self._n.extend(seq[0:2])
                self._e.extend(seq[2:4])
                self._m.extend(seq[4:6])
                self._instr.append(instr)
                self._instr.append(
                    instruction._XC(
                        target=instr.target,
                        domain=seq[6].domain,
                    )
                )
                self._instr.append(
                    instruction._ZC(
                        target=instr.target,
                        domain=seq[7].domain,
                    )
                )
                n_node += 2
            elif kind == instruction.InstructionKind.X:
                ancilla = [n_node, n_node + 1]
                out[instr.target], seq = self._x_command(out[instr.target], ancilla)
                self._n.extend(seq[0:2])
                self._e.extend(seq[2:4])
                self._m.extend(seq[4:6])
                self._instr.append(instr)
                self._instr.append(
                    instruction._XC(
                        target=instr.target,
                        domain=seq[6].domain,
                    )
                )
                self._instr.append(
                    instruction._ZC(
                        target=instr.target,
                        domain=seq[7].domain,
                    )
                )
                n_node += 2
            elif kind == instruction.InstructionKind.Y:
                ancilla = [n_node, n_node + 1, n_node + 2, n_node + 3]
                out[instr.target], seq = self._y_command(out[instr.target], ancilla)
                self._n.extend(seq[0:4])
                self._e.extend(seq[4:8])
                self._m.extend(seq[8:12])
                self._instr.append(instr)
                self._instr.append(
                    instruction._XC(
                        target=instr.target,
                        domain=seq[12].domain,
                    )
                )
                self._instr.append(
                    instruction._ZC(
                        target=instr.target,
                        domain=seq[13].domain,
                    )
                )
                n_node += 4
            elif kind == instruction.InstructionKind.Z:
                ancilla = [n_node, n_node + 1]
                out[instr.target], seq = self._z_command(out[instr.target], ancilla)
                self._n.extend(seq[0:2])
                self._e.extend(seq[2:4])
                self._m.extend(seq[4:6])
                self._instr.append(instr)
                self._instr.append(
                    instruction._XC(
                        target=instr.target,
                        domain=seq[6].domain,
                    )
                )
                self._instr.append(
                    instruction._ZC(
                        target=instr.target,
                        domain=seq[7].domain,
                    )
                )
                n_node += 2
            elif kind == instruction.InstructionKind.RX:
                ancilla = [n_node, n_node + 1]
                out[instr.target], seq = self._rx_command(out[instr.target], ancilla, instr.angle)
                self._n.extend(seq[0:2])
                self._e.extend(seq[2:4])
                self._m.extend(seq[4:6])
                instr_ = deepcopy(instr)
                instr_.meas_index = len(self._m) - 1  # index of arb angle measurement command
                self._instr.append(instr_)
                self._instr.append(
                    instruction._XC(
                        target=instr.target,
                        domain=seq[6].domain,
                    )
                )
                self._instr.append(
                    instruction._ZC(
                        target=instr.target,
                        domain=seq[7].domain,
                    )
                )
                n_node += 2
            elif kind == instruction.InstructionKind.RY:
                ancilla = [n_node, n_node + 1, n_node + 2, n_node + 3]
                out[instr.target], seq = self._ry_command(out[instr.target], ancilla, instr.angle)
                self._n.extend(seq[0:4])
                self._e.extend(seq[4:8])
                self._m.extend(seq[8:12])
                instr_ = deepcopy(instr)
                instr_.meas_index = len(self._m) - 3  # index of arb angle measurement command
                self._instr.append(instr_)
                self._instr.append(
                    instruction._XC(
                        target=instr.target,
                        domain=seq[12].domain,
                    )
                )
                self._instr.append(
                    instruction._ZC(
                        target=instr.target,
                        domain=seq[13].domain,
                    )
                )
                n_node += 4
            elif kind == instruction.InstructionKind.RZ:
                if opt:
                    ancilla = n_node
                    out[instr.target], seq = self._rz_command_opt(out[instr.target], ancilla, instr.angle)
                    self._n.append(seq[0])
                    self._e.append(seq[1])
                    self._m.append(seq[2])
                    instr_ = deepcopy(instr)
                    instr_.meas_index = len(self._m) - 1  # index of arb angle measurement command
                    self._instr.append(instr_)
                    self._instr.append(
                        instruction._ZC(
                            target=instr.target,
                            domain=seq[3].domain,
                        )
                    )
                    n_node += 1
                else:
                    ancilla = [n_node, n_node + 1]
                    out[instr.target], seq = self._rz_command(out[instr.target], ancilla, instr.angle)
                    self._n.extend(seq[0:2])
                    self._e.extend(seq[2:4])
                    self._m.extend(seq[4:6])
                    instr_ = deepcopy(instr)
                    instr_.meas_index = len(self._m) - 2  # index of arb angle measurement command
                    self._instr.append(instr_)
                    self._instr.append(
                        instruction._XC(
                            target=instr.target,
                            domain=seq[6].domain,
                        )
                    )
                    self._instr.append(
                        instruction._ZC(
                            target=instr.target,
                            domain=seq[7].domain,
                        )
                    )
                    n_node += 2
            elif kind == instruction.InstructionKind.RZZ:
                ancilla = n_node
                out[instr.control], out[instr.target], seq = self._rzz_command_opt(
                    out[instr.control], out[instr.target], ancilla, instr.angle
                )
                self._n.append(seq[0])
                self._e.extend(seq[1:3])
                self._m.append(seq[3])
                n_node += 1
                instr_ = deepcopy(instr)
                instr_.meas_index = len(self._m) - 1  # index of arb angle measurement command
                self._instr.append(instr_)
                self._instr.append(
                    instruction._ZC(
                        target=instr.target,
                        domain=seq[4].domain,
                    )
                )
                self._instr.append(
                    instruction._ZC(
                        target=instr.control,
                        domain=seq[5].domain,
                    )
                )
            else:
                raise ValueError("Unknown instruction, commands not added")

        # move xc, zc to the end of the self._instr, so they will be applied last
        self._move_byproduct_to_right()

        # create command sequence
        command_seq = [*self._n, *reversed(self._e), *self._m]
        bpx_added = dict()
        bpz_added = dict()
        # byproduct command buffer
        z_cmds: list[command.Z] = []
        x_cmds: list[command.X] = []
        for i in range(len(self._instr)):
            instr = self._instr[i]
            if instr.kind == instruction.InstructionKind._XC:
                if instr.target in bpx_added.keys():
                    x_cmds[bpx_added[instr.target]].domain ^= instr.domain
                else:
                    bpx_added[instr.target] = len(x_cmds)
                    x_cmds.append(X(node=out[instr.target], domain=deepcopy(instr.domain)))
            elif instr.kind == instruction.InstructionKind._ZC:
                if instr.target in bpz_added.keys():
                    z_cmds[bpz_added[instr.target]].domain ^= instr.domain
                else:
                    bpz_added[instr.target] = len(z_cmds)
                    z_cmds.append(Z(node=out[instr.target], domain=deepcopy(instr.domain)))
        # append z commands first (X and Z commute up to global phase)
        command_seq.extend(z_cmds)
        command_seq.extend(x_cmds)
        pattern = Pattern(input_nodes=inputs)
        pattern.extend(command_seq)
        out = filter(lambda node: node is not None, out)
        pattern.reorder_output_nodes(out)
        return TranspileResult(pattern, classical_outputs)

    def _commute_with_swap(self, target: int):
        correction_instr = self._instr[target]
        swap_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind._XC
            or correction_instr.kind == instruction.InstructionKind._ZC
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
            correction_instr.kind == instruction.InstructionKind._XC
            or correction_instr.kind == instruction.InstructionKind._ZC
        )
        assert cnot_instr.kind == instruction.InstructionKind.CNOT
        if (
            correction_instr.kind == instruction.InstructionKind._XC and correction_instr.target == cnot_instr.control
        ):  # control
            new_cmd = instruction._XC(
                target=cnot_instr.target,
                domain=correction_instr.domain,
            )
            self._commute_with_following(target)
            self._instr.insert(target + 1, new_cmd)
            return target + 1
        elif (
            correction_instr.kind == instruction.InstructionKind._ZC and correction_instr.target == cnot_instr.target
        ):  # target
            new_cmd = instruction._ZC(
                target=cnot_instr.control,
                domain=correction_instr.domain,
            )
            self._commute_with_following(target)
            self._instr.insert(target + 1, new_cmd)
            return target + 1
        else:
            self._commute_with_following(target)
        return target

    def _commute_with_h(self, target: int):
        correction_instr = self._instr[target]
        h_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind._XC
            or correction_instr.kind == instruction.InstructionKind._ZC
        )
        assert h_instr.kind == instruction.InstructionKind.H
        if correction_instr.target == h_instr.target:
            if correction_instr.kind == instruction.InstructionKind._XC:
                self._instr[target] = instruction._ZC(
                    target=correction_instr.target, domain=correction_instr.domain
                )  # byproduct changes to Z
                self._commute_with_following(target)
            else:
                self._instr[target] = instruction._XC(
                    target=correction_instr.target, domain=correction_instr.domain
                )  # byproduct changes to X
                self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_s(self, target: int):
        correction_instr = self._instr[target]
        s_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind._XC
            or correction_instr.kind == instruction.InstructionKind._ZC
        )
        assert s_instr.kind == instruction.InstructionKind.S
        if correction_instr.target == s_instr.target:
            if correction_instr.kind == instruction.InstructionKind._XC:
                self._commute_with_following(target)
                # changes to Y = XZ
                self._instr.insert(
                    target + 1,
                    instruction._ZC(
                        target=correction_instr.target,
                        domain=correction_instr.domain,
                    ),
                )
                return target + 1
        self._commute_with_following(target)
        return target

    def _commute_with_rx(self, target: int):
        correction_instr = self._instr[target]
        rx_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind._XC
            or correction_instr.kind == instruction.InstructionKind._ZC
        )
        assert rx_instr.kind == instruction.InstructionKind.RX
        if correction_instr.target == rx_instr.target:
            if correction_instr.kind == instruction.InstructionKind._ZC:
                # add to the s-domain
                _extend_domain(self._m[rx_instr.meas_index], correction_instr.domain)
                self._commute_with_following(target)
            else:
                self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_ry(self, target: int):
        correction_instr = self._instr[target]
        ry_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind._XC
            or correction_instr.kind == instruction.InstructionKind._ZC
        )
        assert ry_instr.kind == instruction.InstructionKind.RY
        if correction_instr.target == ry_instr.target:
            # add to the s-domain
            _extend_domain(self._m[ry_instr.meas_index], correction_instr.domain)
            self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_rz(self, target: int):
        correction_instr = self._instr[target]
        rz_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind._XC
            or correction_instr.kind == instruction.InstructionKind._ZC
        )
        assert rz_instr.kind == instruction.InstructionKind.RZ
        if correction_instr.target == rz_instr.target:
            if correction_instr.kind == instruction.InstructionKind._XC:
                # add to the s-domain
                _extend_domain(self._m[rz_instr.meas_index], correction_instr.domain)
                self._commute_with_following(target)
            else:
                self._commute_with_following(target)
        else:
            self._commute_with_following(target)

    def _commute_with_rzz(self, target: int):
        correction_instr = self._instr[target]
        rzz_instr = self._instr[target + 1]
        assert (
            correction_instr.kind == instruction.InstructionKind._XC
            or correction_instr.kind == instruction.InstructionKind._ZC
        )
        assert rzz_instr.kind == instruction.InstructionKind.RZZ
        if correction_instr.kind == instruction.InstructionKind._XC:
            cond = correction_instr.target == rzz_instr.control
            cond2 = correction_instr.target == rzz_instr.target
            if cond or cond2:
                # add to the s-domain
                _extend_domain(self._m[rzz_instr.meas_index], correction_instr.domain)
        self._commute_with_following(target)

    def _commute_with_following(self, target: int):
        """Perform the commutation of two consecutive commands that commutes.

        Commutes the target command with the following command.

        Parameters
        ----------
        target : int
            target command index
        """
        a = self._instr[target + 1]
        self._instr.pop(target + 1)
        self._instr.insert(target, a)

    def _find_byproduct_to_move(self, rev: bool = False, skipnum: int = 0):
        """Find command to move.

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
                self._instr[target].kind == instruction.InstructionKind._ZC
                or self._instr[target].kind == instruction.InstructionKind._XC
            ):
                num_ops += 1
            if num_ops == skipnum + 1:
                return target
            ite += 1
            target += step
        target = "end"
        return target

    def _move_byproduct_to_right(self):
        """Move the byproduct 'gate' to the end of sequence, using the commutation relations."""
        moved = 0  # number of moved op
        target = self._find_byproduct_to_move(rev=True, skipnum=moved)
        while target != "end":
            if (target == len(self._instr) - 1) or (
                self._instr[target + 1].kind == instruction.InstructionKind._XC
                or self._instr[target + 1].kind == instruction.InstructionKind._ZC
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
                self._commute_with_h(target)
            elif kind == instruction.InstructionKind.S:
                target = self._commute_with_s(target)
            elif kind == instruction.InstructionKind.RX:
                self._commute_with_rx(target)
            elif kind == instruction.InstructionKind.RY:
                self._commute_with_ry(target)
            elif kind == instruction.InstructionKind.RZ:
                self._commute_with_rz(target)
            elif kind == instruction.InstructionKind.RZZ:
                self._commute_with_rzz(target)
            else:
                # Pauli gates commute up to global phase.
                self._commute_with_following(target)
            target += 1

    @classmethod
    def _cnot_command(
        cls, control_node: int, target_node: int, ancilla: Sequence[int]
    ) -> tuple[int, int, list[command.Command]]:
        """MBQC commands for CNOT gate.

        Parameters
        ----------
        control_node : int
            control node on graph
        target : int
            target node on graph
        ancilla : list of two ints
            ancilla node indices to be added to graph

        Returns
        -------
        control_out : int
            control node on graph after the gate
        target_out : int
            target node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(target_node, ancilla[0])),
                E(nodes=(control_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=target_node),
                M(node=ancilla[0]),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={target_node}),
                Z(node=control_node, domain={target_node}),
            )
        )
        return control_node, ancilla[1], seq

    @classmethod
    def _m_command(cls, input_node: int, plane: Plane, angle: float):
        """MBQC commands for measuring qubit.

        Parameters
        ----------
        input_node : int
            target node on graph
        plane : Plane
            plane of the measure
        angle : float
            angle of the measure (unit: pi radian)

        Returns
        -------
        commands : list
            list of MBQC commands
        """
        seq = [M(node=input_node, plane=plane, angle=angle)]
        return seq

    @classmethod
    def _h_command(cls, input_node: int, ancilla: int):
        """MBQC commands for Hadamard gate.

        Parameters
        ----------
        input_node : int
            target node on graph
        ancilla : int
            ancilla node index to be added

        Returns
        -------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        seq = [N(node=ancilla)]
        seq.extend((E(nodes=(input_node, ancilla)), M(node=input_node), X(node=ancilla, domain={input_node})))
        return ancilla, seq

    @classmethod
    def _s_command(cls, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
        """MBQC commands for S gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph

        Returns
        -------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [N(node=ancilla[0]), command.N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node, angle=-0.5),
                M(node=ancilla[0]),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _x_command(cls, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
        """MBQC commands for Pauli X gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph

        Returns
        -------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node),
                M(node=ancilla[0], angle=-1),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _y_command(cls, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
        """MBQC commands for Pauli Y gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of four ints
            ancilla node indices to be added to graph

        Returns
        -------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 4
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend([N(node=ancilla[2]), N(node=ancilla[3])])
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                E(nodes=(ancilla[1], ancilla[2])),
                E(nodes=(ancilla[2], ancilla[3])),
                M(node=input_node, angle=0.5),
                M(node=ancilla[0], angle=1.0, s_domain={input_node}),
                M(node=ancilla[1], angle=-0.5, s_domain={input_node}),
                M(node=ancilla[2]),
                X(node=ancilla[3], domain={ancilla[0], ancilla[2]}),
                Z(node=ancilla[3], domain={ancilla[0], ancilla[1]}),
            )
        )
        return ancilla[3], seq

    @classmethod
    def _z_command(cls, input_node: int, ancilla: Sequence[int]) -> tuple[int, list[command.Command]]:
        """MBQC commands for Pauli Z gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph

        Returns
        -------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node, angle=-1),
                M(node=ancilla[0]),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _rx_command(cls, input_node: int, ancilla: Sequence[int], angle: float) -> tuple[int, list[command.Command]]:
        """MBQC commands for X rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph
        angle : float
            measurement angle in radian

        Returns
        -------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node),
                M(node=ancilla[0], angle=-angle / np.pi, s_domain={input_node}),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _ry_command(cls, input_node: int, ancilla: Sequence[int], angle: float) -> tuple[int, list[command.Command]]:
        """MBQC commands for Y rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of four ints
            ancilla node indices to be added to graph
        angle : float
            rotation angle in radian

        Returns
        -------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 4
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend([N(node=ancilla[2]), N(node=ancilla[3])])
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                E(nodes=(ancilla[1], ancilla[2])),
                E(nodes=(ancilla[2], ancilla[3])),
                M(node=input_node, angle=0.5),
                M(node=ancilla[0], angle=-angle / np.pi, s_domain={input_node}),
                M(node=ancilla[1], angle=-0.5, s_domain={input_node}),
                M(node=ancilla[2]),
                X(node=ancilla[3], domain={ancilla[0], ancilla[2]}),
                Z(node=ancilla[3], domain={ancilla[0], ancilla[1]}),
            )
        )
        return ancilla[3], seq

    @classmethod
    def _rz_command(cls, input_node: int, ancilla: Sequence[int], angle: float) -> tuple[int, list[command.Command]]:
        """MBQC commands for Z rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph
        angle : float
            measurement angle in radian

        Returns
        -------
        out_node : int
            node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]  # assign new qubit labels
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node, angle=-angle / np.pi),
                M(node=ancilla[0]),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _rz_command_opt(cls, input_node: int, ancilla: int, angle: float) -> tuple[int, list[command.Command]]:
        """Optimized MBQC commands for Z rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : int
            ancilla node index to be added to graph
        angle : float
            measurement angle in radian

        Returns
        -------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        seq = [N(node=ancilla)]
        seq.extend(
            (
                E(nodes=(input_node, ancilla)),
                M(node=ancilla, angle=-angle / np.pi).clifford(Clifford.H),
                Z(node=input_node, domain={ancilla}),
            )
        )
        return input_node, seq

    @classmethod
    def _rzz_command_opt(
        cls, control_node: int, target_node: int, ancilla: int, angle: float
    ) -> tuple[int, int, list[command.Command]]:
        """Optimized MBQC commands for ZZ-rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : int
            ancilla node index
        angle : float
            measurement angle in radian

        Returns
        -------
        out_node_control : int
            control node on graph after the gate
        out_node_target : int
            target node on graph after the gate
        commands : list
            list of MBQC commands
        """
        seq = [N(node=ancilla)]
        seq.extend(
            (
                E(nodes=(control_node, ancilla)),
                E(nodes=(target_node, ancilla)),
                M(node=ancilla, angle=-angle / np.pi).clifford(Clifford.H),
                Z(node=control_node, domain={ancilla}),
                Z(node=target_node, domain={ancilla}),
            )
        )
        return control_node, target_node, seq

    @classmethod
    def _ccx_command(
        cls,
        control_node1: int,
        control_node2: int,
        target_node: int,
        ancilla: Sequence[int],
    ) -> tuple[int, int, int, list[command.Command]]:
        """MBQC commands for CCX gate.

        Parameters
        ----------
        control_node1 : int
            first control node on graph
        control_node2 : int
            second control node on graph
        target_node : int
            target node on graph
        ancilla : list of int
            ancilla node indices to be added to graph

        Returns
        -------
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
        seq.extend(
            (
                E(nodes=(target_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                E(nodes=(ancilla[1], ancilla[2])),
                E(nodes=(ancilla[1], control_node2)),
                E(nodes=(control_node1, ancilla[14])),
                E(nodes=(ancilla[2], ancilla[3])),
                E(nodes=(ancilla[14], ancilla[4])),
                E(nodes=(ancilla[3], ancilla[5])),
                E(nodes=(ancilla[3], ancilla[4])),
                E(nodes=(ancilla[5], ancilla[6])),
                E(nodes=(control_node2, ancilla[6])),
                E(nodes=(control_node2, ancilla[9])),
                E(nodes=(ancilla[6], ancilla[7])),
                E(nodes=(ancilla[9], ancilla[4])),
                E(nodes=(ancilla[9], ancilla[10])),
                E(nodes=(ancilla[7], ancilla[8])),
                E(nodes=(ancilla[10], ancilla[11])),
                E(nodes=(ancilla[4], ancilla[8])),
                E(nodes=(ancilla[4], ancilla[11])),
                E(nodes=(ancilla[4], ancilla[16])),
                E(nodes=(ancilla[8], ancilla[12])),
                E(nodes=(ancilla[11], ancilla[15])),
                E(nodes=(ancilla[12], ancilla[13])),
                E(nodes=(ancilla[16], ancilla[17])),
                M(node=target_node),
                M(node=ancilla[0], s_domain={target_node}),
                M(node=ancilla[1], s_domain={ancilla[0]}),
                M(node=control_node1),
                M(node=ancilla[2], angle=-1.75, s_domain={ancilla[1], target_node}),
                M(node=ancilla[14], s_domain={control_node1}),
                M(node=ancilla[3], s_domain={ancilla[2], ancilla[0]}),
                M(node=ancilla[5], angle=-0.25, s_domain={ancilla[3], ancilla[1], ancilla[14], target_node}),
                M(node=control_node2, angle=-0.25),
                M(node=ancilla[6], s_domain={ancilla[5], ancilla[2], ancilla[0]}),
                M(node=ancilla[9], s_domain={control_node2, ancilla[5], ancilla[2]}),
                M(
                    node=ancilla[7],
                    angle=-1.75,
                    s_domain={ancilla[6], ancilla[3], ancilla[1], ancilla[14], target_node},
                ),
                M(node=ancilla[10], angle=-1.75, s_domain={ancilla[9], ancilla[14]}),
                M(node=ancilla[4], angle=-0.25, s_domain={ancilla[14]}),
                M(node=ancilla[8], s_domain={ancilla[7], ancilla[5], ancilla[2], ancilla[0]}),
                M(node=ancilla[11], s_domain={ancilla[10], control_node2, ancilla[5], ancilla[2]}),
                M(
                    node=ancilla[12],
                    angle=-0.25,
                    s_domain={ancilla[8], ancilla[6], ancilla[3], ancilla[1], target_node},
                ),
                M(
                    node=ancilla[16],
                    s_domain={
                        ancilla[4],
                        control_node1,
                        ancilla[2],
                        control_node2,
                        ancilla[7],
                        ancilla[10],
                        ancilla[2],
                        control_node2,
                        ancilla[5],
                    },
                ),
                X(node=ancilla[17], domain={ancilla[14], ancilla[16]}),
                X(node=ancilla[15], domain={ancilla[9], ancilla[11]}),
                X(node=ancilla[13], domain={ancilla[0], ancilla[2], ancilla[5], ancilla[7], ancilla[12]}),
                Z(node=ancilla[17], domain={ancilla[4], ancilla[5], ancilla[7], ancilla[10], control_node1}),
                Z(node=ancilla[15], domain={control_node2, ancilla[2], ancilla[5], ancilla[10]}),
                Z(node=ancilla[13], domain={ancilla[1], ancilla[3], ancilla[6], ancilla[8], target_node}),
            )
        )
        return ancilla[17], ancilla[15], ancilla[13], seq

    @classmethod
    def _ccx_command_opt(
        cls,
        control_node1: int,
        control_node2: int,
        target_node: int,
        ancilla: Sequence[int],
    ) -> tuple[int, int, int, list[command.Command]]:
        """Optimized MBQC commands for CCX gate.

        Parameters
        ----------
        control_node1 : int
            first control node on graph
        control_node2 : int
            second control node on graph
        target_node : int
            target node on graph
        ancilla : list of int
            ancilla node indices to be added to graph

        Returns
        -------
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
        seq.extend(
            (
                E(nodes=(control_node1, ancilla[8])),
                E(nodes=(control_node2, ancilla[4])),
                E(nodes=(control_node2, ancilla[5])),
                E(nodes=(control_node2, ancilla[2])),
                E(nodes=(control_node2, ancilla[0])),
                E(nodes=(target_node, ancilla[6])),
                E(nodes=(ancilla[0], ancilla[6])),
                E(nodes=(ancilla[1], ancilla[10])),
                E(nodes=(ancilla[2], ancilla[10])),
                E(nodes=(ancilla[2], ancilla[6])),
                E(nodes=(ancilla[3], ancilla[6])),
                E(nodes=(ancilla[3], ancilla[10])),
                E(nodes=(ancilla[4], ancilla[10])),
                E(nodes=(ancilla[5], ancilla[9])),
                E(nodes=(ancilla[6], ancilla[7])),
                E(nodes=(ancilla[8], ancilla[10])),
                M(node=target_node),
                M(node=control_node1),
                M(node=ancilla[0], angle=-1.75, s_domain={target_node}).clifford(Clifford.H),
                M(node=ancilla[8], s_domain={control_node1}),
                M(node=ancilla[2], angle=-0.25, s_domain={target_node, ancilla[8]}).clifford(Clifford.H),
                M(node=control_node2, angle=-0.25),
                M(node=ancilla[3], angle=-1.75, s_domain={ancilla[8], target_node}).clifford(Clifford.H),
                M(node=ancilla[4], angle=-1.75, s_domain={ancilla[8]}).clifford(Clifford.H),
                M(node=ancilla[1], angle=-0.25, s_domain={ancilla[8]}).clifford(Clifford.H),
                M(node=ancilla[5], s_domain={control_node2, ancilla[0], ancilla[2], ancilla[4]}),
                M(node=ancilla[6], angle=-0.25, s_domain={target_node}),
                X(node=ancilla[10], domain={ancilla[8]}),
                X(node=ancilla[9], domain={ancilla[5]}),
                X(node=ancilla[7], domain={ancilla[0], ancilla[2], ancilla[3], ancilla[6]}),
                Z(node=ancilla[10], domain={control_node1, ancilla[1], ancilla[2], ancilla[3], ancilla[4]}),
                Z(node=ancilla[9], domain={control_node2, ancilla[0], ancilla[2], ancilla[4]}),
                Z(node=ancilla[7], domain={target_node}),
            )
        )

        return ancilla[10], ancilla[9], ancilla[7], seq

    @classmethod
    def _sort_outputs(cls, pattern: Pattern, output_nodes: Sequence[int]):
        """Sort the node indices of ouput qubits.

        Parameters
        ----------
        pattern : :meth:`~graphix.pattern.Pattern`
            pattern object
        output_nodes : list of int
            output node indices

        Returns
        -------
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

    def simulate_statevector(self, input_state: Data | None = None) -> SimulateResult:
        """Run statevector simulation of the gate sequence.

        Parameters
        ----------
        input_state : :class:`graphix.sim.statevec.Statevec`

        Returns
        -------
        result : :class:`SimulateResult`
            output state of the statevector simulation and results of classical measures.
        """
        if input_state is None:
            state = Statevec(nqubit=self.width)
        else:
            state = Statevec(nqubit=self.width, data=input_state)

        classical_measures = []

        for i in range(len(self.instruction)):
            instr = self.instruction[i]
            kind = instr.kind
            if kind == instruction.InstructionKind.CNOT:
                state.cnot((instr.control, instr.target))
            elif kind == instruction.InstructionKind.SWAP:
                state.swap(instr.targets)
            elif kind == instruction.InstructionKind.I:
                pass
            elif kind == instruction.InstructionKind.S:
                state.evolve_single(Ops.S, instr.target)
            elif kind == instruction.InstructionKind.H:
                state.evolve_single(Ops.H, instr.target)
            elif kind == instruction.InstructionKind.X:
                state.evolve_single(Ops.X, instr.target)
            elif kind == instruction.InstructionKind.Y:
                state.evolve_single(Ops.Y, instr.target)
            elif kind == instruction.InstructionKind.Z:
                state.evolve_single(Ops.Z, instr.target)
            elif kind == instruction.InstructionKind.RX:
                state.evolve_single(Ops.rx(instr.angle), instr.target)
            elif kind == instruction.InstructionKind.RY:
                state.evolve_single(Ops.ry(instr.angle), instr.target)
            elif kind == instruction.InstructionKind.RZ:
                state.evolve_single(Ops.rz(instr.angle), instr.target)
            elif kind == instruction.InstructionKind.RZZ:
                state.evolve(Ops.rzz(instr.angle), [instr.control, instr.target])
            elif kind == instruction.InstructionKind.CCX:
                state.evolve(Ops.CCX, [instr.controls[0], instr.controls[1], instr.target])
            elif kind == instruction.InstructionKind.M:
                result = base_backend.perform_measure(instr.target, instr.plane, instr.angle * np.pi, state, np.random)
                classical_measures.append(result)
            else:
                raise ValueError(f"Unknown instruction: {instr}")

        return SimulateResult(state, classical_measures)


def _extend_domain(measure: M, domain: set[int]) -> None:
    if measure.plane == Plane.XY:
        measure.s_domain ^= domain
    else:
        measure.t_domain ^= domain
