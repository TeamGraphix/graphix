"""Gate-to-MBQC transpiler.

accepts desired gate operations and transpile into MBQC measurement patterns.

"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, SupportsFloat

import numpy as np
from typing_extensions import assert_never

from graphix import command, instruction, parameter
from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.command import E, M, N, X, Z
from graphix.fundamentals import Plane
from graphix.instruction import Instruction, InstructionKind
from graphix.ops import Ops
from graphix.parameter import ExpressionOrFloat, Parameter
from graphix.pattern import Pattern
from graphix.sim import Data, Statevec, base_backend

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from numpy.random import Generator

    from graphix.command import Command


@dataclasses.dataclass
class TranspileResult:
    """
    The result of a transpilation.

    pattern : :class:`graphix.pattern.Pattern` object
    classical_outputs : tuple[int,...], index of nodes measured with *M* gates
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


Angle = ExpressionOrFloat


def _check_target(out: Sequence[int | None], index: int) -> int:
    target = out[index]
    if target is None:
        msg = f"Qubit {index} has already been measured."
        raise ValueError(msg)
    return target


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

    instruction: list[Instruction]

    def __init__(self, width: int, instr: Iterable[Instruction] | None = None) -> None:
        """
        Construct a circuit.

        Parameters
        ----------
        width : int
            number of logical qubits for the gate network
        instr : list[instruction.Instruction] | None
            Optional. List of initial instructions.
        """
        self.width = width
        self.instruction = []
        self.active_qubits = set(range(width))
        if instr is not None:
            self.extend(instr)

    def add(self, instr: Instruction) -> None:
        """Add an instruction to the circuit."""
        if instr.kind == InstructionKind.CCX:
            self.ccx(instr.controls[0], instr.controls[1], instr.target)
        elif instr.kind == InstructionKind.RZZ:
            self.rzz(instr.control, instr.target, instr.angle)
        elif instr.kind == InstructionKind.CNOT:
            self.cnot(instr.control, instr.target)
        elif instr.kind == InstructionKind.SWAP:
            self.swap(instr.targets[0], instr.targets[1])
        elif instr.kind == InstructionKind.H:
            self.h(instr.target)
        elif instr.kind == InstructionKind.S:
            self.s(instr.target)
        elif instr.kind == InstructionKind.X:
            self.x(instr.target)
        elif instr.kind == InstructionKind.Y:
            self.y(instr.target)
        elif instr.kind == InstructionKind.Z:
            self.z(instr.target)
        elif instr.kind == InstructionKind.I:
            self.i(instr.target)
        elif instr.kind == InstructionKind.M:
            self.m(instr.target, instr.plane, instr.angle)
        elif instr.kind == InstructionKind.RX:
            self.rx(instr.target, instr.angle)
        elif instr.kind == InstructionKind.RY:
            self.ry(instr.target, instr.angle)
        elif instr.kind == InstructionKind.RZ:
            self.rz(instr.target, instr.angle)
        # Use of `==` here for mypy
        elif instr.kind == InstructionKind._XC or instr.kind == InstructionKind._ZC:  # noqa: PLR1714
            raise ValueError(f"Unsupported instruction: {instr}")
        else:
            assert_never(instr.kind)

    def extend(self, instrs: Iterable[Instruction]) -> None:
        """Add instructions to the circuit."""
        for instr in instrs:
            self.add(instr)

    def __repr__(self) -> str:
        """Return a representation of the Circuit."""
        return f"Circuit(width={self.width}, instr={self.instruction})"

    def cnot(self, control: int, target: int) -> None:
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

    def swap(self, qubit1: int, qubit2: int) -> None:
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

    def h(self, qubit: int) -> None:
        """Apply a Hadamard gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.H(target=qubit))

    def s(self, qubit: int) -> None:
        """Apply an S gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.S(target=qubit))

    def x(self, qubit: int) -> None:
        """Apply a Pauli X gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.X(target=qubit))

    def y(self, qubit: int) -> None:
        """Apply a Pauli Y gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.Y(target=qubit))

    def z(self, qubit: int) -> None:
        """Apply a Pauli Z gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.Z(target=qubit))

    def rx(self, qubit: int, angle: Angle) -> None:
        """Apply an X rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : Angle
            rotation angle in radian
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RX(target=qubit, angle=angle))

    def ry(self, qubit: int, angle: Angle) -> None:
        """Apply a Y rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : Angle
            angle in radian
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RY(target=qubit, angle=angle))

    def rz(self, qubit: int, angle: Angle) -> None:
        """Apply a Z rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : Angle
            rotation angle in radian
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RZ(target=qubit, angle=angle))

    def rzz(self, control: int, target: int, angle: Angle) -> None:
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
        angle : Angle
            rotation angle in radian
        """
        assert control in self.active_qubits
        assert target in self.active_qubits
        self.instruction.append(instruction.RZZ(control=control, target=target, angle=angle))

    def ccx(self, control1: int, control2: int, target: int) -> None:
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

    def i(self, qubit: int) -> None:
        """Apply an identity (teleportation) gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.I(target=qubit))

    def m(self, qubit: int, plane: Plane, angle: Angle) -> None:
        """Measure a quantum qubit.

        The measured qubit cannot be used afterwards.

        Parameters
        ----------
        qubit : int
            target qubit
        plane : Plane
        angle : Angle
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.M(target=qubit, plane=plane, angle=angle))
        self.active_qubits.remove(qubit)

    def transpile(self) -> TranspileResult:
        """Transpile the circuit to a pattern.

        Returns
        -------
        result : :class:`TranspileResult` object
        """
        n_node = self.width
        out: list[int | None] = list(range(self.width))
        pattern = Pattern(input_nodes=list(range(self.width)))
        classical_outputs = []
        for instr in _transpile_rzz(self.instruction):
            if instr.kind == instruction.InstructionKind.CNOT:
                ancilla = [n_node, n_node + 1]
                control = _check_target(out, instr.control)
                target = _check_target(out, instr.target)
                out[instr.control], out[instr.target], seq = self._cnot_command(control, target, ancilla)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.SWAP:
                target0 = _check_target(out, instr.targets[0])
                target1 = _check_target(out, instr.targets[1])
                out[instr.targets[0]], out[instr.targets[1]] = (
                    target1,
                    target0,
                )
            elif instr.kind == instruction.InstructionKind.I:
                pass
            elif instr.kind == instruction.InstructionKind.H:
                single_ancilla = n_node
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._h_command(target, single_ancilla)
                pattern.extend(seq)
                n_node += 1
            elif instr.kind == instruction.InstructionKind.S:
                ancilla = [n_node, n_node + 1]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._s_command(target, ancilla)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.X:
                ancilla = [n_node, n_node + 1]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._x_command(target, ancilla)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.Y:
                ancilla = [n_node, n_node + 1, n_node + 2, n_node + 3]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._y_command(target, ancilla)
                pattern.extend(seq)
                n_node += 4
            elif instr.kind == instruction.InstructionKind.Z:
                ancilla = [n_node, n_node + 1]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._z_command(target, ancilla)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.RX:
                ancilla = [n_node, n_node + 1]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._rx_command(target, ancilla, instr.angle)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.RY:
                ancilla = [n_node, n_node + 1, n_node + 2, n_node + 3]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._ry_command(target, ancilla, instr.angle)
                pattern.extend(seq)
                n_node += 4
            elif instr.kind == instruction.InstructionKind.RZ:
                ancilla = [n_node, n_node + 1]
                target = _check_target(out, instr.target)
                out[instr.target], seq = self._rz_command(target, ancilla, instr.angle)
                pattern.extend(seq)
                n_node += 2
            elif instr.kind == instruction.InstructionKind.CCX:
                ancilla = [n_node + i for i in range(18)]
                control0 = _check_target(out, instr.controls[0])
                control1 = _check_target(out, instr.controls[1])
                target = _check_target(out, instr.target)
                (
                    out[instr.controls[0]],
                    out[instr.controls[1]],
                    out[instr.target],
                    seq,
                ) = self._ccx_command(
                    control0,
                    control1,
                    target,
                    ancilla,
                )
                pattern.extend(seq)
                n_node += 18
            elif instr.kind == instruction.InstructionKind.M:
                target = _check_target(out, instr.target)
                seq = self._m_command(target, instr.plane, instr.angle)
                pattern.extend(seq)
                classical_outputs.append(target)
                out[instr.target] = None
            else:
                raise ValueError("Unknown instruction, commands not added")
        output_nodes = [node for node in out if node is not None]
        pattern.reorder_output_nodes(output_nodes)
        return TranspileResult(pattern, tuple(classical_outputs))

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
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(target_node, ancilla[0])),
                E(nodes=(control_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=target_node),
                M(node=ancilla[0], s_domain={target_node}),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={target_node}),
                Z(node=control_node, domain={target_node}),
            )
        )
        return control_node, ancilla[1], seq

    @classmethod
    def _m_command(cls, input_node: int, plane: Plane, angle: Angle) -> list[Command]:
        """MBQC commands for measuring qubit.

        Parameters
        ----------
        input_node : int
            target node on graph
        plane : Plane
            plane of the measure
        angle : Angle
            angle of the measure (unit: pi radian)

        Returns
        -------
        commands : list
            list of MBQC commands
        """
        return [M(node=input_node, plane=plane, angle=angle)]

    @classmethod
    def _h_command(cls, input_node: int, ancilla: int) -> tuple[int, list[Command]]:
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
        seq: list[Command] = [N(node=ancilla)]
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
        seq: list[Command] = [N(node=ancilla[0]), command.N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node, angle=-0.5),
                M(node=ancilla[0], s_domain={input_node}),
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
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node),
                M(node=ancilla[0], angle=-1, s_domain={input_node}),
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
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend([N(node=ancilla[2]), N(node=ancilla[3])])
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                E(nodes=(ancilla[1], ancilla[2])),
                E(nodes=(ancilla[2], ancilla[3])),
                M(node=input_node, angle=0.5),
                M(node=ancilla[0], angle=1.0, s_domain={input_node}),
                M(node=ancilla[1], angle=-0.5, s_domain={ancilla[0]}, t_domain={input_node}),
                M(node=ancilla[2], s_domain={ancilla[1]}, t_domain={ancilla[0]}),
                X(node=ancilla[3], domain={ancilla[2]}),
                Z(node=ancilla[3], domain={ancilla[1]}),
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
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node, angle=-1),
                M(node=ancilla[0], s_domain={input_node}),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _rx_command(cls, input_node: int, ancilla: Sequence[int], angle: Angle) -> tuple[int, list[command.Command]]:
        """MBQC commands for X rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph
        angle : Angle
            measurement angle in radian

        Returns
        -------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
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
    def _ry_command(cls, input_node: int, ancilla: Sequence[int], angle: Angle) -> tuple[int, list[command.Command]]:
        """MBQC commands for Y rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of four ints
            ancilla node indices to be added to graph
        angle : Angle
            rotation angle in radian

        Returns
        -------
        out_node : int
            control node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 4
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend([N(node=ancilla[2]), N(node=ancilla[3])])
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                E(nodes=(ancilla[1], ancilla[2])),
                E(nodes=(ancilla[2], ancilla[3])),
                M(node=input_node, angle=0.5),
                M(node=ancilla[0], angle=-angle / np.pi, s_domain={input_node}),
                M(node=ancilla[1], angle=-0.5, s_domain={ancilla[0]}, t_domain={input_node}),
                M(node=ancilla[2], s_domain={ancilla[1]}, t_domain={ancilla[0]}),
                X(node=ancilla[3], domain={ancilla[2]}),
                Z(node=ancilla[3], domain={ancilla[1]}),
            )
        )
        return ancilla[3], seq

    @classmethod
    def _rz_command(cls, input_node: int, ancilla: Sequence[int], angle: Angle) -> tuple[int, list[command.Command]]:
        """MBQC commands for Z rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph
        angle : Angle
            measurement angle in radian

        Returns
        -------
        out_node : int
            node on graph after the gate
        commands : list
            list of MBQC commands
        """
        assert len(ancilla) == 2
        seq: list[Command] = [N(node=ancilla[0]), N(node=ancilla[1])]  # assign new qubit labels
        seq.extend(
            (
                E(nodes=(input_node, ancilla[0])),
                E(nodes=(ancilla[0], ancilla[1])),
                M(node=input_node, angle=-angle / np.pi),
                M(node=ancilla[0], s_domain={input_node}),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

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
        seq: list[Command] = [N(node=ancilla[i]) for i in range(18)]  # assign new qubit labels
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
                M(node=ancilla[1], s_domain={ancilla[0]}, t_domain={target_node}),
                M(node=control_node1),
                M(node=ancilla[2], angle=-1.75, s_domain={ancilla[1]}, t_domain={ancilla[0]}),
                M(node=ancilla[14], s_domain={control_node1}),
                M(node=ancilla[3], s_domain={ancilla[2]}, t_domain={ancilla[1], ancilla[14]}),
                M(node=ancilla[5], angle=-0.25, s_domain={ancilla[3]}, t_domain={ancilla[2]}),
                M(node=control_node2, angle=-0.25, t_domain={ancilla[5], ancilla[0]}),
                M(node=ancilla[6], s_domain={ancilla[5]}, t_domain={ancilla[3]}),
                M(node=ancilla[9], s_domain={control_node2}, t_domain={ancilla[14]}),
                M(node=ancilla[7], angle=-1.75, s_domain={ancilla[6]}, t_domain={ancilla[5]}),
                M(node=ancilla[10], angle=-1.75, s_domain={ancilla[9]}, t_domain={control_node2}),
                M(
                    node=ancilla[4],
                    angle=-0.25,
                    s_domain={ancilla[14]},
                    t_domain={control_node1, control_node2, ancilla[2], ancilla[7], ancilla[10]},
                ),
                M(node=ancilla[8], s_domain={ancilla[7]}, t_domain={ancilla[14], ancilla[6]}),
                M(node=ancilla[11], s_domain={ancilla[10]}, t_domain={ancilla[9], ancilla[14]}),
                M(node=ancilla[12], angle=-0.25, s_domain={ancilla[8]}, t_domain={ancilla[7]}),
                M(
                    node=ancilla[16],
                    s_domain={ancilla[4]},
                    t_domain={ancilla[14]},
                ),
                X(node=ancilla[17], domain={ancilla[16]}),
                X(node=ancilla[15], domain={ancilla[11]}),
                X(node=ancilla[13], domain={ancilla[12]}),
                Z(node=ancilla[17], domain={ancilla[4]}),
                Z(node=ancilla[15], domain={ancilla[10]}),
                Z(node=ancilla[13], domain={ancilla[8]}),
            )
        )
        return ancilla[17], ancilla[15], ancilla[13], seq

    def simulate_statevector(
        self,
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
        rng: Generator | None = None,
    ) -> SimulateResult:
        """Run statevector simulation of the gate sequence.

        Parameters
        ----------
        input_state : Data
        branch_selector: :class:`graphix.branch_selector.BranchSelector`
            branch selector for measures (default: :class:`RandomBranchSelector`).
        rng: Generator, optional
            Random-number generator for measurements.
            This generator is used only in case of random branch selection
            (see :class:`RandomBranchSelector`).

        Returns
        -------
        result : :class:`SimulateResult`
            output state of the statevector simulation and results of classical measures.
        """
        symbolic = self.is_parameterized()
        if branch_selector is None:
            branch_selector = RandomBranchSelector()

        state = Statevec(nqubit=self.width) if input_state is None else Statevec(nqubit=self.width, data=input_state)

        classical_measures = []

        for i in range(len(self.instruction)):
            instr = self.instruction[i]
            if instr.kind == instruction.InstructionKind.CNOT:
                state.cnot((instr.control, instr.target))
            elif instr.kind == instruction.InstructionKind.SWAP:
                state.swap(instr.targets)
            elif instr.kind == instruction.InstructionKind.I:
                pass
            elif instr.kind == instruction.InstructionKind.S:
                state.evolve_single(Ops.S, instr.target)
            elif instr.kind == instruction.InstructionKind.H:
                state.evolve_single(Ops.H, instr.target)
            elif instr.kind == instruction.InstructionKind.X:
                state.evolve_single(Ops.X, instr.target)
            elif instr.kind == instruction.InstructionKind.Y:
                state.evolve_single(Ops.Y, instr.target)
            elif instr.kind == instruction.InstructionKind.Z:
                state.evolve_single(Ops.Z, instr.target)
            elif instr.kind == instruction.InstructionKind.RX:
                state.evolve_single(Ops.rx(instr.angle), instr.target)
            elif instr.kind == instruction.InstructionKind.RY:
                state.evolve_single(Ops.ry(instr.angle), instr.target)
            elif instr.kind == instruction.InstructionKind.RZ:
                state.evolve_single(Ops.rz(instr.angle), instr.target)
            elif instr.kind == instruction.InstructionKind.RZZ:
                state.evolve(Ops.rzz(instr.angle), [instr.control, instr.target])
            elif instr.kind == instruction.InstructionKind.CCX:
                state.evolve(Ops.CCX, [instr.controls[0], instr.controls[1], instr.target])
            elif instr.kind == instruction.InstructionKind.M:
                result = base_backend.perform_measure(
                    instr.target,
                    instr.target,
                    instr.plane,
                    instr.angle * np.pi,
                    state,
                    branch_selector,
                    rng=rng,
                    symbolic=symbolic,
                )
                classical_measures.append(result)
            else:
                raise ValueError(f"Unknown instruction: {instr}")
        return SimulateResult(state, tuple(classical_measures))

    def map_angle(self, f: Callable[[Angle], Angle]) -> Circuit:
        """Apply `f` to all angles that occur in the circuit."""
        result = Circuit(self.width)
        for instr in self.instruction:
            # Use == for mypy
            if (
                instr.kind == InstructionKind.RZZ  # noqa: PLR1714
                or instr.kind == InstructionKind.M
                or instr.kind == InstructionKind.RX
                or instr.kind == InstructionKind.RY
                or instr.kind == InstructionKind.RZ
            ):
                new_instr = dataclasses.replace(instr, angle=f(instr.angle))
                result.instruction.append(new_instr)
            else:
                result.instruction.append(instr)
        return result

    def is_parameterized(self) -> bool:
        """
        Return `True` if there is at least one measurement angle that is not just an instance of `SupportsFloat`.

        A parameterized circuit is a circuit where at least one
        measurement angle is an expression that is not a number,
        typically an instance of `sympy.Expr` (but we don't force to
        choose `sympy` here).

        """
        # Use of `==` here for mypy
        return any(
            not isinstance(instr.angle, SupportsFloat)
            for instr in self.instruction
            if instr.kind == InstructionKind.RZZ  # noqa: PLR1714
            or instr.kind == InstructionKind.M
            or instr.kind == InstructionKind.RX
            or instr.kind == InstructionKind.RY
            or instr.kind == InstructionKind.RZ
        )

    def subs(self, variable: Parameter, substitute: ExpressionOrFloat) -> Circuit:
        """Return a copy of the circuit where all occurrences of the given variable in measurement angles are substituted by the given value."""
        return self.map_angle(lambda angle: parameter.subs(angle, variable, substitute))

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrFloat]) -> Circuit:
        """Return a copy of the circuit where all occurrences of the given keys in measurement angles are substituted by the given values in parallel."""
        return self.map_angle(lambda angle: parameter.xreplace(angle, assignment))


def _extend_domain(measure: M, domain: set[int]) -> None:
    """Extend the correction domain of ``measure`` by ``domain``.

    Parameters
    ----------
    measure : M
        Measurement command to modify.
    domain : set[int]
        Set of nodes to XOR into the appropriate domain of ``measure``.
    """
    if measure.plane == Plane.XY:
        measure.s_domain ^= domain
    else:
        measure.t_domain ^= domain


def _transpile_rzz(instructions: Iterable[Instruction]) -> Iterator[Instruction]:
    for instr in instructions:
        if instr.kind == InstructionKind.RZZ:
            yield instruction.CNOT(control=instr.control, target=instr.target)
            yield instruction.RZ(target=instr.target, angle=instr.angle)
            yield instruction.CNOT(control=instr.control, target=instr.target)
        else:
            yield instr
