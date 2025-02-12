"""Gate-to-MBQC transpiler.

accepts desired gate operations and transpile into MBQC measurement patterns.

"""

from __future__ import annotations

import dataclasses
from copy import deepcopy
from typing import TYPE_CHECKING, Callable

import numpy as np

from graphix import command, instruction, parameter
from graphix.command import CommandKind, E, M, N, X, Z
from graphix.fundamentals import Plane
from graphix.ops import Ops
from graphix.parameter import ExpressionOrSupportsFloat, Parameter
from graphix.pattern import Pattern
from graphix.sim import base_backend
from graphix.sim.statevec import Data, Statevec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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


Angle = ExpressionOrSupportsFloat


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

    def rx(self, qubit: int, angle: Angle):
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

    def ry(self, qubit: int, angle: Angle):
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

    def rz(self, qubit: int, angle: Angle):
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

    def rzz(self, control: int, target: int, angle: Angle):
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

    def m(self, qubit: int, plane: Plane, angle: Angle):
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
        out = list(range(self.width))
        pattern = Pattern(input_nodes=list(range(self.width)))
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
                ancilla = [n_node, n_node + 1]
                out[instr.target], seq = self._rz_command(out[instr.target], ancilla, instr.angle)
                pattern.extend(seq)
                n_node += 2
            elif kind == instruction.InstructionKind.CCX:
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
        seq.append(E(nodes=(target_node, ancilla[0])))
        seq.append(E(nodes=(control_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=target_node))
        seq.append(M(node=ancilla[0]))
        seq.append(X(node=ancilla[1], domain={ancilla[0]}))
        seq.append(Z(node=ancilla[1], domain={target_node}))
        seq.append(Z(node=control_node, domain={target_node}))
        return control_node, ancilla[1], seq

    @classmethod
    def _m_command(cls, input_node: int, plane: Plane, angle: Angle):
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
        seq.append(E(nodes=(input_node, ancilla)))
        seq.append(M(node=input_node))
        seq.append(X(node=ancilla, domain={input_node}))
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
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=input_node, angle=-0.5))
        seq.append(M(node=ancilla[0]))
        seq.append(X(node=ancilla[1], domain={ancilla[0]}))
        seq.append(Z(node=ancilla[1], domain={input_node}))
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
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=input_node))
        seq.append(M(node=ancilla[0], angle=-1))
        seq.append(X(node=ancilla[1], domain={ancilla[0]}))
        seq.append(Z(node=ancilla[1], domain={input_node}))
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
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(E(nodes=(ancilla[1], ancilla[2])))
        seq.append(E(nodes=(ancilla[2], ancilla[3])))
        seq.append(M(node=input_node, angle=0.5))
        seq.append(M(node=ancilla[0], angle=1.0, s_domain={input_node}))
        seq.append(M(node=ancilla[1], angle=-0.5, s_domain={input_node}))
        seq.append(M(node=ancilla[2]))
        seq.append(X(node=ancilla[3], domain={ancilla[0], ancilla[2]}))
        seq.append(Z(node=ancilla[3], domain={ancilla[0], ancilla[1]}))
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
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=input_node, angle=-1))
        seq.append(M(node=ancilla[0]))
        seq.append(X(node=ancilla[1], domain={ancilla[0]}))
        seq.append(Z(node=ancilla[1], domain={input_node}))
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
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=input_node))
        seq.append(M(node=ancilla[0], angle=-angle / np.pi, s_domain={input_node}))
        seq.append(X(node=ancilla[1], domain={ancilla[0]}))
        seq.append(Z(node=ancilla[1], domain={input_node}))
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
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]
        seq.extend([N(node=ancilla[2]), N(node=ancilla[3])])
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(E(nodes=(ancilla[1], ancilla[2])))
        seq.append(E(nodes=(ancilla[2], ancilla[3])))
        seq.append(M(node=input_node, angle=0.5))
        seq.append(M(node=ancilla[0], angle=-angle / np.pi, s_domain={input_node}))
        seq.append(M(node=ancilla[1], angle=-0.5, s_domain={input_node}))
        seq.append(M(node=ancilla[2]))
        seq.append(X(node=ancilla[3], domain={ancilla[0], ancilla[2]}))
        seq.append(Z(node=ancilla[3], domain={ancilla[0], ancilla[1]}))
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
        seq = [N(node=ancilla[0]), N(node=ancilla[1])]  # assign new qubit labels
        seq.append(E(nodes=(input_node, ancilla[0])))
        seq.append(E(nodes=(ancilla[0], ancilla[1])))
        seq.append(M(node=input_node, angle=-angle / np.pi))
        seq.append(M(node=ancilla[0]))
        seq.append(X(node=ancilla[1], domain={ancilla[0]}))
        seq.append(Z(node=ancilla[1], domain={input_node}))
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
        seq.append(M(node=ancilla[0], s_domain={target_node}))
        seq.append(M(node=ancilla[1], s_domain={ancilla[0]}))
        seq.append(M(node=control_node1))
        seq.append(M(node=ancilla[2], angle=-1.75, s_domain={ancilla[1], target_node}))
        seq.append(M(node=ancilla[14], s_domain={control_node1}))
        seq.append(M(node=ancilla[3], s_domain={ancilla[2], ancilla[0]}))
        seq.append(
            M(
                node=ancilla[5],
                angle=-0.25,
                s_domain={ancilla[3], ancilla[1], ancilla[14], target_node},
            )
        )
        seq.append(M(node=control_node2, angle=-0.25))
        seq.append(M(node=ancilla[6], s_domain={ancilla[5], ancilla[2], ancilla[0]}))
        seq.append(
            M(
                node=ancilla[9],
                s_domain={control_node2, ancilla[5], ancilla[2]},
            )
        )
        seq.append(
            M(
                node=ancilla[7],
                angle=-1.75,
                s_domain={ancilla[6], ancilla[3], ancilla[1], ancilla[14], target_node},
            )
        )
        seq.append(M(node=ancilla[10], angle=-1.75, s_domain={ancilla[9], ancilla[14]}))
        seq.append(M(node=ancilla[4], angle=-0.25, s_domain={ancilla[14]}))
        seq.append(
            M(
                node=ancilla[8],
                s_domain={ancilla[7], ancilla[5], ancilla[2], ancilla[0]},
            )
        )
        seq.append(
            M(
                node=ancilla[11],
                s_domain={ancilla[10], control_node2, ancilla[5], ancilla[2]},
            )
        )
        seq.append(
            M(
                node=ancilla[12],
                angle=-0.25,
                s_domain={
                    ancilla[8],
                    ancilla[6],
                    ancilla[3],
                    ancilla[1],
                    target_node,
                },
            )
        )
        seq.append(
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
            )
        )
        seq.append(X(node=ancilla[17], domain={ancilla[14], ancilla[16]}))
        seq.append(X(node=ancilla[15], domain={ancilla[9], ancilla[11]}))
        seq.append(
            X(
                node=ancilla[13],
                domain={ancilla[0], ancilla[2], ancilla[5], ancilla[7], ancilla[12]},
            )
        )
        seq.append(
            Z(
                node=ancilla[17],
                domain={ancilla[4], ancilla[5], ancilla[7], ancilla[10], control_node1},
            )
        )
        seq.append(
            Z(
                node=ancilla[15],
                domain={control_node2, ancilla[2], ancilla[5], ancilla[10]},
            )
        )
        seq.append(
            Z(
                node=ancilla[13],
                domain={ancilla[1], ancilla[3], ancilla[6], ancilla[8], target_node},
            )
        )
        return ancilla[17], ancilla[15], ancilla[13], seq

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
        state = Statevec(nqubit=self.width) if input_state is None else Statevec(nqubit=self.width, data=input_state)

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

    def map_angle(self, f: Callable[[Angle], Angle]) -> Circuit:
        """Apply `f` to all angles that occur in the circuit."""
        result = Circuit(self.width)
        for instr in self.instruction:
            angle = getattr(instr, "angle", None)
            if angle is None:
                result.instruction.append(instr)
            else:
                new_instr = dataclasses.replace(instr, angle=f(angle))
                result.instruction.append(new_instr)
        return result

    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Circuit:
        """Return a copy of the circuit where all occurrences of the given variable in measurement angles are substituted by the given value."""
        return self.map_angle(lambda angle: parameter.subs(angle, variable, substitute))

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> Circuit:
        """Return a copy of the circuit where all occurrences of the given keys in measurement angles are substituted by the given values in parallel."""
        return self.map_angle(lambda angle: parameter.xreplace(angle, assignment))


def _extend_domain(measure: M, domain: set[int]) -> None:
    if measure.plane == Plane.XY:
        measure.s_domain ^= domain
    else:
        measure.t_domain ^= domain
