"""Gate-to-MBQC transpiler.

accepts desired gate operations and transpile into MBQC measurement patterns.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsFloat

import networkx as nx

# assert_never introduced in Python 3.11
# override introduced in Python 3.12
from typing_extensions import assert_never, override

from graphix import command, instruction, parameter
from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.command import E, M, N, X, Z
from graphix.flow.core import CausalFlow, _corrections_to_partial_order_layers
from graphix.fundamentals import ANGLE_PI, Axis
from graphix.instruction import Instruction, InstructionKind, InstructionVisitor
from graphix.measurements import BlochMeasurement, Measurement, PauliMeasurement
from graphix.opengraph import OpenGraph
from graphix.ops import Ops
from graphix.optimization import StandardizedPattern
from graphix.sim.statevec import Statevec, StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from numpy.random import Generator

    from graphix.command import Command
    from graphix.fundamentals import ParameterizedAngle
    from graphix.parameter import ExpressionOrFloat, Parameter
    from graphix.pattern import Pattern
    from graphix.sim import Data
    from graphix.sim.base_backend import Matrix


@dataclass
class TranspileResult:
    """
    The result of a transpilation.

    pattern : :class:`graphix.pattern.Pattern` object
    classical_outputs : tuple[int,...], index of nodes measured with *M* gates
    """

    pattern: Pattern
    classical_outputs: tuple[int, ...]


@dataclass
class SimulateResult:
    """
    The result of a simulation.

    statevec : :class:`graphix.sim.statevec.Statevec` object
    classical_measures : tuple[int,...], classical measures
    """

    statevec: Statevec
    classical_measures: tuple[int, ...]


def _check_target(out: Sequence[int | None], index: int) -> int:
    target = out[index]
    if target is None:
        msg = f"Qubit {index} has already been measured."
        raise ValueError(msg)
    return target


@dataclass
class _MapAngleVisitor(InstructionVisitor):
    f: Callable[[ParameterizedAngle], ParameterizedAngle]

    @override
    def visit_angle(self, angle: ParameterizedAngle) -> ParameterizedAngle:
        return self.f(angle)


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
        match instr.kind:
            case InstructionKind.CCX:
                self.ccx(instr.controls[0], instr.controls[1], instr.target)
            case InstructionKind.RZZ:
                self.rzz(instr.control, instr.target, instr.angle)
            case InstructionKind.CNOT:
                self.cnot(instr.control, instr.target)
            case InstructionKind.SWAP:
                self.swap(instr.targets[0], instr.targets[1])
            case InstructionKind.CZ:
                self.cz(instr.targets[0], instr.targets[1])
            case InstructionKind.H:
                self.h(instr.target)
            case InstructionKind.S:
                self.s(instr.target)
            case InstructionKind.X:
                self.x(instr.target)
            case InstructionKind.Y:
                self.y(instr.target)
            case InstructionKind.Z:
                self.z(instr.target)
            case InstructionKind.I:
                self.i(instr.target)
            case InstructionKind.M:
                self.m(instr.target, instr.axis)
            case InstructionKind.RX:
                self.rx(instr.target, instr.angle)
            case InstructionKind.RY:
                self.ry(instr.target, instr.angle)
            case InstructionKind.RZ:
                self.rz(instr.target, instr.angle)
            case InstructionKind.J:
                self.j(instr.target, instr.angle)
            case _:
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

    def cz(self, qubit1: int, qubit2: int) -> None:
        """Apply a CNOT gate.

        Parameters
        ----------
        qubit1 : int
            control qubit
        qubit2 : int
            target qubit
        """
        assert qubit1 in self.active_qubits
        assert qubit2 in self.active_qubits
        assert qubit1 != qubit2
        self.instruction.append(instruction.CZ(targets=(qubit1, qubit2)))

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

    def rx(self, qubit: int, angle: ParameterizedAngle) -> None:
        """Apply an X rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : ParameterizedAngle
            rotation angle in units of π
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RX(target=qubit, angle=angle))

    def ry(self, qubit: int, angle: ParameterizedAngle) -> None:
        """Apply a Y rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : ParameterizedAngle
            angle in units of π
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RY(target=qubit, angle=angle))

    def rz(self, qubit: int, angle: ParameterizedAngle) -> None:
        """Apply a Z rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : ParameterizedAngle
            rotation angle in units of π
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.RZ(target=qubit, angle=angle))

    def j(self, qubit: int, angle: ParameterizedAngle) -> None:
        """Apply a J rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : ParameterizedAngle
            rotation angle in units of π
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.J(target=qubit, angle=angle))

    def r(self, qubit: int, axis: Axis, angle: ParameterizedAngle) -> None:
        """Apply a rotation gate on the given axis.

        Parameters
        ----------
        qubit : int
            target qubit
        axis : Axis
            rotation axis
        angle : ParameterizedAngle
            rotation angle in units of π
        """
        match axis:
            case Axis.X:
                self.rx(qubit, angle)
            case Axis.Y:
                self.ry(qubit, angle)
            case Axis.Z:
                self.rz(qubit, angle)
            case _:
                assert_never(axis)

    def rzz(self, control: int, target: int, angle: ParameterizedAngle) -> None:
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
        angle : ParameterizedAngle
            rotation angle in units of π
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

    def m(self, qubit: int, axis: Axis) -> None:
        """Measure a quantum qubit.

        The measured qubit cannot be used afterwards.

        Parameters
        ----------
        qubit : int
            target qubit
        axis : Axis
            measurement basis
        """
        assert qubit in self.active_qubits
        self.instruction.append(instruction.M(target=qubit, axis=axis))
        self.active_qubits.remove(qubit)

    def transpile(self) -> TranspileResult:
        """Transpile a circuit via J-∧z decomposition to a pattern.

        Parameters
        ----------
            self: the circuit to transpile.

        Returns
        -------
            the result of the transpilation: a pattern.

        Raises
        ------
            IllformedCircuitError: if the pattern is ill-formed (operation on already measured node)
            CircuitWithMeasurementError: if the circuit contains measurements.

        """
        indices: list[int | None] = list(range(self.width))
        n_nodes = self.width
        measurements: dict[int, BlochMeasurement] = {}
        classical_outputs: dict[int, command.M] = {}
        inputs = list(range(n_nodes))
        graph: nx.Graph[int] = nx.Graph()  # CHANGE TO OPEN GRAPH NOT NX
        graph.add_nodes_from(inputs)
        x_corrections: dict[int, set[int]] = {}
        for instr in self.instruction:
            if instr.kind == InstructionKind.M:
                target = indices[instr.target]
                if target is None:
                    raise IllformedCircuitError
                classical_outputs[target] = command.M(target, PauliMeasurement(instr.axis))
                indices[instr.target] = None
                continue
            for instr_jcz in instruction_to_jcz(instr):
                if instr_jcz.kind == InstructionKind.J:
                    target = indices[instr_jcz.target]
                    if target is None:
                        raise IllformedCircuitError
                    graph.add_edge(target, n_nodes)  # Also adds nodes
                    measurements[target] = Measurement.XY(normalize_angle(-instr_jcz.angle))
                    indices[instr_jcz.target] = n_nodes
                    x_corrections[target] = {n_nodes}  # X correction on ancilla
                    n_nodes += 1
                    continue
                if instr_jcz.kind == InstructionKind.CZ:
                    t0, t1 = instr_jcz.targets
                    i0, i1 = indices[t0], indices[t1]
                    if i0 is None or i1 is None:
                        raise IllformedCircuitError
                    # If edge exists, remove it; else, add it
                    if graph.has_edge(i0, i1):
                        graph.remove_edge(i0, i1)
                    else:
                        graph.add_edge(i0, i1)
                    continue
                assert_never(instr_jcz.kind)
        outputs = [i for i in indices if i is not None]
        outputs.extend(classical_outputs.keys())
        og = OpenGraph(
            graph=graph,
            input_nodes=inputs,
            output_nodes=outputs,
            measurements=measurements,
        )
        z_corrections: dict[int, set[int]] = {}
        for node, correctors in x_corrections.items():
            (corrector,) = correctors
            z_targets = set(graph.neighbors(corrector)) - {node}
            if z_targets:
                z_corrections[node] = z_targets
        partial_order_layers = _corrections_to_partial_order_layers(og, x_corrections, z_corrections)
        f: CausalFlow[BlochMeasurement] = CausalFlow(og, x_corrections, partial_order_layers)
        pattern = StandardizedPattern.from_pattern(f.to_corrections().to_pattern()).to_space_optimal_pattern()
        pattern.extend(classical_outputs.values())
        return TranspileResult(pattern, tuple(classical_outputs.keys()))

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
    def _cz_command(cls, target_1: int, target_2: int) -> list[Command]:
        """MBQC commands for CZ gate.

        Parameters
        ----------
        target_1 : int
            target node on graph
        target_2 : int
            other target node on graph

        Returns
        -------
        commands : list
            list of MBQC commands
        """
        return [E(nodes=(target_1, target_2))]

    @classmethod
    def _m_command(cls, input_node: int, axis: Axis) -> list[Command]:
        """MBQC commands for measuring qubit.

        Parameters
        ----------
        input_node : int
            target node on graph
        axis : Axis
            measurement basis

        Returns
        -------
        commands : list
            list of MBQC commands
        """
        # `measurement.angle` and `M.angle` are both expressed in units of π.
        return [M(input_node, PauliMeasurement(axis))]

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
                M(input_node, -Measurement.Y),
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
                M(ancilla[0], -Measurement.X, s_domain={input_node}),
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
                M(input_node, Measurement.Y),
                M(ancilla[0], -Measurement.X, s_domain={input_node}),
                M(ancilla[1], -Measurement.Y, s_domain={ancilla[0]}, t_domain={input_node}),
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
                M(input_node, -Measurement.X),
                M(node=ancilla[0], s_domain={input_node}),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _rx_command(
        cls, input_node: int, ancilla: Sequence[int], angle: ParameterizedAngle
    ) -> tuple[int, list[command.Command]]:
        """MBQC commands for X rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph
        angle : ParameterizedAngle
            measurement angle in units of π

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
                M(ancilla[0], Measurement.XY(-angle), s_domain={input_node}),
                X(node=ancilla[1], domain={ancilla[0]}),
                Z(node=ancilla[1], domain={input_node}),
            )
        )
        return ancilla[1], seq

    @classmethod
    def _ry_command(
        cls, input_node: int, ancilla: Sequence[int], angle: ParameterizedAngle
    ) -> tuple[int, list[command.Command]]:
        """MBQC commands for Y rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of four ints
            ancilla node indices to be added to graph
        angle : ParameterizedAngle
            rotation angle in units of π

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
                M(input_node, Measurement.Y),
                M(ancilla[0], Measurement.XY(-angle), s_domain={input_node}),
                M(ancilla[1], -Measurement.Y, s_domain={ancilla[0]}, t_domain={input_node}),
                M(node=ancilla[2], s_domain={ancilla[1]}, t_domain={ancilla[0]}),
                X(node=ancilla[3], domain={ancilla[2]}),
                Z(node=ancilla[3], domain={ancilla[1]}),
            )
        )
        return ancilla[3], seq

    @classmethod
    def _rz_command(
        cls, input_node: int, ancilla: Sequence[int], angle: ParameterizedAngle
    ) -> tuple[int, list[command.Command]]:
        """MBQC commands for Z rotation gate.

        Parameters
        ----------
        input_node : int
            input node index
        ancilla : list of two ints
            ancilla node indices to be added to graph
        angle : ParameterizedAngle
            measurement angle in units of π

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
                M(input_node, Measurement.XY(-angle)),
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
                M(ancilla[2], Measurement.XY(-7 * ANGLE_PI / 4), s_domain={ancilla[1]}, t_domain={ancilla[0]}),
                M(node=ancilla[14], s_domain={control_node1}),
                M(node=ancilla[3], s_domain={ancilla[2]}, t_domain={ancilla[1], ancilla[14]}),
                M(ancilla[5], Measurement.XY(-ANGLE_PI / 4), s_domain={ancilla[3]}, t_domain={ancilla[2]}),
                M(control_node2, Measurement.XY(-ANGLE_PI / 4), t_domain={ancilla[5], ancilla[0]}),
                M(node=ancilla[6], s_domain={ancilla[5]}, t_domain={ancilla[3]}),
                M(node=ancilla[9], s_domain={control_node2}, t_domain={ancilla[14]}),
                M(ancilla[7], Measurement.XY(-7 * ANGLE_PI / 4), s_domain={ancilla[6]}, t_domain={ancilla[5]}),
                M(ancilla[10], Measurement.XY(-7 * ANGLE_PI / 4), s_domain={ancilla[9]}, t_domain={control_node2}),
                M(
                    ancilla[4],
                    Measurement.XY(-ANGLE_PI / 4),
                    s_domain={ancilla[14]},
                    t_domain={control_node1, control_node2, ancilla[2], ancilla[7], ancilla[10]},
                ),
                M(node=ancilla[8], s_domain={ancilla[7]}, t_domain={ancilla[14], ancilla[6]}),
                M(node=ancilla[11], s_domain={ancilla[10]}, t_domain={ancilla[9], ancilla[14]}),
                M(ancilla[12], Measurement.XY(-ANGLE_PI / 4), s_domain={ancilla[8]}, t_domain={ancilla[7]}),
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
        *,
        stacklevel: int = 1,
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
        if branch_selector is None:
            branch_selector = RandomBranchSelector()

        backend = StatevectorBackend(branch_selector=branch_selector)
        if input_state is None:
            backend.add_nodes(range(self.width))
        else:
            backend.add_nodes(range(self.width), input_state)

        classical_measures = []

        for i in range(len(self.instruction)):
            instr = self.instruction[i]

            def evolve_single(op: Matrix, target: int) -> None:
                backend.state.evolve_single(op, backend.node_index.index(target))

            def evolve(op: Matrix, qargs: Iterable[int]) -> None:
                backend.state.evolve(op, [backend.node_index.index(qarg) for qarg in qargs])

            match instr.kind:
                case instruction.InstructionKind.CNOT:
                    backend.state.cnot(
                        (backend.node_index.index(instr.control), backend.node_index.index(instr.target))
                    )
                case instruction.InstructionKind.SWAP:
                    u, v = instr.targets
                    backend.state.swap((backend.node_index.index(u), backend.node_index.index(v)))
                case instruction.InstructionKind.CZ:
                    u, v = instr.targets
                    backend.state.entangle((backend.node_index.index(u), backend.node_index.index(v)))
                case instruction.InstructionKind.I:
                    pass
                case instruction.InstructionKind.S:
                    evolve_single(Ops.S, instr.target)
                case instruction.InstructionKind.H:
                    evolve_single(Ops.H, instr.target)
                case instruction.InstructionKind.X:
                    evolve_single(Ops.X, instr.target)
                case instruction.InstructionKind.Y:
                    evolve_single(Ops.Y, instr.target)
                case instruction.InstructionKind.Z:
                    evolve_single(Ops.Z, instr.target)
                case instruction.InstructionKind.RX:
                    evolve_single(Ops.rx(instr.angle), instr.target)
                case instruction.InstructionKind.RY:
                    evolve_single(Ops.ry(instr.angle), instr.target)
                case instruction.InstructionKind.RZ:
                    evolve_single(Ops.rz(instr.angle), instr.target)
                case instruction.InstructionKind.RZZ:
                    evolve(Ops.rzz(instr.angle), [instr.control, instr.target])
                case instruction.InstructionKind.CCX:
                    evolve(Ops.CCX, [instr.controls[0], instr.controls[1], instr.target])
                case instruction.InstructionKind.M:
                    result = backend.measure(
                        instr.target, PauliMeasurement(instr.axis), rng=rng, stacklevel=stacklevel + 1
                    )
                    classical_measures.append(result)
                case _:
                    raise ValueError(f"Unknown instruction: {instr}")
        return SimulateResult(backend.state, tuple(classical_measures))

    def visit(self, visitor: InstructionVisitor) -> Circuit:
        """Apply `visitor` to all instructions in the circuit."""
        result = Circuit(self.width)
        for instr in self.instruction:
            result.instruction.append(instr.visit(visitor))
        return result

    def map_angle(self, f: Callable[[ParameterizedAngle], ParameterizedAngle]) -> Circuit:
        """Apply `f` to all angles that occur in the circuit."""
        return self.visit(_MapAngleVisitor(f))

    def is_parameterized(self) -> bool:
        """
        Return `True` if there is at least one measurement angle that is not just an instance of `SupportsFloat`.

        A parameterized circuit is a circuit where at least one
        measurement angle is an expression that is not a number,
        typically an instance of `sympy.Expr` (but we don't force to
        choose `sympy` here).

        """
        for instr in self.instruction:
            match instr.kind:
                case InstructionKind.RZZ | InstructionKind.RX | InstructionKind.RY | InstructionKind.RZ:
                    if not isinstance(instr.angle, SupportsFloat):
                        return True
        return False

    def subs(self, variable: Parameter, substitute: ExpressionOrFloat) -> Circuit:
        """Return a copy of the circuit where all occurrences of the given variable in measurement angles are substituted by the given value."""
        return self.map_angle(lambda angle: parameter.subs(angle, variable, substitute))

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrFloat]) -> Circuit:
        """Return a copy of the circuit where all occurrences of the given keys in measurement angles are substituted by the given values in parallel."""
        return self.map_angle(lambda angle: parameter.xreplace(angle, assignment))

    def transpile_measurements_to_z_axis(self) -> Circuit:
        """Return an equivalent circuit where all measurements are on Z axis."""
        circuit = Circuit(width=self.width)
        for instr in self.instruction:
            if instr.kind == InstructionKind.M:
                match instr.axis:
                    case Axis.X:
                        circuit.h(instr.target)
                        circuit.m(instr.target, Axis.Z)
                    case Axis.Y:
                        circuit.rx(instr.target, ANGLE_PI / 2)
                        circuit.m(instr.target, Axis.Z)
                    case Axis.Z:
                        circuit.add(instr)
                    case _:
                        assert_never(instr.axis)
            else:
                circuit.add(instr)
        return circuit


def decompose_rzz(instr: instruction.RZZ) -> list[instruction.CNOT | instruction.RZ]:
    """Return a decomposition of RZZ(α) gate as CNOT(control, target)·Rz(target, α)·CNOT(control, target).

    Parameters
    ----------
        instr: the RZZ instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [
        instruction.CNOT(target=instr.target, control=instr.control),
        instruction.RZ(instr.target, instr.angle),
        instruction.CNOT(target=instr.target, control=instr.control),
    ]


def decompose_ccx(
    instr: instruction.CCX,
) -> list[instruction.H | instruction.CNOT | instruction.RZ]:
    """Return a decomposition of the CCX gate into H, CNOT, T and T-dagger gates.

    This decomposition of the Toffoli gate can be found in
    Michael A. Nielsen and Isaac L. Chuang,
    Quantum Computation and Quantum Information,
    Cambridge University Press, 2000
    (p. 182 in the 10th Anniversary Edition).

    Parameters
    ----------
        instr: the CCX instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [
        instruction.H(instr.target),
        instruction.CNOT(control=instr.controls[1], target=instr.target),
        instruction.RZ(instr.target, -ANGLE_PI / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.target),
        instruction.RZ(instr.target, ANGLE_PI / 4),
        instruction.CNOT(control=instr.controls[1], target=instr.target),
        instruction.RZ(instr.target, -ANGLE_PI / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.target),
        instruction.RZ(instr.controls[1], -ANGLE_PI / 4),
        instruction.RZ(instr.target, ANGLE_PI / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.controls[1]),
        instruction.H(instr.target),
        instruction.RZ(instr.controls[1], -ANGLE_PI / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.controls[1]),
        instruction.RZ(instr.controls[0], ANGLE_PI / 4),
        instruction.RZ(instr.controls[1], ANGLE_PI / 2),
    ]


def decompose_cnot(instr: instruction.CNOT) -> list[instruction.H | instruction.CZ]:
    """Return a decomposition of the CNOT gate as H·∧z·H.

    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Parameters
    ----------
        instr: the CNOT instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [
        instruction.H(instr.target),
        instruction.CZ((instr.control, instr.target)),
        instruction.H(instr.target),
    ]


def decompose_swap(instr: instruction.SWAP) -> list[instruction.CNOT]:
    """Return a decomposition of the SWAP gate as CNOT(0, 1)·CNOT(1, 0)·CNOT(0, 1).

    Michael A. Nielsen and Isaac L. Chuang,
    Quantum Computation and Quantum Information,
    Cambridge University Press, 2000
    (p. 23 in the 10th Anniversary Edition).

    Parameters
    ----------
        instr: the SWAP instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [
        instruction.CNOT(control=instr.targets[0], target=instr.targets[1]),
        instruction.CNOT(control=instr.targets[1], target=instr.targets[0]),
        instruction.CNOT(control=instr.targets[0], target=instr.targets[1]),
    ]


def decompose_y(instr: instruction.Y) -> list[instruction.X | instruction.Z]:
    """Return a decomposition of the Y gate as X·Z.

    Parameters
    ----------
        instr: the Y instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return list(reversed([instruction.X(instr.target), instruction.Z(instr.target)]))


def decompose_rx(instr: instruction.RX) -> list[instruction.J]:
    """Return a J decomposition of the RX gate.

    The Rx(α) gate is decomposed into J(α)·H (that is to say, J(α)·J(0)).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Parameters
    ----------
        instr: the RX instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [instruction.J(target=instr.target, angle=angle) for angle in reversed((instr.angle, 0))]


def decompose_ry(instr: instruction.RY) -> list[instruction.J]:
    """Return a J decomposition of the RY gate.

    The Ry(α) gate is decomposed into J(0)·J(π/2)·J(α)·J(-π/2).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, Robust and parsimonious realisations of unitaries in the one-way
    model, 2004.

    Parameters
    ----------
        instr: the RY instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [
        instruction.J(target=instr.target, angle=angle)
        for angle in reversed((0, ANGLE_PI / 2, instr.angle, -ANGLE_PI / 2))
    ]


def decompose_rz(instr: instruction.RZ) -> list[instruction.J]:
    """Return a J decomposition of the RZ gate.

    The Rz(α) gate is decomposed into H·J(α) (that is to say, J(0)·J(α)).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Parameters
    ----------
        instr: the RZ instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [instruction.J(target=instr.target, angle=angle) for angle in reversed((0, instr.angle))]


def instruction_to_jcz(instr: Instruction) -> Sequence[instruction.J | instruction.CZ]:
    """Return a J-∧z decomposition of the instruction.

    Parameters
    ----------
        instr: the instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    # Use == for mypy
    if instr.kind == InstructionKind.J:
        return [instr]
    if instr.kind == InstructionKind.CZ:
        return [instr]
    if instr.kind == InstructionKind.I:
        return []
    if instr.kind == InstructionKind.H:
        return [instruction.J(instr.target, 0)]
    if instr.kind == InstructionKind.S:
        return instruction_to_jcz(instruction.RZ(instr.target, ANGLE_PI / 2))
    if instr.kind == InstructionKind.X:
        return instruction_to_jcz(instruction.RX(instr.target, ANGLE_PI))
    if instr.kind == InstructionKind.Y:
        return instruction_list_to_jcz(decompose_y(instr))
    if instr.kind == InstructionKind.Z:
        return instruction_to_jcz(instruction.RZ(instr.target, ANGLE_PI))
    if instr.kind == InstructionKind.RX:
        return decompose_rx(instr)
    if instr.kind == InstructionKind.RY:
        return decompose_ry(instr)
    if instr.kind == InstructionKind.RZ:
        return decompose_rz(instr)
    if instr.kind == InstructionKind.CCX:
        return instruction_list_to_jcz(decompose_ccx(instr))
    if instr.kind == InstructionKind.RZZ:
        return instruction_list_to_jcz(decompose_rzz(instr))
    if instr.kind == InstructionKind.CNOT:
        return instruction_list_to_jcz(decompose_cnot(instr))
    if instr.kind == InstructionKind.SWAP:
        return instruction_list_to_jcz(decompose_swap(instr))
    if instr.kind == InstructionKind.M:
        raise ValueError("Measurement instructions cannot be decomposed into J and CZ gates.")
    assert_never(instr.kind)


def instruction_list_to_jcz(
    instrs: Iterable[Instruction],
) -> list[instruction.J | instruction.CZ]:
    """Return a J-∧z decomposition of the sequence of instructions.

    Parameters
    ----------
        instrs: the instruction sequence to decompose.

    Returns
    -------
        the decomposition.

    """
    return [jcz_instr for instr in instrs for jcz_instr in instruction_to_jcz(instr)]


def j_commands(current_node: int, next_node: int, angle: ParameterizedAngle) -> list[command.Command]:
    """Return the MBQC pattern commands for a J gate.

    Parameters
    ----------
        current_node: the current node.
        next_node: the next node.
        angle: the angle of the J gate.

    Returns
    -------
        the MBQC pattern commands for a J gate as a list

    """
    return [
        command.N(node=next_node),
        command.E(nodes=(current_node, next_node)),
        command.M(current_node, Measurement.XY(angle)),
        command.X(node=next_node, domain={current_node}),
    ]


def normalize_angle(angle: ParameterizedAngle) -> ParameterizedAngle:
    r"""Return an equivalent angle in range :math:`[0, 2 \cdot \pi)` if ``angle`` is instantiated.

    Parameters
    ----------
    angle: ParameterizedAngle
        An angle.

    Returns
    -------
    ParameterizedAngle
        An equivalent angle in range :math:`[0, 2 \cdot \pi)` if ``angle`` is instantiated.
        If ``angle`` is parameterized, ``angle`` is returned unchanged.
    """
    if isinstance(angle, float):
        return angle % (2 * ANGLE_PI)
    return angle


@dataclass(frozen=True)
class TranspileSwapsResult:
    """The result returned by :func:`transpile_swaps`."""

    circuit: Circuit
    """Circuit without SWAP gates."""

    qubits: tuple[int | None, ...]
    """
    Tuple which has the same width as the circuit and which for
    every qubit of the original circuit provides the index of the
    corresponding qubit in the output of the returned
    circuit. Measured qubits are mapped to ``None`` in this tuple.
    """


class _TranspileSwapVisitor(InstructionVisitor):
    qubits: list[int | None]

    def __init__(self, width: int) -> None:
        self.qubits = list(range(width))

    @override
    def visit_qubit(self, qubit: int) -> int:
        return _check_target(self.qubits, qubit)


def transpile_swaps(circuit: Circuit) -> TranspileSwapsResult:
    """Return a new circuit equivalent to the original one but without SWAP gates.

    Parameters
    ----------
    circuit : Circuit
        The original circuit

    Returns
    -------
    TranspileSwapsResult
        The field ``circuit`` contains an equivalent circuit without
        SWAP gates.
        The field ``qubits`` contains a tuple which has the same width
        as the circuit and which for every qubit of the original
        circuit provides the index of the corresponding qubit in the
        output of the returned circuit. Measured qubits are mapped to
        ``None`` in this tuple.
    """
    new_circuit = Circuit(circuit.width)
    visitor = _TranspileSwapVisitor(circuit.width)
    for instr in circuit.instruction:
        if instr.kind == InstructionKind.SWAP:
            u, v = instr.targets
            visitor.qubits[u], visitor.qubits[v] = visitor.qubits[v], visitor.qubits[u]
        else:
            new_circuit.add(instr.visit(visitor))
            if instr.kind == InstructionKind.M:
                visitor.qubits[instr.target] = None
    return TranspileSwapsResult(new_circuit, tuple(visitor.qubits))


class IllformedCircuitError(Exception):
    """Raised if the circuit is ill-formed."""

    def __init__(self) -> None:
        """Build the exception."""
        super().__init__("Ill-formed pattern")


class CircuitWithMeasurementError(Exception):
    """Raised if the circuit contains measurements."""

    def __init__(self) -> None:
        """Build the exception."""
        super().__init__("Circuits containing measurements are not supported by the transpiler.")


class InternalInstructionError(Exception):
    """Raised if the circuit contains internal _XC or _ZC instructions."""

    def __init__(self, instr: instruction.Instruction) -> None:
        """Build the exception."""
        super().__init__(f"Internal instruction: {instr}")


class MeasurementsNoPreproceddedError(Exception):
    """Raised if the circuit contains measurements that have not been preprocessed before the transpile step."""

    def __init__(self) -> None:
        """Build the exception."""
        super().__init__("Circuits containing measurements were incorrectly passed to the transpiler.")
