"""Gate-to-MBQC transpiler.

accepts desired gate operations and transpile into MBQC measurement patterns.

"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Generic, SupportsFloat, TypeVar, overload

import networkx as nx

# assert_never introduced in Python 3.11
# override introduced in Python 3.12
from typing_extensions import assert_never, override

from graphix import command, instruction, parameter
from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.flow.core import CausalFlow, _corrections_to_partial_order_layers
from graphix.fundamentals import ANGLE_PI, Axis
from graphix.instruction import InstructionKind, InstructionVisitor
from graphix.measurements import BlochMeasurement, Measurement, Outcome, PauliMeasurement
from graphix.opengraph import OpenGraph
from graphix.ops import Ops
from graphix.optimization import StandardizedPattern
from graphix.pattern import Pattern
from graphix.sim.base_backend import DenseStateBackend
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import Statevector, StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from typing import Literal

    from numpy.random import Generator

    from graphix.command import Node
    from graphix.fundamentals import ParameterizedAngle
    from graphix.instruction import InstructionType
    from graphix.parameter import ExpressionOrFloat, Parameter
    from graphix.pattern import Pattern
    from graphix.sim import Data
    from graphix.sim.base_backend import DenseState, Matrix
    from graphix.sim.density_matrix import DensityMatrix

    _BuiltinDenseStateBackend = DensityMatrixBackend | StatevectorBackend
    _DenseStateBackendLiteral = Literal["statevector", "densitymatrix"]

_DenseStateT = TypeVar("_DenseStateT", bound="DenseState")


@dataclass(frozen=True, slots=True)
class TranspiledPattern:
    """A transpiled pattern."""

    pattern: Pattern

    classical_outputs: tuple[Node, ...]
    """Nodes measured with circuit measurements, in the order of the gates."""


@dataclass(frozen=True, slots=True)
class TranspiledFlow:
    """A transpiled causal flow."""

    flow: CausalFlow[BlochMeasurement]

    classical_outputs: dict[int, command.M]
    """M commands for nodes measured with circuit measurements."""

    def to_pattern(self) -> TranspiledPattern:
        """Return the transpiled pattern."""
        pattern = StandardizedPattern.from_pattern(self.flow.to_xzcorrections().to_pattern()).to_space_optimal_pattern()
        pattern.extend(self.classical_outputs.values())
        return TranspiledPattern(pattern, tuple(self.classical_outputs.keys()))


@dataclass(frozen=True)
class SimulateResult(Generic[_DenseStateT]):
    """
    Result of a circuit simulation.

    statevec : _DenseStateT
        State representation of the simulation output.
    classical_measures : tuple[int,...]
        Results of classical measurements.
    """

    statevec: _DenseStateT  # mypy rejects covariant types as dataclass parameters as of Python 3.13
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

    instruction: list[InstructionType]

    def __init__(self, width: int, instr: Iterable[InstructionType] | None = None) -> None:
        """
        Construct a circuit.

        Parameters
        ----------
        width : int
            number of logical qubits for the gate network
        instr : list[instruction.InstructionType] | None
            Optional. List of initial instructions.
        """
        self.width = width
        self.instruction = []
        self.active_qubits = set(range(width))
        if instr is not None:
            self.extend(instr)

    def add(self, instr: InstructionType) -> None:
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

    def extend(self, instrs: Iterable[InstructionType]) -> None:
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

    def transpile_to_causalflow(self) -> TranspiledFlow:
        """Transpile a circuit via J-∧z decomposition to a causal flow.

        Parameters
        ----------
            self: the circuit to transpile.

        Returns
        -------
            the result of the transpilation: a causal flow and classical outputs.
        """
        indices: list[int | None] = list(range(self.width))
        n_nodes = self.width
        measurements: dict[int, BlochMeasurement] = {}
        classical_outputs: dict[int, command.M] = {}
        inputs = list(range(n_nodes))
        graph: nx.Graph[int] = nx.Graph()
        graph.add_nodes_from(inputs)
        x_corrections: dict[int, set[int]] = {}
        for instr in instructions_to_jcz(self.instruction):
            match instr.kind:
                case InstructionKind.M:
                    target = indices[instr.target]
                    if target is None:
                        raise RuntimeError("Ill-formed circuit")
                    classical_outputs[target] = command.M(target, PauliMeasurement(instr.axis))
                    indices[instr.target] = None
                    continue
                case InstructionKind.J:
                    target = indices[instr.target]
                    if target is None:
                        raise RuntimeError("Ill-formed circuit")
                    graph.add_edge(target, n_nodes)  # Also adds nodes
                    measurements[target] = Measurement.XY(normalize_angle(-instr.angle))
                    indices[instr.target] = n_nodes
                    x_corrections[target] = {n_nodes}  # X correction on ancilla
                    n_nodes += 1
                    continue
                case InstructionKind.CZ:
                    t0, t1 = instr.targets
                    i0, i1 = indices[t0], indices[t1]
                    if i0 is None or i1 is None:
                        raise RuntimeError("Ill-formed circuit")
                    # If edge exists, remove it; else, add it
                    if graph.has_edge(i0, i1):
                        graph.remove_edge(i0, i1)
                    else:
                        graph.add_edge(i0, i1)
                    continue
                case _:
                    assert_never(instr.kind)
        outputs = [i for i in indices if i is not None]
        outputs.extend(classical_outputs.keys())  # Necessary for flow-finding step
        og = OpenGraph(
            graph=graph,
            input_nodes=inputs,
            output_nodes=outputs,
            measurements=measurements,
        )
        z_corrections: dict[int, set[int]] = {}
        for node, correctors in x_corrections.items():
            z_targets = og.neighbors(correctors) - {node}
            if z_targets:
                z_corrections[node] = z_targets
        partial_order_layers = _corrections_to_partial_order_layers(og, x_corrections, z_corrections)
        f: CausalFlow[BlochMeasurement] = CausalFlow(og, x_corrections, partial_order_layers)
        return TranspiledFlow(f, classical_outputs)

    def transpile(self, *, transpile_swaps: bool = True) -> TranspiledPattern:
        """Transpile a circuit via J-∧z decomposition to a pattern.

        Parameters
        ----------
        transpile_swaps: bool, optional
            If ``True`` (the default), SWAP gates are eliminated by switching the qubits.
            If ``False``, SWAP gates are transpiled into a sequence of three CNOT gates.

        Returns
        -------
        TranspiledPattern
            The result of the transpilation: a pattern and classical outputs.
        """
        if not transpile_swaps:
            return self.transpile_to_causalflow().to_pattern()
        swap = _transpile_swaps(self)
        result = swap.circuit.transpile_to_causalflow().to_pattern()
        result.pattern.reorder_output_nodes(swap.swap_output_nodes(result.pattern.output_nodes))
        classical_outputs = swap.swap_classical_outputs(result.classical_outputs)
        return TranspiledPattern(result.pattern, classical_outputs)

    @overload
    def simulate_statevector(
        self,
        backend: StatevectorBackend | Literal["statevector"] = ...,
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> SimulateResult[Statevector]: ...

    @overload
    def simulate_statevector(
        self,
        backend: DensityMatrixBackend | Literal["densitymatrix"],
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> SimulateResult[DensityMatrix]: ...

    @overload
    def simulate_statevector(
        self,
        backend: DenseStateBackend[_DenseStateT],
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> SimulateResult[_DenseStateT]: ...

    def simulate_statevector(
        self,
        backend: DenseStateBackend[_DenseStateT] | _DenseStateBackendLiteral = "statevector",
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> SimulateResult[_DenseStateT] | SimulateResult[_DenseStateT | Statevector | DensityMatrix]:
        # `SimulateResult` is not covariant in `_DenseStateT` so `SimulateResult[_DenseStateT]` is not a subtype of `SimulateResult[_DenseStateT | Statevector | DensityMatrix]`
        r"""Simulate the gate sequence with a backend and input state of choice.

        By default, this method uses the statevector backend and initializes the register to :math:`|+\rangle^{\otimes n}`.

        Parameters
        ----------
        input_state : Data
        backend: :class:`graphix.sim.base_backend.DenseStateBackend[_DenseStateT]`, 'statevector', or 'densitymatrix'
            Simulator backend to use. Optional, defaults to "statevector".
        branch_selector: :class:`graphix.branch_selector.BranchSelector`
            branch selector for measures (default: :class:`RandomBranchSelector`). It cannot be specified if ``backend`` is already instantiated.
        rng: Generator, optional
            Random-number generator for measurements.
            This generator is used only in case of random branch selection
            (see :class:`RandomBranchSelector`).

        Returns
        -------
        result : :class:`SimulateResult`
            output state of the statevector simulation and results of classical measures.
        """
        _backend = _initialize_backend(backend, branch_selector)

        if input_state is None:
            _backend.add_nodes(range(self.width))
        else:
            _backend.add_nodes(range(self.width), input_state)

        classical_measures: list[Outcome] = []

        for i in range(len(self.instruction)):
            instr = self.instruction[i]

            def evolve_single(op: Matrix, target: int) -> None:
                _backend.state.evolve_single(op, _backend.node_index.index(target))

            def evolve(op: Matrix, qargs: Iterable[int]) -> None:
                _backend.state.evolve(op, [_backend.node_index.index(qarg) for qarg in qargs])

            match instr.kind:
                case instruction.InstructionKind.CNOT:
                    evolve(Ops.CNOT, [instr.control, instr.target])
                case instruction.InstructionKind.SWAP:
                    u, v = instr.targets
                    _backend.state.swap((_backend.node_index.index(u), _backend.node_index.index(v)))
                case instruction.InstructionKind.CZ:
                    u, v = instr.targets
                    _backend.state.entangle((_backend.node_index.index(u), _backend.node_index.index(v)))
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
                case instruction.InstructionKind.J:
                    evolve_single(Ops.j(instr.angle), instr.target)
                case instruction.InstructionKind.RZZ:
                    evolve(Ops.rzz(instr.angle), [instr.control, instr.target])
                case instruction.InstructionKind.CCX:
                    evolve(Ops.CCX, [instr.controls[0], instr.controls[1], instr.target])
                case instruction.InstructionKind.M:
                    result = _backend.measure(
                        instr.target, PauliMeasurement(instr.axis), rng=rng, stacklevel=stacklevel + 1
                    )
                    classical_measures.append(result)
                case _:
                    raise ValueError(f"Unknown instruction: {instr}")
        return SimulateResult(_backend.state, tuple(classical_measures))

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

    def transpile_j_to_rzh(self) -> Circuit:
        """Return an equivalent circuit where all J gates have been replaced with RZ and H gates."""
        new_circuit = Circuit(self.width)
        for instr in self.instruction:
            match instr.kind:
                case InstructionKind.J:
                    new_circuit.add(instruction.RZ(target=instr.target, angle=instr.angle))
                    new_circuit.add(instruction.H(target=instr.target))
                case _:
                    new_circuit.add(instr)
        return new_circuit


def decompose_rzz(instr: instruction.RZZ) -> Iterator[instruction.CNOT | instruction.RZ]:
    """Yield a decomposition of RZZ(α) gate as CNOT(control, target)·Rz(target, α)·CNOT(control, target).

    Parameters
    ----------
        instr: the RZZ instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    yield instruction.CNOT(target=instr.target, control=instr.control)
    yield instruction.RZ(instr.target, instr.angle)
    yield instruction.CNOT(target=instr.target, control=instr.control)


def decompose_ccx(
    instr: instruction.CCX,
) -> Iterator[instruction.H | instruction.CNOT | instruction.RZ]:
    """Yield a decomposition of the CCX gate into H, CNOT, T and T-dagger gates.

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
    c0, c1, t = instr.controls[0], instr.controls[1], instr.target
    yield instruction.H(t)
    yield instruction.CNOT(control=c1, target=t)
    yield instruction.RZ(t, -ANGLE_PI / 4)
    yield instruction.CNOT(control=c0, target=t)
    yield instruction.RZ(t, ANGLE_PI / 4)
    yield instruction.CNOT(control=c1, target=t)
    yield instruction.RZ(t, -ANGLE_PI / 4)
    yield instruction.CNOT(control=c0, target=t)
    yield instruction.RZ(c1, -ANGLE_PI / 4)
    yield instruction.RZ(t, ANGLE_PI / 4)
    yield instruction.CNOT(control=c0, target=c1)
    yield instruction.H(t)
    yield instruction.RZ(c1, -ANGLE_PI / 4)
    yield instruction.CNOT(control=c0, target=c1)
    yield instruction.RZ(c0, ANGLE_PI / 4)
    yield instruction.RZ(c1, ANGLE_PI / 2)


def decompose_cnot(instr: instruction.CNOT) -> Iterator[instruction.H | instruction.CZ]:
    """Yield a decomposition of the CNOT gate as H·∧z·H.

    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Parameters
    ----------
        instr: the CNOT instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    yield instruction.H(instr.target)
    yield instruction.CZ((instr.control, instr.target))
    yield instruction.H(instr.target)


def decompose_swap(instr: instruction.SWAP) -> Iterator[instruction.CNOT]:
    """Yield a decomposition of the SWAP gate as CNOT(0, 1)·CNOT(1, 0)·CNOT(0, 1).

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
    yield instruction.CNOT(control=instr.targets[0], target=instr.targets[1])
    yield instruction.CNOT(control=instr.targets[1], target=instr.targets[0])
    yield instruction.CNOT(control=instr.targets[0], target=instr.targets[1])


def decompose_y(instr: instruction.Y) -> Iterator[instruction.X | instruction.Z]:
    """Return a decomposition of the Y gate as X·Z.

    Parameters
    ----------
        instr: the Y instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    yield instruction.Z(instr.target)
    yield instruction.X(instr.target)


def decompose_rx(instr: instruction.RX) -> Iterator[instruction.J]:
    """Yield a J decomposition of the RX gate.

    The Rx(α) gate is decomposed into J(α)·H (that is to say, J(α)·J(0)).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Parameters
    ----------
        instr: the RX instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    yield instruction.J(instr.target, 0)
    yield instruction.J(instr.target, instr.angle)


def decompose_ry(instr: instruction.RY) -> Iterator[instruction.J]:
    """Yield a J decomposition of the RY gate.

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
    yield instruction.J(target=instr.target, angle=-ANGLE_PI / 2)
    yield instruction.J(target=instr.target, angle=instr.angle)
    yield instruction.J(target=instr.target, angle=ANGLE_PI / 2)
    yield instruction.J(target=instr.target, angle=0)


def decompose_rz(instr: instruction.RZ) -> Iterator[instruction.J]:
    """Yield a J decomposition of the RZ gate.

    The Rz(α) gate is decomposed into H·J(α) (that is to say, J(0)·J(α)).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Parameters
    ----------
        instr: the RZ instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    yield instruction.J(target=instr.target, angle=instr.angle)
    yield instruction.J(target=instr.target, angle=0)


def instructions_to_jcz(instrs: Iterable[InstructionType]) -> Iterator[instruction.J | instruction.CZ | instruction.M]:
    """Yield a J-∧z decomposition of the instruction.

    Parameters
    ----------
        instr: the instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    for instr in instrs:
        match instr.kind:
            case InstructionKind.J | InstructionKind.CZ | InstructionKind.M:
                yield instr
            case InstructionKind.I:
                return
            case InstructionKind.H:
                yield instruction.J(instr.target, 0)
            case InstructionKind.S:
                yield from decompose_rz(instruction.RZ(instr.target, ANGLE_PI / 2))
            case InstructionKind.X:
                yield from decompose_rx(instruction.RX(instr.target, ANGLE_PI))
            case InstructionKind.Y:
                yield from instructions_to_jcz(decompose_y(instr))
            case InstructionKind.Z:
                yield from decompose_rz(instruction.RZ(instr.target, ANGLE_PI))
            case InstructionKind.RX:
                yield from decompose_rx(instr)
            case InstructionKind.RY:
                yield from decompose_ry(instr)
            case InstructionKind.RZ:
                yield from decompose_rz(instr)
            case InstructionKind.CCX:
                yield from instructions_to_jcz(decompose_ccx(instr))
            case InstructionKind.RZZ:
                yield from instructions_to_jcz(decompose_rzz(instr))
            case InstructionKind.CNOT:
                yield from instructions_to_jcz(decompose_cnot(instr))
            case InstructionKind.SWAP:
                yield from instructions_to_jcz(decompose_swap(instr))
            case _:
                assert_never(instr.kind)


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

    outputs: tuple[OutputIndex, ...]
    """
    Tuple which has the same width as the circuit and which for
    every qubit of the original circuit provides the index of the
    corresponding qubit in the output of the swapped circuit
    (either measured or not).
    """

    def extract_outputs(self, kind: OutputKind) -> tuple[int, ...]:
        """Return the sequence of outputs of the given kind."""
        return tuple(output.index for output in self.outputs if output.kind == kind)

    def extract_output_node_indices(self) -> tuple[int, ...]:
        """Return for each output node, sorted in the order of the original circuit, the index of the corresponding output node in the order of the swapped circuit."""
        reduced_index = {}
        reduced_counter = 0
        for index, output in enumerate(self.outputs):
            if output.kind == OutputKind.Qubit:
                reduced_index[index] = reduced_counter
                reduced_counter += 1
        return tuple(reduced_index[index] for index in self.extract_outputs(OutputKind.Qubit))

    def swap_output_nodes(self, output_nodes: Sequence[Node]) -> tuple[Node, ...]:
        """Reorder the output nodes of a pattern obtained from a swapped circuit to restore the qubit ordering of the original circuit."""
        return tuple(output_nodes[index] for index in self.extract_output_node_indices())

    def swap_classical_outputs(self, classical_outputs: Sequence[Node]) -> tuple[int, ...]:
        """Reorder the classical outpus of a pattern obtained from a swapped circuit to restore the output ordering of the original circuit."""
        return tuple(classical_outputs[index] for index in self.extract_outputs(OutputKind.Bit))


class OutputKind(Enum):
    """Specify whether a qubit is measured or not."""

    Qubit = enum.auto()
    Bit = enum.auto()


@dataclass(frozen=True)
class OutputIndex:
    """Index of a swapped qubit.

    If the qubit is measured, ``kind`` equals to `OutputKind.Bit` and
    ``index`` is the index of the measurement.

    If the qubit is not measured, ``kind`` equals to `OutputKind.qubit`
    and ``index`` is the index of the qubit in the swapped circuit.
    """

    kind: OutputKind
    index: int


class _TranspileSwapVisitor(InstructionVisitor):
    outputs: list[OutputIndex]

    def __init__(self, width: int) -> None:
        self.outputs = [OutputIndex(OutputKind.Qubit, index) for index in range(width)]

    @override
    def visit_qubit(self, qubit: int) -> int:
        target = self.outputs[qubit]
        if target.kind == OutputKind.Bit:
            raise RuntimeError(f"Qubit {qubit} has already been measured.")
        return target.index


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
        SWAP gates.  The field ``outputs`` contains a tuple which has
        the same width as the circuit. For every qubit of the original
        circuit, either the qubit is not measured, and ``outputs``
        provides the index of the corresponding qubit in the output of
        the returned circuit; or the qubit has been measured, and
        ``outputs`` provides the index of the measurement.
    """
    new_circuit = Circuit(circuit.width)
    visitor = _TranspileSwapVisitor(circuit.width)
    measurement_index = 0
    for instr in circuit.instruction:
        if instr.kind == InstructionKind.SWAP:
            u, v = instr.targets
            # We apply the visitor to check that the qubits have not been measured.
            visitor.visit_qubit(u)
            visitor.visit_qubit(v)
            visitor.outputs[u], visitor.outputs[v] = visitor.outputs[v], visitor.outputs[u]
        else:
            new_circuit.add(instr.visit(visitor))
            if instr.kind == InstructionKind.M:
                visitor.outputs[instr.target] = OutputIndex(OutputKind.Bit, measurement_index)
                measurement_index += 1
    return TranspileSwapsResult(new_circuit, tuple(visitor.outputs))


_transpile_swaps = transpile_swaps


@overload
def _initialize_backend(
    backend: StatevectorBackend | Literal["statevector"],
    branch_selector: BranchSelector | None,
) -> StatevectorBackend: ...


@overload
def _initialize_backend(
    backend: DensityMatrixBackend | Literal["densitymatrix"],
    branch_selector: BranchSelector | None,
) -> DensityMatrixBackend: ...


@overload
def _initialize_backend(
    backend: DenseStateBackend[_DenseStateT],
    branch_selector: BranchSelector | None,
) -> DenseStateBackend[_DenseStateT]: ...


def _initialize_backend(
    backend: DenseStateBackend[_DenseStateT] | _DenseStateBackendLiteral,
    branch_selector: BranchSelector | None,
) -> _BuiltinDenseStateBackend | DenseStateBackend[_DenseStateT]:
    """Initialize backend for circuit simulation.

    Parameters
    ----------
    backend: :class:`graphix.sim.base_backend.DenseStateBackend[_DenseStateT]`, 'statevector', or 'densitymatrix'
        Simulation backend
    branch_selector: :class:`BranchSelector`
        Branch selector used for measurements. Can only be specified if ``backend`` is not an already instantiated :class:`Backend` object.  If ``None``, it defaults to :class:`RandomBranchSelector`.

    Returns
    -------
    :class:`DenseStateBackend`
        matching the appropriate backend
    """
    if isinstance(backend, DenseStateBackend):
        if branch_selector is not None:
            raise ValueError("`branch_selector` cannot be specified if `backend` is already instantiated.")
        return backend

    if branch_selector is None:
        branch_selector = RandomBranchSelector()

    match backend:
        case "statevector":
            return StatevectorBackend(branch_selector=branch_selector)
        case "densitymatrix":
            return DensityMatrixBackend(branch_selector=branch_selector)
        case _:
            raise ValueError(f"Unknown backend {backend}.")
