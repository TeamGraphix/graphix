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
from graphix.parameter import InplaceParameterizable
from graphix.pattern import Pattern
from graphix.sim.base_backend import DenseStateBackend
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import Statevector, StatevectorBackend
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from collections.abc import Set as AbstractSet
    from typing import Literal

    from numpy.random import Generator

    from graphix.command import Node
    from graphix.fundamentals import ParameterizedAngle
    from graphix.instruction import InstructionType
    from graphix.parameter import ExpressionOrSupportsFloat, Parameter
    from graphix.pattern import Pattern
    from graphix.sim import Data
    from graphix.sim.base_backend import DenseState, Matrix
    from graphix.sim.density_matrix import DensityMatrix
    from graphix.states import State

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

    state : _DenseStateT
        State representation of the simulation output.
    classical_measures : tuple[int,...]
        Results of classical measurements.
    """

    state: _DenseStateT  # mypy rejects covariant types as dataclass parameters as of Python 3.13
    classical_measures: tuple[int, ...]


@dataclass
class _MapAngleVisitor(InstructionVisitor):
    f: Callable[[ParameterizedAngle], ParameterizedAngle]

    @override
    def visit_angle(self, angle: ParameterizedAngle) -> ParameterizedAngle:
        return self.f(angle)


class _InstructionValidatorVisitor(InstructionVisitor):
    """Instruction visitor that validates operations on active qubits.

    The ``active_qubits`` attribute references the same mutable set as the
    circuit's ``active_qubits`` attribute. The circuit updates this set after measurement gates are applied. See :meth:`Circuit.m`.
    """

    output_kind: dict[int, OutputKind]

    def __init__(self, nqubit: int) -> None:
        self.output_kind = dict.fromkeys(range(nqubit), OutputKind.Qubit)

    @override
    def visit_qubit(self, qubit: int) -> int:
        if self.output_kind.get(qubit) is not OutputKind.Qubit:
            raise ValueError(f"Qubit {qubit} is not an active qubit.")
        return qubit

    @override
    def visit_domain(self, domain: set[int]) -> set[int]:
        for bit in domain:
            if self.output_kind.get(bit) is not OutputKind.Bit:
                raise ValueError(f"Qubit {bit} is not a measured qubit.")
        return domain


class Circuit(InplaceParameterizable):
    """Quantum circuit.

    Stores a sequence of gate operations. Supports transpilation into
    measurement-based quantum computing (MBQC) measurement patterns and
    state vector simulation.

    Attributes
    ----------
    width : int
        Number of logical qubits in the gate network.
    instruction : list of InstructionType
        Sequence of gate instructions applied to the circuit.
    ancillas : int
        Number of ancilla qubits.
    ancilla_state : State
        Initial state of the ancilla qubits.
    active_qubits : set of int
        Indices of qubits currently active in the circuit, including logical
        and ancilla qubits.
    """

    instruction: list[InstructionType]

    def __init__(
        self,
        width: int,
        instr: Iterable[InstructionType] | None = None,
        *,
        ancillas: int = 0,
        ancilla_state: State = BasicStates.PLUS,
    ) -> None:
        """Initialize a circuit.

        Parameters
        ----------
        width : int
            Number of logical qubits in the gate network.
        instr : Iterable[InstructionType] or None, optional
            Initial sequence of instructions to add to the circuit. If
            ``None``, no instructions are added.
        ancillas : int, default=0
            Number of ancilla qubits.
        ancilla_state : State, default=BasicStates.PLUS
            Initial state assigned to the ancilla qubits.

        Notes
        -----
        Circuit simulation is supported for any ``ancilla_state``. However,
        transpilation to a measurement pattern is currently supported only when
        ``ancilla_state`` is a member of ``BasicStates``.
        """
        self.width = width
        self.ancillas = ancillas
        self.ancilla_state = ancilla_state
        self.instruction = []
        self.active_qubits = set(range(width + ancillas))
        # The visitor contains a reference to the `active_qubits` mutable set.
        # Changes on `self.active_qubits` also occur on `self._visitor.active_qubits`.
        self._visitor = _InstructionValidatorVisitor(width + ancillas)
        if instr is not None:
            self.extend(instr)

    @property
    def nqubit(self) -> int:
        """Total number of qubits in the circuit.

        It includes logical and ancilla qubits, whether they are active or not.
        """
        return self.width + self.ancillas

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
            case InstructionKind.CONDINSTR:
                self.cond_instr(instr.instructions, instr.domain)
            case _:
                assert_never(instr.kind)

    def extend(self, instrs: Iterable[InstructionType]) -> None:
        """Add instructions to the circuit."""
        for instr in instrs:
            self.add(instr)

    def __repr__(self) -> str:
        """Return a representation of the Circuit."""
        return f"Circuit(width={self.width}, instr={self.instruction}, ancillas={self.ancillas}, ancilla_state={self.ancilla_state!r})"

    def cnot(self, control: int, target: int) -> None:
        """Apply a CNOT gate.

        Parameters
        ----------
        control : int
            control qubit
        target : int
            target qubit
        """
        self.instruction.append(instruction.CNOT(control=control, target=target).visit(self._visitor))

    def swap(self, qubit1: int, qubit2: int) -> None:
        """Apply a SWAP gate.

        Parameters
        ----------
        qubit1 : int
            first qubit to be swapped
        qubit2 : int
            second qubit to be swapped
        """
        self.instruction.append(instruction.SWAP(targets=(qubit1, qubit2)).visit(self._visitor))

    def cz(self, qubit1: int, qubit2: int) -> None:
        """Apply a CZ gate.

        Parameters
        ----------
        qubit1 : int
            control qubit
        qubit2 : int
            target qubit
        """
        self.instruction.append(instruction.CZ(targets=(qubit1, qubit2)).visit(self._visitor))

    def h(self, qubit: int) -> None:
        """Apply a Hadamard gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        self.instruction.append(instruction.H(target=qubit).visit(self._visitor))

    def s(self, qubit: int) -> None:
        """Apply an S gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        self.instruction.append(instruction.S(target=qubit).visit(self._visitor))

    def x(self, qubit: int) -> None:
        """Apply a Pauli X gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        self.instruction.append(instruction.X(target=qubit).visit(self._visitor))

    def y(self, qubit: int) -> None:
        """Apply a Pauli Y gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        self.instruction.append(instruction.Y(target=qubit).visit(self._visitor))

    def z(self, qubit: int) -> None:
        """Apply a Pauli Z gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        self.instruction.append(instruction.Z(target=qubit).visit(self._visitor))

    def rx(self, qubit: int, angle: ParameterizedAngle) -> None:
        """Apply an X rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : ParameterizedAngle
            rotation angle in units of π
        """
        self.instruction.append(instruction.RX(target=qubit, angle=angle).visit(self._visitor))

    def ry(self, qubit: int, angle: ParameterizedAngle) -> None:
        """Apply a Y rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : ParameterizedAngle
            angle in units of π
        """
        self.instruction.append(instruction.RY(target=qubit, angle=angle).visit(self._visitor))

    def rz(self, qubit: int, angle: ParameterizedAngle) -> None:
        """Apply a Z rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : ParameterizedAngle
            rotation angle in units of π
        """
        self.instruction.append(instruction.RZ(target=qubit, angle=angle).visit(self._visitor))

    def j(self, qubit: int, angle: ParameterizedAngle) -> None:
        """Apply a J rotation gate.

        Parameters
        ----------
        qubit : int
            target qubit
        angle : ParameterizedAngle
            rotation angle in units of π
        """
        self.instruction.append(instruction.J(target=qubit, angle=angle).visit(self._visitor))

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
        self.instruction.append(instruction.RZZ(control=control, target=target, angle=angle).visit(self._visitor))

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
        self.instruction.append(instruction.CCX(controls=(control1, control2), target=target).visit(self._visitor))

    def i(self, qubit: int) -> None:
        """Apply an identity (teleportation) gate.

        Parameters
        ----------
        qubit : int
            target qubit
        """
        self.instruction.append(instruction.I(target=qubit).visit(self._visitor))

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
        self.instruction.append(instruction.M(target=qubit, axis=axis).visit(self._visitor))
        self.active_qubits.remove(qubit)
        self._visitor.output_kind[qubit] = OutputKind.Bit

    def cond_instr(self, instrs: Iterable[InstructionType], domain: AbstractSet[int] | None) -> None:
        """Apply a conditional sequence of gates.

        Parameters
        ----------
        instrs : Iterable[InstructionType]
            Sequence of instructions to apply conditionally.
        domain : AbstractSet[int] or None, optional
            Indices of measured qubits whose outcomes determine the condition. Defaults to ``None``.

        Notes
        -----
        The instruction sequence is applied when the XOR of the measurement outcomes of the qubits in ``domain`` evaluates to ``1``.
        """
        domain_set = set(domain) if domain is not None else set()
        self.instruction.append(
            instruction.CONDINSTR(instructions=tuple(instrs), domain=domain_set).visit(self._visitor)
        )

    def transpile_to_causalflow(self) -> TranspiledFlow:
        """Transpile a circuit via J-∧z decomposition to a causal flow.

        Parameters
        ----------
            self: the circuit to transpile.

        Returns
        -------
            the result of the transpilation: a causal flow and classical outputs.
        """
        n_nodes = self.nqubit
        indices: list[int | None] = list(range(n_nodes))
        measurements: dict[int, BlochMeasurement] = {}
        classical_outputs: dict[int, command.M] = {}
        inputs = list(range(self.width))
        graph: nx.Graph[int] = nx.empty_graph(n_nodes)
        x_corrections: dict[int, set[int]] = {}

        if self.ancillas and self.ancilla_state is not BasicStates.PLUS:
            new_circuit = self.transpile_ancilla_to_plus()
            instructions = new_circuit.instruction
        else:
            instructions = self.instruction

        for instr in instructions_to_jcz(instructions):
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
        swap = _transpile_swaps(self, copy=True)
        result = swap.circuit.transpile_to_causalflow().to_pattern()
        result.pattern.reorder_output_nodes(swap.swap_output_nodes(result.pattern.output_nodes))
        classical_outputs = swap.swap_classical_outputs(result.classical_outputs)
        return TranspiledPattern(result.pattern, classical_outputs)

    @overload
    def simulate(
        self,
        backend: StatevectorBackend | Literal["statevector"] = ...,
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> SimulateResult[Statevector]: ...

    @overload
    def simulate(
        self,
        backend: DensityMatrixBackend | Literal["densitymatrix"],
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> SimulateResult[DensityMatrix]: ...

    @overload
    def simulate(
        self,
        backend: DenseStateBackend[_DenseStateT],
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> SimulateResult[_DenseStateT]: ...

    def simulate(
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
        _backend = _initialize_backend(backend, branch_selector, self.nqubit)

        if input_state is None:
            _backend.add_nodes(range(self.width))
        else:
            _backend.add_nodes(range(self.width), input_state)
        if self.ancillas:
            _backend.add_nodes(range(self.width, self.nqubit), self.ancilla_state)

        classical_measures: list[Outcome] = []
        results: dict[int, Outcome] = {}  # Mimics `DefaultMeasureMethod.results`

        # Modifies in place `_backend`, `results`, `classical_measures`
        simulate_instructions(self.instruction, _backend, rng, results, classical_measures, stacklevel=stacklevel + 1)

        return SimulateResult(_backend.state, tuple(classical_measures))

    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Circuit:
        """Apply ``visitor`` to all instructions in the circuit.

        Parameters
        ----------
        visitor : InstructionVisitor
            The visitor specifying the rewriting.

        copy : bool, optional
            If ``True``, the current circuit remains unchanged, and a
            new circuit is returned. The default is ``False``, meaning
            that changes are performed in place.

        Returns
        -------
        Self
            The rewritten circuit. Equal to ``self`` if ``copy`` is ``False``.
        """
        if copy:
            result = Circuit(self.width)
            for instr in self.instruction:
                result.instruction.append(instr.visit(visitor, copy=True))
            return result
        for instr in self.instruction:
            instr.visit(visitor, copy=False)
        return self

    def apply_angle(self, f: Callable[[ParameterizedAngle], ParameterizedAngle], *, copy: bool = False) -> Circuit:
        """Apply ``f`` to all angles that occur in the circuit.

        Parameters
        ----------
        f : Callable[[ParameterizedAngle], ParameterizedAngle]
            The function to apply to every angle.

        copy : bool, optional
            If ``True``, the current circuit remains unchanged, and a
            new circuit is returned. The default is ``False``, meaning
            that changes are performed in place.

        Returns
        -------
        Self
            The rewritten circuit. Equal to ``self`` if ``copy`` is ``False``.
        """
        return self.visit(_MapAngleVisitor(f), copy=copy)

    def is_parameterized(self) -> bool:
        """
        Return ``True`` if there is at least one measurement angle that is not just an instance of :class:`SupportsFloat`.

        A parameterized circuit is a circuit where at least one
        measurement angle is an expression that is not a number,
        typically an instance of :class:`sympy.Expr` (but we don't force to
        choose ``sympy`` here).

        """
        for instr in self.instruction:
            match instr.kind:
                case InstructionKind.RZZ | InstructionKind.RX | InstructionKind.RY | InstructionKind.RZ:
                    if not isinstance(instr.angle, SupportsFloat):
                        return True
        return False

    @override
    def replace_parameter(
        self, variable: Parameter, substitute: ExpressionOrSupportsFloat, *, copy: bool = False
    ) -> Circuit:
        return self.apply_angle(lambda angle: parameter.with_parameter(angle, variable, substitute), copy=copy)

    @override
    def replace_parameters(
        self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat], *, copy: bool = False
    ) -> Circuit:
        return self.apply_angle(lambda angle: parameter.with_parameters(angle, assignment), copy=copy)

    def transpile_measurements_to_z_axis(self) -> Circuit:
        """Return an equivalent circuit where all measurements are on Z axis."""
        circuit = Circuit(width=self.width, ancillas=self.ancillas, ancilla_state=self.ancilla_state)
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
        new_circuit = Circuit(width=self.width, ancillas=self.ancillas, ancilla_state=self.ancilla_state)
        for instr in self.instruction:
            match instr.kind:
                case InstructionKind.J:
                    new_circuit.add(instruction.RZ(target=instr.target, angle=instr.angle))
                    new_circuit.add(instruction.H(target=instr.target))
                case _:
                    new_circuit.add(instr)
        return new_circuit

    def transpile_ancilla_to_plus(self) -> Circuit:
        r"""Return an equivalent circuit where ancilla states are replaced with :math:`|+\rangle`."""
        new_circuit = Circuit(self.width, ancillas=self.ancillas)
        instructions_prepend: list[Callable[[int], InstructionType]] = []
        match self.ancilla_state:
            case BasicStates.PLUS:
                pass
            case BasicStates.MINUS:
                instructions_prepend.append(instruction.Z)
            case BasicStates.ZERO:
                instructions_prepend.append(instruction.H)
            case BasicStates.ONE:
                instructions_prepend.extend((instruction.H, instruction.X))
            case BasicStates.PLUS_I:
                instructions_prepend.append(instruction.S)
            case BasicStates.MINUS_I:
                instructions_prepend.extend((instruction.S, instruction.Z))
            case _:
                raise NotImplementedError(
                    f"Transpilation only supports `BasicStates` ancillas. Ancilla state is {self.ancilla_state}"
                )

        for qubit in range(self.width, self.nqubit):
            for instr in instructions_prepend:
                new_circuit.add(instr(qubit))
        new_circuit.instruction += self.instruction
        return new_circuit

    def transpile_rzz(self) -> Circuit:
        """Return an equivalent circuit where all RZZ gates have been replaced with OpenQASM gates."""
        new_circuit = Circuit(self.width)
        for instr in self.instruction:
            match instr.kind:
                case InstructionKind.RZZ:
                    new_circuit.extend(decompose_rzz(instr))
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
            case InstructionKind.CONDINSTR:
                raise NotImplementedError("Transpilation of conditional instructions is not supported.")
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
        """Return for each output node, sorted in the order of the original circuit, the index of the corresponding output node in the order of the swapped circuit.

        This method returns a permutation of ``range(number_of_output_qubits)``.
        """
        qubit_indices = self.extract_outputs(OutputKind.Qubit)
        rank = {q: i for i, q in enumerate(sorted(qubit_indices))}
        return tuple(rank[q] for q in qubit_indices)

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


def transpile_swaps(circuit: Circuit, *, copy: bool = False) -> TranspileSwapsResult:
    """Return a new circuit equivalent to the original one but without SWAP gates.

    Parameters
    ----------
    circuit : Circuit
        The original circuit

    copy : bool, optional
        If ``True``, the current pattern remains unchanged, and a
        new pattern is returned. The default is ``False``, meaning
        that changes are performed in place.

    Returns
    -------
    TranspileSwapsResult
        The field ``circuit`` contains an equivalent circuit without
        SWAP gates. Equal to ``self`` if ``copy`` is ``False``.

        The field ``outputs`` contains a tuple which has
        the same width as the circuit. For every qubit of the original
        circuit, either the qubit is not measured, and ``outputs``
        provides the index of the corresponding qubit in the output of
        the returned circuit; or the qubit has been measured, and
        ``outputs`` provides the index of the measurement.
    """
    new_circuit = Circuit(circuit.width, ancillas=circuit.ancillas, ancilla_state=circuit.ancilla_state)
    visitor = _TranspileSwapVisitor(circuit.nqubit)
    measurement_index = 0
    for instr in circuit.instruction:
        if instr.kind == InstructionKind.SWAP:
            u, v = instr.targets
            # We apply the visitor to check that the qubits have not been measured.
            visitor.visit_qubit(u)
            visitor.visit_qubit(v)
            visitor.outputs[u], visitor.outputs[v] = visitor.outputs[v], visitor.outputs[u]
        elif instr.kind == InstructionKind.M:
            old_target = instr.target
            new_circuit.add(instr.visit(visitor, copy=copy))
            visitor.outputs[old_target] = OutputIndex(OutputKind.Bit, measurement_index)
            measurement_index += 1
        else:
            new_circuit.add(instr.visit(visitor, copy=copy))
    if not copy:
        circuit.instruction = new_circuit.instruction
        new_circuit = circuit
    return TranspileSwapsResult(new_circuit, tuple(visitor.outputs))


# Alias `_transpile_swaps` to call the function in `Circuit.transpile`
# method where `transpile_swaps` is shadowed by the keyword parameter.
_transpile_swaps = transpile_swaps


@overload
def _initialize_backend(
    backend: StatevectorBackend | Literal["statevector"],
    branch_selector: BranchSelector | None,
    width: int,
) -> StatevectorBackend: ...


@overload
def _initialize_backend(
    backend: DensityMatrixBackend | Literal["densitymatrix"],
    branch_selector: BranchSelector | None,
    width: int,
) -> DensityMatrixBackend: ...


@overload
def _initialize_backend(
    backend: DenseStateBackend[_DenseStateT],
    branch_selector: BranchSelector | None,
    width: int,
) -> DenseStateBackend[_DenseStateT]: ...


def _initialize_backend(
    backend: DenseStateBackend[_DenseStateT] | _DenseStateBackendLiteral,
    branch_selector: BranchSelector | None,
    width: int,
) -> _BuiltinDenseStateBackend | DenseStateBackend[_DenseStateT]:
    """Initialize backend for circuit simulation.

    Parameters
    ----------
    backend: :class:`graphix.sim.base_backend.DenseStateBackend[_DenseStateT]`, 'statevector', or 'densitymatrix'
        Simulation backend
    branch_selector: :class:`BranchSelector`
        Branch selector used for measurements. Can only be specified if ``backend`` is not an already instantiated :class:`Backend` object.  If ``None``, it defaults to :class:`RandomBranchSelector`.
    width : int
        Number of qubits in circuit. It is required to initialize the :class:`StatevectorBackend` with the appropriate
        capacity.

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
            return StatevectorBackend.with_capacity(width, branch_selector=branch_selector)
        case "densitymatrix":
            return DensityMatrixBackend(branch_selector=branch_selector)
        case _:
            raise ValueError(f"Unknown backend {backend}.")


def simulate_instructions(
    instructions: Iterable[InstructionType],
    backend: _BuiltinDenseStateBackend | DenseStateBackend[_DenseStateT],
    rng: Generator | None,
    results: dict[int, Outcome],
    classical_outputs: list[Outcome],
    *,
    stacklevel: int = 1,
) -> None:
    """Simulate a sequence of quantum instructions.

    Parameters
    ----------
    instructions : Iterable[InstructionType]
        Sequence of instructions to simulate.
    backend : _BuiltinDenseStateBackend or DenseStateBackend[_DenseStateT]
        Backend containing the quantum state.
    rng : Generator or None
        Random number generator used for stochastic measurements. If ``None``, the backend's default random number generation behavior is used.
    results : dict[int, Outcome]
        Mapping from measured qubit indices to their measurement outcomes. This mapping is updated in place as measurement instructions are simulated.
    classical_outputs : list[Outcome]
        List of measurement outcomes. This list is updated in place and retained for backwards compatibility.
    stacklevel : int, default=1
        Stack level used when reporting warnings generated during measurement.
    """

    def evolve_single(op: Matrix, target: int) -> None:
        backend.state.evolve_single(op, backend.node_index.index(target))

    def evolve(op: Matrix, qargs: Iterable[int]) -> None:
        backend.state.evolve(op, [backend.node_index.index(qarg) for qarg in qargs])

    # Mimics `MeasureMethod.check_domain`
    def check_domain(domain: set[int]) -> bool:
        return sum(results[j] for j in domain) % 2 == 1

    for instr in instructions:
        match instr.kind:
            case instruction.InstructionKind.CNOT:
                evolve(Ops.CNOT, [instr.control, instr.target])
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
            case instruction.InstructionKind.J:
                evolve_single(Ops.j(instr.angle), instr.target)
            case instruction.InstructionKind.RZZ:
                evolve(Ops.rzz(instr.angle), [instr.control, instr.target])
            case instruction.InstructionKind.CCX:
                evolve(Ops.CCX, [instr.controls[0], instr.controls[1], instr.target])
            case instruction.InstructionKind.M:
                result = backend.measure(instr.target, PauliMeasurement(instr.axis), rng=rng, stacklevel=stacklevel + 1)
                # We keep `classical_outputs` for backwards compatibility
                classical_outputs.append(result)
                results[instr.target] = result
            case instruction.InstructionKind.CONDINSTR:
                if check_domain(instr.domain):
                    simulate_instructions(
                        instr.instructions, backend, rng, results, classical_outputs, stacklevel=stacklevel + 1
                    )
            case _:
                raise ValueError(f"Unknown instruction: {instr}")
