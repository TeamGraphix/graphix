"""Exporter to OpenQASM3."""

from __future__ import annotations

from typing import TYPE_CHECKING

# assert_never added in Python 3.11
from typing_extensions import assert_never

from graphix.command import CommandKind
from graphix.fundamentals import Axis, ParameterizedAngle, Plane, angle_to_rad
from graphix.instruction import Instruction, InstructionKind
from graphix.pretty_print import OutputFormat, angle_to_str
from graphix.states import BasicStates, State

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from graphix import Circuit, Pattern
    from graphix.command import Command


def circuit_to_qasm3(circuit: Circuit) -> str:
    """Export circuit instructions to OpenQASM 3.0 representation.

    Returns
    -------
    str
        The OpenQASM 3.0 string representation of the circuit.
    """
    return "\n".join(circuit_to_qasm3_lines(circuit))


def circuit_to_qasm3_lines(circuit: Circuit) -> Iterator[str]:
    """Export circuit instructions to line-by-line OpenQASM 3.0 representation.

    Returns
    -------
    Iterator[str]
        The OpenQASM 3.0 lines that represent the circuit.
    """
    yield "OPENQASM 3;"
    yield 'include "stdgates.inc";'
    yield f"qubit[{circuit.width}] q;"
    if any(instr.kind == InstructionKind.M for instr in circuit.instruction):
        yield f"bit[{circuit.width}] b;"
    for instr in circuit.instruction:
        yield f"{instruction_to_qasm3(instr)};"


def qasm3_qubit(index: int) -> str:
    """Return the name of the indexed qubit."""
    return f"q[{index}]"


def qasm3_gate_call(gate: str, operands: Iterable[str], args: Iterable[str] | None = None) -> str:
    """Return the OpenQASM3 gate call."""
    operands_str = ", ".join(operands)
    if args is None:
        return f"{gate} {operands_str}"
    args_str = ", ".join(args)
    return f"{gate}({args_str}) {operands_str}"


def angle_to_qasm3(angle: ParameterizedAngle) -> str:
    """Get the OpenQASM3 representation of an angle."""
    if not isinstance(angle, float):
        raise TypeError("QASM export of symbolic pattern is not supported")
    return angle_to_str(angle, output=OutputFormat.ASCII, multiplication_sign=True)


def instruction_to_qasm3(instruction: Instruction) -> str:
    """Get the OpenQASM3 representation of a single circuit instruction."""
    if instruction.kind == InstructionKind.M:
        if instruction.axis != Axis.Z:
            raise ValueError(
                "OpenQASM3 only supports measurements on Z axis. Use `Circuit.transpile_measurements_to_z_axis` to rewrite measurements on X and Y axes."
            )
        return f"b[{instruction.target}] = measure q[{instruction.target}]"
    # Use of `==` here for mypy
    if (
        instruction.kind == InstructionKind.RX  # noqa: PLR1714
        or instruction.kind == InstructionKind.RY
        or instruction.kind == InstructionKind.RZ
    ):
        angle = angle_to_qasm3(instruction.angle)
        return qasm3_gate_call(instruction.kind.name.lower(), args=[angle], operands=[qasm3_qubit(instruction.target)])

    # Use of `==` here for mypy
    if (
        instruction.kind == InstructionKind.H  # noqa: PLR1714
        or instruction.kind == InstructionKind.S
        or instruction.kind == InstructionKind.X
        or instruction.kind == InstructionKind.Y
        or instruction.kind == InstructionKind.Z
    ):
        return qasm3_gate_call(instruction.kind.name.lower(), [qasm3_qubit(instruction.target)])
    if instruction.kind == InstructionKind.I:
        return qasm3_gate_call("id", [qasm3_qubit(instruction.target)])
    if instruction.kind == InstructionKind.CNOT:
        return qasm3_gate_call("cx", [qasm3_qubit(instruction.control), qasm3_qubit(instruction.target)])
    if instruction.kind == InstructionKind.SWAP:
        return qasm3_gate_call("swap", [qasm3_qubit(instruction.targets[i]) for i in (0, 1)])
    if instruction.kind == InstructionKind.CZ:
        return qasm3_gate_call("cz", [qasm3_qubit(instruction.targets[i]) for i in (0, 1)])
    if instruction.kind == InstructionKind.RZZ:
        angle = angle_to_qasm3(instruction.angle)
        return qasm3_gate_call(
            "crz", args=[angle], operands=[qasm3_qubit(instruction.control), qasm3_qubit(instruction.target)]
        )
    if instruction.kind == InstructionKind.CCX:
        return qasm3_gate_call(
            "ccx",
            [
                qasm3_qubit(instruction.controls[0]),
                qasm3_qubit(instruction.controls[1]),
                qasm3_qubit(instruction.target),
            ],
        )
    # Use of `==` here for mypy
    if instruction.kind == InstructionKind._XC or instruction.kind == InstructionKind._ZC:  # noqa: PLR1714
        raise ValueError("Internal instruction should not appear")
    assert_never(instruction.kind)


def pattern_to_qasm3(pattern: Pattern, input_state: dict[int, State] | State = BasicStates.PLUS) -> str:
    """Export a pattern to OpenQASM 3.0 representation.

    The generated OpenQASM may include initializations of classical
    qubits if the pattern has been Pauli-presimulated, and it may include
    Boolean expressions using xor (`^`) if some domains contain
    multiple qubits. These features are not supported by
    `qiskit-qasm3-import`. The functions
    :func:`graphix.optimization.incorporate_pauli_results` and
    :func:`graphix.optimization.single_qubit_domains` transform any
    pattern into an equivalent one such that exporting to OpenQASM 3.0
    produces a circuit that can be imported into Qiskit.

    Parameters
    ----------
    pattern : Pattern
        The pattern to export.

    input_state : dict[int, State] | State, default BasicStates.PLUS
        The initial state for each input node. Only |0⟩ or |+⟩ states are supported.
    """
    return "".join(pattern_to_qasm3_lines(pattern, input_state=input_state))


def pattern_to_qasm3_lines(pattern: Pattern, input_state: dict[int, State] | State = BasicStates.PLUS) -> Iterator[str]:
    """Export pattern to line-by-line OpenQASM 3.0 representation.

    See :func:`pattern_to_qasm3`.
    """
    yield "// generated by graphix\n"
    yield "OPENQASM 3;\n"
    yield 'include "stdgates.inc";\n'
    yield "\n"
    for node in pattern.input_nodes:
        yield f"// prepare input qubit {node} in |+⟩\n"
        yield f"qubit q{node};\n"
        state = input_state if isinstance(input_state, State) else input_state[node]
        yield from state_to_qasm3_lines(node, state)
        yield "\n"
    if pattern.results != {}:
        for i in pattern.results:
            res = pattern.results[i]
            yield f"// measurement result of qubit q{i}\n"
            yield f"bit c{i} = {res};\n"
            yield "\n"
    for cmd in pattern:
        yield from command_to_qasm3_lines(cmd)


def command_to_qasm3_lines(cmd: Command) -> Iterator[str]:
    """Convert a command in the pattern into OpenQASM 3.0 statement.

    Parameter
    ---------
    cmd : Command
        command

    Yields
    ------
    string
        translated pattern commands in OpenQASM 3.0 language

    """
    yield f"// {cmd}\n"
    if cmd.kind == CommandKind.N:
        yield f"qubit q{cmd.node};\n"
        yield from state_to_qasm3_lines(cmd.node, cmd.state)

    elif cmd.kind == CommandKind.E:
        n0, n1 = cmd.nodes
        yield f"cz q{n0}, q{n1};\n"

    elif cmd.kind == CommandKind.M:
        yield from domain_to_qasm3_lines(cmd.s_domain, f"x q{cmd.node}")
        yield from domain_to_qasm3_lines(cmd.t_domain, f"z q{cmd.node}")
        if cmd.plane == Plane.XY:
            yield f"h q{cmd.node};\n"
        if cmd.angle != 0:
            rad_angle = angle_to_rad(cmd.angle)
            if cmd.plane == Plane.XY:
                yield f"rx({-rad_angle}) q{cmd.node};\n"
            elif cmd.plane == Plane.XZ:
                yield f"ry({-rad_angle}) q{cmd.node};\n"
            else:
                yield f"rx({rad_angle}) q{cmd.node};\n"
        yield f"bit c{cmd.node};\n"
        yield f"c{cmd.node} = measure q{cmd.node};\n"

    elif cmd.kind == CommandKind.X:
        yield from domain_to_qasm3_lines(cmd.domain, f"x q{cmd.node}")

    elif cmd.kind == CommandKind.Z:
        yield from domain_to_qasm3_lines(cmd.domain, f"z q{cmd.node}")

    elif cmd.kind == CommandKind.C:
        for op in cmd.clifford.qasm3:
            yield str(op) + " q" + str(cmd.node) + ";\n"

    else:
        raise ValueError(f"invalid command {cmd}")

    yield "\n"


def state_to_qasm3_lines(node: int, state: State) -> Iterator[str]:
    """Convert initial state into OpenQASM 3.0 statement."""
    if state == BasicStates.ZERO:
        pass
    elif state == BasicStates.PLUS:
        yield f"h q{node};\n"
    else:
        raise ValueError("QASM3 conversion only supports |0⟩ or |+⟩ initial states.")


def domain_to_qasm3_lines(domain: Iterable[int], cmd: str) -> Iterator[str]:
    """Convert domain controlled-command into OpenQASM 3.0 statement.

    Parameter
    ---------
    domain : Iterable[int]
        measured nodes
    cmd : str
        controlled command

    Yields
    ------
    string
        translated controlled command in OpenQASM 3.0 language
    """
    condition = " ^ ".join(f"c{node}" for node in domain)
    if not condition:
        return
    yield f"if ({condition}) {{\n"
    yield f"  {cmd};\n"
    yield "}\n"
