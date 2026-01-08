"""Exporter to OpenQASM3."""

from __future__ import annotations

from typing import TYPE_CHECKING

# assert_never added in Python 3.11
from typing_extensions import assert_never

from graphix.fundamentals import Axis, ParameterizedAngle
from graphix.instruction import Instruction, InstructionKind
from graphix.pretty_print import OutputFormat, angle_to_str

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from graphix import Circuit


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
    assert_never(instruction.kind)
