"""Exporter to OpenQASM3."""

from __future__ import annotations

from fractions import Fraction
from math import pi
from typing import TYPE_CHECKING

# assert_never added in Python 3.11
from typing_extensions import assert_never

from graphix.instruction import Instruction, InstructionKind

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def instruction_to_qasm3(instruction: Instruction) -> str:
    """Get the qasm3 representation of a single circuit instruction."""
    if instruction.kind == InstructionKind.M:
        return f"b[{instruction.target}] = measure q[{instruction.target}]"
    # Use of `==` here for mypy
    if (
        instruction.kind == InstructionKind.RX  # noqa: PLR1714
        or instruction.kind == InstructionKind.RY
        or instruction.kind == InstructionKind.RZ
    ):
        if not isinstance(instruction.angle, float):
            raise ValueError("QASM export of symbolic pattern is not supported")
        rad_over_pi = instruction.angle / pi
        tol = 1e-9
        frac = Fraction(rad_over_pi).limit_denominator(1000)
        if abs(rad_over_pi - float(frac)) > tol:
            angle = f"{rad_over_pi}*pi"
        num, den = frac.numerator, frac.denominator
        sign = "-" if num < 0 else ""
        num = abs(num)
        if den == 1:
            angle = f"{sign}pi" if num == 1 else f"{sign}{num}*pi"
        else:
            angle = f"{sign}pi/{den}" if num == 1 else f"{sign}{num}*pi/{den}"
        return f"{instruction.kind.name.lower()}({angle}) q[{instruction.target}]"

    # Use of `==` here for mypy
    if (
        instruction.kind == InstructionKind.H  # noqa: PLR1714
        or instruction.kind == InstructionKind.I
        or instruction.kind == InstructionKind.S
        or instruction.kind == InstructionKind.X
        or instruction.kind == InstructionKind.Y
        or instruction.kind == InstructionKind.Z
    ):
        return f"{instruction.kind.name.lower()} q[{instruction.target}]"
    if instruction.kind == InstructionKind.CNOT:
        return f"cx q[{instruction.control}], q[{instruction.target}]"
    if instruction.kind == InstructionKind.SWAP:
        return f"swap q[{instruction.targets[0]}], q[{instruction.targets[1]}]"
    if instruction.kind == InstructionKind.RZZ:
        return f"rzz q[{instruction.control}], q[{instruction.target}]"
    if instruction.kind == InstructionKind.CCX:
        return f"ccx q[{instruction.controls[0]}], q[{instruction.controls[1]}], q[{instruction.target}]"
    # Use of `==` here for mypy
    if instruction.kind == InstructionKind._XC or instruction.kind == InstructionKind._ZC:  # noqa: PLR1714
        raise ValueError("Internal instruction should not appear")
    assert_never(instruction.kind)
