"""Test exporter to OpenQASM3."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import pytest
from numpy.random import PCG64, Generator

from graphix import Circuit, instruction
from graphix.fundamentals import Plane
from graphix.qasm3_exporter import angle_to_qasm3, circuit_to_qasm3
from graphix.random_objects import rand_circuit

if TYPE_CHECKING:
    from graphix.instruction import Instruction

try:
    from graphix_qasm_parser import OpenQASMParser
except ImportError:
    pytestmark = pytest.mark.skip(reason="graphix-qasm-parser not installed")

    if TYPE_CHECKING:
        import sys

        # We skip type-checking the case where there is no
        # graphix-qasm-parser, since pyright cannot figure out that
        # tests are skipped in this case.
        sys.exit(1)


def check_round_trip(circuit: Circuit) -> None:
    qasm = circuit_to_qasm3(circuit)
    parser = OpenQASMParser()
    parsed_circuit = parser.parse_str(qasm)
    assert parsed_circuit.instruction == circuit.instruction


@pytest.mark.parametrize("jumps", range(1, 11))
def test_circuit_to_qasm3(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 4
    check_round_trip(rand_circuit(nqubits, depth, rng))


@pytest.mark.parametrize(
    "instruction",
    [
        instruction.CCX(target=0, controls=(1, 2)),
        instruction.RZZ(target=0, control=1, angle=pi / 4),
        instruction.CNOT(target=0, control=1),
        instruction.SWAP(targets=(0, 1)),
        instruction.H(target=0),
        instruction.S(target=0),
        instruction.X(target=0),
        instruction.Y(target=0),
        instruction.Z(target=0),
        instruction.I(target=0),
        instruction.RX(target=0, angle=pi / 4),
        instruction.RY(target=0, angle=pi / 4),
        instruction.RZ(target=0, angle=pi / 4),
    ],
)
def test_instruction_to_qasm3(instruction: Instruction) -> None:
    check_round_trip(Circuit(3, instr=[instruction]))


@pytest.mark.parametrize("check", [(pi / 4, "pi/4"), (3 * pi / 4, "3*pi/4"), (0.5, "0.5")])
def test_angle_to_qasm3(check: tuple[float, str]) -> None:
    angle, expected = check
    assert angle_to_qasm3(angle) == expected


def test_measurement() -> None:
    # Measurements are not supported yet by the parser.
    # https://github.com/TeamGraphix/graphix-qasm-parser/issues/3
    # The best we can do is to check if the measurement instruction
    # is exported as expected.
    circuit = Circuit(1, instr=[instruction.M(target=0, plane=Plane.XZ, angle=0)])
    qasm = circuit_to_qasm3(circuit)
    assert (
        qasm
        == """OPENQASM 3;
include "stdgates.inc";
qubit[1] q;
bit[1] b;
b[0] = measure q[0];"""
    )
