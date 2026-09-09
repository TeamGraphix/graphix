"""Test exporter to OpenQASM3 using graphix-qasm-parser to check the round-trip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy.random import PCG64, Generator

from graphix import Circuit, Instruction
from graphix.fundamentals import ANGLE_PI, Axis
from graphix.instruction import InstructionKind
from graphix.qasm3_exporter import circuit_to_qasm3
from graphix.random_objects import rand_circuit
from tests.test_instruction import ALL_INSTRUCTIONS

if TYPE_CHECKING:
    from graphix.instruction import InstructionType

try:
    from graphix_qasm_parser import OpenQASMParser  # type: ignore[import-not-found, unused-ignore]
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
    check_circuit = circuit.transpile_j_to_rzh()
    parser = OpenQASMParser()
    parsed_circuit = parser.parse_str(qasm)
    assert parsed_circuit.instruction == check_circuit.instruction


@pytest.mark.parametrize("jumps", range(1, 11))
def test_circuit_to_qasm3(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 4
    # See https://github.com/TeamGraphix/graphix-qasm-parser/pull/5
    check_round_trip(rand_circuit(nqubits, depth, rng, use_j=True, use_cz=True))


@pytest.mark.parametrize("instruction", ALL_INSTRUCTIONS)
def test_instruction_to_qasm3(instruction: InstructionType) -> None:
    if instruction.kind in {InstructionKind.RZZ, InstructionKind.M}:
        pytest.skip()
    check_round_trip(Circuit(3, instr=[instruction]))


def test_j_to_qasm3() -> None:
    circuit = Circuit(3, instr=[Instruction.J(target=0, angle=ANGLE_PI / 4)])
    qasm = circuit_to_qasm3(circuit)
    parser = OpenQASMParser()
    parsed_circuit = parser.parse_str(qasm)
    assert parsed_circuit.instruction == circuit.transpile_j_to_rzh().instruction


def test_j_to_qasm3_failure() -> None:
    circuit = Circuit(3, instr=[Instruction.J(target=0, angle=ANGLE_PI / 4)])
    with pytest.raises(ValueError):
        circuit_to_qasm3(circuit, transpile=False)


def test_measurement() -> None:
    circuit = Circuit(1, instr=[Instruction.M(target=0, axis=Axis.Z)])
    check_round_trip(circuit)
