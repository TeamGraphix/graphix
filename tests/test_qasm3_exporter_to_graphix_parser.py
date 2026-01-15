"""Test exporter to OpenQASM3 using graphix-qasm-parser to check the round-trip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy.random import PCG64, Generator

from graphix import Circuit, instruction
from graphix.fundamentals import ANGLE_PI
from graphix.qasm3_exporter import circuit_to_qasm3
from graphix.random_objects import rand_circuit

if TYPE_CHECKING:
    from graphix.instruction import Instruction

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
    parser = OpenQASMParser()
    parsed_circuit = parser.parse_str(qasm)
    assert parsed_circuit.instruction == circuit.instruction


@pytest.mark.parametrize("jumps", range(1, 11))
def test_circuit_to_qasm3(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 4
    # See https://github.com/TeamGraphix/graphix-qasm-parser/pull/5
    check_round_trip(rand_circuit(nqubits, depth, rng, use_cz=False))


@pytest.mark.parametrize(
    "instruction",
    [
        instruction.CCX(target=0, controls=(1, 2)),
        instruction.RZZ(target=0, control=1, angle=ANGLE_PI / 4),
        instruction.CNOT(target=0, control=1),
        instruction.SWAP(targets=(0, 1)),
        # See https://github.com/TeamGraphix/graphix-qasm-parser/pull/5
        # instruction.CZ(targets=(0, 1)),
        instruction.H(target=0),
        instruction.S(target=0),
        instruction.X(target=0),
        instruction.Y(target=0),
        instruction.Z(target=0),
        instruction.I(target=0),
        instruction.RX(target=0, angle=ANGLE_PI / 4),
        instruction.RY(target=0, angle=ANGLE_PI / 4),
        instruction.RZ(target=0, angle=ANGLE_PI / 4),
    ],
)
def test_instruction_to_qasm3(instruction: Instruction) -> None:
    check_round_trip(Circuit(3, instr=[instruction]))
