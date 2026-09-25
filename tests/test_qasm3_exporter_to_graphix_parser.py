"""Test exporter to OpenQASM3 using graphix-qasm-parser to check the round-trip."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import pytest
from numpy.random import PCG64, Generator

from graphix import Circuit, Instruction
from graphix.fundamentals import ANGLE_PI, Axis
from graphix.instruction import InstructionKind
from graphix.qasm3_exporter import circuit_to_qasm3
from graphix.random_objects import rand_circuit
from graphix.states import BasicStates
from tests.test_instruction import INSTRUCTION_TEST_CASES

# `graphix-qasm-parser` depends on the `graphix` package, so we cannot have
# `graphix-qasm-parser` as a dependency of `graphix`, as this would create
# a dependency loop, which is disallowed by PyPI.
# Instead, `graphix-qasm-parser` is made optional and installed separately
# in `noxfile.py` (for `tests_all` and as a reverse dependency) and in the
# `cov` pipeline.
# The version of `graphix-qasm-parser` to use is defined in two places:
# - `nox`'s `tests_all` session and `cov` pipeline rely on the requirement
#   defined in `.github/qasm-parser-requirements.txt`;
# - `nox`'s reverse dependency check is declared in `noxfile.py` itself.
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

if TYPE_CHECKING:
    from graphix.states import PlanarState
    from tests.test_instruction import InstructionTestCase


def check_round_trip(circuit: Circuit) -> None:
    qasm = circuit_to_qasm3(circuit)
    check_circuit = circuit.transpile_to_qasm_gates().transpile_ancilla_state(BasicStates.ZERO)
    parser = OpenQASMParser()
    parsed_circuit = parser.parse_str(qasm)
    for parsed_instr, instr in zip(parsed_circuit.instruction, check_circuit.instruction, strict=True):
        assert parsed_instr.kind == instr.kind
        assert all(
            math.isclose(x, y) if isinstance(x, float) and isinstance(y, float) else x == y
            for field in dataclasses.fields(parsed_instr)
            for x, y in [(getattr(parsed_instr, field.name), getattr(instr, field.name))]
        )


@pytest.mark.parametrize("jumps", range(1, 11))
def test_circuit_to_qasm3(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 4
    # See https://github.com/TeamGraphix/graphix-qasm-parser/pull/5
    check_round_trip(rand_circuit(nqubits, depth, rng, use_j=True, use_cz=True))


@pytest.mark.parametrize("test_case", INSTRUCTION_TEST_CASES)
def test_instruction_to_qasm3(fx_rng: Generator, test_case: InstructionTestCase) -> None:
    instruction = test_case.instruction(fx_rng)
    if instruction.kind in {InstructionKind.CJ, InstructionKind.RZZ, InstructionKind.M}:
        pytest.skip()
    check_round_trip(Circuit(3, instr=[instruction]))


def test_j_to_qasm3() -> None:
    circuit = Circuit(1, instr=[Instruction.J(target=0, angle=ANGLE_PI / 4)])
    qasm = circuit_to_qasm3(circuit)
    parser = OpenQASMParser()
    parsed_circuit = parser.parse_str(qasm)
    assert parsed_circuit.instruction == circuit.transpile_to_qasm_gates().instruction


def test_j_to_qasm3_failure() -> None:
    circuit = Circuit(3, instr=[Instruction.J(target=0, angle=ANGLE_PI / 4)])
    with pytest.raises(ValueError):
        circuit_to_qasm3(circuit, transpile=False)


def test_measurement() -> None:
    circuit = Circuit(1, instr=[Instruction.M(target=0, axis=Axis.Z)])
    check_round_trip(circuit)


def test_cj_to_qasm3() -> None:
    circuit = Circuit(2, instr=[Instruction.CJ(control=0, target=1, angle=ANGLE_PI / 4)])
    qasm = circuit_to_qasm3(circuit)
    parser = OpenQASMParser()
    parsed_circuit = parser.parse_str(qasm)
    assert parsed_circuit.instruction == circuit.transpile_to_qasm_gates().instruction


def test_rzz_to_qasm3() -> None:
    circuit = Circuit(2, instr=[Instruction.RZZ(control=0, target=1, angle=ANGLE_PI / 4)])
    qasm = circuit_to_qasm3(circuit)
    parser = OpenQASMParser()
    parsed_circuit = parser.parse_str(qasm)
    assert parsed_circuit.instruction == circuit.transpile_to_qasm_gates().instruction


def test_gphase_to_qasm3() -> None:
    instr = Instruction.GPHASE(ANGLE_PI / 4)
    circuit = Circuit(1, instr=[instr])
    qasm = circuit_to_qasm3(circuit)
    parser = OpenQASMParser()
    parsed_circuit = parser.parse_str(qasm)
    assert parsed_circuit.instruction == [instr]


def test_condinstr_to_qasm3() -> None:
    circuit = Circuit(
        3,
        instr=[
            Instruction.M(2, Axis.Z),
            Instruction.CONDINSTR((Instruction.X(0),), {2}),
            Instruction.M(0, Axis.Z),
            Instruction.M(3, Axis.X),
            Instruction.CONDINSTR(
                (
                    Instruction.X(1),
                    Instruction.Z(1),
                ),
                {0, 2, 3},
            ),
        ],
        ancillas=1,
    )
    check_round_trip(circuit)


@pytest.mark.parametrize(
    "ancilla_state",
    [
        BasicStates.PLUS,
        BasicStates.MINUS,
        BasicStates.ZERO,
        BasicStates.ONE,
        BasicStates.PLUS_I,
        BasicStates.MINUS_I,
    ],
)
def test_to_qasm3_ancillas(ancilla_state: PlanarState) -> None:
    circuit = Circuit(
        1,
        instr=[Instruction.RX(0, 0.3), Instruction.RY(1, 0.4), Instruction.RZ(2, 0.35)],
        ancillas=2,
        ancilla_state=ancilla_state,
    )
    check_round_trip(circuit)
