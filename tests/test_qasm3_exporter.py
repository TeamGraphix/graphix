"""Test exporter to OpenQASM3."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy.random import PCG64, Generator

from graphix.qasm3_exporter import circuit_to_qasm3
from graphix.random_objects import rand_circuit

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


@pytest.mark.parametrize("jumps", range(1, 11))
def test_circuit_to_qasm3(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    qasm = circuit_to_qasm3(circuit)
    parser = OpenQASMParser()
    parsed_circuit = parser.parse_str(qasm)
    assert parsed_circuit.instruction == circuit.instruction
