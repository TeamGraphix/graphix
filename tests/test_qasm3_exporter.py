"""Test exporter to OpenQASM3 without external dependencies.

See Also
--------
- :mod:`test_qasm3_exporter_to_graphix_parser`, which checks the round trip with ``graphix-qasm-parser``;
- :mod:`test_qasm3_exporter_to_qiskit`, which checks against Qiskit simulation.
"""

from __future__ import annotations

import pytest
from numpy.random import PCG64, Generator

from graphix import Circuit, instruction
from graphix.fundamentals import ANGLE_PI, Axis
from graphix.qasm3_exporter import angle_to_qasm3, circuit_to_qasm3, pattern_to_qasm3
from graphix.random_objects import rand_circuit


@pytest.mark.parametrize("check", [(ANGLE_PI / 4, "pi/4"), (3 * ANGLE_PI / 4, "3*pi/4"), (ANGLE_PI / 2, "pi/2")])
def test_angle_to_qasm3(check: tuple[float, str]) -> None:
    angle, expected = check
    assert angle_to_qasm3(angle) == expected


def test_measurement() -> None:
    # Measurements are not supported yet by the parser.
    # https://github.com/TeamGraphix/graphix-qasm-parser/issues/3
    # The best we can do is to check if the measurement instruction
    # is exported as expected.
    circuit = Circuit(1, instr=[instruction.M(target=0, axis=Axis.Z)])
    qasm = circuit_to_qasm3(circuit)
    assert (
        qasm
        == """OPENQASM 3;
include "stdgates.inc";
qubit[1] q;
bit[1] b;
b[0] = measure q[0];"""
    )


@pytest.mark.parametrize("jumps", range(1, 11))
def test_to_qasm3_random_circuit(fx_bg: PCG64, jumps: int) -> None:
    """Check the export to OpenQASM 3 without validating the result.

    See
    :func:`test_qasm3_exporter_to_qiskit:test_to_qasm3_random_circuit`,
    where the result is validated. The current test does not go through the
    normalization passes ``incorporate_pauli_results`` and ``single_qubit_domains``,
    so it exercises execution paths that are not tested elsewhere.
    """
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng=rng)
    pattern = circuit.transpile().pattern
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern.minimize_space()
    _qasm3 = pattern_to_qasm3(pattern)
