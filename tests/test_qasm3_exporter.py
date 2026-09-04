"""Test exporter to OpenQASM3 without external dependencies.

See Also
--------
- :mod:`test_qasm3_exporter_to_graphix_parser`, which checks the round trip with ``graphix-qasm-parser``;
- :mod:`test_qasm3_exporter_to_qiskit`, which checks against Qiskit simulation.
"""

from __future__ import annotations

import pytest
from numpy.random import PCG64, Generator

from graphix import Circuit
from graphix.fundamentals import ANGLE_PI, Axis
from graphix.qasm3_exporter import angle_to_qasm3, circuit_to_qasm3, pattern_to_qasm3
from graphix.random_objects import rand_circuit


@pytest.mark.parametrize("check", [(ANGLE_PI / 4, "pi/4"), (3 * ANGLE_PI / 4, "3*pi/4"), (ANGLE_PI / 2, "pi/2")])
def test_angle_to_qasm3(check: tuple[float, str]) -> None:
    angle, expected = check
    assert angle_to_qasm3(angle) == expected


@pytest.mark.parametrize("jumps", range(1, 11))
def test_to_qasm3_random_circuit(fx_bg: PCG64, jumps: int) -> None:
    """Check the export to OpenQASM 3 without validating the result.

    See
    :func:`test_qasm3_exporter_to_qiskit:test_to_qasm3_random_circuit`,
    where the result is validated. The current test does not go
    through the normalization pass ``single_qubit_domains``, so it
    exercises execution paths that are not tested elsewhere.
    """
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng=rng)
    pattern = circuit.transpile().pattern
    pattern.infer_pauli_measurements()
    pattern.remove_pauli_measurements()
    pattern.minimize_space()
    _qasm3 = pattern_to_qasm3(pattern)


def test_to_qasm3_measure_on_x_axis() -> None:
    circuit = Circuit(1)
    circuit.m(0, Axis.X)
    with pytest.raises(ValueError, match="OpenQASM3 only supports measurements on Z axis"):
        circuit_to_qasm3(circuit, transpile=False)
    _qasm3 = circuit_to_qasm3(circuit)


def test_to_qasm3_j() -> None:
    circuit = Circuit(1)
    circuit.j(0, 0.25)
    with pytest.raises(ValueError, match="J gates must be decomposed before QASM3 export"):
        circuit_to_qasm3(circuit, transpile=False)
    _qasm3 = circuit_to_qasm3(circuit)


def test_to_qasm3_rzz() -> None:
    circuit = Circuit(2)
    circuit.rzz(0, 1, 0.25)
    with pytest.raises(ValueError, match="RZZ gates must be decomposed before QASM3 export"):
        circuit_to_qasm3(circuit, transpile=False)
    _qasm3 = circuit_to_qasm3(circuit)
