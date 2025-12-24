"""Test exporter to OpenQASM3 targetting Qiskit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix import Circuit, Pattern
from graphix.branch_selector import FixedBranchSelector
from graphix.clifford import Clifford
from graphix.command import C, CommandKind, E, M, N
from graphix.fundamentals import Plane
from graphix.measurements import Measurement, outcome
from graphix.optimization import incorporate_pauli_results, single_qubit_domains
from graphix.qasm3_exporter import pattern_to_qasm3
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates

if TYPE_CHECKING:
    from graphix.measurements import Outcome
    from graphix.states import State

try:
    import qiskit
    import qiskit_qasm3_import
    from qiskit_aer import AerSimulator  # type:ignore[attr-defined]
except ImportError:
    pytestmark = pytest.mark.skip(reason="Missing packages: qiskit, qiskit_qasm3_import, qiskit_aer")

    if TYPE_CHECKING:
        import sys

        # We skip type-checking the case where there is no
        # graphix-qasm-parser, since pyright cannot figure out that
        # tests are skipped in this case.
        sys.exit(1)


def check_qasm3(pattern: Pattern) -> None:
    """Check that we obtain equivalent statevectors whether we simulate the pattern with Graphix or we use Qiskit AER simulator."""
    qasm3 = pattern_to_qasm3(pattern)
    qc = qiskit_qasm3_import.parse(qasm3)
    qc.save_statevector()  # type:ignore[attr-defined]
    aer_backend = AerSimulator(method="statevector")
    transpiled = qiskit.transpile(qc, aer_backend)
    result = aer_backend.run(transpiled, shots=1, memory=True).result()
    if qc.clbits:
        # One bitstring per shot; we ran exactly one shot.
        memory = result.get_memory()[0]
        # Qiskit reports measurement outcomes in reversed order:
        # the first measured qubit appears at the end of the string.
        results: dict[int, Outcome] = {
            cmd.node: outcome(measurement == "1")
            for cmd, measurement in zip(pattern.extract_measurement_commands(), reversed(memory), strict=True)
        }
    else:
        results = {}
    branch_selector = FixedBranchSelector(results)
    backend = StatevectorBackend(branch_selector=branch_selector)
    # Qiskit and Graphix order qubits in opposite directions.
    nodes = [
        node
        for src in (pattern.input_nodes, (cmd.node for cmd in pattern if cmd.kind == CommandKind.N))
        for node in src
    ]
    nodes.reverse()
    backend.add_nodes(nodes=nodes, data=np.asarray(result.get_statevector()))
    # Trace out measured qubits.
    for cmd in pattern.extract_measurement_commands():
        backend.measure(cmd.node, Measurement(angle=0, plane=Plane.XZ))
    # Reorder qubits to match the pattern's expected output ordering.
    backend.finalize(pattern.output_nodes)
    state_qiskit = backend.state
    state_mbqc = pattern.simulate_pattern(branch_selector=branch_selector)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state_qiskit.flatten())) == pytest.approx(1)


def test_to_qasm3_qubits_preparation() -> None:
    check_qasm3(Pattern(cmds=[N(0), N(1)]))
    check_qasm3(Pattern(input_nodes=[0], cmds=[N(1)]))


def test_to_qasm3_entanglement() -> None:
    check_qasm3(Pattern(input_nodes=[0, 1], cmds=[E((0, 1))]))
    check_qasm3(Pattern(input_nodes=[0, 1], cmds=[N(2), E((1, 2))]))


@pytest.mark.parametrize("clifford", Clifford)
@pytest.mark.parametrize("state", [BasicStates.ZERO, BasicStates.PLUS])
def test_to_qasm3_clifford(clifford: Clifford, state: State) -> None:
    check_qasm3(Pattern(cmds=[N(0, state), C(0, clifford)]))


@pytest.mark.parametrize("state", [BasicStates.ZERO, BasicStates.PLUS])
@pytest.mark.parametrize("plane", list(Plane))
@pytest.mark.parametrize("angle", [0, 0.25, 1.75])
def test_to_qasm3_measurement(state: State, plane: Plane, angle: float) -> None:
    check_qasm3(Pattern(cmds=[N(0, state), N(1), E((0, 1)), M(0, plane=plane, angle=angle)]))


def test_to_qasm3_hadamard() -> None:
    circuit = Circuit(1)
    circuit.h(0)
    pattern = circuit.transpile().pattern
    check_qasm3(pattern)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_to_qasm3_random_circuit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng=rng)
    pattern = circuit.transpile().pattern
    pattern.perform_pauli_measurements()
    pattern.minimize_space()

    # qiskit_qasm3_import.exceptions.ConversionError: initialisation of classical bits is not supported
    pattern = incorporate_pauli_results(pattern)

    # qiskit_qasm3_import.exceptions.ConversionError: unhandled binary operator '^'
    pattern = single_qubit_domains(pattern)

    check_qasm3(pattern)
