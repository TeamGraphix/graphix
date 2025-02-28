from __future__ import annotations

from shutil import which

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix.fundamentals import Plane
from graphix.random_objects import rand_circuit, rand_gate
from graphix.transpiler import Circuit


class TestTranspilerUnitGates:
    def test_cnot(self, fx_rng: Generator) -> None:
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_hadamard(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.h(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_s(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.s(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_x(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.x(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_y(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.y(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_z(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.z(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_rx(self, fx_rng: Generator) -> None:
        theta = fx_rng.uniform() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rx(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_ry(self, fx_rng: Generator) -> None:
        theta = fx_rng.uniform() * 2 * np.pi
        circuit = Circuit(1)
        circuit.ry(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_rz(self, fx_rng: Generator) -> None:
        theta = fx_rng.uniform() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rz(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_i(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.i(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_ccx(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 4
        depth = 6
        circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
        pattern = circuit.transpile().pattern
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_transpiled(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rand_gate(nqubits, depth, pairs, fx_rng, use_rzz=True)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_measure(self) -> None:
        circuit = Circuit(2)
        circuit.h(1)
        circuit.cnot(0, 1)
        circuit.m(0, Plane.XY, 0.5)
        _ = circuit.transpile()

        def simulate_and_measure() -> int:
            circuit_simulate = circuit.simulate_statevector()
            assert circuit_simulate.classical_measures[0] == (circuit_simulate.statevec.psi[0][1].imag > 0)
            return circuit_simulate.classical_measures[0]

        nb_shots = 10000
        count = sum(1 for _ in range(nb_shots) if simulate_and_measure())
        assert abs(count - nb_shots / 2) < nb_shots / 20


def test_circuit_draw() -> None:
    circuit = Circuit(10)
    try:
        circuit.draw("text")
        circuit.draw("mpl")
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.skipif(which("latex") is None, reason="latex not installed")
def test_circuit_draw_latex() -> None:
    circuit = Circuit(10)
    try:
        circuit.draw("latex")
        circuit.draw("latex_source")
    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize("jumps", range(1, 11))
def test_to_qasm3(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    qasm = circuit.to_qasm3()
    import pyzx as zx

    from graphix.pyzx import from_pyzx_graph

    print(qasm)
    z = zx.qasm(qasm)
    g = z.to_graph()
    og = from_pyzx_graph(g)
    pattern = og.to_pattern()
    pattern.minimize_space()
    print(pattern.draw("unicode"))
    state = circuit.simulate_statevector().statevec
    state_mbqc = pattern.simulate_pattern()
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)
