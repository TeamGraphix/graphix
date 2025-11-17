from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix import instruction
from graphix.fundamentals import Plane
from graphix.gflow import flow_from_pattern
from graphix.random_objects import rand_circuit, rand_gate
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from graphix.instruction import Instruction

INSTRUCTION_TEST_CASES = [
    instruction.CCX(0, (1, 2)),
    instruction.RZZ(0, 1, np.pi / 4),
    instruction.CNOT(0, 1),
    instruction.SWAP((0, 1)),
    instruction.H(0),
    instruction.S(0),
    instruction.X(0),
    instruction.Y(0),
    instruction.Z(0),
    instruction.I(0),
    instruction.RX(0, 0),
    instruction.RY(0, 0),
    instruction.RZ(0, 0),
]


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

    def test_measure(self, fx_rng: Generator) -> None:
        circuit = Circuit(2)
        circuit.h(1)
        circuit.cnot(0, 1)
        circuit.m(0, Plane.XY, 0.5)
        _ = circuit.transpile()

        def simulate_and_measure() -> int:
            circuit_simulate = circuit.simulate_statevector(rng=fx_rng)
            assert circuit_simulate.classical_measures[0] == (circuit_simulate.statevec.psi[0][1].imag > 0)
            return circuit_simulate.classical_measures[0]

        nb_shots = 10000
        count = sum(1 for _ in range(nb_shots) if simulate_and_measure())
        assert abs(count - nb_shots / 2) < nb_shots / 20

    def test_add_extend(self) -> None:
        circuit = Circuit(3)
        circuit.ccx(0, 1, 2)
        circuit.rzz(0, 1, 2)
        circuit.cnot(0, 1)
        circuit.swap(0, 1)
        circuit.h(0)
        circuit.s(0)
        circuit.x(0)
        circuit.y(0)
        circuit.z(0)
        circuit.i(0)
        circuit.m(0, Plane.XY, 0.5)
        circuit.rx(1, 0.5)
        circuit.ry(2, 0.5)
        circuit.rz(1, 0.5)
        circuit2 = Circuit(3, instr=circuit.instruction)
        assert circuit.instruction == circuit2.instruction

    @pytest.mark.parametrize("instruction", INSTRUCTION_TEST_CASES)
    def test_instruction_flow(self, instruction: Instruction) -> None:
        circuit = Circuit(3, instr=[instruction])
        pattern = circuit.transpile().pattern
        pattern.standardize()
        f, _l = flow_from_pattern(pattern)
        assert f is not None

    @pytest.mark.parametrize("instruction", INSTRUCTION_TEST_CASES)
    def test_instructions(self, fx_rng: Generator, instruction: Instruction) -> None:
        circuit = Circuit(3, instr=[instruction])
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)
