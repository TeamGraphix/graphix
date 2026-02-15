from __future__ import annotations

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix.clifford import Clifford
from graphix.command import C, Command, CommandKind, E, M, N, X, Z
from graphix.fundamentals import ANGLE_PI, Plane
from graphix.optimization import StandardizedPattern, incorporate_pauli_results, remove_useless_domains
from graphix.pattern import Pattern
from graphix.random_objects import rand_circuit
from graphix.states import PlanarState


def test_standardize_clifford_entanglement(fx_rng: Generator) -> None:
    alpha = 2 * ANGLE_PI * fx_rng.random()
    i_lst = [0]
    o_lst = [0, 1]

    supported_gates = {0, 1, 2, 3, 4, 5, 9, 10}

    for i in range(24):
        for j in range(24):
            cmds: list[Command] = [N(1), C(0, Clifford(i)), C(1, Clifford(j)), E((0, 1))]
            p = Pattern(input_nodes=i_lst, output_nodes=o_lst, cmds=cmds)
            p_ref = p.copy()

            if i not in supported_gates:
                with pytest.raises(
                    NotImplementedError,
                    match=r"Pattern contains a Clifford followed by an E command on qubit 0 which only commute up to a two-qubit Clifford. Standarization is not supported.",
                ):
                    p.standardize()
            elif j not in supported_gates:
                with pytest.raises(
                    NotImplementedError,
                    match=r"Pattern contains a Clifford followed by an E command on qubit 1 which only commute up to a two-qubit Clifford. Standarization is not supported.",
                ):
                    p.standardize()
            else:
                p.standardize()

                # check C commands are at the end
                assert p[0].kind == CommandKind.N
                assert p[1].kind == CommandKind.E
                assert p[2].kind == CommandKind.C
                assert p[3].kind == CommandKind.C

                state_ref = p_ref.simulate_pattern(input_state=PlanarState(Plane.XY, alpha))
                state_p = p.simulate_pattern(input_state=PlanarState(Plane.XY, alpha))
                assert state_ref.isclose(state_p)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_incorporate_pauli_results(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 3
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern2 = incorporate_pauli_results(pattern)
    state = pattern.simulate_pattern(rng=rng)
    state2 = pattern2.simulate_pattern(rng=rng)
    assert state.isclose(state2)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_flow_after_pauli_preprocessing(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 3
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    # pattern.move_pauli_measurements_to_the_front()
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern2 = incorporate_pauli_results(pattern)
    gflow = pattern2.extract_gflow()
    gflow.check_well_formed()


@pytest.mark.parametrize("jumps", range(1, 11))
def test_remove_useless_domains(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 3
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern2 = remove_useless_domains(pattern)
    state = pattern.simulate_pattern(rng=rng)
    state2 = pattern2.simulate_pattern(rng=rng)
    assert state.isclose(state2)


def test_to_space_optimal_pattern() -> None:
    pattern = Pattern(
        cmds=[
            N(8),
            N(17),
            N(18),
            E((8, 17)),
            E((17, 18)),
            M(8, angle=-0.75),
            Z(18, {8}),
            X(17, {8}),
            C(17, (Clifford.S @ Clifford.Z)),
            C(18, (Clifford.S @ Clifford.Z)),
        ],
        output_nodes=[17, 18],
    )
    pattern2 = StandardizedPattern.from_pattern(pattern).to_space_optimal_pattern()
    state = pattern.simulate_pattern()
    state2 = pattern2.simulate_pattern()
    assert state.isclose(state2)
