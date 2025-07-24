from __future__ import annotations

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix.clifford import Clifford
from graphix.command import C, Command, CommandKind, E, N
from graphix.fundamentals import Plane
from graphix.gflow import gflow_from_pattern
from graphix.optimization import incorporate_pauli_results
from graphix.pattern import Pattern
from graphix.random_objects import rand_circuit
from graphix.states import PlanarState


def test_standardize_clifford_entanglement(fx_rng: Generator) -> None:
    alpha = 2 * np.pi * fx_rng.random()
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
                assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_incorporate_pauli_results(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 3
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.perform_pauli_measurements()
    pattern2 = incorporate_pauli_results(pattern)
    state = pattern.simulate_pattern(rng=rng)
    state2 = pattern2.simulate_pattern(rng=rng)
    assert np.abs(np.dot(state.flatten().conjugate(), state2.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_flow_after_pauli_preprocessing(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 3
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.perform_pauli_measurements()
    pattern2 = incorporate_pauli_results(pattern)
    pattern2.standardize()
    f, _l = gflow_from_pattern(pattern2)
    assert f is not None
