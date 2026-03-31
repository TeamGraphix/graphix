from __future__ import annotations

from typing import TYPE_CHECKING

from graphix import Circuit
from graphix.sim.statevec import Statevec, StatevectorBackend
from graphix.states import BasicStates

if TYPE_CHECKING:
    from numpy.random import Generator


def test_input_state_none(fx_rng: Generator) -> None:
    circuit = Circuit(1)
    circuit.h(0)
    pattern = circuit.transpile().pattern
    # By default, the initial state is |+>, therefore H|+> = |0>.
    state = pattern.simulate_pattern(rng=fx_rng)
    assert state.isclose(Statevec(BasicStates.ZERO))
    # With the initial state |0>, we obtain H|0> = |+>.
    input_state = BasicStates.ZERO
    state = pattern.simulate_pattern(input_state=input_state, rng=fx_rng)
    assert state.isclose(Statevec(BasicStates.PLUS))
    # With the initial state |0> prepared in the backend, we obtain
    # H|0> = |+>.
    backend = StatevectorBackend()
    backend.add_nodes(pattern.input_nodes, input_state)
    state = pattern.simulate_pattern(backend=backend, input_state=None, rng=fx_rng)
    assert state.isclose(Statevec(BasicStates.PLUS))
    # The backend already prepares |0>. If the simulator also prepares
    # the input qubits in |+> (because we do not pass
    # `input_state=None`), an additional qubit is introduced. The
    # simulation therefore applies (I ⊗ H) on |0+>, resulting in |00>.
    backend = StatevectorBackend()
    backend.add_nodes(pattern.input_nodes, input_state)
    state = pattern.simulate_pattern(backend=backend, rng=fx_rng)
    assert state.isclose(Statevec(BasicStates.ZERO, nqubit=2))
