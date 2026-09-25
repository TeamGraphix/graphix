from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from graphix import BasicStates, Pattern, Statevector, StatevectorBackend
from graphix.command import BaseN

if TYPE_CHECKING:
    from numpy.random import Generator

    from graphix.command import CommandType


def test_no_explicit_input_state(hadamardpattern: Pattern, fx_rng: Generator) -> None:
    # No explicit input state: the default initial state is |+⟩.
    # H|+⟩ = |0⟩, so we expect the final state to be |0⟩.
    state = hadamardpattern.simulate(rng=fx_rng)
    assert state.isclose(Statevector(BasicStates.ZERO))


def test_explicit_input_state_zero(hadamardpattern: Pattern, fx_rng: Generator) -> None:
    # Provide an explicit input state |0⟩.
    # H|0⟩ = |+⟩, so the final state should be |+⟩.
    state = hadamardpattern.simulate(input_state=BasicStates.ZERO, rng=fx_rng)
    assert state.isclose(Statevector(BasicStates.PLUS))


def test_backend_prepared_zero(hadamardpattern: Pattern, fx_rng: Generator) -> None:
    # Prepare the initial state in a backend and pass `input_state=None`.
    # The backend already contains |0⟩ on its input nodes,
    # therefore H|0⟩ = |+⟩.
    backend = StatevectorBackend()
    backend.add_nodes(hadamardpattern.input_nodes, BasicStates.ZERO)
    state = hadamardpattern.simulate(backend=backend, input_state=None, rng=fx_rng)
    assert state.isclose(Statevector(BasicStates.PLUS))


def test_no_prepared_qubits_and_input_state_none(hadamardpattern: Pattern, fx_rng: Generator) -> None:
    # No prepared qubits in the backend and `input_state=None`.
    # This is ambiguous, so a ValueError must be raised.
    backend = StatevectorBackend()
    with pytest.raises(ValueError, match="the backend is expected to have 1 input nodes already prepared"):
        hadamardpattern.simulate(backend=backend, input_state=None, rng=fx_rng)


def test_prepared_qubits_and_input_state(hadamardpattern: Pattern, fx_rng: Generator) -> None:
    # Backend already contains a state (|0⟩) **and** we ask the
    # simulator to prepare its own input state (by omitting `input_state`).
    # This would lead to double-allocation of qubits, so a ValueError is
    # raised.
    backend = StatevectorBackend()
    backend.add_nodes(hadamardpattern.input_nodes, BasicStates.ZERO)
    with pytest.raises(ValueError, match="the backend is expected to have no pre-allocated qubits"):
        hadamardpattern.simulate(backend=backend, rng=fx_rng)


def test_node_index_after_finalize() -> None:
    pattern = Pattern(input_nodes=[0, 1], output_nodes=[1, 0])
    backend = StatevectorBackend()
    pattern.simulate(backend=backend)
    assert list(backend.node_index) == [1, 0]


def test_default_prepare_method_requires_n() -> None:
    # The type annotations require the pattern to contain only
    # elements of type `CommandType`, which excludes `BaseN`. We hope
    # pattern types will become more precise in the near future.
    # See https://github.com/TeamGraphix/graphix/issues/266
    pattern = Pattern(cmds=[cast("CommandType", BaseN(0))])
    with pytest.raises(
        TypeError, match=r"The default prepare method requires all preparation commands to be of type `N`."
    ):
        pattern.simulate()
