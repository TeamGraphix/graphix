from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest

from graphix import Command, OpenGraph, Pattern, PauliMeasurement
from graphix._db import (
    CLIFFORD,
    CLIFFORD_CONJ,
    CLIFFORD_HSZ_DECOMPOSITION,
    CLIFFORD_MEASURE,
    CLIFFORD_MUL,
    CLIFFORD_PAULI_DECOMPOSITION,
)
from graphix.clifford import Clifford
from graphix.opengraph import OpenGraphError
from graphix.ops import Ops
from graphix.random_objects import rand_state_vector

if TYPE_CHECKING:
    from typing import TypeVar

    from numpy.random import Generator

    T = TypeVar("T")


class TestCliffordDB:
    @pytest.mark.parametrize(("i", "j"), itertools.product(range(24), range(3)))
    def test_measure(self, i: int, j: int) -> None:
        pauli = CLIFFORD[j + 1]
        arr = CLIFFORD[i].conjugate().T @ pauli @ CLIFFORD[i]
        sym, sgn = CLIFFORD_MEASURE[i][j]
        arr_ = complex(sgn) * Ops.from_ixyz(sym)
        assert np.allclose(arr, arr_)

    @pytest.mark.parametrize(("i", "j"), itertools.product(range(24), range(24)))
    def test_multiplication(self, i: int, j: int) -> None:
        op = CLIFFORD[i] @ CLIFFORD[j]
        assert Clifford.try_from_matrix(op) == Clifford(CLIFFORD_MUL[i][j])

    @pytest.mark.parametrize("i", range(24))
    def test_conjugation(self, i: int) -> None:
        op = CLIFFORD[i].conjugate().T
        assert Clifford.try_from_matrix(op) == Clifford(CLIFFORD_CONJ[i])

    @pytest.mark.parametrize("i", range(24))
    def test_decomposition(self, i: int) -> None:
        op = np.eye(2, dtype=np.complex128)
        for j in CLIFFORD_HSZ_DECOMPOSITION[i]:
            op @= CLIFFORD[j]
        assert Clifford.try_from_matrix(op) == Clifford(i)

    @pytest.mark.parametrize("i", range(24))
    def test_safety(self, i: int) -> None:
        with pytest.raises(TypeError):
            # Cannot replace
            CLIFFORD[i] = np.eye(2)  # type: ignore[index]
        m = CLIFFORD[i]
        with pytest.raises(ValueError):
            # Cannot modify
            m[0, 0] = 42
        with pytest.raises(ValueError):
            # Cannot make it writeable
            m.flags.writeable = True
        v = m.view()
        with pytest.raises(ValueError):
            # Cannot create writeable view
            v.flags.writeable = True


def generate_clifford_pauli_decomposition(rng: Generator) -> tuple[tuple[PauliMeasurement, ...], ...]:
    """Compute the value of CLIFFORD_PAULI_DECOMPOSITION.

    This function ensures that the length of the decomposition is
    optimal by searching exhaustively for increasing length.
    """
    pauli_measurements = tuple(PauliMeasurement)
    input_states = tuple(rand_state_vector(nqubits=1, rng=rng) for _ in range(10))
    clifford_output_states_ref = tuple(
        (
            clifford,
            tuple(
                Pattern(input_nodes=[0], cmds=[Command.C(0, clifford)]).simulate_pattern(input_state=input_state)
                for input_state in input_states
            ),
        )
        for clifford in Clifford
    )

    patterns: list[tuple[PauliMeasurement, ...] | None] = [None] * len(Clifford)

    def explore(n: int) -> None:
        graph = nx.path_graph(n + 1)
        for measurement_list in itertools.product(pauli_measurements, repeat=n):
            measurements = dict(zip(range(n), measurement_list, strict=True))
            og = OpenGraph(graph=graph, input_nodes=[0], output_nodes=[n], measurements=measurements)
            try:
                pattern = og.to_pattern()
            except OpenGraphError:
                continue
            for clifford, output_states_ref in clifford_output_states_ref:
                if patterns[clifford.value] is not None:
                    continue
                if all(
                    pattern.simulate_pattern(input_state=input_state, rng=rng).isclose(output_state_ref)
                    for input_state, output_state_ref in zip(input_states, output_states_ref, strict=True)
                ):
                    patterns[clifford.value] = measurement_list

    for n in range(4):
        explore(n)

    if any(pattern is None for pattern in patterns):
        raise RuntimeError("Local Cliffords are guaranteed to have a decomposition in 3 or less measured nodes.")

    return tuple(map(unwrap, patterns))


def unwrap(v: T | None) -> T:
    """Return ``v`` if it is not ``None``, or raise an exception."""
    if v is None:
        raise ValueError("Unexpected `None`.")
    return v


def test_generate_clifford_pauli_decomposition(fx_rng: Generator) -> None:
    clifford_pauli_decomposition = generate_clifford_pauli_decomposition(fx_rng)
    assert clifford_pauli_decomposition == CLIFFORD_PAULI_DECOMPOSITION
