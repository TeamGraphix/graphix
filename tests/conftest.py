from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy.random import PCG64, Generator

from graphix.random_objects import rand_circuit
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from graphix.pattern import Pattern

SEED = 25
DEPTH = 1


@pytest.fixture()
def fx_rng() -> Generator:
    return Generator(PCG64(SEED))


@pytest.fixture()
def fx_bg() -> PCG64:
    return PCG64(SEED)


@pytest.fixture
def hadamardpattern() -> Pattern:
    circ = Circuit(1)
    circ.h(0)
    return circ.transpile().pattern


@pytest.fixture
def nqb(fx_rng: Generator) -> int:
    return fx_rng.integers(2, 5)


@pytest.fixture
def rand_circ(nqb, fx_rng: Generator) -> Circuit:
    return rand_circuit(nqb, DEPTH, fx_rng)


@pytest.fixture
def randpattern(rand_circ: Circuit) -> Pattern:
    return rand_circ.transpile().pattern
