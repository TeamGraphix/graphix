import pytest
from numpy.random import PCG64, Generator
import tests.random_circuit
import graphix.transpiler

SEED = 25


@pytest.fixture()
def fx_rng() -> Generator:
    return Generator(PCG64(SEED))


@pytest.fixture()
def fx_bg() -> PCG64:
    return PCG64(SEED)


@pytest.fixture
def hadamardpattern():
    circ = graphix.transpiler.Circuit(1)
    circ.h(0)
    return circ.transpile().pattern


@pytest.fixture
def nqb(fx_rng: Generator):
    return fx_rng.integers(2, 5)


@pytest.fixture
def depth():
    return 1


@pytest.fixture
def rand_circ(nqb, depth, fx_rng: Generator):
    return tests.random_circuit.get_rand_circuit(nqb, depth, fx_rng)


@pytest.fixture
def randpattern(rand_circ):
    return rand_circ.transpile().pattern
