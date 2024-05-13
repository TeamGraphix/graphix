from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np

from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.random import Generator


def first_rotation(circuit: Circuit, nqubits: int, rng: Generator) -> None:
    for qubit in range(nqubits):
        circuit.rx(qubit, rng.random())


def mid_rotation(circuit: Circuit, nqubits: int, rng: Generator) -> None:
    for qubit in range(nqubits):
        circuit.rx(qubit, rng.random())
        circuit.rz(qubit, rng.random())


def last_rotation(circuit: Circuit, nqubits: int, rng: Generator) -> None:
    for qubit in range(nqubits):
        circuit.rz(qubit, rng.random())


def entangler(circuit: Circuit, pairs: Iterable[tuple[int, int]]) -> None:
    for a, b in pairs:
        circuit.cnot(a, b)


def entangler_rzz(circuit: Circuit, pairs: Iterable[tuple[int, int]], rng: Generator) -> None:
    for a, b in pairs:
        circuit.rzz(a, b, rng.random())


def generate_gate(
    nqubits: int,
    depth: int,
    pairs: Iterable[tuple[int, int]],
    rng: Generator,
    *,
    use_rzz: bool = False,
) -> Circuit:
    circuit = Circuit(nqubits)
    first_rotation(circuit, nqubits, rng)
    entangler(circuit, pairs)
    for _ in range(depth - 1):
        mid_rotation(circuit, nqubits, rng)
        if use_rzz:
            entangler_rzz(circuit, pairs, rng)
        else:
            entangler(circuit, pairs)
    last_rotation(circuit, nqubits, rng)
    return circuit


def genpair(n_qubits: int, count: int, rng: Generator) -> Iterator[tuple[int, int]]:
    choice = list(range(n_qubits))
    for _ in range(count):
        rng.shuffle(choice)
        x, y = choice[:2]
        yield (x, y)


def gentriplet(n_qubits: int, count: int, rng: Generator) -> Iterator[tuple[int, int, int]]:
    choice = list(range(n_qubits))
    for _ in range(count):
        rng.shuffle(choice)
        x, y, z = choice[:3]
        yield (x, y, z)


def get_rand_circuit(
    nqubits: int,
    depth: int,
    rng: Generator,
    *,
    use_rzz: bool = False,
    use_ccx: bool = False,
) -> Circuit:
    circuit = Circuit(nqubits)
    gate_choice = (
        functools.partial(circuit.ry, angle=np.pi / 4),
        functools.partial(circuit.rz, angle=-np.pi / 4),
        functools.partial(circuit.rx, angle=-np.pi / 4),
        circuit.h,
        circuit.s,
        circuit.x,
        circuit.z,
        circuit.y,
    )
    for _ in range(depth):
        for j, k in genpair(nqubits, 2, rng):
            circuit.cnot(j, k)
        if use_rzz:
            for j, k in genpair(nqubits, 2, rng):
                circuit.rzz(j, k, np.pi / 4)
        if use_ccx:
            for j, k, l in gentriplet(nqubits, 2, rng):
                circuit.ccx(j, k, l)
        for j, k in genpair(nqubits, 4, rng):
            circuit.swap(j, k)
        for j in range(nqubits):
            rng.choice(gate_choice)(j)
    return circuit
