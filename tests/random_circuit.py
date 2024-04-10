from copy import deepcopy

import numpy as np

from graphix.transpiler import Circuit

GLOBAL_SEED = None


def set_seed(seed):
    global GLOBAL_SEED
    GLOBAL_SEED = seed


def get_rng(seed=None):
    if seed is not None:
        return np.random.default_rng(seed)
    elif seed is None and GLOBAL_SEED is not None:
        return np.random.default_rng(GLOBAL_SEED)
    else:
        return np.random.default_rng()


def first_rotation(circuit, nqubits, rng):
    for qubit in range(nqubits):
        circuit.rx(qubit, rng.random())


def mid_rotation(circuit, nqubits, rng):
    for qubit in range(nqubits):
        circuit.rx(qubit, rng.random())
        circuit.rz(qubit, rng.random())


def last_rotation(circuit, nqubits, rng):
    for qubit in range(nqubits):
        circuit.rz(qubit, rng.random())


def entangler(circuit, pairs):
    for a, b in pairs:
        circuit.cnot(a, b)


def entangler_rzz(circuit, pairs, rng):
    for a, b in pairs:
        circuit.rzz(a, b, rng.random())


def generate_gate(nqubits, depth, pairs, use_rzz=False, seed=None):
    rng = get_rng(seed)
    circuit = Circuit(nqubits)
    first_rotation(circuit, nqubits, rng)
    entangler(circuit, pairs)
    for k in range(depth - 1):
        mid_rotation(circuit, nqubits, rng)
        if use_rzz:
            entangler_rzz(circuit, pairs, rng)
        else:
            entangler(circuit, pairs)
    last_rotation(circuit, nqubits, rng)
    return circuit


def genpair(n_qubits, count, rng):
    pairs = []
    for i in range(count):
        choice = [j for j in range(n_qubits)]
        x = rng.choice(choice)
        choice.pop(x)
        y = rng.choice(choice)
        pairs.append((x, y))
    return pairs


def gentriplet(n_qubits, count, rng):
    triplets = []
    for i in range(count):
        choice = [j for j in range(n_qubits)]
        x = rng.choice(choice)
        choice.pop(x)
        y = rng.choice(choice)
        locy = np.where(y == np.array(deepcopy(choice)))[0][0]
        choice.pop(locy)
        z = rng.choice(choice)
        triplets.append((x, y, z))
    return triplets


def get_rand_circuit(nqubits, depth, use_rzz=False, use_ccx=False, seed=None):
    rng = get_rng(seed)
    circuit = Circuit(nqubits)
    gate_choice = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(depth):
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
            match rng.choice(gate_choice):
                case 0:
                    circuit.ry(j, np.pi / 4)
                case 1:
                    circuit.rz(j, -np.pi / 4)
                case 2:
                    circuit.rx(j, -np.pi / 4)
                case 3:
                    circuit.h(j)
                case 4:
                    circuit.s(j)
                case 5:
                    circuit.x(j)
                case 6:
                    circuit.z(j)
                case 7:
                    circuit.y(j)
                case _:
                    pass
    return circuit
