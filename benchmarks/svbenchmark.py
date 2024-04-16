# %%
import numpy as np
import cotengra as ctg
import quimb as qu
from copy import deepcopy
from time import perf_counter
from graphix.transpiler import Circuit

# %%
GLOBAL_SEED = 2
qu.core._NUM_THREAD_WORKERS = 1


def get_rng(seed=None):
    if seed is not None:
        return np.random.default_rng(seed)
    elif seed is None and GLOBAL_SEED is not None:
        return np.random.default_rng(GLOBAL_SEED)
    else:
        return np.random.default_rng()


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
            k = rng.choice(gate_choice)
            if k == 0:
                circuit.ry(j, np.pi / 4)
            elif k == 1:
                circuit.rz(j, -np.pi / 4)
            elif k == 2:
                circuit.rx(j, -np.pi / 4)
            elif k == 3:
                circuit.h(j)
            elif k == 4:
                circuit.s(j)
            elif k == 5:
                circuit.x(j)
            elif k == 6:
                circuit.z(j)
            elif k == 7:
                circuit.y(j)
            else:
                pass
    return circuit


# %%

# NOTE: to increase n_qubit to more than 11, increase the max_qubit_num argument in the StatevectorBackend class
n_qubit_list = list(range(2, 10))


circ_sim = []
sv_sim = []
eco_sim = []
max_space_ls = []

for n_qubit in n_qubit_list:
    circuit = get_rand_circuit(n_qubit, n_qubit)
    pattern = circuit.transpile()

    start = perf_counter()
    circ_sv = circuit.simulate_statevector()
    end = perf_counter()
    circ_sim.append(end - start)
    del circ_sv

    pattern.standardize()
    pat_original = deepcopy(pattern)
    pat_original.minimize_space()
    max_sp = pat_original.max_space()
    max_space_ls.append(max_sp)
    if max_sp >= 20:
        sv_sim.append(None)
    else:
        start = perf_counter()
        sv = pat_original.simulate_pattern()
        end = perf_counter()
        sv_sim.append(end - start)
        del sv

    # tn
    tn = pattern.simulate_pattern("tensornetwork")
    output_inds = [tn._dangling[str(index)] for index in tn.default_output_nodes]

    start = perf_counter()
    tn_sv = tn.to_statevector(
        backend="numpy",
        skip=False,
        optimize=ctg.HyperOptimizer(minimize="combo", max_time=600, progbar=True),
    )
    end = perf_counter()
    eco_sim.append(end - start)
    del tn_sv, tn
