"""
Efficient contraction order statevector simulation of MBQC patterns
===================================================================

Here we benchmark our efficient contraction order statevector simulator for MBQC,
which uses the TensorNetwork backend.

The methods and modules we use are the followings:
    1. :meth:`graphix.pattern.Pattern.simulate_pattern`
        Pattern simulation with Statevector backend.
        :meth:`graphix.pattern.Pattern.minimize_space` locally minimizes the space of the pattern.
    2. :mod:`graphix.sim.tensornet`
        Pattern simulation with TensorNetwork backend.
        This enables the efficient contraction order simulation of the pattern.
        Here we use the `cotengra` optimizer for the contraction order optimization.
"""

# %%
# Firstly, let us import relevant modules:

from copy import deepcopy
from time import perf_counter

import cotengra as ctg
import quimb as qu
from numpy.random import PCG64, Generator

import graphix
from graphix.random_objects import get_rand_circuit

# %%
# Next, set global seed and number of thread workers
GLOBAL_SEED = 2
qu.core._NUM_THREAD_WORKERS = 1

# %%
# We then run simulations.
# Let's benchmark the simulation time of the statevector simulator and the efficient contraction order simulator.

n_qubit_list = list(range(2, 25))


def prepare_pattern(n_qubit: int) -> tuple[graphix.pattern.Pattern, graphix.pattern.Pattern]:
    rng = Generator(PCG64(GLOBAL_SEED))
    circuit = get_rand_circuit(n_qubit, n_qubit, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pat_original = deepcopy(pattern)
    pattern.standardize()
    pat_original = deepcopy(pattern)
    return pattern, pat_original


def statevector_sim(n_qubit: int, sv_sim: list, pat_original: graphix.pattern.Pattern) -> None:
    if n_qubit > 17:
        sv_sim.append(-1)
    else:
        start = perf_counter()
        pat_original.minimize_space()
        max_sp = pat_original.max_space()
        sv = pat_original.simulate_pattern(max_qubit_num=max_sp)  # noqa: F841
        end = perf_counter()
        sv_sim.append(end - start)
        del sv


def efficient_contraction_order_sim(eco_sim: list, pattern: graphix.pattern.Pattern) -> None:
    tn = pattern.simulate_pattern("tensornetwork")
    start = perf_counter()
    tn_sv = tn.to_statevector(
        backend="numpy",
        optimize=ctg.HyperOptimizer(minimize="combo", max_time=600, progbar=True),
    )
    end = perf_counter()
    eco_sim.append(end - start)
    del tn_sv, tn


def single_iter(n_qubit: int, sv_sim: list, eco_sim: list) -> None:
    pattern, pat_original = prepare_pattern(n_qubit)
    statevector_sim(n_qubit, sv_sim, pat_original)
    efficient_contraction_order_sim(eco_sim, pattern)


def benchmark(n_qubit_list: list) -> tuple[list, list]:
    sv_sim = []
    eco_sim = []
    for n_qubit in n_qubit_list:
        single_iter(n_qubit, sv_sim, eco_sim)
    return sv_sim, eco_sim


sv_sim, eco_sim = benchmark(n_qubit_list)


# %%
# Finally, we save the results to a text file.
with open("results.txt", "w") as f:
    f.write("n_qubit, sv_sim, eco_sim\n")
    for i in range(len(n_qubit_list)):
        f.write(f"{n_qubit_list[i]}, {sv_sim[i]}, {eco_sim[i]}\n")

# %%
# We can now plot the simulation time results.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

data = np.loadtxt("results.txt", delimiter=",", skiprows=1)
n_qubits = data[:, 0].astype(int)
sv_sim = data[:16, 1].astype(float)
eco_sim = data[:, 2].astype(float)

fig, ax = plt.subplots(figsize=(8, 5))
color = "tab:red"
ax.set_xlabel("Original Circuit Size [qubit]")
ax.set_ylabel("Simulation time [sec]")
ax.set_yscale("log")
ax.scatter(n_qubits[:16], sv_sim, marker="x", label="MBQC Statevector (minimizing sp)")
ax.scatter(n_qubits, eco_sim, marker="x", label="MBQC TN base")
ax.tick_params(axis="y")
ax.legend(loc="upper left")
ax.set_title("Simulation time")
plt.grid(True, which="Major")
plt.rcParams["svg.fonttype"] = "none"
plt.savefig("simulation_time.png")
plt.close()

# %%
