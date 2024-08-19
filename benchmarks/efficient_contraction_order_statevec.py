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
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import cotengra as ctg
import numpy as np
import numpy.typing as npt
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


def benchmark(n_qubit_list: list) -> tuple[npt.NDArray, npt.NDArray]:
    sv_sim = []
    eco_sim = []
    for n_qubit in n_qubit_list:
        single_iter(n_qubit, sv_sim, eco_sim)
    return np.array(sv_sim), np.array(eco_sim)


# %%
# Here, we calculate the simulation multiple times so that we can make error bars.
n_qubit_list = list(range(2, 25))
sv_sim_results: list[npt.NDArray] = []
eco_sim_results: list[npt.NDArray] = []

for _ in range(6):
    sv_sim, eco_sim = benchmark(n_qubit_list)
    sv_sim_results.append(sv_sim)
    eco_sim_results.append(eco_sim)


sv_sim_means = np.mean(sv_sim_results, axis=0)
eco_sim_means = np.mean(eco_sim_results, axis=0)
sv_sim_std = np.std(sv_sim_results, axis=0)
eco_sim_std = np.std(eco_sim_results, axis=0)

# %%
# Finally, we save the results to a text file.
with open("results.txt", "w") as f:
    f.write("n_qubit, sv_sim_mean, sv_sim_std, eco_sim_mean, eco_sim_std\n")
    for i in range(len(n_qubit_list)):
        f.write(f"{n_qubit_list[i]}, {sv_sim_means[i]}, {sv_sim_std[i]}, {eco_sim_means[i]}, {eco_sim_std[i]}\n")

# %%
# We can now plot the simulation time results.
import matplotlib.pyplot as plt  # noqa: E402

data = np.loadtxt("results.txt", delimiter=",", skiprows=1)
n_qubits = data[:, 0].astype(int)
sv_sim_mean = data[:16, 1].astype(float)
sv_sim_std = data[:16, 2].astype(float)
eco_sim_mean = data[:, 3].astype(float)
eco_sim_std = data[:, 4].astype(float)

fig, ax = plt.subplots(figsize=(8, 5))
color = "tab:red"
ax.set_xlabel("Original Circuit Size [qubit]", fontsize=20)
ax.set_ylabel("Simulation time [sec]", fontsize=20)
ax.set_yscale("log")
ax.errorbar(n_qubits[:16], sv_sim_mean, yerr=sv_sim_std, color=color, fmt="o", label="MBQC Statevector (minimizing sp)")
ax.errorbar(n_qubits, eco_sim_mean, yerr=eco_sim_std, fmt="x", label="MBQC TN base")
ax.tick_params(axis="y")
ax.legend(loc="upper left")
ax.set_title("Simulation time", fontsize=20)
plt.grid(True, which="Major")
plt.rcParams["svg.fonttype"] = "none"
plt.savefig("simulation_time.png")
plt.close()

# %%
