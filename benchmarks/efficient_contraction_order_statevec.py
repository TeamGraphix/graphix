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

from graphix.random_objects import get_rand_circuit

# %%
# Next, set global seed and number of thread workers
GLOBAL_SEED = 2
qu.core._NUM_THREAD_WORKERS = 1

# %%
# We then run simulations.
# Let's benchmark the simulation time of the statevector simulator and the efficient contraction order simulator.

n_qubit_list = list(range(2, 31))

sv_sim = []
neco_sim = []
eco_sim = []
eco_sim_wo_ctg = []
max_space_ls = []

for n_qubit in n_qubit_list:
    rng = Generator(PCG64(GLOBAL_SEED))
    circuit = get_rand_circuit(n_qubit, n_qubit, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pat_original = deepcopy(pattern)

    # statevector simulation: sv
    if n_qubit > 17:
        sv_sim.append(-1)
        max_space_ls.append(-1)
    else:
        start = perf_counter()
        pat_original.minimize_space()
        max_sp = pat_original.max_space()
        max_space_ls.append(max_sp)
        sv = pat_original.simulate_pattern(max_qubit_num=max_sp)
        end = perf_counter()
        sv_sim.append(end - start)
        del sv

    # no-contraction order search simulation: tn
    tn = pattern.simulate_pattern("tensornetwork")
    output_inds = [tn._dangling[str(index)] for index in tn.default_output_nodes]

    start = perf_counter()
    tn_sv = tn.to_statevector(backend="numpy", skip_tn_simp=True, optimize=None)
    end = perf_counter()
    neco_sim.append(end - start)
    del tn_sv

    # efficient contraction order simulation (eco-sim): tn
    tn = pattern.simulate_pattern("tensornetwork")
    output_inds = [tn._dangling[str(index)] for index in tn.default_output_nodes]

    start = perf_counter()
    tn_sv = tn.to_statevector(
        backend="numpy",
        skip_tn_simp=False,
        optimize=ctg.HyperOptimizer(minimize="combo", max_time=600, progbar=True),
    )
    end = perf_counter()
    eco_sim.append(end - start)
    del tn_sv

    # eco-sim: tn without cotengra optimizer
    start = perf_counter()
    tn_sv = tn.to_statevector(
        backend="numpy",
        skip_tn_simp=False,
        optimize=None,
    )
    end = perf_counter()
    eco_sim_wo_ctg.append(end - start)
    del tn_sv, tn

# %%
# Finally, we save the results to a text file.
with open("sqrqcresults.txt", "w") as f:
    f.write("n_qubit, neco_sim, sv_sim, eco_sim, eco_sim_wo_ctg, max_space_ls\n")
    for i in range(len(n_qubit_list)):
        f.write(
            f"{n_qubit_list[i]}, {sv_sim[i]}, {neco_sim[i]}, {eco_sim[i]}, {eco_sim_wo_ctg[i]}, {max_space_ls[i]}\n"
        )

# %%
# We can now plot the simulation time results.
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("sqrqcresults.txt", delimiter=",", skiprows=1)
n_qubits = data[:, 0].astype(int)
sv_sim = data[:16, 1].astype(float)
neco_sim = data[:, 2].astype(float)
eco_sim = data[:, 3].astype(float)
eco_sim_wo_ctg = data[:, 4].astype(float)
max_sp = data[:16, 5].astype(int)

fig, ax1 = plt.subplots(figsize=(8, 5))
color = "tab:red"
ax1.set_xlabel("Original Circuit Size [qubit]")
ax1.set_ylabel("Simulation time [sec]")
ax1.set_yscale("log")
ax1.scatter(n_qubits[:16], sv_sim, marker="x", label="MBQC Statevector (minimizing sp)")
ax1.scatter(n_qubits, neco_sim, marker="x", label="MBQC TN base (contraction skipped)")
ax1.scatter(n_qubits, eco_sim, marker="x", label="MBQC TN base (with cotengra)")
ax1.scatter(n_qubits, eco_sim_wo_ctg, marker="x", label="MBQC TN base (without cotengra)")
ax1.tick_params(axis="y")
ax1.legend(loc="upper left")
ax1.set_title("Simulation time (Square RQC)")
plt.grid(True, which="Major")

ax2 = ax1.twinx()
color = "tab:blue"
ax2.set_ylabel("Max Space [qubit]")
ax2.plot(n_qubits[:16], max_sp, color=color, linestyle="--", label="Max Space")
ax2.tick_params(axis="y")
ax2.legend(loc="lower right")

plt.rcParams["svg.fonttype"] = "none"
plt.savefig("simulation_time_wo_p.png")
plt.close()

# %%
