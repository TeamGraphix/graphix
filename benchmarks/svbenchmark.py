# %%
from copy import deepcopy
from time import perf_counter

import cotengra as ctg
import quimb as qu
from graphix.random_circuit import get_rand_circuit

# %%
GLOBAL_SEED = 2
qu.core._NUM_THREAD_WORKERS = 1

# %%

# NOTE: to increase n_qubit to more than 11, increase the max_qubit_num argument in the StatevectorBackend class
n_qubit_list = list(range(2, 21))

sv_sim = []
eco_sim = []
eco_sim_wo_ctg = []
max_space_ls = []

for n_qubit in n_qubit_list:
    circuit = get_rand_circuit(n_qubit, n_qubit)
    pattern = circuit.transpile()

    pattern.standardize()
    pat_original = deepcopy(pattern)
    pat_original.minimize_space()
    max_sp = pat_original.max_space()
    max_space_ls.append(max_sp)
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
    del tn_sv

    # tn without cotengra optimizer
    start = perf_counter()
    tn_sv = tn.to_statevector(
        backend="numpy",
        skip=False,
        optimize=None,
    )
    end = perf_counter()
    eco_sim_wo_ctg.append(end - start)
    del tn_sv, tn

# %%
# write results into a file
# with open("sqrqcresults.txt", "w") as f:
#     f.write("n_qubit, sv_sim, eco_sim, eco_sim_wo_ctg, max_space_ls\n")
#     for i in range(len(n_qubit_list)):
#         f.write(
#             f"{n_qubit_list[i]}, {sv_sim[i]}, {eco_sim[i]}, {eco_sim_wo_ctg[i]}, {max_space_ls[i]}\n"
#         )
