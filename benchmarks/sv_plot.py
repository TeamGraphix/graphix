# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
data = np.loadtxt("sqrqcresults.txt", delimiter=",", skiprows=1)
n_qubits = data[:, 0].astype(int)
sv_sim = data[:, 1].astype(float)
eco_sim = data[:, 2].astype(float)
eco_sim_wo_ctg = data[:, 3].astype(float)
max_sp = data[:, 4].astype(int)

# %%

# plot with pauli data
fig, ax1 = plt.subplots(figsize=(8, 5))
color = "tab:red"
ax1.set_xlabel("Original Circuit Size [qubit]")
ax1.set_ylabel("Simulation time [sec]")
ax1.set_yscale("log")
ax1.scatter(n_qubits, sv_sim, marker="x", label="MBQC Statevector(minimizing sp)")
ax1.scatter(n_qubits, eco_sim, marker="x", label="MBQC TN base(with cotengra)")
ax1.scatter(n_qubits, eco_sim_wo_ctg, marker="x", label="MBQC TN base(w/o cotengra)")
ax1.tick_params(axis="y")
ax1.legend(loc="upper left")
ax1.set_title("Simulation time(Square RQC)")
plt.grid(True, which="Major")

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color = "tab:blue"
ax2.set_ylabel("Max Space [qubit]")  # We already handled the x-label with ax1
ax2.plot(n_qubits, max_sp, color=color, linestyle="--", label="Max Space")
ax2.tick_params(axis="y")
ax2.legend(loc="lower right")
plt.savefig("simulation_time_wo_p.png")
# plt.close()
