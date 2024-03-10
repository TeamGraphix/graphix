"""
Simulating noisy MBQC
=====================

:class:`~graphix.density_matrix.DensityMatrix` class (through :class:`~graphix.simulator.PatternSimulator`) 
allows the simulation of MBQC with customizable noise model.

In this example, we simulate a simple MBQC pattern with various noise models to see their effects.
First, let us import relevant modules and define a pattern
"""
# %%
import numpy as np
from graphix import Circuit

circuit = Circuit(2)
theta = np.random.rand(2)
circuit.rz(0, theta[0])
circuit.rz(1, theta[1])
circuit.cnot(0,1)

# %%
# Now we transpile into measurement pattern using :meth:`~graphix.transpiler.Circuit.transpile` method.
# This returns :class:`~graphix.pattern.Pattern` object containing measurement pattern:

pattern = circuit.transpile()
pattern.print_pattern(lim=30)
# pattern.draw_graph()

# %%
# simulate with statevector backend:
out_state = pattern.simulate_pattern(backend="statevector")
print(out_state.flatten())

# %%
# Now let us define a noise model. We specify Kraus channels for each of the command executions.
# Here, we apply dephasing noise to the qubit preparation.
from graphix.noise_models.noiseless_noise_model import NoiselessNoiseModel
from graphix.noise_models.noise_model import NoiseModel
from graphix.channels import (
    KrausChannel,
    dephasing_channel,
)

class NoisyGraphState(NoiseModel):

    def __init__(self, p_z=0.1):
        self.p_z = p_z

    def prepare_qubit(self):
        """return the channel to apply after clean single-qubit preparation. Here just identity."""
        return dephasing_channel(self.p_z)

    def entangle(self):
        """return noise model to qubits that happens after the CZ gate"""
        return KrausChannel([{"coef": 1.0, "operator": np.eye(4)}])

    def measure(self):
        """apply noise to qubit to be measured."""
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def confuse_result(self, cmd):
        """assign wrong measurement result
        cmd = "M"
        """
        pass

    def byproduct_x(self):
        """apply noise to qubits after X gate correction"""
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def byproduct_z(self):
        """apply noise to qubits after Z gate correction"""
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def clifford(self):
        """apply noise to qubits that happens in the Clifford gate process"""
        # TODO list separate different Cliffords to allow customization
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def tick_clock(self):
        """notion of time in real devices - this is where we apply effect of T1 and T2.
        we assume commands that lie between 'T' commands run simultaneously on the device.
        """
        pass

#%%
# simulate with the noise model
from graphix.simulator import PatternSimulator

simulator = PatternSimulator(pattern, backend="densitymatrix", noise_model=NoisyGraphState(p_z=0.01))
dm_result = simulator.run()
print(dm_result.fidelity(out_state.psi.flatten()))

# %%
import matplotlib.pyplot as plt

err_arr = np.logspace(-4, -1, 10)
fidelity = np.zeros(10)
for i in range(10):
    simulator = PatternSimulator(pattern, backend="densitymatrix", noise_model=NoisyGraphState(p_z=err_arr[i]))
    dm_result = simulator.run()
    fidelity[i] = dm_result.fidelity(out_state.psi.flatten())

plt.semilogx(err_arr, fidelity, "o:")
plt.xlabel("p_z for state prep and entanglement operations")
plt.ylabel("Final fidelity")
plt.show()
# %%
