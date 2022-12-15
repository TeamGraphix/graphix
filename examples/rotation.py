from graphix.ops import Ops
import numpy as np
from graphix.transpiler import Circuit
from graphix.sim.statevec import Statevec

circuit = Circuit(2)

# initialize all states in |0>, not |+>
circuit.h(1)
circuit.h(0)

# rot gates
theta = -0.4 * np.pi
circuit.rx(1, theta)
circuit.rx(0, theta)

# run with MBQC simulator
pat = circuit.transpile()
pat.standardize()
pat.shift_signals()
pat.perform_pauli_measurements()
pat.minimize_space()
out_state = pat.simulate_pattern()

# statevector sim
state = Statevec(nqubit=2, plus_states=False)  # starts with |0> states
state.evolve_single(Ops.Rx(theta), 0)
state.evolve_single(Ops.Rx(theta), 1)
print("overlap of states: ", np.abs(np.dot(state.psi.flatten().conjugate(), out_state.psi.flatten())))
