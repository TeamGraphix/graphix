from graphix.ops import States, Ops
import numpy as np
from graphix.transpiler import Circuit


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
pat.optimize_pattern()
out_state = pat.simulate_pattern()
print('MBQC sampling result: ', out_state.sample_counts(1000,))

# statevector sim
state = States.zplus_state.copy()
state = state.tensor(States.zplus_state)
state = state.evolve(Ops.Rx(theta), [0])
state = state.evolve(Ops.Rx(theta), [1])
print('statevector sim sampling result: ', state.sample_counts(1000))
print('overlap of states: ', np.abs(np.dot(state.data.conjugate(), out_state.data)))
