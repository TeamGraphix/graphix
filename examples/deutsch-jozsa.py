"""
Preprocessing Clifford gates
============================

In this example, we implement the Deutsch-Jozsa algorithm which determines whether
a function is *balanced* or *constant*.
Since this algorithm is written only with Clifford gates, we can expect the preprocessing of Clifford gates
would significantly improve the MBQC pattern simulation.
You can find nice description of the algorithm `here <https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm>`_.

You can run this code on your browser with `mybinder.org <https://mybinder.org/>`_ - click the badge below.


.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=deutsch-jozsa.ipynb

First, let us import relevant modules:
"""

import matplotlib.pyplot as plt
import networkx as nx

# %%
import numpy as np

from graphix import Circuit

# %%
# Now we implement the algorithm with quantum circuit, which we can transpile into MBQC.
# As an example, we look at balanced oracle for 4 qubits.

circuit = Circuit(4)

# prepare all qubits in |0> for easier comparison with original algorithm
for i in range(4):
    circuit.h(i)

# initialization
circuit.h(0)
circuit.h(1)
circuit.h(2)

# prepare ancilla
circuit.x(3)
circuit.h(3)

# balanced oracle - flip qubits 0 and 2
circuit.x(0)
circuit.x(2)

# algorithm
circuit.cnot(0, 3)
circuit.cnot(1, 3)
circuit.cnot(2, 3)

circuit.x(0)
circuit.x(2)

circuit.h(0)
circuit.h(1)
circuit.h(2)

# %%
# Now let us transpile into MBQC measurement pattern and inspect the pattern sequence and graph state

pattern = circuit.transpile().pattern
pattern.print_pattern(lim=15)
pattern.draw_graph(flow_from_pattern=False)

# %%
# this seems to require quite a large graph state.
# However, we know that Pauli measurements can be preprocessed with graph state simulator.
# To do so, let us first standardize and shift signals, so that measurements are less interdependent.

pattern.standardize()
pattern.shift_signals()
pattern.print_pattern(lim=15)

# %%
# Now we preprocess all Pauli measurements

pattern.perform_pauli_measurements()
pattern.print_pattern(lim=16, filter=["N", "M", "C"])
pattern.draw_graph(flow_from_pattern=True)

# %%
# Since all operations of the original circuit are Clifford, all measurements in the measurement pattern are Pauli measurements:
# So the preprocessing has done all the necessary computations, and all nodes are isolated with no further measurements required.
# Let us make sure the result is correct:

out_state = pattern.simulate_pattern()
state = circuit.simulate_statevector().statevec
print("overlap of states: ", np.abs(np.dot(state.psi.flatten().conjugate(), out_state.psi.flatten())))

# %%
