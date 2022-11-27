Graphix demo
============

QFT
---

Three-qubit QFT circuit

.. code-block:: python

    from graphix.transpiler import Circuit
    import numpy as np

    def cp(circuit, theta, control, target):
        """Controlled rotation gate, decomposed
        """
        circuit.rz(control, theta / 2)
        circuit.rz(target, theta / 2)
        circuit.cnot(control, target)
        circuit.rz(target, -1 * theta / 2)
        circuit.cnot(control, target)


    def swap(circuit, a, b):
        """swap gate
        """
        circuit.cnot(a, b)
        circuit.cnot(b, a)
        circuit.cnot(a, b)


    circuit = Circuit(3)

    # prepare all states in |0>
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)

    # prepare input state for QFT
    circuit.x(1)
    circuit.x(2)

    # QFT
    circuit.h(2)
    cp(circuit, np.pi / 4, 0, 2)
    cp(circuit, np.pi / 2, 1, 2)
    circuit.h(1)
    cp(circuit, np.pi / 2, 0, 1)
    circuit.h(0)
    swap(circuit, 0, 2)

    # run with MBQC simulator
    pat = circuit.transpile()
    pat.standardize()
    pat.shift_signals()
    pat.perform_pauli_measurements()
    pat.minimize_space()
    out_state = pat.simulate_pattern()

    state = circuit.simulate_statevector()
    print('overlap of states: ', np.abs(np.dot(state.data.conjugate(), out_state.data)))

QAOA
----

QAOA circuit for max-cut of complete graph with four nodes

.. code-block:: python

    from graphix.transpiler import Circuit
    import networkx as nx
    import numpy as np
    n=4
    xi = np.random.rand(6)
    theta = np.random.rand(4)
    g = nx.complete_graph(n)
    circuit = Circuit(n)
    for i, (u,v) in enumerate(g.edges):
        circuit.cnot(u, v)
        circuit.rz(v, xi[i])
        circuit.cnot(u, v)
    for v in g.nodes:
        circuit.rx(v, theta[v])
    pattern = circuit.transpile()
    pattern.standardize()
    pattern.shift_signals()
    pattern.perform_pauli_measurements()
    pat.minimize_space()
    out_state = pattern.simulate_pattern()
