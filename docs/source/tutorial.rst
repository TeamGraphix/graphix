Tutorial
========

Graphix provides a high-level interface to generate, optimize and classically simulate the measurement-based quantum computing (MBQC).

In this tutorial, we look at how to program MBQC using graphix library.
We will explain the basics here along with the code, and you can go to :doc:`intro` to learn more about the theoretical background of MBQC and :doc:`references` for module references.

Generating measurement patterns
-------------------------------

First, install ``graphix`` by

>>> pip install graphix

Graphix is centered around the measurement `pattern`, which is a sequence of commands such as qubit preparattion, entanglement and single-qubit measurement commands.
The most basic measurement pattern is that for realizing Hadamard gate, which we will use to see how `graphix` works.

For any gate network, we can use :class:`~graphix.transpiler.Circuit` class to generate the measurement pattern to realize the unitary evolution.

.. code-block:: python

    from graphix.transpiler import Circuit
    # apply H gate to a qubit in + state
    circuit = Circuit(1)
    circuit.h(0)
    pattern = circuit.transpile()

the :class:`~graphix.pattern.Pattern` object contains the sequence of commands according to the measurement calculus framework [1].

>>> pattern.print_pattern() # show the command sequence (pattern)
N, node=0
N, node=1
E, nodes=(0, 1)
M, node=0, plane=XY, angle(pi)=0, s-domain=[], t_domain=[]
X byproduct, node=1, domain=[0]

The command sequence represents the following sequence:

    * starting with an input qubit :math:`|\psi_{in}\rangle_0`, we first prepare an ancilla qubit :math:`|+\rangle_1` with ['N', 1] command
    * We then apply CZ-gate by ['E', (0, 1)] command to create entanglement.
    * We measure the qubit 0 in Pauli X basis, by ['M'] command.
    * If the measurement outcome is :math:`s_0 = 1` (i.e. if the qubit is projected to :math:`|-\rangle`, the Pauli X eigenstate with eigenvalue of :math:`(-1)^{s_0} = -1`), the 'X' command is applied to qubit 1 to 'correct' the measurement byproduct (see :doc:`intro`) that ensure deterministic computation.
    * Tracing out the qubit 0 (since the measurement is destructive), we have :math:`H|\psi_{in}\rangle_1` - the input qubit has teleported to qubit 1, while being transformed by Hadamard gate.

This MBQC pattern seqeunce uses a resource state as shown below:

.. figure:: ./../imgs/graphH.png
   :scale: 100 %
   :alt: resource state to perform computation

This is the simplest of the `graph state
<https://en.wikipedia.org/wiki/Graph_state>`_ with nodes = [0, 1] and edge = (0, 1). Any MBQC pattern has a corresponding resource graph state on which the computation occurs only with single-qubit measurements.

We can use the :class:`~graphix.simulator.PatternSimulator` to classically simulate the pattern above and obtain the output state, for default input state of :math:`|+\rangle`.
Alternatively, we can simply call :meth:`~graphix.pattern.Pattern.simulate_pattern` of :class:`~graphix.pattern.Pattern` object to do it in one line:

>>> pattern.simulate_pattern(backend='statevector')
statevector([0, 1])

Note again that we started with :math:`|+\rangle` state so the answer is correct.

Universal gatesets
------------------

As a more complex example than above, we show measurement patterns and graph states for CNOT and single-qubit general rotation which makes MBQC universal:

+----------------------------------------------------------------------+----------------------------------------------------------------------------+
| CNOT                                                                 |   general rotation (an example with Euler angles 0.2pi, 0.15pi and 0.1 pi) |
+======================================================================+============================================================================+
|.. figure:: ./../imgs/graph_cnot.png                                  |.. figure:: ./../imgs/graph_rot.png                                         |
|   :scale: 100 %                                                      |   :scale: 100 %                                                            |
|   :alt: resource state                                               |   :alt: resource state                                                     |
|                                                                      |                                                                            |
|   control: input=0, output=0; target: input=1, output=3              |   input = 0, output = 4                                                    |
+----------------------------------------------------------------------+----------------------------------------------------------------------------+
| >>> cnot_pattern.print_pattern()                                     | >>> euler_rot_pattern.print_pattern()                                      |
| N, node = 0                                                          | N, node = 0                                                                |
| N, node = 1                                                          | N, node = 1                                                                |
| N, node = 2                                                          | N, node = 2                                                                |
| N, node = 3                                                          | N, node = 3                                                                |
| E, nodes = (1, 2)                                                    | N, node = 4                                                                |
| E, nodes = (0, 2)                                                    | M, node = 0, plane = XY, angle(pi) = -0.2, s-domain = [], t_domain = []    |
| E, nodes = (2, 3)                                                    | M, node = 1, plane = XY, angle(pi) = -0.15, s-domain = [0], t_domain = []  |
| M, node = 1, plane = XY, angle(pi) = 0, s-domain = [], t_domain = [] | M, node = 2, plane = XY, angle(pi) = -0.1, s-domain = [1], t_domain = []   |
| M, node = 2, plane = XY, angle(pi) = 0, s-domain = [], t_domain = [] | M, node = 3, plane = XY, angle(pi) = 0, s-domain = [], t_domain = []       |
| X byproduct, node = 3, domain = [2]                                  | Z byproduct, node = 4, domain = [0,2]                                      |
| Z byproduct, node = 3, domain = [1]                                  | X byproduct, node = 4, domain = [1,3]                                      |
| Z byproduct, node = 0, domain = [1]                                  |                                                                            |
+----------------------------------------------------------------------+----------------------------------------------------------------------------+


We can concatenate these commands to perform any quantum information processing tasks, which we will look at in more detail below.
Of course, we also have many other gates that can be transpiled into MBQC; see :class:`graphix.transpiler.Circuit` class.


Optimizing patterns
-------------------------------
We provide a number of optimization routines to improve the measurement pattern.
As an example, let us prepare a pattern to rotate two qubits in :math:`|+\rangle` with a random angle and entangle them with a CNOT gate:

.. code-block:: python

    from graphix.transpiler import Circuit
    import numpy as np
    circuit = Circuit(2) # initialize with |+> \otimes |+>
    circuit.rz(0, np.random.rand())
    circuit.rz(1, np.random.rand())
    circuit.cnot(0, 1)
    pattern = circuit.transpile()

This produces a rather long and complicated command sequence. As we see below, we can significantly optimize this for better simulation performance and even operations in quantum hardware.

>>> pattern.print_pattern() # show the command sequence (pattern)
N, node = 0
N, node = 1
N, node = 2
N, node = 3
E, nodes = (0, 2)
E, nodes = (2, 3)
M, node = 0, plane = XY, angle(pi) = -0.11530492922405373, s-domain = [], t_domain = []
M, node = 2, plane = XY, angle(pi) = 0, s-domain = [], t_domain = []
X byproduct, node = 3, domain = [2]
Z byproduct, node = 3, domain = [0]
N, node = 4
N, node = 5
E, nodes = (1, 4)
E, nodes = (4, 5)
M, node = 1, plane = XY, angle(pi) = -0.08641619841768139, s-domain = [], t_domain = []
M, node = 4, plane = XY, angle(pi) = 0, s-domain = [], t_domain = []
X byproduct, node = 5, domain = [4]
Z byproduct, node = 5, domain = [1]
N, node = 6
N, node = 7
E, nodes = (5, 6)
E, nodes = (3, 6)
E, nodes = (6, 7)
M, node = 5, plane = XY, angle(pi) = 0, s-domain = [], t_domain = []
M, node = 6, plane = XY, angle(pi) = 0, s-domain = [], t_domain = []
X byproduct, node = 7, domain = [6]
Z byproduct, node = 7, domain = [5]
Z byproduct, node = 3, domain = [5]



Standardization and signal shifting
+++++++++++++++++++++++++++++++++++

The `standard` pattern is a pattern where the commands are sorted in the order of N, E, M, (X, Z, C) where X, Z and C commands in bracket can be in any order but must apply only to output nodes.
Any command sequence has a standard form, which can be obtained by the `standarziation` algorithm in [1] that runs in polynomial time on the number of commands.

An additional `signal shifting` procedure simplifies the dependence structure of the pattern to minimize the feedforward operations.
These can be called with :meth:`~graphix.pattern.Pattern.standardize` and :meth:`~graphix.pattern.Pattern.shift_signals` and result in a simpler pattern sequence.

>>> pattern.standardize()
>>> pattern.shift_signals()
>>> pattern.print_pattern()
N, node = 0
N, node = 1
N, node = 2
N, node = 3
N, node = 4
N, node = 5
N, node = 6
N, node = 7
E, nodes = (6, 7)
E, nodes = (3, 6)
E, nodes = (5, 6)
E, nodes = (4, 5)
E, nodes = (1, 4)
E, nodes = (2, 3)
E, nodes = (0, 2)
M, node = 0, plane = XY, angle(pi) = -0.11530492922405373, s-domain = [], t_domain = []
M, node = 2, plane = XY, angle(pi) = 0, s-domain = [], t_domain = []
M, node = 1, plane = XY, angle(pi) = -0.08641619841768139, s-domain = [], t_domain = []
M, node = 4, plane = XY, angle(pi) = 0, s-domain = [], t_domain = []
M, node = 5, plane = XY, angle(pi) = 0, s-domain = [4], t_domain = []
M, node = 6, plane = XY, angle(pi) = 0, s-domain = [], t_domain = []
Z byproduct, node = 3, domain = [0, 1, 5]
Z byproduct, node = 7, domain = [1, 5]
X byproduct, node = 7, domain = [2, 4, 6]
X byproduct, node = 3, domain = [2]

The command sequence is now much clear and note that the byproduct commands now only apply to output nodes (3, 7).
This reveals the graph structure of the resource state which we can inspect:

.. code-block:: python

    import networkx as nx
    nodes, edges = pattern.get_graph()
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = {0: (0, 0), 1: (0, -0.5), 2: (1, 0), 3: (4, 0), 4: (1, -0.5), 5: (2, -0.5), 6: (3, -0.5), 7: (4, -0.5)}
    graph_params = {'node_size': 240, 'node_color': 'w', 'edgecolors': 'k', 'with_labels': True}
    nx.draw(g, pos=pos, **graph_params)

.. figure:: ./../imgs/graph.png
   :scale: 100 %
   :alt: resource state to perform computation

0 and 1 are the input nodes and 3 and 7 are the output nodes of this graph.

Performing Pauli measurements
+++++++++++++++++++++++++++++

It is known that quantum circuit consisting of Pauli basis states, Clifford gates and Pauli measurements can be simulated classically (see `Gottesman-Knill theorem
<https://en.wikipedia.org/wiki/Gottesman%E2%80%93Knill_theorem>`_; e.g. the graph state simulator runs in :math:`\mathcal{O}(n \log n)` time).
The Pauli measurement part of the MBQC is exactly this, and they can be preprocessed by our graph state simulator :class:`~graphix.graphsim.GraphState` - see :doc:`lc-mbqc` for more detailed description.

We can call this in a line by calling :meth:`~graphix.pattern.Pattern.perform_pauli_measurements()` of :class:`~graphix.pattern.Pattern` object, which acts as the optimization routine of the measurement pattern.
We get an updated measurement pattern without Pauli measurements as follows:

>>> pattern.perform_pauli_measurements()
>>> pattern.print_pattern()
N, node = 0
N, node = 1
N, node = 3
N, node = 7
E, nodes = (0, 3)
E, nodes = (0, 7)
E, nodes = (1, 7)
M, node = 0, plane = XY, angle(pi) = -0.11530492922405373, s-domain = [], t_domain = [], Clifford index = 0
M, node = 1, plane = XY, angle(pi) = -0.08641619841768139, s-domain = [], t_domain = [], Clifford index = 0
Clifford, node = 3, Clifford index = 6
Clifford, node = 7, Clifford index = 6
Z byproduct, node = 3, domain = [0, 1, 5]
Z byproduct, node = 7, domain = [1, 5]
X byproduct, node = 7, domain = [2, 4, 6]
X byproduct, node = 3, domain = [2]

Notice that all measurements with angle=0 (Pauli X measurements) dissapeared - this means that a part of quantum computation was `classically` (and efficiently) preprocessed such that we only need much smaller quantum resource.
The additional Clifford commands, along with byproduct operations, can be dealt with by simply rotating the final readout measurements from the standard Z basis, so there is no downside in doing this preprocessing.

As you can see below, the resource state has shrank significantly (factor of two reduction in the number of nodes), but again we know that they both serve as the quantum resource state for the same quantum computation task as defined above.

+---------------------------------+---------------------------------+
| before                          | after                           |
+=================================+=================================+
|.. figure:: ./../imgs/graph.png  |.. figure:: ./../imgs/graph2.png |
|   :scale: 100 %                 |   :scale: 100 %                 |
|   :alt: resource state          |   :alt: resource state          |
+---------------------------------+---------------------------------+

As we mention in :doc:`intro`, all Clifford gates translates into MBQC only consisting of Pauli measurements. So this procedure is equivalent to classically preprocessing all Clifford operations from quantum algorithms.


Minimizing 'space' of a pattern
+++++++++++++++++++++++++++++++

The `space` of a pettern is the largest number of qubits that must be present in the graph state during the execution of the pattern.
For standard patterns, this is exactly the size of the resource graph state, since we prepare all ancilla qubits at the start of the computation.
However, we do not always need to prepare all qubits at the start; in fact preparing all the adjacent (connected) qubits of the ones that you are about measure, is sufficient to run MBQC.
We exploit this fact to minimize the `space` of the pattern, which is crucial for running statevector simulation of MBQC since they are typically limited by the available computer memory.
We can simply call :meth:`~graphix.pattern.Pattern.minimize_space()` to reduce the `space`:

>>> pattern.minimize_space()
>>> pattern.print_pattern()
N, node = 1
N, node = 7
E, nodes = (1, 7)
M, node = 1, plane = XY, angle(pi) = -0.08641619841768139, s-domain = [], t_domain = [], Clifford index = 0
N, node = 0
N, node = 3
E, nodes = (0, 3)
E, nodes = (0, 7)
M, node = 0, plane = XY, angle(pi) = -0.11530492922405373, s-domain = [], t_domain = [], Clifford index = 0
Clifford, node = 3, Clifford index = 6
Clifford, node = 7, Clifford index = 6
Z byproduct, node = 3, domain = [0, 1, 5]
Z byproduct, node = 7, domain = [1, 5]
X byproduct, node = 7, domain = [2, 4, 6]
X byproduct, node = 3, domain = [2]

With the original measurement pattern, the simulation should have proceeded as follows, with maximum of four qubits on the memory.

.. figure:: ./../imgs/graph_space1.png
   :scale: 100 %
   :alt: simulation order

With the optimization with :meth:`~graphix.pattern.Pattern.minimize_space()`, the simulation proceeds as below, where we measure and trace out qubit 1 before preparing qubits 0 and 3.
Because the graph state only has short-range correlations (only adjacent qubits are entangled), this does not affect the outcome of the computation.
With this, we only need the memory space for three qubits.

.. figure:: ./../imgs/graph_space2.png
   :scale: 100 %
   :alt: simulation order after optimization


This procedure is more effective when the resource state size is large compared to the logical input qubit count;
for example, the three-qubit `quantum Fourier transform (QFT)
<https://en.wikipedia.org/wiki/Quantum_Fourier_transform>`_ circuit requires 12 qubits in the resource state after :meth:`~graphix.pattern.Pattern.perform_pauli_measurements()` (we show the code in :ref:`gallery:qft`); with the proper reordering of the commands, the max_space reduces to 4.
In fact, for patterns transpiled from gate network, the minimum `space` we can realize is typically :math:`n_w+1` where :math:`n_w` is the width of the circuit.


Running pattern on quantum devices
++++++++++++++++++++++++++++++++++

We are currently adding cloud-based quantum devices to run MBQC pattern. Our first such interface is for IBMQ devices, and is available as `graphix-ibmq <https://github.com/TeamGraphix/graphix-ibmq>`_ module.

First, install ``graphix-ibmq`` by

>>> pip install graphix-ibmq

With graphix-ibmq installed, we can turn a measurement pattern into a qiskit dynamic circuit.

.. code-block:: python

    from graphix_ibmq.runner import IBMQBackend

    # minimize space and convert to qiskit circuit
    pattern.minimize_space()
    backend = IBMQBackend(pattern)
    backend.to_qiskit()
    print(type(backend.circ))

    #set the rondom input state
    psi = []
    for i in range(n):
        psi.append(qi.random_statevector(2, seed=100+i))
    backend.set_input(psi)

.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    <class 'qiskit.circuit.quantumcircuit.QuantumCircuit'>

This can be run on Aer simulator or IBMQ devices. See `documentation page for graphix-ibmq interface <https://graphix-ibmq.readthedocs.io/en/latest/tutorial.html>`_ for more details, as well as `a detailed example showing how to run pattern on IBMQ devices <https://graphix-ibmq.readthedocs.io/en/latest/gallery/aer_sim.html#sphx-glr-gallery-aer-sim-py>`_.


Generating QASM file
++++++++++++++++++++

For other systems, we can generate QASM3 instruction set corresponding to the pattern, following

.. code-block:: python

    qasm_inst = pattern.to_qasm3('pattern')

Now check the generated qasm file:

.. code-block:: bash

    $ cat pattern.qasm

.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    // generated by graphix
    OPENQASM 3;
    include "stdgates.inc";

    // prepare qubit q1
    qubit q1;
    h q1;

    // prepare qubit q4
    qubit q4;
    h q4;

    // entangle qubit q1 and q4
    cz q1, q4;

    // measure qubit q1
    bit c1;
    float theta1 = 0;
    p(-theta1) q1;
    h q1;
    c1 = measure q1;
    h q1;
    p(theta1) q1;
    
    ...


References
----------

[1] `V. Danos, E Kashefi and P. Panangaden, "The Measurement Calculus", Journal of the ACM 54, 2 (2007) <https://doi.org/10.48550/arXiv.0704.1263>`_


