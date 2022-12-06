Introduction to MBQC
====================

Here, we provide an introduction to the measurement-based quantum computing (MBQC), more specifically the one-way model of MBQC.

If you already know the basics of MBQC and would like to read about LC-MBQC (MBQC on local-Clifford decorated graph state), go to :doc:`lc-mbqc`.


Introduction
------------

| Quantum computing utilizes entanglment to accelerate the computation of some class of problems, such as the `prime factorization <https://en.wikipedia.org/wiki/Shor%27s_algorithm>`_.
| Quantum algorithms are very often expressed in the `gate network` model, which is a direct analog of the classical computers expressed in the network of logical bit operations (AND, OR, XOR, ...).
| The familiar quantum `circuits` thus express the time evolution of quantum bits (qubits) as they 'pass through' the quantum version of the logical operation.
| Here, the entanglement, which is arguably the source of the power of quantum computing, is created and destroyed continuously - in a way, this means that we don't really know where the `quantum` comes in [#gktheorem]_.

cartoon of gate network. analogy with classical one (XOR etc)

| Measurement-based (one-way) quantum computing, introduced by Raussendorf [#raussendorf]_, has a completely different approach which we can call `consumption of initially entangled resource state` (we will explain the `resource state` later); first you create a large entangled quantum state, and the computation goes by measurements of qubits which drives the evolution of the quantum state. There is no classical analog and it is difficult to picture, but MBQC has several remarkable advantages that motivate us to study further:

- Only single-qubit measurements are needed to perform the computation on the resource state (no single- or multi-qubit gates).
- The resource state is a [low-entangled state]() with only two-qubit entanglement which can be prepared in a depth of one. Further, they can be prepared offline [], and can be based on probabilistic entangling operations []
- The depth of the computation is usually significantly smaller than the equivalent quantum circuit model, which means computation is less affected by finite coherence time.

cartoon of graph state


One-way quantum computing
-------------------------

In one-way model, we perform quantum computation on the `resource state`, or equivalently the `graph state <https://en.wikipedia.org/wiki/Graph_state>`_, defined on methematical graph :math:`G = (N, E)` where :math:`N` is the set of nodes (qubits) and :math:`E` is a set of pairs of node indices, specifying the set of edges, by

.. math::
    \begin{equation}
    |g\rangle = \prod_{(i,j) \in E} CZ_{i,j} \bigotimes_{i\in N} |+\rangle, \label{1}   \tag{1}
    \end{equation}

where :math:`\bigotimes_{i\in N} = |+\rangle_{i_1}\otimes|+\rangle_{i_2} \otimes ... `, tensor product of :math:`|+\rangle` states.
A simplest example is the graph state with two qubits, :math:`|g'\rangle = CZ_{0,1}|+\rangle_1 \otimes |+\rangle_0`, which is local-unitary equivalent to the Bell state.

Measurement of a qubit in Pauli X basis is expressed by the application of one of projection operators corresponding to measurement outcome :math:`(-1)^s = -1` or :math:`1`` for :math:`s=0, 1` ,

.. math::
    \begin{equation}
    P_{X, s=0} = |+\rangle \langle+|, \ \ P_{X, s=1} = |-\rangle \langle-|. \label{2}   \tag{2}
    \end{equation}

Since measurements can be considered destructive, we can trace out (partial trace) the measured qubits and  the application of bras (:math:`\langle+|, \langle-|`) is sufficient.
For our simplest graph state :math:`|g'\rangle`, measurement of qubit 0 in the X bases gives

.. math::
    \begin{align}
    |g_{s=0}\rangle &= \langle+|_0 CZ_{0,1}|+\rangle_1 \otimes |+\rangle_0 = H|+\rangle_1,\ \ s = 0, \label{3}   \tag{3} \\
    |g_{s=1}\rangle &= \langle-|_0 CZ_{0,1}|+\rangle_1 \otimes |+\rangle_0 = XH|+\rangle_1, \ \ s = 1. \label{4}   \tag{4}
    \end{align}

If the measurement outcome was :math:`s=0`, the output state is the initial :math:`|+\rangle` state with a Hadamard gate applied. If :math:`s=1`, there is additional :math:`X` gate applied. \
In fact, this process of entangling with another qubit and then measuring with X basis the original qubit (qubit 0) is the MBQC version of Hadamard gate, and we treat the randomness of the measurement outcome with feedforward operations, as we describe below.

In MBQC, measurements with :math:`s=0` is to be considered `default`, and the additional :math:`X` term is called `byproduct` of the measurement. The simplest construction of MBQC would be to apply adaptive :math:`X` gate to `correct` for this byproduct, which we can express as follows

.. math::
    \begin{equation}
    |g_{out}\rangle = X^{s_0} \langle\pm|_0 CZ_{0,1}|+\rangle_1 \otimes |+\rangle_0,  \label{5}   \tag{5}
    \end{equation}

where :math:`X^{s_0}` is applied if the measurement outcome of qubit 0 is :math:`s_0=1`.
Most basic quantum gates (unitary operations) have corresponding graph state and a sequence of measurements and byproduct corrections, as we show below as an example.

.. math::
    \begin{align}
        |g_{out, CNOT}\rangle = X^{s_0} \langle\pm|_0 CZ_{0,1}|+\rangle_1 \otimes |+\rangle_0,  \label{6}   \tag{6} \\
        |g_{out, CNOT}\rangle = X^{s_0} \langle\pm|_0 CZ_{0,1}|+\rangle_1 \otimes |+\rangle_0,  \label{7}   \tag{7} \\
        |g_{out, CNOT}\rangle = X^{s_0} \langle\pm|_0 CZ_{0,1}|+\rangle_1 \otimes |+\rangle_0.  \label{8}   \tag{8} \\
    \end{align}

We can concatenate them to create a larger graph state that realizes a more complex unitary evolution, and because these building blocks include the single-qubit rotation (:math:`R_x` and :math:`R_z`) and CNOT gate, MBQC is said to be universal (can determinisitically realize any multi-qubit unitary operations).

..
    We can inspect the graph state using :class:`~graphix.graphsim.GraphState` class:

    .. code-block:: python

        from graphix import GraphState
        g = GraphState(nodes=[0,1],edges=[(0,1)])

    >>> print(g.to_statevector())
    Statevec, data=[[ 0.5+0.j  0.5+0.j]
    [ 0.5+0.j -0.5+0.j]], shape=(2, 2)




Measurement Calculus
--------------------
..
    It is tedious to treat the MBQC by bras and kets as we show in eqs (:math:`\ref{6}` - :math:`\ref{8}`) - it is impossible to track all the feedforwards and ancillas by hand if the number of operations grow as we try larger quantum algorithms.
    Instead, we can resort to

References and footnotes
------------------------

.. [#gktheorem] For example, we know that `a certain type of quantum gates are not so essential for quantum computations (efficiently simulatable on classical computers) <https://en.wikipedia.org/wiki/Gottesman%E2%80%93Knill_theorem>`_. However, in gate sequences these 'classical' parts are interleaved with 'quantum' parts of the algorithm. In fact, by translating the problem into MBQC, one can :doc:`classically preprocess such a part<stabilizer>`.

.. [#raussendorf] Raussendorf et al., `PRL 86, 5188 (2001) <https://link.aps.org/doi/10.1103/PhysRevLett.86.5188>`_, `PRA 68, 022312 (2003) <https://link.aps.org/doi/10.1103/PhysRevA.68.022312>`_. Here, by MBQC we refer to one-way quantum computing by Raussendorf among `several measurement-based schemes <>`_. As shown in [Danos], those differnt models can be mapped between one another, and since the measurement calculus framework can express most of them (see danos), graphix is capable of expressing these other models too.



a


