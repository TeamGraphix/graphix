.. _introduction-tutorial:

Introduction
============

In the measurement-based quantum computation (MBQC) model, computation works differently from the circuit model. Instead of applying a sequence of unitary gates to an input state, we first prepare a `graph state <https://en.wikipedia.org/wiki/Graph_state>`__ that acts as a computational resource, with the input embedded in it. We then run the computation by performing a sequence of single-qubit measurements. To compensate for the randomness of quantum measurements, we apply Pauli corrections conditioned on earlier measurement outcomes, a mechanism known as *feed-forward*. By the end, the resource state has been "consumed": only the output qubits, which now hold the result, remain.

Let's see this principle in action on the small circuit below.

.. image:: ../plots/circuits/hadamard_ff.svg
   :align: center
   :width: 400px
   :alt: Hadamard gate circuit

The procedure has three steps:

1. Prepare an ancilla qubit (qubit 1) in the state :math:`\ketplus` and apply a :math:`CZ` gate between it and the input qubit (qubit 0).
2. Measure qubit 0 along the :math:`\Xaxis` axis. The outcome :math:`s_0` is :math:`0` if the qubit is projected onto :math:`\ketplus` and :math:`1` if it is projected onto :math:`\ketminus`.
3. Apply a Pauli :math:`X` correction to qubit 1 if :math:`s_0 = 1`.

For an arbitrary input state :math:`\ket{\psi} = a\ket{0}_0 + b\ket{1}_0`, the computation unfolds as follow (up to a global factor):

.. math::

   \ket{\psi'}_1
   &\sim X_1^{s_0} \, {}_0\langle \pm | \left( a\ket{0}_0 \ketplus_1 + b\ket{1}_0 \ketminus_1 \right) \\
   &\sim X_1^{s_0}
   \begin{cases}
       a\ketplus_1 + b\ketminus_1 & \text{if projected onto } \ketplus \ (s_0 = 0), \\
       a\ketplus_1 - b\ketminus_1 & \text{if projected onto } \ketminus \ (s_0 = 1),
   \end{cases} \\
   &\sim a\ketplus_1 + b\ketminus_1 .

Both measurement outcomes lead to the same final state. This is the whole point of feed-forward: when :math:`s_0 = 1`, the :math:`X` correction flips the sign of the :math:`\ketminus` component and cancels the randomness of the measurement. If you compare the input and output states, you will notice that this circuit implements a Hadamard gate, since :math:`H\ket{\psi} = a\ketplus + b\ketminus`. Note also that the result now lives on qubit 1. The information has been *teleported* from the input register to the output register as a side effect of the computation.

Our example is a circuit enriched with *ancillas*, *mid-circuit measurements* and *classical feed-forward*. MBQC generalizes this idea: it provides a language and tools for expressing computations independently of any circuit representation.

In the next tutorials, we will see that it is possible to express *any* quantum computation using just four primitives:

- preparing ancilla qubits in the :math:`\ketplus` state,
- entangling qubits with :math:`CZ` gates,
- performing single-qubit measurements along an *arbitrary angle* on the :math:`\XYplane` plane,
- applying Pauli :math:`X` and :math:`Z` corrections conditioned on previous measurement outcomes.