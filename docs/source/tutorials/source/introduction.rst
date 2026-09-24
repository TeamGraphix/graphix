.. _introduction-tutorial:

Introduction
============

In the MBQC model, a quantum computation is carried out in a fundamentally different way from the circuit model. Rather than applying a sequence of unitary gates on an input state, one first prepares an entangled multi-qubit state that serves as a computational resource --- with the input state embedded in it --- and then performs a sequence of single-qubit measurements. In the context of Graphix, the resource is always a graph state. To compensate for the inherent randomness of quantum measurements, Pauli corrections conditioned on previous measurement outcomes are applied. At the end, the initial resource state is ''consumed'' by the computation and only the output qubits encoding the computation result remain.



.. image:: ../plots/circuits/hadamard.svg
   :align: center
   :alt: Hadamard gate circuit