Why MBQC?
=========


Comparison with gate-based QC
-----------------------------

The pros and cons of MBQC, in comparison to gate network method, would be as follows:

**Pros**

:Simplicity: Computation can be performed only by single-qubit measurements of the resource state. This is particularly useful for quantum harware where qubits can be added easily to graph states but it is hard to apply arbitrary unitary gates, such as optical systems [1].

:Logical depth: The logical depth can be significantly smaller than the corresponding gate network because many operations in MBQC can be parallelized.

**Cons**

:Qubit number: MBQC requires large number of ancillas.

:Feedforward: MBQC requires feedforward measurements to ensure determinism by 'correcting' for randomness of measurement outcomes.


MBQC requires the number of ancilla that scales almost linearly to the number of gates.
At first sight, this seems daunting for classical simulation where the statevector simulation require the memory space that scales exponentially with the number of qubits.
Graphix is designed to mitigate such a downside of MBQC by optimization of measurement patterns.

How graphix can help
--------------------

With graphix, we can optimize the measurement pattern and use tensor network simulator to simulate much larger quantum algorithms running on MBQC.

:stabilizer simulator: Using the efficient stabilizer (graph state) simulator, graphix classically preprocess the Pauli measurements of the measurement pattern to significantly reduce the number of ancilla required in the resource state. This corresponds to the complete elimination of Clifford part of the original gate network and our modification to the measurement calculus allows us this integration without losing determinism of the pattern.

:pattern optimization: command sequence can be reordered to optimize the pattern characteristics according to the hardware requirements. If the number of qubit is limited, graphix can optimize the pattern to reduce the required qubit space (while increasing the depth); on the other hand if parallelism is desired (e.g. GPU simulation), graphix can parallelize the commands to minimze the depth of the command execution.

:tensor-network simulator: We provide matrix-product state (MPS) simulation backend to run MBQC, which require only polynomial memory space on the number of nodes [2]. For a number of quantum algorithms with limited amount of entanglement required, this significantly improve the clasical simulation speed. For example, with max_edge < 5 we can simulate MBQC with non-Pauli measurements running on thousands of qubits, on modest computer such as laptop.



References and notes
--------------------

[1] For example, with time-multiplexed continuous-variable optical systems generation of graph state with thousands or even millions of qubits is already demonstrated [Science 366, 373 (2019), Science 366, 369 (2019), APL Photonics 1, 060801 (2016)].
With realization of Bosonic codes they can readily run MBQC but realizing deterministic multi-qubit gate with these systems is difficult.

[2] if the number of edges from a single node scales :math:`\mathcal{O}(1)`. This is easily achievable by utilizing the teleportation pattern to avoid attaching many edges to a certain node. The translation from the MPS to statevector requires exponential operations however this can be massively parallelized. For more details, see our upcoming publication: M. Fukushima, Y. Matsuda and S. Sunami, in preparation.
