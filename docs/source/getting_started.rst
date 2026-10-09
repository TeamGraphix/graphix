Getting started
===============

Once you have Graphix installed in your system, let's run your first measurement-based computation!

In the MBQC model, all computations rely on first preparing an entangled multi-qubit state and then performing a series of local measurements. In Graphix, the resource graph state and the choice of measurements to be performed can be represented using an `open graph <graphmbqc-tutorial>`.

In your terminal or a new python file, type the following:

.. jupyter-execute::

   import networkx as nx
   from graphix import OpenGraph, Measurement

   og = OpenGraph(
      graph=nx.Graph([(1, 2), (2, 3), (0, 3), (3, 4)]),
      input_nodes=[0, 1],
      output_nodes=[0, 4],
      measurements={node: Measurement.X for node in [1, 2, 3]}
   )
   
   og.draw()

To fully describe an MBQC computation however, an open graph needs to be supplemented with a `correction strategy <og-corrections>` that assigns conditional Pauli operations to nodes based on previous measurement results. 

If an open graph is compatible with a valid correction strategy, the resulting computation can be described as an `MBQC pattern <measurement-calculus>`, which contains a sequence of commands including qubit preparation, entanglement, single-qubit measurement, and Pauli corrections.

Let's try to extract a pattern from the above open graph and simulate the computation.

.. jupyter-execute::

   from numpy.random import default_rng

   pattern = og.to_pattern()
   print(pattern)
   
   state = pattern.simulate(rng=default_rng(seed=42))
   print(state)

Congratulations! You've just run your first MBQC simulation in Graphix and created a Bell state. 

To dive deeper into MBQC concepts and learn how to use Graphix to build, optimize and simulate measurement-based quantum programs, head over to the :ref:`Tutorials <tutorials>` section.