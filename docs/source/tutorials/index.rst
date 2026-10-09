.. _tutorials:

Tutorials
=========

Measurement-based quantum computation (MBQC) takes a different route to quantum computing from the standard circuit model. These tutorials will take you from the core ideas behind this model to building and running your own MBQC programs with Graphix.

Along the way, you'll learn the fundamental concepts of MBQC and how to put them into practice: expressing computations as graphs with correction strategies or as measurement patterns, optimizing them, and simulating the results in Python. Each tutorial builds on the last, so we recommend following them in order. We assume you're comfortable with the basics of quantum computation (qubits, gates, and measurement), but you don't need any prior experience with MBQC.

Throughout the tutorials, we link to the original research articles so you can dig deeper whenever something sparks your interest.

.. toctree::
    :maxdepth: 1

    source/introduction
    source/graph_mbqc
    source/measurement_calculus
    source/pattern_xzcorrections
    source/flows
    source/transpilation
    source/space_minimization
    source/pauli_removal
    source/pattern_simulation
    source/circuit_extraction

.. toctree::
    :caption: Advanced Tutorials

    source/symbolic
    source/measurement_types
    source/clifford_commands
