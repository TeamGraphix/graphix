.. _tutorials:

Tutorials
=========

Measurement-based quantum computation (MBQC) takes a different route to quantum computing from the standard circuit model. These tutorials will take you from the core ideas behind this model to building and running your own MBQC programs with Graphix.

Along the way, you'll learn the fundamental concepts of MBQC and how to put them into practice: expressing computations as graphs with correction strategies or as measurement patterns, optimizing them, and simulating the results in Python. Each tutorial builds on the last, so we recommend following them in order. We assume you're comfortable with the basics of quantum computation (qubits, gates, and measurement), but you don't need any prior experience with MBQC.

Throughout the tutorials, we link to the original research articles so you can dig deeper whenever something sparks your interest.

.. raw:: html

   <div class="section-divider"></div>


.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Introduction
      :link: source/introduction
      :link-type: doc
      :class-card: tutorial-card
      :shadow: md

      Learn what differentiates MBQC from the circuit model with the help of a simple example.

   .. grid-item-card:: Open graphs and correction functions
      :link: source/graph_mbqc
      :link-type: doc
      :class-card: tutorial-card
      :shadow: md

      Describe MBQC computations in terms of the resource graph state and a correction strategy.

   .. grid-item-card:: Patterns and the measurement calculus
      :link: source/measurement_calculus
      :link-type: doc
      :class-card: tutorial-card
      :shadow: md

      An introduction to MBQC patterns and the measurment calculus formalism.

   .. grid-item-card:: From patterns to corrections and back
      :link: source/pattern_xzcorrections
      :link-type: doc
      :class-card: tutorial-card
      :shadow: md

      Convert freely between patterns and their underlying graphical structure.

   .. grid-item-card:: Flows and determinism
      :link: source/flows
      :link-type: doc
      :class-card: tutorial-card
      :shadow: md

      An introduction to flows and how to build deterministic MBQC computations.

   .. grid-item-card:: From circuits to patterns
      :link: source/transpilation
      :link-type: doc
      :class-card: tutorial-card
      :shadow: md

      Transpile any quantum circuit into an MBQC pattern using Graphix.


.. toctree::
    :hidden:
    :maxdepth: 1

    source/introduction
    source/graph_mbqc
    source/measurement_calculus
    source/pattern_xzcorrections
    source/flows
    source/transpilation

