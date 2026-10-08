Getting started
===============

Once you have Graphix installed in your system, let's run your first MBQC program.

If you are not very familiar with MBQC yet, you might find it more natural to think of quantum programs in the circuit 
framework. With the help of the :class:`.Circuit` class, Graphix allows you to build any abritrary quantum circuit. 
Open your terminal or a python file, and type

.. jupyter-execute::

   from graphix import Circuit

   circuit = Circuit(3)

   circuit.h(0)
   circuit.cnot(0, 1)
   circuit.cz(1, 2)

   state_circ = circuit.simulate().state
   print(state_circ)


The core functionality of Graphix, however, is centered around the :ref:`MBQC pattern <measurement-calculus>`, 
which is a sequence of commands including qubit preparation, entanglement and single-qubit measurements. Any quantum 
circuit can be transpiled into a pattern that implements the same unitary. To see this in action with the above 3-qubit 
circuit, run

.. jupyter-execute::

   pattern = circuit.transpile().pattern

   state_pat = pattern.simulate()
   print(state_pat)

Congratulations! You've just run an MBQC simulation. To dive deeper into MBQC and how to use Graphix to build, optimize 
and simulate measurement-based quantum programs, head over to the :ref:`Tutorials <tutorials>` section.