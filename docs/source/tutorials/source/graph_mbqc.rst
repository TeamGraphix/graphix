.. _graphmbqc-tutorial:

Open graphs and correction functions
====================================

A measurement-based computation has three ingredients: a large entangled resource state, a sequence of single-qubit measurements, and Pauli corrections that depend on the measurement outcomes. This page introduces how Graphix represents these objects.


Open graphs
-----------

*Open graphs* (sometimes denoted *labelled open graphs*) describe the resource and the choice of measurements. Formally, an open graph :math:`\Gamma = (G, I, O, \lambda)` consists of:

* an undirected graph :math:`G = (V, E)`. Its nodes :math:`V` are qubits, and its edges :math:`E` describe how they are entangled;
* two sequences :math:`I, O` of distinct nodes, denoting the input and output qubits of the computation. Their ordering fixes the order of the qubit registers in the input and output Hilbert spaces;
* a map :math:`\lambda: O^c \to \{\XYplane, \XZplane, \YZplane\}` that assigns a measurement plane to each non-output qubit, where :math:`O^c := V \setminus O`.

If you are familiar with graph states, an open graph is one whose input nodes are left free. Instead of starting in :math:`\ketplus`, they hold an arbitrary input state :math:`\ket{\psi}_I`. This gives the *partial graph state*

.. math::
   :label: og-state

   |\Gamma, \psi\rangle :=
   \prod_{(i,j) \in E} CZ_{ij}
   \left( \ketplus^{\otimes |I^c|} \otimes \ket{\psi}_I \right),

where :math:`CZ_{ij}` is the controlled-:math:`Z` gate acting on qubits :math:`i` and :math:`j`, and :math:`I^c := V \setminus I`. Because of this construction, we use the terms *qubit* and *node* interchangeably.

Partial graph states are also *partial stabilizer states*:

.. math::
   :label: og-stabilizers

   K_j |\Gamma, \psi\rangle = |\Gamma, \psi\rangle
   \quad \forall j \in I^c,
   \qquad
   K_j := X_j \prod_{i \in N_G(j)} Z_i,

where :math:`N_G(j)` is the neighbourhood of node :math:`j` in :math:`G`. This holds for any input state :math:`\ket{\psi}_I` :cite:`graph-BKMP07:gflow`.

In Graphix, open graphs are represented by the :class:`.OpenGraph` class. The example below builds and draws the smallest non-trivial one, with two nodes joined by a single edge:

.. jupyter-execute::

    import networkx as nx
    from graphix import OpenGraph, Plane

    og = OpenGraph(
        graph=nx.Graph([(0, 1)]),
        input_nodes=[0],
        output_nodes=[1],
        measurements={0: Plane.XY},
    )
    og.draw()


.. _og-corrections:

From open graphs to computations
--------------------------------

A labelled open graph says which qubits are measured and in which plane, but does not have information about the classical feed-forward. A complete description of an MBQC computation is the tuple :math:`(\Gamma, \alpha, \boldsymbol{x}, \boldsymbol{z})`, made of:

* a labelled open graph :math:`\Gamma = (G, I, O, \lambda)`;
* a map :math:`\alpha: O^c \to [0, 2\pi)` that assigns a measurement angle to each non-output qubit;
* a *correction strategy*, given by two functions :math:`\boldsymbol{x}, \boldsymbol{z}: O^c \to \mathcal{P}(I^c)`, where :math:`\mathcal{P}(I^c)` is the power set of the non-input nodes. The set :math:`\boldsymbol{x}(i)` contains the nodes that receive a Pauli :math:`X^{s_i}` correction, and :math:`\boldsymbol{z}(i)` those that receive a :math:`Z^{s_i}` correction. Here :math:`s_i \in \{0, 1\}` is the outcome of the measurement on qubit :math:`i`, so the correction is applied only if that outcome is :math:`1`.

For the computation to be *runnable*, corrections may only depend on qubits that have already been measured. In other words, the dependencies must contain no cycles.

.. _measurement-bases:

Measurement bases
^^^^^^^^^^^^^^^^^

The pair :math:`(\lambda(i), \alpha(i))` defines a single-qubit measurement on qubit :math:`i`, namely a projection onto the states :math:`|\pm_{\lambda(i), \alpha(i)}\rangle`:

.. math::
   :label: meas-states

   \begin{aligned}
   |\pm_{\XYplane,\alpha}\rangle &= \tfrac{1}{\sqrt{2}}\left(|0\rangle \pm e^{i\alpha}|1\rangle\right), \\
   |\pm_{\XZplane,\alpha}\rangle &= t^{\alpha}_{\pm}|0\rangle \pm t^{\alpha}_{\mp}|1\rangle, \\
   |\pm_{\YZplane,\alpha}\rangle &= t^{\alpha}_{\pm}|0\rangle \pm i\, t^{\alpha}_{\mp}|1\rangle,
   \end{aligned}

with :math:`t^{\alpha}_{+} = \cos(\alpha/2)` and :math:`t^{\alpha}_{-} = \sin(\alpha/2)`. The classical bit :math:`s_i` from the correction strategy is precisely this measurement outcome: :math:`s_i = 0` means the qubit was projected onto :math:`|+_{\lambda(i),\alpha(i)}\rangle`, and :math:`s_i = 1` onto :math:`|-_{\lambda(i),\alpha(i)}\rangle`.

When :math:`\alpha(i) \in \{0, \pi/2, \pi, 3\pi/2\}`, the measurement is along a Pauli axis, and the pair :math:`(\lambda(i), \alpha(i))` can be replaced by a signed axis. For instance, :math:`(\XYplane, 0) := +\Xaxis`. These are called *Pauli measurements*, as opposed to the more general *Bloch* (or *planar*) *measurements*.

.. plot:: tutorials/plots/measurement_bases.py
     :include-source: false

.. note::
  :math:`\XYplane` measurements with an arbitrary angle :math:`\alpha` are enough for universal quantum computation, but allowing measurements on the :math:`\XZplane` and :math:`\YZplane` planes is very practical.

.. _hadamard-example:

Example: the Hadamard gate
--------------------------

Let us revisit the Hadamard gate from the :ref:`introduction tutorial <introduction-tutorial>`. Its open graph has two entangled qubits: node 0 is the input and node 1 is the output. Node 0 is measured in the :math:`\XYplane` plane with angle :math:`\alpha(0) = 0`, and the correction strategy is simply :math:`\boldsymbol{x}(0) = \{1\}`, :math:`\boldsymbol{z}(0) = \emptyset`. If the measurement returns :math:`s_0 = 1`, an :math:`X` gate on node 1 fixes the output.

In Graphix, an MBQC computation can be represented with the :class:`.XZCorrections` class. You instantiate it from an :class:`.OpenGraph` and the two correction maps. The constructor checks that the correction strategy is runnable.

.. jupyter-execute::

    import networkx as nx
    from graphix import Measurement, OpenGraph, XZCorrections

    og = OpenGraph(
        graph=nx.Graph([(0, 1)]),
        input_nodes=[0],
        output_nodes=[1],
        measurements={0: Measurement.XY(0)},
    )

    # Node 1 receives an X correction depending on the measuring outcome of node 0.
    # The Z-correction map is empty.
    x_map = {0: {1}}

    xz_corr = XZCorrections.from_measured_nodes_mapping(og, x_map)
    xz_corr.draw()

The red arrow from node 0 to node 1 represents the correction :math:`X_1^{s_0}`. The nodes are arranged in layers that show the measurement order, from left to right. In this basic example only node 0 is measured, and node 1 is an output node.

.. attention::
   All angles in Graphix are expressed in units of :math:`\pi`.

Finally, you can simulate the MBQC computation by calling the method :meth:`~.Simulable.simulate`.
We cover simulation in depth in the :ref:`dedicated tutorial <simulation-tutorial>`, but two points are worth knowing right away:

- **Input state.** By default, every input qubit starts in :math:`\ket{+}`. You
  can pass any other input state. Below, we start from the all-:math:`\ket{0}`
  state. The output confirms that we are implementing a Hadamard.
- **Randomness.** Measurement outcomes are drawn at random. To make your runs
  reproducible, you can pass a seeded random number generator (:class:`numpy.random.Generator`).
  We recommend always doing so.

.. jupyter-execute::

   from numpy.random import default_rng
   from graphix import BasicStates
   
   state = xz_corr.simulate(input_state=BasicStates.ZERO, rng=default_rng(seed=42))
   print(state) 

.. note::

   **Planes or measurements?**
   
   You may have noticed that the two examples above describe the measurements in the :class:`.OpenGraph` differently. Graphix bundles planes and angles into a single object :class:`.Measurement` instead of keeping two separate maps :math:`\lambda` and :math:`\alpha` as in the formal definitions presented above, but it lets you build open graphs either *with or without* information about the measurement direction. Additionally, there are specific classes to represent axes and Pauli measurements.

   .. list-table::
      :header-rows: 1

      * - Direction-agnostic
        - Direction-specific
      * - ``Plane.XY``
        - ``Measurement.XY(angle)``
      * - ``Plane.XZ``
        - ``Measurement.XZ(angle)``
      * - ``Plane.YZ``
        - ``Measurement.YZ(angle)``
      * - ``Axis.X``
        - ``+Measurement.X`` or ``-Measurement.X``
      * - ``Axis.Y``
        - ``+Measurement.Y`` or ``-Measurement.Y``
      * - ``Axis.Z``
        - ``+Measurement.Z`` or ``-Measurement.Z``
   
   However, it is only possible to simulate objects with defined measurement angles.
   To learn more about the differences between the two representations, see the advanced :ref:`tutorial on measurement types <types-tutorial>`.



Beyond this example
-------------------

The Hadamard example implements a single-qubit unitary, but the formalism is more general: the input and output Hilbert spaces can have arbitrary and different dimensions. We also chose the corrections so that the computation is *deterministic*, meaning the resulting transformation does not depend on the intermediate measurement outcomes. This is the most common case, but the formalism also accommodates non-deterministic computations. To learn more about determinism in MBQC, see the :ref:`tutorial on flows <flows-tutorial>`.

References
----------

.. bibliography::
   :cited:
   :keyprefix: graph-