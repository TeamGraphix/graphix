.. _flows-tutorial:

Flows and determinism
=====================

Every measurement in MBQC has a random outcome, so a computation can unfold along
many different *branches*. The key idea we saw in the :ref:`introduction <introduction-tutorial>` is that
outcome-dependent Pauli corrections can "correct" this randomness. With the right
corrections, the pattern implements the same transformation, up to a global phase,
**in every branch**. This property is called *strong determinism*.

Strongly deterministic patterns realize *unitary* transformations, which makes them the MBQC counterpart of quantum circuits.

A central question in MBQC is which resource states can support a unitary pattern (or, more generally, an isometry, when the number of outputs exceeds the number of inputs, :math:`|O| > |I|`). Put differently:
given an open graph, is there a correction strategy that makes the computation
strongly deterministic?

Previous work answers this with graph-theoretic criteria on labelled open graphs,
known as *flow conditions*
:cite:`flows-MP08:finding_flows,flows-BKMP07:gflow,flows-S21:pauli_flow,flows-BMBdF+21,flows-MB24:algebraic`.
There are different flavors of flows depending on the measurements of the open graph, but they all share the same structure. Each consists of:

* a **correction function**, which maps each measured qubit to a prepared qubit that
  will absorb its correction, and
* a **partial order** on the nodes, which fixes the order in which measurements may
  be performed.

Whenever a flow exists, it can be found in polynomial time.

Causal flow
-----------

The simplest flow is the *causal flow*. It applies to open graphs in which every
measurement lies in the :math:`\XYplane` plane.

.. definition:: Causal flow :cite:`flows-DK06:determinism`

   An open graph :math:`(G, I, O, \lambda: O^c \rightarrow \{\XYplane\})` has
   *causal flow* if there exist a map :math:`c: O^c \rightarrow I^c` and a strict
   partial order :math:`\prec` over :math:`V` such that, for every
   :math:`i \in O^c`:

   * **(C1)** :math:`i` and :math:`c(i)` are neighbors,
   * **(C2)** :math:`i \prec c(i)`,
   * **(C3)** :math:`i \prec j` for every neighbor :math:`j \neq i` of :math:`c(i)`.

Here :math:`O^c` is the set of measured (non-output) nodes and :math:`I^c` the set of
prepared (non-input) nodes.

Graphix enables you to analyze the determinism of computations on open graphs and offers abstractions to represent flows. The following example shows how to extract and draw a causal flow from an open graph:

.. jupyter-execute::

    import networkx as nx
    from graphix import Measurement, OpenGraph

    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5)]),
        input_nodes=[0, 1],
        output_nodes=[4, 5],
        measurements={
            0: Measurement.XY(0.1),
            1: Measurement.XY(0.2),
            2: Measurement.XY(0.3),
            3: Measurement.XY(0.4)})

    cf = og.to_causalflow()
    print("Causal flow: ", cf)
    cf.draw()

The method :meth:`.OpenGraph.to_causalflow` runs in :math:`O(N^2)` with :math:`N` the number of nodes and returns a :class:`.CausalFlow` instance when the open graph has causal flow. If it doesn't, it will raise an :class:`.OpenGraphError` exception. For convenience, there is the :meth:`.OpenGraph.to_causalflow_or_none` method which handles the exception and returns ``None``.

We visualize flows similarly to :ref:`XZ-correction maps <hadamard-example>`: black arrows represent the flow's correction function on the open graph. The nodes are arranged in layers that represent the flow's partial order, from left to right.  

.. note::

    The layer numbering follows this convention:

    * **Layer 0** holds the output nodes, if there are any.
    * **Higher layers are measured first.** If :math:`l_i < l_j`, every node in layer :math:`l_j` is measured before the nodes in layer :math:`l_i`. In terms of the flow, this says that nodes in layer :math:`l_j` precede nodes in layer :math:`l_i` in the partial order :math:`\prec`.


From flows to computations
--------------------------
A flow does more than promise that a deterministic computation exists. It also
gives a specific pattern implementing the computation itself. Concretely, a causal flow :math:`(c, \prec)` yields the following XZ-correction maps:

.. math::
  :label: eq-causal-flow-corrections

   \boldsymbol{x}_{\text{c}}(i) = c(i), \qquad
   \boldsymbol{z}_{\text{c}}(i) = N_G(c(i)) \setminus \{i\},

where :math:`N_G(j)` is the neighborhood of node :math:`j` in the open graph. From these, we can :ref:`reconstruct a pattern <patterns_to_corrections>`:

.. math::
   :label: eq-causal-flow-pattern

   \mathcal{P}_{c,G} =
   \prod^{\prec}_{i \in O^c}
   \left( \X^{s_i}_{c(i)} \, \Z^{s_i}_{N_G(c(i)) \setminus \{i\}} \,
   \M^{\XYplane,\alpha_i}_i \right) \prod_{(i,j) \in E} \E_{ij}
   \prod_{i \in I^c} \N_i,

Recall that we read patterns from right to left: prepare the non-input nodes,
entangle them according to the graph, then go through the measured
nodes in the order given by :math:`\prec`. Each node is measured, and its outcome
:math:`s_i` is immediately used to correct the nodes :math:`c(i)` and
:math:`N_G(c(i)) \setminus \{i\}`.

.. note::

    To see why these corrections make the pattern deterministic notice that we
    can make a single measurement :math:`\M_i^{\XYplane,\alpha_i}` deterministic by
    preceding it with a Pauli-:math:`Z` correction :math:`\Z_i^{s_i}`. For example, for a
    one-qubit state :math:`\lvert\psi_0\rangle = a\lvert 0\rangle + b\lvert 1\rangle`,
    both outcomes give the same amplitude:

    .. math::
        :label: eq-meas-xy-acausal

        \M_0^{\XYplane,\alpha_0} \Z^{s_0} \lvert\psi_0\rangle \sim
        \begin{cases}
            \left(\langle 0\rvert + e^{i\alpha_0}\langle 1\rvert\right)
            \left(a\lvert 0\rangle + b\lvert 1\rangle\right)
            \sim a + b\,e^{i\alpha_0} & \text{if } s_0 = 0, \\[4pt]
            \left(\langle 0\rvert - e^{i\alpha_0}\langle 1\rvert\right)
            \left(a\lvert 0\rangle - b\lvert 1\rangle\right)
            \sim a + b\,e^{i\alpha_0} & \text{if } s_0 = 1.
        \end{cases}

    However, the correction :math:`\Z_i^{s_i}` depends on the outcome of the
    measurement it precedes, which is not yet known, so it's unphysical.
    The way out is that open graphs are partial stabilizer states:

    .. math::
        :label: eq-og-stab-reminder

        K_j \, \E_G \N_{I^c} = \E_G \N_{I^c},
        \qquad K_j = \X_j \prod_{k \in N_G(j)} \Z_k,
        \qquad \text{for all } j \in I^c,

    where :math:`\E_G \N_{I^c} = \prod_{(i,j) \in E} \E_{ij} \prod_{i \in I^c} \N_i`. 

    Inserting the stabilizer :math:`K_{c(i)}^{s_i}` makes the unphysical
    :math:`\Z_i^{s_i}` cancel out and leaves a *runnable* pattern:

    .. math::
        :label: eq-causal-flow-explanation

        \begin{aligned}
        \M_i^{\XYplane,\alpha_i} \Z_i^{s_i} \E_G \N_{I^c}
        &= \M_i^{\XYplane,\alpha_i} \Z_i^{s_i} K^{s_i}_{c(i)} \E_G \N_{I^c} \\
        &= \M_i^{\XYplane,\alpha_i} \Z_i^{s_i} \X^{s_i}_{c(i)} \Z^{s_i}_{N_G(c(i))} \E_G \N_{I^c} \\
        &= \X^{s_i}_{c(i)} \Z^{s_i}_{N_G(c(i)) \setminus \{i\}} \M_i^{\XYplane,\alpha_i}\E_G \N_{I^c}.
        \end{aligned}

    The first equality uses :eq:`eq-og-stab-reminder`. The third one is where the
    flow conditions come in:

    * **(C1)** :math:`i \in N_G(c(i))`, so the stabilizer contains a :math:`\Z_i` that cancels the unphysical :math:`\Z_i^{s_i}`.

    * **(C2)** :math:`c(i)` is measured after :math:`i`, so :math:`\X_{c(i)}^{s_i}` commutes past :math:`\M_i^{\XYplane,\alpha_i}`.

    * **(C3)** The nodes in :math:`N_G(c(i)) \setminus \{i\}` are measured after :math:`i`, so :math:`\Z_{N_G(c(i)) \setminus \{i\}}^{s_i}` also commutes past :math:`\M_i^{\XYplane,\alpha_i}`.

    Repeating this for every measured node :math:`i \in O^c`, in the appropriate
    order, gives the pattern of :eq:`eq-causal-flow-pattern`.


Using :eq:`eq-causal-flow-corrections`, Graphix converts any flow into an :class:`.XZCorrections` object that holds
the X and Z corrections, and from there you can recover the corresponding deterministic :class:`.Pattern`:

.. jupyter-execute::

    xz_corr = cf.to_xzcorrections()
    pattern = xz_corr.to_pattern()

    xz_corr.draw()
    print("Pattern: ", pattern)

Gflow and Pauli flow
--------------------
Causal flow only covers measurements in the :math:`\XYplane` plane. Once
measurements can also lie in the :math:`\XZplane` or :math:`\YZplane` planes,
strong determinism is characterized by the existence of a *generalised flow*,
or *gflow*. Both causal flow and gflow guarantee strong determinism for **any** choice of
measurement angles. This property is called *robust determinism*.

Robust determinism can be relaxed by fixing some measurement angles to Pauli
angles, that is, integer multiples of :math:`\pi/2`. At these angles, a
measurement belongs to two measurement planes at once. For example, an
:math:`\XYplane` measurement at angle :math:`0` is also an :math:`\XZplane`
measurement. This extra freedom can be exploited to establish strong determinism
even when no gflow exists. This more general setting is captured by the *Pauli flow*,
which strictly generalizes gflow by taking advantage of these fixed Pauli measurements.

As with causal flow, gflow and Pauli flow also yield a specific deterministic
correction strategy on the open graph. For the definitions and the ensuing corrections, see
:cite:`flows-BKMP07:gflow, flows-UNGSMG26:graphix`.

Graphix provides :meth:`.Opengraph.to_gflow` and :meth:`.Opengraph.to_pauliflow`.
Both run in :math:`O(N^3)`, with :math:`N` is the number of nodes. They return a
:class:`.GFlow` or a :class:`.PauliFlow` instance, respectively, or raise an
:class:`.OpenGraphError` if the flow does not exist. The methods :meth:`.Opengraph.to_gflow_or_none`, :meth:`.Opengraph.to_pauliflow_or_none` handle the exception and return ``None`` if it doesn't exist any flow.

For example:

.. jupyter-execute::

    import networkx as nx
    from graphix import Measurement, OpenGraph

    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 2), (0, 4), (1, 5), (2, 4), (2, 5), (3, 5)]),
        input_nodes=[0, 1],
        output_nodes=[4, 5],
        measurements={
            0: Measurement.XY(0.1),
            1: Measurement.XY(0.1),
            2: Measurement.XZ(0.2),
            3: Measurement.YZ(0.3),
        },
    )

    gf = og.to_gflow()
    gf.draw()

Flow extraction algorithms don't identify Pauli measurements. This means that if your open graph contains for instance an ``Measurement.XY(0)``, it will be treated as a planar measurement and not as a Pauli measurement ``PauliMeasurement.X``. However, you can manually cast planar measurements with a Pauli angle into Pauli measurements with the method :meth:`OpenGraph.infer_pauli_measurements`. This distinction can allow to extract a Pauli flow where a gflow does not exist or a flow with lower depth:

.. jupyter-execute::

    import networkx as nx
    from graphix import Measurement, OpenGraph

    og = OpenGraph(
        graph=nx.Graph(
            [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (1, 3), (4, 6)]),
        input_nodes=[0, 3, 6],
        output_nodes=[2, 5, 8],
        measurements=dict.fromkeys((0, 1, 3, 4, 6, 7), Measurement.XY(0)))

    gf = og.to_gflow()
    pf = og.infer_pauli_measurements().to_pauliflow()

    gf.draw()
    pf.draw()

The two plots share the same graph structure (nodes and dashed lines). What
changes is the correction function and the number of layers. If you skip the
Pauli-measurement inference and call ``og.to_pauliflow()``
directly on the previous open graph, you still get a :class:`.PauliFlow`
object. Its correction function and partial order, however, are identical
to those of the gflow. This is not a problem: every gflow is a special case
of a Pauli flow.

.. warning::

    Extracting a gflow or a causal flow from an open graph that contains Pauli
    measurements is not a valid operation. For instance,
    ``og.infer_pauli_measurements().to_gflow()`` in the previous example fails
    at runtime. The type checker ``mypy`` catches the mistake before you run
    the code, because the inferred Pauli measurements have a different type.
    To learn more, see the advanced :ref:`tutorial on measurement types
    <types-tutorial>`. 

Both the gflow and the Pauli flow let us extract a deterministic pattern. Here
the two correction functions differ, so the patterns differ too. However, they
implement the same unitary transformation:

.. jupyter-execute::

    from numpy.random import default_rng
    rng = default_rng(42)

    pattern_gf = gf.to_xzcorrections().to_pattern()
    pattern_pf = pf.to_xzcorrections().to_pattern()
    
    # The "gflow pattern" has more commands:
    assert len(pattern_gf) > len(pattern_pf)

    # By the default, the input state is |+++〉
    state_gf = pattern_gf.simulate(rng=rng)
    state_pf = pattern_pf.simulate(rng=rng)

    assert state_gf.isclose(state_pf)
    print(state_gf)

This example shows an important property of open graphs: once the measurement angles are fixed,
an open graph implements a single unitary, but many correction strategies can implement it.
For a given open graph, the correction functions of gflow and Pauli flow are not unique.

Graphix extracts *a* valid correction function, not all of them. It does not
currently support enumerating every correction function that exists on an open
graph.

Robust determinism is uniform
-----------------------------

Whether an open graph has a flow depends only on the measurement *planes* and
*axes* assigned to its nodes, never on the specific value of an angle.

.. note::

   This may look at odds with the previous section, where we said that fixing Pauli angles
   can make a Pauli flow possible even without a gflow. To be fully precise, angles matter
   only through whether they are Pauli:

   * A **non-Pauli angle** in a plane behaves like any other non-Pauli angle in
     that plane. Changing its value never changes whether a flow exists. This
     is robust determinism.
   * A **Pauli angle** places the measurement on an axis (:math:`\Xaxis`, :math:`\Yaxis`
     or :math:`\Zaxis`). That extra structure is what Pauli flow exploits, and an
     :class:`.Axis` records it directly.

   So the question "does a flow exist?" is answered by the planes and axes
   alone, with no need to know the continuous angle values.

This means we can extract flows from open graphs where ``measurements`` is
specified as a mapping from nodes to :class:`.Plane` or :class:`.Axis`, instead of
:class:`.Measurement`, and convert them to an :class:`.XZCorrections` instance:

.. jupyter-execute::

    import networkx as nx
    from graphix import Axis, Plane, OpenGraph

    og = OpenGraph(
        graph=nx.Graph([(0, 2), (2, 4), (3, 4), (4, 6), (1, 4), (1, 6), (2, 3), (3, 5), (2, 6), (3, 6)]),
        input_nodes=[0],
        output_nodes=[5, 6],
        measurements={
            0: Plane.XY,
            1: Plane.XZ,
            2: Axis.Y,
            3: Plane.XY,
            4: Axis.Z,
        },
    )

    pf = og.to_pauliflow()
    xz_corr = pf.to_xzcorrections()
    
    print(pf)
    xz_corr.draw()

However, building a pattern needs the measurement angles. Calling
``xz_corr.to_pattern()`` on the previous example is therefore incorrect.
``mypy`` flags it, and running it anyway produces a faulty :class:`.Pattern`. If you don't want to commit to *concrete* angle values yet, you can use :ref:`parametric placeholders <symbolic-tutorial>` instead. They let you build the pattern now and assign the angles later.

Extracting flows from corrections
---------------------------------

The previous examples showed how to go from a flow to its X and Z corrections.
Graphix also supports the opposite direction. Given a specific deterministic
correction strategy on an open graph, you can recover the flow it implements
with :meth:`.XZCorrections.to_causalflow`, :meth:`.XZCorrections.to_gflow`, and
:meth:`.XZCorrections.to_pauliflow`.

These methods do **not** call the open-graph flow-finding algorithms. They read
the flow's correction function directly from the X and Z correction maps.

This distinction matters. Running a flow-finding algorithm on the open graph
underlying your corrections returns *a* gflow or Pauli flow, not necessarily the
one that produced them, because the flow of an open graph is not unique. The
corrections derived from that flow may therefore differ from your original
strategy. Extracting the flow from the corrections themselves avoids this
ambiguity. We illustrate this effect in the example below:

.. jupyter-execute::

    import networkx as nx
    from graphix import OpenGraph, Plane, XZCorrections

    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
        input_nodes=[0],
        output_nodes=[3],
        measurements=dict.fromkeys(range(3), Plane.XY))

    x_map = {0: {1}, 1: {2}, 2: {3}}
    z_map = {0: {2}, 1: {3}}
    xz_corr = XZCorrections.from_measured_nodes_mapping(og, x_map, z_map)

    gf_from_xz = xz_corr.to_gflow()    # Gflow from X and Z corrections
    gf_from_og = xz_corr.og.to_gflow() # Gflow extraction algorithm on the open graph

    # The flow correction functions differ!
    print(gf_from_xz)
    print(gf_from_og)

    gf_from_xz.to_xzcorrections().draw() # Equal to the original ``xz_corr``
    gf_from_og.to_xzcorrections().draw() # Different from the original

.. tip::

   Patterns have their own shortcuts, :meth:`.Pattern.to_causalflow`,
   :meth:`.Pattern.to_gflow`, and :meth:`.Pattern.to_pauliflow`. Each one
   extracts the X and Z corrections from the pattern and then recovers the flow
   from them.

   Going the other way, :meth:`.OpenGraph.to_pattern` is a convenient shorthand
   for open graphs. It tries to find a flow and builds the pattern from it.


Building flows by hand
----------------------

Most of the time you will obtain flows programmatically, from open graphs or
patterns, as in the previous examples. However, you can also instantiate a flow directly
by specifying its correction function and its partial order.

Graphix lets you represent flows that are **not** valid. To check one, call
:meth:`.CausalFlow.check_well_formed`, :meth:`.GFlow.check_well_formed`, or
:meth:`.PauliFlow.check_well_formed`. If the flow is not well formed, they raise
a :exc:`.FlowException` that tells you exactly which condition is violated.
This can be useful when you are learning about flows and want to experiment, or when
you are designing a new flow-extraction algorithm.

.. jupyter-execute::

    import networkx as nx
    from graphix import CausalFlow, OpenGraph, Plane
    from graphix.flow.exceptions import FlowError

    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
        input_nodes=[0],
        output_nodes=[3],
        measurements=dict.fromkeys(range(3), Plane.XY))

    cf = CausalFlow(
        og,
        correction_function={0: {1}, 1: {2, 3}, 2: {3}},
        partial_order_layers=({3}, {2}, {1}, {0}))

    try:
        cf.check_well_formed()
    except FlowError as err:
        print(err)

.. tip::

   If you only need a yes-or-no answer, :meth:`.PauliFlow.is_well_formed` (and its
   counterparts on the other flow classes) catches the exception and returns a
   boolean.

References
----------

.. bibliography::
   :cited:
   :keyprefix: flows-