.. _flows-tutorial:

Flows and determinism
=====================

Every measurement in MBQC has a random outcome, so a computation can unfold along
many different *branches*. The key idea we saw in the :ref:`introduction <introduction-tutorial>` is that
outcome-dependent Pauli corrections can "correct" this randomness. With the right
corrections, the pattern implements the same transformation, up to a global phase,
**in every branch**. This property is called *strong determinism*.

Strongly deterministic patterns realize *unitary* transformations, which makes them the MBQC counterpart of quantum circuits.

Which resource states support a deterministic computation?
----------------------------------------------------------

A central question in MBQC is which resource states can support a unitary pattern (or, more generally, an isometry, when the number of outputs exceeds the number of inputs, :math:`|O| > |I|`). Put differently:
given an open graph, is there a correction strategy that makes the computation
strongly deterministic?

Previous work answers this with graph-theoretic criteria on labelled open graphs,
known as *flow conditions*
:cite:`MP08:finding_flows,BKMP07:gflow,Simmons21,BMBdF+21,MB24:algebraic`.
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

.. admonition:: Definition (Causal flow :cite:`DK06:determinism`)
   :class: note

   An open graph :math:`(G, I, O, \lambda: O^c \rightarrow \{\XYplane\})` has
   *causal flow* if there exist a map :math:`c: O^c \rightarrow I^c` and a strict
   partial order :math:`\prec` over :math:`V` such that, for every
   :math:`i \in O^c`:

   * **(C1)** :math:`i` and :math:`c(i)` are neighbors,
   * **(C2)** :math:`i \prec c(i)`,
   * **(C3)** :math:`i \prec j` for every neighbor :math:`j \neq i` of :math:`c(i)`.

Here :math:`O^c` is the set of measured (non-output) nodes and :math:`I^c` the set of
prepared (non-input) nodes. So :math:`c(i)` is the node that takes over the
correction for measuring :math:`i`.

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

We visualize flows similarly to :ref:`XZ-correction maps <hadamard_example>`: black arrows represent the flow's correction fuction on the open graph. The nodes are arranged in layers that represent the flow's partial order, from left to right.  

.. note::

    The layer numbering follows this convention:

    * **Layer 0** holds the output nodes, if there are any.
    * **Higher layers are measured first.** If :math:`l_i < l_j`, every node in layer :math:`l_j` is measured before the nodes in layer :math:`l_i`. In terms of the flow, this says that nodes in layer :math:`l_j` precede nodes in layer :math:`l_i` in the partial order :math:`\prec`.


From flows to computations
--------------------------
A flow does more than promise that a deterministic computation exists. It also
gives you the pattern itself. Concretely, a causal flow :math:`(c, \prec)` yields
the following XZ-correction maps:

.. math::

   \boldsymbol{x}_{\text{c}}(i) = c(i), \qquad
   \boldsymbol{z}_{\text{c}}(i) = N_G(c(i)) \setminus \{i\}.

From these, we can :ref:`reconstruct a pattern <patterns_to_corrections>`:

.. math::
   :label: eq-causal-flow-pattern

   \mathcal{P}_{c,G} =
   \prod^{\prec}_{i \in O^c}
   \left( \X^{s_i}_{c(i)} \, \Z^{s_i}_{N_G(c(i)) \setminus \{i\}} \,
   M^{\XYplane,\alpha_i}_i \right) \prod_{(i,j) \in E} \E_{ij}
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

        K_j \, E_G N_{I^c} = E_G N_{I^c},
        \qquad K_j = \X_j \prod_{k \in N_G(j)} \Z_k,
        \qquad \text{for all } j \in I^c,

    where :math:`E_G N_{I^c} = \prod_{(i,j) \in E} \E_{ij} \prod_{i \in I^c} \N_i`. 

    Inserting the stabilizer :math:`K_{c(i)}^{s_i}` makes the unphysical
    :math:`\Z_i^{s_i}` cancel out and leaves a *runnable* pattern:

    .. math::
        :label: eq-causal-flow-explanation

        \begin{aligned}
        \M_i^{\XYplane,\alpha_i} \Z_i^{s_i} \E_G \N_{I^c}
        &= \M_i^{\XYplane,\alpha_i} \Z_i^{s_i} K^{s_i}_{c(i)} \E_G \N_{I^c} \\
        &= \M_i^{\XYplane,\alpha_i} \Z_i^{s_i} \X^{s_i}_{c(i)} \Z^{s_i}_{N_G(c(i))} \E_G \N_{I^c} \\
        &= \X^{s_i}_{c(i)} \Z^{s_i}_{N_G(c(i)) \setminus \{i\}} \M_i^{\XYplane,\alpha_i}E_G \N_{I^c}.
        \end{aligned}

    The first equality uses :eq:`eq-og-stab-reminder`. The third one is where the
    flow conditions come in:

    * **(C1)** :math:`i \in N_G(c(i))`, so the stabilizer contains a :math:`\Z_i` that cancels the unphysical :math:`\Z_i^{s_i}`.

    * **(C2)** :math:`c(i)` is measured after :math:`i`, so :math:`\X_{c(i)}^{s_i}` commutes past :math:`\M_i^{\XYplane,\alpha_i}`.

    * **(C3)** The nodes in :math:`N_G(c(i)) \setminus \{i\}` are measured after :math:`i`, so :math:`\Z_{N_G(c(i)) \setminus \{i\}}^{s_i}` also commutes past :math:`\M_i^{\XYplane,\alpha_i}`.

    Repeating this for every measured node :math:`i \in O^c`, in the appropriate
    order, gives the pattern of :eq:`eq-causal-flow-pattern`.


Graphix allows to convert any flow into an :class:`.XZCorrections` object that holds
these X and Z corrections, and from there you can recover the corresponding deterministic :class:`.Pattern`:

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
pattern on the open graph. For the definitions and the exact corrections, see
:cite:`...`.

Graphix provides :meth:`.Opengraph.to_gflow` and :meth:`.Opengraph.to_pauliflow`.
Both run in :math:`O(N^3)`, with :math:`N` is the number of nodes. They return a
:class:`.GFlow` or a :class:`.PauliFlow` instance, respectively, or raise an
:class:`.OpenGraphError` if the flow does not exist.

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

---> Add link to measurement types
---> ads discussions planes vs meas

.. jupyter-execute::

    import networkx as nx
    from graphix import Measurement, OpenGraph

    og = OpenGraph(
        graph=nx.Graph(
            [(0, 1), (1, 2), (3, 4), (4, 5),
            (6, 7), (7, 8), (1, 3), (4, 6)]),
        input_nodes=[0, 3, 6],
        output_nodes=[2, 5, 8],
        measurements=dict.fromkeys(
            (0, 1, 3, 4, 6, 7),
            Measurement.XY(0)))

    gf = og.to_gflow()
    pf = og.infer_pauli_measurements().to_pauliflow()

    gf.draw()
    pf.draw()
