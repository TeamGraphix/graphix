.. _measurement-calculus:

Patterns and the measurement calculus
=====================================

Open graphs and correction strategies give a compact picture of an MBQC
computation. To reason about computations more systematically, however, it
helps to switch to a more operational language: the *measurement calculus*
:cite:`patterns-DKP07:calculus`. It describes a computation as a sequence of elementary
instructions, called *commands*, of four kinds:

- **Preparation** :math:`\N_i`: prepare qubit :math:`i`.
- **Entanglement** :math:`\E_{ij}`: entangle qubits :math:`i` and :math:`j`.
- **Measurement** :math:`\M_i^{\lambda,\alpha}`: measure qubit :math:`i`
  destructively in the plane :math:`\lambda(i) \in \{\XYplane, \XZplane, \YZplane\}`, at angle
  :math:`\alpha(i) \in [0, 2\pi)`.
- **Pauli correction** :math:`\X_i^s` and :math:`\Z_i^s`: apply a Pauli
  correction to qubit :math:`i`, conditional on a classical bit
  :math:`s \in \{0, 1\}`. This bit is the parity
  :math:`s = \bigoplus_{j \in D} s_j` of the outcomes of a set :math:`D` of
  previously measured nodes, called the correction *domain*.

These commands map directly onto the ingredients we have already met:

- :math:`\N` and :math:`\E` together build the resource graph state.
- :math:`\lambda` and :math:`\alpha` assign a measurement plane and angle to
  every measured qubit (the set :math:`O^c`), exactly as in an open graph.
- The Pauli corrections implement the XZ-correction maps defined in
  the :ref:`previous tutorial <og-corrections>`.

The precise meaning of each command is summarised in the table below. The
states :math:`\ket{\pm_{\lambda,\alpha}}` are the measurement basis states
defined in :ref:`the earlier section on measurements <measurement-bases>`;
the two outcomes :math:`s = 0, 1` correspond to the signs :math:`+` and
:math:`-`.

.. list-table:: Syntax and semantics of MBQC commands
   :header-rows: 1
   :widths: 10 10 10 10

   * - :math:`\N_i`
     - :math:`\E_{ij}`
     - :math:`\M_i^{\lambda,\alpha}`
     - :math:`\X_i^s,\ \Z_i^s`
   * - :math:`\otimes \ketplus`
     - :math:`CZ_{ij}`
     - :math:`\bra{\pm_{\lambda(i),\alpha(i)}}`
     - :math:`X_i^s,\ Z_i^s`

As with open graphs, when :math:`\alpha(i)` is a Pauli angle, i.e. one of
:math:`\{0, \pi/2, \pi, 3\pi/2\}`, the pair :math:`(\lambda(i), \alpha(i))`
can be replaced by a signed axis :math:`\pm \Xaxis`, :math:`\pm \Yaxis` or
:math:`\pm \Zaxis`.

Patterns
--------

.. _patterns:

A computation in this formalism is a *measurement pattern* (or simply
*pattern*). A pattern consists of:

- a finite sequence of commands acting on a set of qubits, and
- two designated qubit sets, the input and output registers.

Not every sequence of commands makes sense physically, so patterns must
satisfy *runnability* conditions :cite:`patterns-DKP07:calculus`:

- A command cannot depend on a measurement outcome that is not yet available.
- A command can only act on qubits that have not been measured yet. These qubits must be inputs or must have been already prepared.
- Qubits are measured if and only if they are not outputs.
- Only non-input qubits are prepared, and they are prepared only once.

Example: the Hadamard gate
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Hadamard gate example from the :ref:`introduction tutorial <introduction-tutorial>` has the following pattern representation:

.. math::
   :label: eq-pattern-h

   \mathcal{P}_H = \X_1^{s_0} \, \M_0^{X} \, \E_{01} \, \N_1 .

Commands are executed from right to left, so the pattern reads as follows:

1. :math:`\N_1`: prepare the output qubit 1 in :math:`\ket{+}`.
2. :math:`\E_{01}`: entangle qubits 0 and 1 with a :math:`CZ` gate.
3. :math:`\M_0^{X}`: measure the input qubit 0 in the :math:`\Xaxis` basis, which
   yields the outcome :math:`s_0`.
4. :math:`\X_1^{s_0}`: apply an :math:`X` correction to qubit 1 if
   :math:`s_0 = 1`.

In Graphix, patterns are represented by the :class:`.Pattern` class. The example below direclty instantiates the Hadamard pattern by specifying the input qubits and the sequence of commands. The output qubits are inferred from the non-measured qubits.

.. jupyter-execute::

    from graphix import Pattern
    from graphix.command import E, M, N, X

    pattern = Pattern(input_nodes=[0], cmds=[N(1), E((0, 1)), M(0), X(1, {0})])
    print(pattern)

Once a pattern is built, you can run it with :meth:`.Pattern.simulate` :ref:`as we did with the XZ-corrections <hadamard-example>`.

.. jupyter-execute::

    from numpy.random import default_rng
    from graphix import BasicStates

    rng = default_rng(42)

    state = pattern.simulate(input_state=BasicStates.ZERO, rng=rng)
    print(state)


Writing patterns in code
~~~~~~~~~~~~~~~~~~~~~~~~

As you saw in the previous examples, measurement calculus commands
have a counterpart in Graphix:
:class:`~.command.N`, :class:`~.command.E`, :class:`~.command.M`,
:class:`~.command.X` and :class:`~.command.Z`. We create one by giving the
indices of the nodes it acts on. For the corrections :class:`~.command.X` and
:class:`~.command.Z`, the correction domain is passed as a ``set`` of nodes.

By default, :class:`~.command.M` measures along the :math:`X` axis. To choose
another plane and angle, pass any :class:`~.measurements.Measurement` object.

The example below is a slightly bigger pattern that puts all five commands to
work. Note that it has two input qubits, and it also contains an :math:`\XYplane`-measurement
with angle :math:`\alpha=3\pi/2` on node 2:

.. jupyter-execute::

    from graphix import Pattern, Measurement
    from graphix.command import E, M, N, X, Z

    cmds = [
        N(2), E((0, 2)), M(0),
        N(3), E((2, 3)), X(1, {0}), M(2, Measurement.XY(1.5)),
        E((3, 1)), Z(3, {0}), X(3, {2}), Z(1, {2})]

    pattern = Pattern(input_nodes=[0, 1], cmds=cmds)

Patterns can be visualized with the method :meth:`.Pattern.draw`. By default it shows the pattern's :ref:`flow <flows-tutorial>`, but it is possible to show the XZ-correction maps instead:

.. jupyter-execute::

    from graphix import DrawPatternAnnotations
    pattern.draw(annotations=DrawPatternAnnotations.XZCorrections)

Notice that input qubits can be output qubits too!

The pattern in the example implements a :math:`R_x(\frac{\pi}{2})` rotation on qubit 0 followed by a :math:`CZ` on qubits 0 and 1. You can try to prove it as an exercice, but don't worry if it looks too cumbersome. Typically, we don't instantiate large patterns by hand. In later tutorials, you will learn how to :ref:`extract patterns from open graphs <flows-tutorial>` and how to :ref:`transpile patterns from circuits <transpilation-tutorial>`.
      


Pattern standardization
-----------------------

Patterns in the measurement calculus can be
manipulated with rewrite rules which allow to absorb a
correction into later measurements. This transformation reflects the
adaptivity of MBQC: a Pauli correction applied just before a measurement
is equivalent to measuring at a modified angle.

Concretely, applying :math:`\X_i^s \Z_i^t` to qubit :math:`i` right before
measuring it gives

.. math::
   :label: eq-calc-domains-update

   {}_t\!\left[\M_i^{\lambda,\alpha}\right]^s
   := \M_i^{\lambda,\alpha} \, \X_i^s \, \Z_i^t
   = \M_i^{\lambda,\alpha'_{s,t}}

where the new angle depends on the measurement plane:

.. math::
   :label: eq-alpha-update

   \alpha'_{s,t} =
   \begin{cases}
       (-1)^s \alpha + t\pi, & \text{if } \lambda = XY,\\
       (-1)^{s+t} \alpha + t\pi, & \text{if } \lambda = XZ,\\
       (-1)^t \alpha + s\pi, & \text{if } \lambda = YZ.
   \end{cases}

The bits :math:`s = \bigoplus_{j \in D_s} s_j` and
:math:`t = \bigoplus_{j \in D_t} s_j` are called *signals*. Each one is the
parity of earlier measurement outcomes, taken over the sets of measured nodes
:math:`D_s, D_t \subseteq O^c`, which are the *domains* of the measurement.

In Graphix, an :class:`~.command.M` command stores these domains in its
``s_domain`` and ``t_domain`` attributes. As for :class:`~.command.X` and
:class:`~.command.Z`, each domain is given as a ``set`` of nodes.

Applying this rule repeatedly, together with the other rules of the
calculus :cite:`patterns-DKP07:calculus`, allows to bring any pattern into
*standard form*:

1. all qubit preparations (:math:`N` commands),
2. then all entanglements (:math:`E` commands),
3. then all measurements (:math:`M` commands),
4. and finally the corrections on the output qubits (:math:`X` and
   :math:`Z` commands).

Graphix implements these rules in :meth:`.Pattern.standardize`. In the example
below, the corrections scattered between the measurements are absorbed into
the measurement domains, leaving only output corrections at the end:

.. jupyter-execute::

    from graphix import Pattern, Measurement
    from graphix.command import N, E, M, X, Z

    cmds = [
        N(1), N(3), E((0, 3)), N(4), E((0, 4)), E((4, 1)), N(2), E((4, 2)),
        M(0, Measurement.XY(0.3)), X(3, {0}),
        M(2, Measurement.XZ(0.7)), Z(1, {2}), Z(4, {2}), X(3, {2}), X(4, {2}),
        M(1, Measurement.YZ(0.4)), Z(4, {1}),
    ]

    pattern = Pattern(input_nodes=[0], cmds=cmds)
    pattern.standardize()
    print(pattern)

.. note::

    **When are two patterns the same?**

    Pattern's standard form leaves freedom to arrange commands of the same kind, so it is not a normal form. In Graphix, pattern equality ``==`` compares the input, output and commands sequences, so two patterns implementing the same operation may still fail an equality test. In practice, whenever we want to check if two patterns implement the same transformation, we test if they produce the same output state given the same input state up to a global phase. Of course, this only works if the pattern is deterministic, otherwise, you need to compare all possible :ref:`execution branches <flows-tutorial>`.

    .. jupyter-execute::

        from graphix import Pattern
        from graphix.command import N, E, M, X, Z
        from graphix.random_objects import rand_state_vector
        from numpy.random import default_rng

        rng = default_rng(42)

        pattern = Pattern(input_nodes=[0, 1], cmds=[N(2), E((1, 2)), M(1), E((2, 0)), N(3), E((2, 3)), M(2, s_domain={1}), Z(0, {1}), Z(3, {1}), X(3, {2})])

        # ``standardize`` acts in place, so we make a copy.
        pattern_std = pattern.copy()
        pattern_std.standardize()
        assert pattern != pattern_std

        input_state = rand_state_vector(2, rng=rng)
        output_state = pattern.simulate(input_state=input_state, rng=rng)
        output_state_std = pattern_std.simulate(input_state=input_state, rng=rng)
        assert output_state.isclose(output_state_std)

Clifford commands
-----------------
Beyond the standard measurement calculus, Graphix adds one command,
:math:`\C_i` (:class:`~.command.C`), which applies an unconditional local
Clifford operation to node :math:`i`. Clifford commands appear as by-products of
certain optimization routines, such as :ref:`Pauli-measurement removal
<pauli-removal>`. They are covered in a :ref:`dedicated tutorial
<clifford-commands>`.

References
----------

.. bibliography::
   :cited:
   :keyprefix: patterns-