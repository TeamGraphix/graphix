.. _patterns_to_corrections:

From patterns to corrections and back
=====================================

So far, we have seen two ways to describe an MBQC computation:
:ref:`XZ-correction maps defined on an open graph <og-corrections>` and
:ref:`patterns in the measurement calculus <patterns>`. These two
representations are closely related, and Graphix lets you convert freely from
one to the other.

Every pattern has a unique underlying open graph and defines a unique
correction map. Conversely, a computation given as an open graph with
correction maps, :math:`(\Gamma, \alpha, \boldsymbol{x}, \boldsymbol{z})`, corresponds to the
following pattern :cite:`patternscorr-PS16`:

.. math::
   :label: eq-pattern-xzcorr

   \mathcal{P} =
   \prod^{\prec}_{i \in O^c}
   \left( \X_{\boldsymbol{x}(i)}^{s_i} \, \Z_{\boldsymbol{z}(i)}^{s_i} \, \M_i^{\lambda,\alpha} \right)
   \prod_{(i,j) \in E} \E_{ij}
   \prod_{i \in I^c} \N_i,

where the :math:`\prec` is the partial order induced by the correction maps.

The method :meth:`.Pattern.to_xzcorrections` gives the graph-based
representation of a pattern. Conversely, you can go from
:math:`(\Gamma, \alpha, \boldsymbol{x}, \boldsymbol{z})` back to a pattern with
:meth:`.XZCorrections.to_pattern`:

.. jupyter-execute::
    
    from graphix import Pattern
    from graphix.command import N, E, M, X, Z

    pattern = Pattern(input_nodes=[0], cmds=[N(1), E((0, 1)), M(0), N(2), E((1, 2)), X(1, {0}), M(1), Z(2, {0}), X(2, {1})])

    xz_corr = pattern.to_xzcorrections()
    print(xz_corr)

    pattern_bis = xz_corr.to_pattern()
    print(pattern_bis)

If you only need the underlying open graph, :meth:`.Pattern.to_opengraph`
returns it directly. The opposite direction is trickier: an open graph does
not carry correction information, so turning it into a pattern is only
possible when a valid correction strategy can be found. That is the job of
the flow machinery, which we cover in the :ref:`next tutorial <flows-tutorial>`.



References
----------

.. bibliography::
   :cited:
   :keyprefix: patternscorr-