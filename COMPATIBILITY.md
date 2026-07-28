# Backwards compatibility

Graphix does not guarantee backwards compatibility between 0.x version releases. Here we list API changes which may break existing code using Graphix. 

## Unreleased

- #518: Numba-jit statevector backend.
  - Changed statevector representation from a tensor to a flat array.
  - Dropped support for symbolic simulation and removed `Statevector.xreplace` and `Statevector.subs` methods. Symbolic simulation via SymPy is still possible with the plugin `graphix-symbolic`.

- #557: Namespace homogenisation.

  The following methods and classes have been renamed to provide a homogeneous, consistent and pythonic naming convention:

    | Old | New |
    |---|---|
    | `Pattern.extract_opengraph` | `Pattern.to_opengraph` |
    | `Pattern.extract_xzcorrections` | `Pattern.to_xzcorrections` |
    | `Pattern.extract_causal_flow` | `Pattern.to_causalflow` |
    | `Pattern.extract_gflow` | `Pattern.to_gflow` |
    | `Pattern.extract_pauli_flow` | `Pattern.to_pauliflow` |
    | `Pattern.extract_signals` | `Pattern.signals` |
    | `Pattern.extract_partial_order_layers` | `Pattern.partial_order_layers` |
    | `Pattern.extract_measurement_commands` | `Pattern.measurement_commands` |
    | `Pattern.compute_max_degree` | `Pattern.max_degree` |
    | `Pattern.extract_graph` | `Pattern.graph` |
    | `Pattern.extract_nodes` | `Pattern.nodes` |
    | `Pattern.extract_isolated_nodes` | `Pattern.isolated_nodes` |
    | `Pattern.extract_clifford` | `Pattern.clifford_commands` |
    | `StandardizedPattern.extract_opengraph` | `StandardizedPattern.to_opengraph` |
    | `StandardizedPattern.extract_xzcorrections` | `StandardizedPattern.to_xzcorrections` |
    | `XZCorrections.to_causal_flow` | `XZCorrections.to_causalflow` |
    | `XZCorrections.to_pauli_flow` | `XZCorrections.to_pauliflow` |
    | `XZCorrections.extract_dag` | `XZCorrections.to_dag` |
    | `PauliFlow.try_from_correction_matrix` | `PauliFlow.from_correctionmatrix_or_none` |
    | `PauliFlow.to_corrections` | `PauliFlow.to_xzcorrections` |
    | `GFlow.to_corrections` | `GFlow.to_xzcorrections` |
    | `CausalFlow.to_corrections` | `CausalFlow.to_xzcorrections` |
    | `State.to_statevector` | `State.to_statevector_numpy` |
    | `State.to_densitymatrix` | `State.to_densitymatrix_numpy` |
    | `Opengraph.extract_causal_flow` | `Opengraph.to_causal_flow` |
    | `Opengraph.extract_gflow` | `Opengraph.to_gflow` |
    | `Opengraph.extract_pauli_flow` | `Opengraph.to_pauliflow` |
    | `OpenGraph.extract_circuit` | `Opengraph.to_circuit` |
    | `Opengraph.find_causal_flow` | `Opengraph.to_causalflow_or_none` |
    | `Opengraph.find_gflow` | `Opengraph.to_gflow_or_none` |
    | `Opengraph.find_pauli_flow` | `Opengraph.to_pauliflow_or_none` |
    | `Noise.to_kraus_channel` | `Noise.to_krauschannel` |
    | `<...>.to_standardized_pattern` | `<...>.to_standardizedpattern` |
    | `<...>.from_standardized_pattern` | `<...>.from_standardizedpattern` |
    | `Measurement.try_to_pauli` | `Measurement.to_pauli_or_none` |
    | `ComplexUnit.try_from` | `ComplexUnit.from_or_none` |
    | `MatGF2.compute_rank` | `MatGF2.rank` |
    | `Statevec` | `Statevector` |

 





