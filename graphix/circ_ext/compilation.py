"""Compilation passes to transform the result of the circuit extraction algorithm into a quantum circuit."""

from __future__ import annotations

from itertools import chain, pairwise
from typing import TYPE_CHECKING

from graphix.fundamentals import ANGLE_PI
from graphix.sim.base_backend import NodeIndex
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Callable

    from graphix.circ_ext.extraction import CliffordMap, ExtractionResult, PauliExponential, PauliExponentialDAG


def ladder_pass(pexp_dag: PauliExponentialDAG, circuit: Circuit) -> None:
    r"""Compilation pass to synthetize a Pauli exponential DAG by using a ladder decomposition.

    Pauli exponentials in the DAG are compiled sequentially following an arbitrary total order compatible with the DAG. Each Pauli exponential is decomposed into a sequence of basis changes, CNOT gates, and a single :math:`R_Z` rotation:

    .. math::

        R_Z(\phi) = \exp \left(-i \frac{\phi}{2} Z \right),

    with effective angle :math:`\phi = -2\alpha`, where :math:`\alpha` is the angle encoded in `self.angle`. Basis changes map :math:`X` and :math:`Y` operators to the :math:`Z` basis before entangling the qubits in a CNOT ladder.

    Gate set: H, CNOT, RZ, RY

    Notes
    -----
    See https://quantumcomputing.stackexchange.com/questions/5567/circuit-construction-for-hamiltonian-simulation/11373#11373 for additional information.
    """

    def add_to_circuit(pexp_dag: PauliExponentialDAG, circuit: Circuit) -> None:
        """Add a Pauli exponential DAG to a circuit.

        See documentation in :meth:`PauliExponentialDAGCompilationPass.add_to_circuit` for additional information.
        """
        for node in chain(*reversed(pexp_dag.partial_order_layers[1:])):
            pexp = pexp_dag.pauli_exponentials[node]
            add_pexp(pexp, circuit)

    def add_pexp(pexp: PauliExponential, circuit: Circuit) -> None:
        r"""Add the Pauli exponential unitary to a quantum circuit.

        This method modifies the input circuit in-place.

        Parameters
        ----------
        circuit : Circuit
            The quantum circuit to which the Pauli exponential is added.

        Notes
        -----
        It is assumed that the ``x``, ``y``, and ``z`` node sets of the Pauli string in the exponential are well-formed, i.e., contain valid qubit indices and are pairwise disjoint.
        """
        if pexp.angle == 0:  # No rotation
            return

        # We assume that nodes in the Pauli strings have been mapped to qubits.
        modified_qubits = [
            qubit
            for qubit in range(circuit.width)
            if qubit in pexp.pauli_string.x_nodes | pexp.pauli_string.y_nodes | pexp.pauli_string.z_nodes
        ]
        angle = -2 * pexp.angle * pexp.pauli_string.sign

        if len(modified_qubits) == 0:  # Identity
            return

        q0 = modified_qubits[0]

        if len(modified_qubits) == 1:
            if q0 in pexp.pauli_string.x_nodes:
                circuit.rx(q0, angle)
            elif q0 in pexp.pauli_string.y_nodes:
                circuit.ry(q0, angle)
            else:
                circuit.rz(q0, angle)
            return

        add_basis_change(pexp, q0, circuit)

        for q1, q2 in pairwise(modified_qubits):
            add_basis_change(pexp, q2, circuit)
            circuit.cnot(control=q1, target=q2)

        circuit.rz(modified_qubits[-1], angle)

        for q2, q1 in pairwise(modified_qubits[::-1]):
            circuit.cnot(control=q1, target=q2)
            add_basis_change(pexp, q2, circuit)

        add_basis_change(pexp, modified_qubits[0], circuit)

    def add_basis_change(pexp: PauliExponential, qubit: int, circuit: Circuit) -> None:
        """Apply an X or a Y basis change to a given qubit if required by the Pauli string.

        This method modifies the input circuit in-place.

        Parameters
        ----------
        pexp : PauliExponential
            The Pauli exponential under consideration.
        qubit : int
            The qubit on which the basis-change operation is performed.
        circuit : Circuit
            The quantum circuit to which the basis change is added.
        """
        if qubit in pexp.pauli_string.x_nodes:
            circuit.h(qubit)
        elif qubit in pexp.pauli_string.y_nodes:
            add_hy(qubit, circuit)

    def add_hy(qubit: int, circuit: Circuit) -> None:
        """Add a pi rotation around the z + y axis.

        This method modifies the input circuit in-place.
        """
        circuit.rz(qubit, ANGLE_PI / 2)
        circuit.ry(qubit, ANGLE_PI / 2)
        circuit.rz(qubit, ANGLE_PI / 2)

    for node in chain(*reversed(pexp_dag.partial_order_layers[1:])):
        pexp = pexp_dag.pauli_exponentials[node]
        add_pexp(pexp, circuit)


def er_to_circuit(
    er: ExtractionResult,
    pexp_cp: Callable[[PauliExponentialDAG, Circuit], None] | None = None,
    cm_cp: Callable[[CliffordMap, Circuit], None] | None = None,
) -> Circuit:
    """Convert a circuit extraction result into a quantum circuit representation.

    This method synthesizes a circuit by sequentially applying the Clifford map and the Pauli exponential DAG (Directed Acyclic Graph) in the extraction result. It performs a validation check to ensure that the output nodes of both components are identical and it maps the output node numbers to qubit indices.

    Parameters
    ----------
    er : ExtractionResult
        The result of the extraction process, containing both the ``clifford_map`` and the ``pexp_dag``.

    Returns
    -------
    Circuit
        A quantum circuit that combines the Clifford map operations followed by the Pauli exponential operations.

    Raises
    ------
    ValueError
        If the output nodes of ``er.pexp_dag`` and ``er.clifford_map`` do not match, indicating an incompatible extraction result.

    Notes
    -----
    The conversion relies on the internal compilation passes ``self.cm_cp`` (Clifford Map Circuit Processor) and ``self.pexp_cp`` (Pauli Exponential Circuit Processor) to handle the low-level circuit synthesis.
    """
    if list(er.pexp_dag.output_nodes) != list(er.clifford_map.output_nodes):
        raise ValueError(
            "The Pauli Exponential DAG and the Clifford Map in the Extraction Result are incompatible since they have different output nodes."
        )
    if pexp_cp is None:
        pexp_cp = ladder_pass
    if cm_cp is None:
        raise ValueError(
            "Clifford-map pass is missing: there is still no default pass for Clifford map integrated in Graphix. You may use graphix-stim-compiler plugin."
        )

    n_qubits = len(er.pexp_dag.output_nodes)
    circuit = Circuit(n_qubits)
    outputs_mapping = NodeIndex()
    outputs_mapping.extend(er.pexp_dag.output_nodes)

    inputs_mapping = NodeIndex()
    inputs_mapping.extend(er.clifford_map.input_nodes)

    cm_cp(er.clifford_map.remap(inputs_mapping.index, outputs_mapping.index), circuit)
    pexp_cp(er.pexp_dag.remap(outputs_mapping.index), circuit)
    return circuit
