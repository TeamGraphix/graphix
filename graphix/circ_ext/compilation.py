from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain, pairwise
from typing import TYPE_CHECKING

from graphix.circ_ext.extraction import PauliExponentialDAG
from graphix.fundamentals import ANGLE_PI
from graphix.sim.base_backend import NodeIndex
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphix.circ_ext.extraction import CliffordMap, ExtractionResult, PauliExponential, PauliExponentialDAG
    from graphix.command import Node


@dataclass(frozen=True)
class CompilationPass:
    pexp_cp: PauliExponentialDAGCompilationPass
    cm_cp: CliffordMapCompilationPass

    def er_to_circuit(self, er: ExtractionResult) -> Circuit:
        if list(er.pexp_dag.output_nodes) != list(er.clifford_map.output_nodes):
            raise ValueError("The Pauli Exponential DAG and the Clifford Map in the Extraction Result are incompatible since they have different output nodes.")
        circuit = self.cm_cp.add_to_circuit(er.clifford_map)
        return self.pexp_cp.add_to_circuit(er.pexp_dag, circuit)


class PauliExponentialDAGCompilationPass(ABC):
    """Abstract base class to implement a compilation procedure for a Pauli Exponential DAG."""

    @staticmethod
    @abstractmethod
    def add_to_circuit(pexp_dag: PauliExponentialDAG, circuit: Circuit | None = None, copy: bool = False) -> Circuit:
        r"""Add a Pauli exponential rotation to a circuit.

        Parameters
        ----------
        pexp_dag: PauliExponentialDAG
            The Pauli exponential rotation to be added to the circuit.
        circuit : Circuit or None, optional
            The circuit to which the operation is added. If ``None``, a new
            ``Circuit`` instance is created. Default is ``None``.
        copy : bool, optional
            If ``True``, the operation is applied to a deep copy of ``circuit`` and
            the modified copy is returned. Otherwise, the input circuit is modified
            in place. Default is ``False``.

        Returns
        -------
        Circuit
            The circuit with the operation applied.

        Raises
        ------
        ValueError
            If the input circuit is not compatible with ``pexp_dag.output_nodes``.
        """


class CliffordMapCompilationPass(ABC):
    """Abstract base class to implement a compilation procedure for a Clifford Map."""

    @abstractmethod
    def add_to_circuit(self, clifford_map: CliffordMap, circuit: Circuit | None = None, copy: bool = False) -> Circuit:
        """Add the Clifford map to a quantum circuit.

        Parameters
        ----------
        clifford_map: CliffordMap
            The Clifford map to be added to the circuit.
        circuit : Circuit
            The quantum circuit to which the Clifford map is added.
        copy : bool, optional
            If ``True``, operate on a deep copy of ``circuit`` and return it.
            Otherwise, the input circuit is modified in place. Default is
            ``False``.

        Returns
        -------
        Circuit
            The circuit with the operation applied.

        Raises
        ------
        ValueError
            If the input circuit is not compatible with ``clifford_map.output_nodes``.
        NotImplementedError
            If the Clifford map represents an isometry, i.e., ``len(clifford_map.input_nodes) != len(clifford_map.output_nodes)``.
        """


class LadderPass(PauliExponentialDAGCompilationPass):

    @staticmethod
    def add_to_circuit(pexp_dag: PauliExponentialDAG, circuit: Circuit | None = None, copy: bool = False) -> Circuit:
        circuit = initialize_circuit(pexp_dag.output_nodes, circuit, copy)
        outputs_mapping = NodeIndex()
        outputs_mapping.extend(pexp_dag.output_nodes)

        for node in chain(*reversed(pexp_dag.partial_order_layers[1:])):
            pexp = pexp_dag.pauli_exponentials[node]
            LadderPass.add_pexp(pexp, outputs_mapping, circuit)

        return circuit

    @staticmethod
    def add_pexp(pexp: PauliExponential, outputs_mapping: NodeIndex, circuit: Circuit) -> None:
        r"""Add the Pauli exponential unitary to a quantum circuit.

        For a Pauli string acting on multiple qubits, the unitary is decomposed into a sequence of basis changes, CNOT gates, and a single :math:`R_Z` rotation:

        .. math::

            R_Z(\phi) = \exp \left(-i \frac{\phi}{2} Z \right),

        with effective angle :math:`\phi = -2\alpha`, where :math:`\alpha` is the angle encoded in `self.angle`. Basis changes map :math:`X` and :math:`Y` operators to the :math:`Z` basis before entangling the qubits in a CNOT ladder.

        Parameters
        ----------
        circuit : CircuitMBQC
            The quantum circuit to which the Pauli exponential is added. `circuit` is modified in place.

        Notes
        -----
        It is assumed that the ``x``, ``y``, and ``z`` node sets of the Pauli string in the exponential are well-formed, i.e., contain only output nodes and are pairwise disjoint.

        See https://quantumcomputing.stackexchange.com/questions/5567/circuit-construction-for-hamiltonian-simulation/11373#11373
        for additional information.
        """
        if pexp.angle == 0:  # No rotation
            return

        nodes = sorted(
            pexp.pauli_string.x_nodes | pexp.pauli_string.y_nodes | pexp.pauli_string.z_nodes,
            key=outputs_mapping.index,
        )
        sign = -1 if pexp.pauli_string.negative_sign else 1
        angle = -2 * pexp.angle * sign

        if len(nodes) == 0:  # Identity
            return

        if len(nodes) == 1:
            n0 = nodes[0]
            q0 = outputs_mapping.index(n0)
            if n0 in pexp.pauli_string.x_nodes:
                circuit.rx(q0, angle)
            elif n0 in pexp.pauli_string.y_nodes:
                circuit.ry(q0, angle)
            else:
                circuit.rz(q0, angle)
            return

        LadderPass.add_basis_change(pexp, outputs_mapping, nodes[0], circuit)

        for n1, n2 in pairwise(nodes):
            LadderPass.add_basis_change(pexp, outputs_mapping, n2, circuit)
            q1, q2 = outputs_mapping.index(n1), outputs_mapping.index(n2)
            circuit.cnot(control=q1, target=q2)

        circuit.rz(q2, angle)

        for n2, n1 in pairwise(nodes[::-1]):
            q1, q2 = outputs_mapping.index(n1), outputs_mapping.index(n2)
            circuit.cnot(control=q1, target=q2)
            LadderPass.add_basis_change(pexp, outputs_mapping, n2, circuit)

        LadderPass.add_basis_change(pexp, outputs_mapping, nodes[0], circuit)

    @staticmethod
    def add_basis_change(pexp: PauliExponential, outputs_mapping: NodeIndex, node: Node, circuit: Circuit) -> None:
        """Apply an X or a Y basis change to a given node."""
        qubit = outputs_mapping.index(node)
        if node in pexp.pauli_string.x_nodes:
            circuit.h(qubit)
        elif node in pexp.pauli_string.y_nodes:
            LadderPass.add_hy(qubit, circuit)

    @staticmethod
    def add_hy(qubit: int, circuit: Circuit) -> None:
        """Add a pi rotation around the z + y axis."""
        circuit.rz(qubit, ANGLE_PI / 2)
        circuit.ry(qubit, ANGLE_PI / 2)
        circuit.rz(qubit, ANGLE_PI / 2)


def initialize_circuit(output_nodes: Sequence[int], circuit: Circuit | None = None, copy: bool = False) -> Circuit:
        n_qubits = len(output_nodes)
        if circuit is None:
            circuit = Circuit(n_qubits)
        else:
            if circuit.width != n_qubits:
                raise ValueError(f"Circuit width ({circuit.width}) differs from number of outputs ({n_qubits}).")
            if copy:
                circuit = deepcopy(circuit)
        return circuit
