"""Compilation passes to transform the result of the circuit extraction algorithm into a quantum circuit."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain, pairwise
from typing import TYPE_CHECKING

from graphix.circ_ext.extraction import PauliExponentialDAG
from graphix.fundamentals import ANGLE_PI
from graphix.sim.base_backend import NodeIndex
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from graphix.circ_ext.extraction import CliffordMap, ExtractionResult, PauliExponential, PauliExponentialDAG
    from graphix.command import Node


@dataclass(frozen=True)
class CompilationPass:
    """Dataclass to bundle the two compilation passes necessary to obtain a quantum circuit from a `ExtractionResult`.

    Attributes
    ----------
    pexp_cp: PauliExponentialDAGCompilationPass
        Compilation pass to synthesize a Pauli exponential DAG.
    cm_cp: CliffordMapCompilationPass
        Compilation pass to synthesize a Clifford map.
    """

    pexp_cp: PauliExponentialDAGCompilationPass
    cm_cp: CliffordMapCompilationPass

    def er_to_circuit(self, er: ExtractionResult) -> Circuit:
        """Convert a circuit extraction result into a quantum circuit representation.

        This method synthesizes a circuit by sequentially applying the Clifford map and the Pauli exponential DAG (Directed Acyclic Graph) extraction result. It performs a validation check to ensure that the output nodes of both components are identical.

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
        n_qubits = len(er.pexp_dag.output_nodes)
        circuit = Circuit(n_qubits)
        outputs_mapping = NodeIndex()
        outputs_mapping.extend(er.pexp_dag.output_nodes)

        self.cm_cp.add_to_circuit(er.clifford_map, circuit)
        self.pexp_cp.add_to_circuit(er.pexp_dag.remap(outputs_mapping), circuit)
        return circuit


class PauliExponentialDAGCompilationPass(ABC):
    """Abstract base class to implement a compilation procedure for a Pauli Exponential DAG."""

    @staticmethod
    @abstractmethod
    def add_to_circuit(pexp_dag: PauliExponentialDAG, circuit: Circuit) -> None:
        r"""Add a Pauli exponential DAG to a circuit.

        The input circuit is modified in-place.

        Parameters
        ----------
        pexp_dag: PauliExponentialDAG
            The Pauli exponential rotation to be added to the circuit.
        circuit : Circuit
            The circuit to which the operation is added. The input circuit is assumed to be compatible with ``pexp_dag.output_nodes``.
        """


class CliffordMapCompilationPass(ABC):
    """Abstract base class to implement a compilation procedure for a Clifford Map."""

    @staticmethod
    @abstractmethod
    def add_to_circuit(clifford_map: CliffordMap, circuit: Circuit) -> None:
        """Add the Clifford map to a quantum circuit.

        The input circuit is modified in-place.

        Parameters
        ----------
        clifford_map: CliffordMap
            The Clifford map to be added to the circuit.
        circuit : Circuit
            The quantum circuit to which the Clifford map is added. The input circuit is assumed to be compatible with ``clifford_map.output_nodes``.

        Raises
        ------
        NotImplementedError
            If the Clifford map represents an isometry, i.e., ``len(clifford_map.input_nodes) != len(clifford_map.output_nodes)``.
        """


class LadderPass(PauliExponentialDAGCompilationPass):
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

    @staticmethod
    def add_to_circuit(pexp_dag: PauliExponentialDAG, circuit: Circuit) -> None:
        """Add a Pauli exponential DAG to a circuit.

        See documentation in :meth:`PauliExponentialDAGCompilationPass.add_to_circuit` for additional information.
        """
        for node in chain(*reversed(pexp_dag.partial_order_layers[1:])):
            pexp = pexp_dag.pauli_exponentials[node]
            LadderPass.add_pexp(pexp, circuit)

    @staticmethod
    def add_pexp(pexp: PauliExponential, circuit: Circuit) -> None:
        r"""Add the Pauli exponential unitary to a quantum circuit.

        This method modifies the input circuit in-place.

        Parameters
        ----------
        circuit : Circuit
            The quantum circuit to which the Pauli exponential is added.

        Notes
        -----
        It is assumed that the ``x``, ``y``, and ``z`` node sets of the Pauli string in the exponential are well-formed, i.e., contain only output nodes and are pairwise disjoint.
        """
        if pexp.angle == 0:  # No rotation
            return

        nodes = sorted(pexp.pauli_string.x_nodes | pexp.pauli_string.y_nodes | pexp.pauli_string.z_nodes)
        angle = -2 * pexp.angle * pexp.pauli_string.sign

        if len(nodes) == 0:  # Identity
            return

        if len(nodes) == 1:
            n0 = nodes[0]
            if n0 in pexp.pauli_string.x_nodes:
                circuit.rx(n0, angle)
            elif n0 in pexp.pauli_string.y_nodes:
                circuit.ry(n0, angle)
            else:
                circuit.rz(n0, angle)
            return

        LadderPass.add_basis_change(pexp, nodes[0], circuit)

        for n1, n2 in pairwise(nodes):
            LadderPass.add_basis_change(pexp, n2, circuit)
            circuit.cnot(control=n1, target=n2)

        circuit.rz(nodes[-1], angle)

        for n2, n1 in pairwise(nodes[::-1]):
            circuit.cnot(control=n1, target=n2)
            LadderPass.add_basis_change(pexp, n2, circuit)

        LadderPass.add_basis_change(pexp, nodes[0], circuit)

    @staticmethod
    def add_basis_change(pexp: PauliExponential, node: Node, circuit: Circuit) -> None:
        """Apply an X or a Y basis change to a given node if required by the Pauli string.

        This method modifies the input circuit in-place.

        Parameters
        ----------
        pexp : PauliExponential
            The Pauli exponential under consideration.
        node : Node
            The node on which the basis-change operation is performed.
        circuit : Circuit
            The quantum circuit to which the basis change is added.
        """
        if node in pexp.pauli_string.x_nodes:
            circuit.h(node)
        elif node in pexp.pauli_string.y_nodes:
            LadderPass.add_hy(node, circuit)

    @staticmethod
    def add_hy(qubit: int, circuit: Circuit) -> None:
        """Add a pi rotation around the z + y axis.

        This method modifies the input circuit in-place.
        """
        circuit.rz(qubit, ANGLE_PI / 2)
        circuit.ry(qubit, ANGLE_PI / 2)
        circuit.rz(qubit, ANGLE_PI / 2)
