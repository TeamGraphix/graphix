"""Compilation passes to transform the result of the circuit extraction algorithm into a quantum circuit."""

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
        circuit = self.cm_cp.add_to_circuit(er.clifford_map)
        return self.pexp_cp.add_to_circuit(er.pexp_dag, circuit)


class PauliExponentialDAGCompilationPass(ABC):
    """Abstract base class to implement a compilation procedure for a Pauli Exponential DAG."""

    @staticmethod
    @abstractmethod
    def add_to_circuit(pexp_dag: PauliExponentialDAG, circuit: Circuit | None = None, copy: bool = False) -> Circuit:
        r"""Add a Pauli exponential DAG to a circuit.

        Parameters
        ----------
        pexp_dag: PauliExponentialDAG
            The Pauli exponential rotation to be added to the circuit.
        circuit : Circuit or ``None``, optional
            The circuit to which the operation is added. If ``None``, a new ``Circuit`` instance is created with a width matching the number of output nodes in ``pexp_dag``. Default is ``None``.
        copy : bool, optional
            If ``True``, the operation is applied to a deep copy of ``circuit`` and the modified copy is returned. Otherwise, the input circuit is modified in place. Default is ``False``.

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

    @staticmethod
    @abstractmethod
    def add_to_circuit(clifford_map: CliffordMap, circuit: Circuit | None = None, copy: bool = False) -> Circuit:
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
    r"""Compilation pass to synthetize a Pauli exponential DAG by using a ladder decomposition.

    Pauli exponentials in the DAG are compiled sequentially following an arbitrary total order compatible with the DAG. Each Pauli exponential is decomposed into a sequence of basis changes, CNOT gates, and a single :math:`R_Z` rotation:

    .. math::

        R_Z(\phi) = \exp \left(-i \frac{\phi}{2} Z \right),

    with effective angle :math:`\phi = -2\alpha`, where :math:`\alpha` is the angle encoded in `self.angle`. Basis changes map :math:`X` and :math:`Y` operators to the :math:`Z` basis before entangling the qubits in a CNOT ladder.

    Notes
    -----
    See https://quantumcomputing.stackexchange.com/questions/5567/circuit-construction-for-hamiltonian-simulation/11373#11373 for additional information.
    """

    @staticmethod
    def add_to_circuit(pexp_dag: PauliExponentialDAG, circuit: Circuit | None = None, copy: bool = False) -> Circuit:
        """Add a Pauli exponential DAG to a circuit.

        See documentation in :meth:`PauliExponentialDAGCompilationPass.add_to_circuit` for additional information.
        """
        circuit = initialize_circuit(pexp_dag.output_nodes, circuit, copy)  # May raise value error
        outputs_mapping = NodeIndex()
        outputs_mapping.extend(pexp_dag.output_nodes)

        for node in chain(*reversed(pexp_dag.partial_order_layers[1:])):
            pexp = pexp_dag.pauli_exponentials[node]
            LadderPass.add_pexp(pexp, outputs_mapping, circuit)

        return circuit

    @staticmethod
    def add_pexp(pexp: PauliExponential, outputs_mapping: NodeIndex, circuit: Circuit) -> None:
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
        """Apply an X or a Y basis change to a given node if required by the Pauli string.

        This method modifies the input circuit in-place.

        Parameters
        ----------
        pexp : PauliExponential
            The Pauli exponential under consideration.
        outputs_mapping : NodeIndex
            Mapping between node numbers of the original MBQC pattern or open graph and qubit indices of the circuit.
        node : Node
            The node on which the basis-change operation is performed.
        circuit : Circuit
            The quantum circuit to which the basis change is added.
        """
        qubit = outputs_mapping.index(node)
        if node in pexp.pauli_string.x_nodes:
            circuit.h(qubit)
        elif node in pexp.pauli_string.y_nodes:
            LadderPass.add_hy(qubit, circuit)

    @staticmethod
    def add_hy(qubit: int, circuit: Circuit) -> None:
        """Add a pi rotation around the z + y axis.

        This method modifies the input circuit in-place.
        """
        circuit.rz(qubit, ANGLE_PI / 2)
        circuit.ry(qubit, ANGLE_PI / 2)
        circuit.rz(qubit, ANGLE_PI / 2)


def initialize_circuit(output_nodes: Sequence[int], circuit: Circuit | None = None, copy: bool = False) -> Circuit:
    """Initialize or validate a quantum circuit based on the provided output nodes.

    If no circuit is provided, a new one is created with a width matching the number of output nodes. If a circuit is provided, its width is validated against the number of output nodes.

    Parameters
    ----------
    output_nodes : Sequence[int]
        A sequence of integers representing the output nodes of the original MBQC pattern or open graph. The length of this sequence determines the required circuit width.
    circuit : Circuit, optional
        An existing circuit to initialize. If ``None`` (default), a new `Circuit` object is instantiated.
    copy : bool, optional
        If ``True`` and an existing `circuit` is provided, a deep copy of the circuit is returned to avoid mutating the original object. Defaults to ``False``.

    Returns
    -------
    Circuit
        The initialized quantum circuit.

    Raises
    ------
    ValueError
        If the provided ``circuit`` width does not match the length of ``output_nodes``.
    """
    n_qubits = len(output_nodes)
    if circuit is None:
        circuit = Circuit(n_qubits)
    else:
        if circuit.width != n_qubits:
            raise ValueError(f"Circuit width ({circuit.width}) differs from number of outputs ({n_qubits}).")
        if copy:
            circuit = deepcopy(circuit)
    return circuit
