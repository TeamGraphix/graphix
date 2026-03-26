"""Compilation passes to transform the result of the circuit extraction algorithm into a quantum circuit."""

from __future__ import annotations

from itertools import batched, chain, pairwise
from typing import TYPE_CHECKING

import numpy as np

from graphix.fundamentals import ANGLE_PI, Axis
from graphix.instruction import CNOT, SWAP, H, S
from graphix.sim.base_backend import NodeIndex
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from graphix._linalg import MatGF2
    from graphix.circ_ext.extraction import CliffordMap, ExtractionResult, PauliExponential, PauliExponentialDAG
    from graphix.instruction import Instruction

    Qubit: TypeAlias = int | np.int_


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
    pexp_cp: Callable[[PauliExponentialDAG, Circuit], None] | None
        Compilation pass to synthetize a Pauli exponential DAG. If ``None`` (default), :func:`pexp_ladder_pass` is employed.
    cm_cp: Callable[[PauliExponentialDAG, Circuit], None] | None
        Compilation pass to synthetize a Clifford map. If ``None`` (default), a `ValueError` is raised since there is still no default pass for Clifford map integrated in Graphix.

    Returns
    -------
    Circuit
        A quantum circuit that combines the Clifford map operations followed by the Pauli exponential operations.

    Raises
    ------
    ValueError
        If the output nodes of ``er.pexp_dag`` and ``er.clifford_map`` do not match, indicating an incompatible extraction result.
    """
    if list(er.pexp_dag.output_nodes) != list(er.clifford_map.output_nodes):
        raise ValueError(
            "The Pauli Exponential DAG and the Clifford Map in the Extraction Result are incompatible since they have different output nodes."
        )
    if pexp_cp is None:
        pexp_cp = pexp_ladder_pass

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


def pexp_ladder_pass(pexp_dag: PauliExponentialDAG, circuit: Circuit) -> None:
    r"""Add a Pauli exponential DAG to a circuit by using a ladder decomposition.

    The input circuit is modified in-place. This function assumes that the Pauli exponential DAG has been remap, i.e., its Pauli strings are defined on qubit indices instead of output nodes. See :meth:`PauliString.remap` for additional information.

    Parameters
    ----------
    pexp_dag: PauliExponentialDAG
        The Pauli exponential rotation to be added to the circuit. Its Pauli strings are assumed to be defined on qubit indices.
    circuit : Circuit
        The circuit to which the operation is added. The input circuit is assumed to be compatible with ``pexp_dag.output_nodes``.

    Notes
    -----
    Pauli exponentials in the DAG are compiled sequentially following an arbitrary total order compatible with the DAG. Each Pauli exponential is decomposed into a sequence of basis changes, CNOT gates, and a single :math:`R_Z` rotation:

    .. math::

        R_Z(\phi) = \exp \left(-i \frac{\phi}{2} Z \right),

    with effective angle :math:`\phi = -2\alpha`, where :math:`\alpha` is the angle encoded in `self.angle`. Basis changes map :math:`X` and :math:`Y` operators to the :math:`Z` basis before entangling the qubits in a CNOT ladder.

    Gate set: H, CNOT, RZ, RY

    See https://quantumcomputing.stackexchange.com/questions/5567/circuit-construction-for-hamiltonian-simulation/11373#11373 for additional information.
    """

    def add_pexp(pexp: PauliExponential, circuit: Circuit) -> None:
        r"""Add the Pauli exponential unitary to a quantum circuit.

        This function modifies the input circuit in-place.

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
        # The order on which we iterate over the modified qubits does not matter.
        modified_qubits = list(pexp.pauli_string.axes)
        angle = -2 * pexp.angle * pexp.pauli_string.sign

        if len(modified_qubits) == 0:  # Identity
            return

        q0 = modified_qubits[0]

        if len(modified_qubits) == 1:
            circuit.r(q0, pexp.pauli_string.axes[q0], angle)
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

        This function modifies the input circuit in-place.

        Parameters
        ----------
        pexp : PauliExponential
            The Pauli exponential under consideration.
        qubit : int
            The qubit on which the basis-change operation is performed.
        circuit : Circuit
            The quantum circuit to which the basis change is added.
        """
        match pexp.pauli_string.axes[qubit]:
            case Axis.X:
                circuit.h(qubit)
            case Axis.Y:
                add_hy(qubit, circuit)

    def add_hy(qubit: int, circuit: Circuit) -> None:
        """Add a pi rotation around the z + y axis.

        This function modifies the input circuit in-place.

        Parameters
        ----------
        qubit : int
            The qubit on which the basis-change operation is performed.
        circuit : Circuit
            The quantum circuit to which the basis change is added.
        """
        circuit.rz(qubit, ANGLE_PI / 2)
        circuit.ry(qubit, ANGLE_PI / 2)
        circuit.rz(qubit, ANGLE_PI / 2)

    for node in chain(*reversed(pexp_dag.partial_order_layers[1:])):
        pexp = pexp_dag.pauli_exponentials[node]
        add_pexp(pexp, circuit)


def cm_berg_pass(clifford_map: CliffordMap, circuit: Circuit) -> None:
    tab = clifford_map.to_tableau()
    n = len(clifford_map.output_nodes)
    instructions: list[Instruction] = []

    def process_qubit(tab: MatGF2, instructions: list[Instruction], q: int) -> None:

        # print(f"Qubit {q}")

        # Step 1
        do_step_1(tab, instructions, row_idx=q)

        # print("After step 1\n", tab)

        # Step 2
        pivot = do_step_2(tab, instructions, row_idx=q)
        # print("After step 2\n", tab)

        # Step 3
        if pivot != q:
            add_swap(tab, circuit, q, pivot)

        # print("After step 3\n", tab)

        # Step 4
        col_idx_z = np.flatnonzero(tab[q + n, :-1])  # xz and zz blocks of qubit q, without sign.
        if not (len(col_idx_z) == 1 and col_idx_z[0] == q + n):
            add_h(tab, instructions, q)
            do_step_1(tab, instructions, row_idx=q + n)
            pivot = do_step_2(tab, instructions, row_idx=q + n)
            assert pivot == q
            add_h(tab, instructions, q)

        # print("After step 4\n", tab)

        # Step 5
        sign_xz = tab[q, -1], tab[q + n, -1]
        match sign_xz:
            case (0, 1):
                circuit.x(q)
            case (1, 1):
                circuit.y(q)
            case (1, 0):
                circuit.z(q)

    def do_step_1(tab: MatGF2, instructions: list[Instruction], row_idx: int) -> None:
        col_idx_zx = np.flatnonzero(tab[row_idx, n : 2 * n])
        for j in col_idx_zx:
            add_s(tab, instructions, int(j)) if tab[row_idx, j] else add_h(tab, instructions, int(j))

    def do_step_2(tab: MatGF2, instructions: list[Instruction], row_idx: int) -> int:
        # Return pivot
        col_idx_xx = np.flatnonzero(tab[row_idx, :n])
        while len(col_idx_xx) > 1:
            for edge in batched(col_idx_xx, 2):
                if len(edge) == 2:
                    add_cnot(tab, instructions, *edge)
            col_idx_xx = col_idx_xx[::2]

        return col_idx_xx[0]

    def add_h(tab: MatGF2, instructions: list[Instruction], q: Qubit) -> None:
        tab[:, -1] ^= tab[:, q] * tab[:, q + n]
        tab[:, [q, q + n]] = tab[:, [q + n, q]]
        instructions.append(H(q))

    def add_s(tab: MatGF2, instructions: list[Instruction], q: Qubit) -> None:
        tab[:, -1] ^= tab[:, q] * tab[:, q + n]
        tab[:, q + n] = tab[:, q] ^ tab[:, q + n]
        instructions.append(S(q))

    def add_cnot(tab: MatGF2, instructions: list[Instruction], qc: Qubit, qt: Qubit) -> None:
        tab[:, -1] ^= tab[:, qc] * tab[:, qt + n] * (tab[:, qt] ^ tab[:, qc + n] ^ 1)
        tab[:, qt] ^= tab[:, qc]
        tab[:, qc + n] ^= tab[:, qt + n]
        instructions.append(CNOT(control=qc, target=qt))

    def add_swap(tab: MatGF2, instructions: list[Instruction], q0: Qubit, q1: Qubit) -> None:
        for shift in [0, n]:
            tab[:, [q0 + shift, q1 + shift]] = tab[:, [q1 + shift, q0 + shift]]

        instructions.append(SWAP((q0, q1)))

    for q in range(n):
        process_qubit(tab, instructions, q)

    for instr in instructions[::-1]:
        circuit.add(instr)
