"""Compilation passes to transform the result of the circuit extraction algorithm into a quantum circuit."""

from __future__ import annotations

from itertools import batched, chain, pairwise
from typing import TYPE_CHECKING

import numpy as np

from graphix.fundamentals import ANGLE_PI, Axis
from graphix.instruction import CNOT, SWAP, H, S, X, Y, Z
from graphix.sim.base_backend import NodeIndex
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from graphix._linalg import MatGF2
    from graphix.circ_ext.extraction import CliffordMap, ExtractionResult, PauliExponential, PauliExponentialDAG
    from graphix.instruction import Instruction

    # NOTE: This alias could be defined at the level of graphix.instruction, and treat all qubit indices as `Qubit`. This change would affect many files in the codebase, so as a temporary solution `Qubit` is casted to `int` in this module.
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
    r"""Add a Clifford map to a circuit by using and adaptation of van den Berg's sweeping algorithm introduced in Ref.[1].

    The input circuit is modified in-place. This function assumes that the Clifford Map has been remap, i.e., its Pauli strings are defined on qubit indices instead of output nodes. See :meth:`PauliString.remap` for additional information.

    Parameters
    ----------
    clifford_map: CliffordMap
        The Clifford map to be transpiled. Its Pauli strings are assumed to be defined on qubit indices.
    circuit : Circuit
        The circuit to which the operation is added. The input circuit is assumed to be compatible with ``CliffordMap.input_nodes`` and ``CliffordMap.output_nodes``.

    Raises
    ------
    NotImplementedError
        If ``len(clifford_map.input_nodes) != len(clifford_map.output_nodes)``.
    AssertionError
        If an unexpected pivot position is encountered during Step 4.

    Notes
    -----
    This pass only handles unitaries so far (Clifford maps with the same number of input and ouptut nodes).

    Gate set: H, S, CNOT, SWAP, X, Y, Z

    This function converts a ``CliffordMap`` into a sequence of quantum
    gate instructions by operating on its binary tableau representation.
    The synthesis proceeds qubit-by-qubit, applying a sequence of local
    transformations (H, S, CNOT, and SWAP gates) to reduce the
    tableau into a canonical form. The resulting sequence represents the
    adjoint of the input Clifford map, therefore it's appended in reverse
    order (and exchanging S by Sdagger) to the provided ``Circuit``.

    The synthesis applies a series of steps on every qubit subtableau:

    .. math::

    T_q = \begin{pmatrix}
        XX & XZ \\
        ZX & ZZ
    \end{pmatrix}

    1. Clear elements in the XZ-block by applying single-qubit gates (H or S).

    2. Use CNOT gates to reduce the XX block to a single pivot column.

    3. Apply a SWAP gae to bring the pivot to the diagonal if neccesary.

    4. Ensure the ZX and ZZ blocks of the tableau have the correct canonical form
    by redoing steps 1. and 2.

    After processing all qubits, a final sign correction step applies
    Pauli gates (X, Y, Z) to fix phase bits in the tableau.

    The generated instructions are accumulated during the forward pass
    and then appended to the circuit in reverse order to yield the
    correct overall transformation.

    For the mapping between tableau updates and Clifford gates (H, S, CNOT) see [2].

    References
    ----------
    [1] Van Den Berg, 2021. A simple method for sampling random Clifford operators (arxiv:2008.06011).
    [2] Aaronson, Gottesman, (2004). Improved Simulation of Stabilizer Circuits (arXiv:quant-ph/0406196).
    """
    tab = clifford_map.to_tableau()
    n = len(clifford_map.output_nodes)
    if len(clifford_map.input_nodes) != n:
        raise NotImplementedError(
            ":func:`cm_berg_pass` does not support circuit compilation if the number of input and output nodes is different (isometry)."
        )
    instructions: list[Instruction] = []

    def process_qubit(tab: MatGF2, instructions: list[Instruction], q: int) -> None:
        """Bring to canonical form two tableau rows corresponding to qubit ``q``."""
        # Step 1
        do_step_1(tab, instructions, row_idx=q)

        # Step 2
        pivot = do_step_2(tab, instructions, row_idx=q)

        # Step 3
        if pivot != q:
            add_swap(tab, instructions, q, pivot)

        # Step 4
        col_idx_z = np.flatnonzero(tab[q + n, :-1])  # ZX and ZZ blocks of qubit q, without sign.
        if not (len(col_idx_z) == 1 and col_idx_z[0] == q + n):
            add_h(tab, instructions, q)
            do_step_1(tab, instructions, row_idx=q + n)
            pivot = do_step_2(tab, instructions, row_idx=q + n)
            if pivot != q:
                raise AssertionError(
                    f"Pivot in block ZZ should be at q = {q}. This error probably means that `CliffordMap` doesn't describe a valid Clifford operation. All Pauli strings must commute, except for `x_map[q]` anticommuting with `z_map[q]` for each q."
                )

            add_h(tab, instructions, q)

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

        return int(col_idx_xx[0])

    def add_h(tab: MatGF2, instructions: list[Instruction], q: Qubit) -> None:
        q = int(q)  # Cast to `int` to avoid typing issues
        tab[:, -1] ^= tab[:, q] & tab[:, q + n]
        tab[:, [q, q + n]] = tab[:, [q + n, q]]
        instructions.append(H(q))

    def add_s(tab: MatGF2, instructions: list[Instruction], q: Qubit) -> None:
        tab[:, -1] ^= tab[:, q] & tab[:, q + n]
        tab[:, q + n] = tab[:, q] ^ tab[:, q + n]
        q = int(q)
        instructions.extend((S(q), Z(q)))  # We append Sdagger to get C instead of C^dagger

    def add_cnot(tab: MatGF2, instructions: list[Instruction], qc: Qubit, qt: Qubit) -> None:
        tab[:, -1] ^= tab[:, qc] * tab[:, qt + n] * (tab[:, qt] ^ tab[:, qc + n] ^ 1)
        tab[:, qt] ^= tab[:, qc]
        tab[:, qc + n] ^= tab[:, qt + n]
        instructions.append(CNOT(control=int(qc), target=int(qt)))

    def add_swap(tab: MatGF2, instructions: list[Instruction], q0: Qubit, q1: Qubit) -> None:
        q0, q1 = int(q0), int(q1)  # Cast to `int` to avoid typing issues
        for shift in [0, n]:
            tab[:, [q0 + shift, q1 + shift]] = tab[:, [q1 + shift, q0 + shift]]

        instructions.append(SWAP((q0, q1)))

    def correct_signs(tab: MatGF2, instructions: list[Instruction]) -> None:
        for q in range(n):
            sign_xz = tab[q, -1], tab[q + n, -1]
            match sign_xz:
                case (0, 1):
                    instructions.append(X(q))
                case (1, 1):
                    instructions.append(Y(q))
                case (1, 0):
                    instructions.append(Z(q))

            # The tableau sign column should be set to 0, but we don't need to do it since it's the last step.
            # tab[q, -1], tab[q + n, -1] = 0, 0

    for q in range(n):
        process_qubit(tab, instructions, q)

    correct_signs(tab, instructions)

    for instr in instructions[::-1]:
        circuit.add(instr)
