from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
import pytest
from numpy.random import Generator

from graphix._linalg import MatGF2
from graphix.circ_ext.compilation import cm_berg_pass, pexp_ladder_pass
from graphix.circ_ext.extraction import CliffordMap, PauliExponential, PauliExponentialDAG, PauliString, extend_input
from graphix.flow.core import PauliFlow
from graphix.fundamentals import ANGLE_PI, Axis, Sign
from graphix.instruction import CNOT, RX, RY, RZ, H, S
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.parameter import Placeholder
from graphix.random_objects import rand_circuit
from graphix.sim.base_backend import NodeIndex
from graphix.states import BasicStates
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from numpy.random import PCG64


class PauliExpTestCase(NamedTuple):
    pexp_dag: PauliExponentialDAG
    qc: Circuit


class TestPauliExponential:
    # Angles of Pauli exponentials are in units of pi
    alpha = 0.3 * ANGLE_PI

    @pytest.mark.parametrize(
        "test_case",
        [
            PauliExpTestCase(
                PauliExponentialDAG(
                    pauli_exponentials={
                        0: PauliExponential(alpha / 2, PauliString({1: Axis.Z}, sign=Sign.MINUS)),
                    },
                    partial_order_layers=[{1}, {0}],
                    output_nodes=[1],
                ),
                Circuit(width=1, instr=[RZ(0, alpha)]),
            ),
            PauliExpTestCase(
                PauliExponentialDAG(
                    pauli_exponentials={
                        0: PauliExponential(alpha / 2, PauliString({1: Axis.X}, sign=Sign.MINUS)),
                    },
                    partial_order_layers=[{1}, {0}],
                    output_nodes=[1],
                ),
                Circuit(width=1, instr=[RX(0, alpha)]),
            ),
            PauliExpTestCase(
                PauliExponentialDAG(
                    pauli_exponentials={
                        0: PauliExponential(alpha / 2, PauliString({1: Axis.Y}, sign=Sign.MINUS)),
                    },
                    partial_order_layers=[{1}, {0}],
                    output_nodes=[1],
                ),
                Circuit(width=1, instr=[RY(0, alpha)]),
            ),
            PauliExpTestCase(
                PauliExponentialDAG(
                    pauli_exponentials={
                        0: PauliExponential(ANGLE_PI / 4, PauliString({3: Axis.Z}, sign=Sign.MINUS)),
                        1: PauliExponential(ANGLE_PI / 4, PauliString({3: Axis.X}, sign=Sign.MINUS)),
                        2: PauliExponential(ANGLE_PI / 4, PauliString({3: Axis.Z}, sign=Sign.MINUS)),
                    },
                    partial_order_layers=[{3}, {2}, {1}, {0}],
                    output_nodes=[3],
                ),
                Circuit(width=1, instr=[H(0)]),
            ),
            PauliExpTestCase(
                PauliExponentialDAG(
                    pauli_exponentials={
                        0: PauliExponential(ANGLE_PI / 4, PauliString({3: Axis.X})),
                        1: PauliExponential(ANGLE_PI / 4, PauliString({5: Axis.Z})),
                        2: PauliExponential(ANGLE_PI / 4, PauliString({3: Axis.X, 5: Axis.Z}, Sign.MINUS)),
                    },
                    partial_order_layers=[{5, 3}, {2}, {0, 1}],
                    output_nodes=[5, 3],  # Node 5 -> qubit 0 (control), node 3 -> qubit 1 (target)
                ),
                Circuit(width=2, instr=[CNOT(1, 0)]),
            ),
            PauliExpTestCase(
                PauliExponentialDAG(
                    pauli_exponentials={
                        0: PauliExponential(alpha / 2, PauliString({1: Axis.X, 2: Axis.Z, 4: Axis.Z})),
                    },
                    partial_order_layers=[{1, 2, 3, 4}, {0}],
                    output_nodes=[2, 1, 3, 4],
                ),
                Circuit(width=4, instr=[H(1), CNOT(3, 1), CNOT(0, 3), RZ(0, -alpha), CNOT(0, 3), CNOT(3, 1), H(1)]),
            ),
        ],
    )
    def test_to_circuit(self, test_case: PauliExpTestCase, fx_rng: Generator) -> None:
        qc = Circuit(len(test_case.pexp_dag.output_nodes))
        outputs_mapping = NodeIndex()
        outputs_mapping.extend(test_case.pexp_dag.output_nodes)
        pexp_ladder_pass(test_case.pexp_dag.remap(outputs_mapping.index), qc)
        state = qc.simulate_statevector(rng=fx_rng).statevec
        state_ref = test_case.qc.simulate_statevector(rng=fx_rng).statevec
        assert state.isclose(state_ref)

    def test_to_circuit_outputs_order(self, fx_rng: Generator) -> None:
        pexp_map = {2: PauliExponential(0.1, PauliString({1: Axis.X, 0: Axis.Z}))}
        pol = [{0, 1}, {2}]

        outputs_1 = [0, 1]
        outputs_2 = [1, 0]

        pexp_dag_1 = PauliExponentialDAG(pauli_exponentials=pexp_map, partial_order_layers=pol, output_nodes=outputs_1)
        qc_1 = Circuit(2)
        outputs_mapping_1 = NodeIndex()
        outputs_mapping_1.extend(pexp_dag_1.output_nodes)
        pexp_ladder_pass(pexp_dag_1.remap(outputs_mapping_1.index), qc_1)
        s_1 = qc_1.simulate_statevector(rng=fx_rng, input_state=[BasicStates.PLUS, BasicStates.MINUS]).statevec

        pexp_dag_2 = PauliExponentialDAG(pauli_exponentials=pexp_map, partial_order_layers=pol, output_nodes=outputs_2)
        qc_2 = Circuit(2)
        qc_2.swap(0, 1)  # We must swap before and after the Pauli exponential!
        outputs_mapping_2 = NodeIndex()
        outputs_mapping_2.extend(pexp_dag_2.output_nodes)
        pexp_ladder_pass(pexp_dag_2.remap(outputs_mapping_2.index), qc_2)

        s_2 = qc_2.simulate_statevector(rng=fx_rng, input_state=[BasicStates.PLUS, BasicStates.MINUS]).statevec
        assert not s_1.isclose(s_2)

        qc_2.swap(0, 1)
        s_2 = qc_2.simulate_statevector(rng=fx_rng, input_state=[BasicStates.PLUS, BasicStates.MINUS]).statevec

        assert s_1.isclose(s_2)

    def test_from_focused_flow(self) -> None:
        """Test example C.13. in Simmons, 2021."""
        og = OpenGraph(
            graph=nx.Graph([(0, 2), (1, 2), (2, 4), (1, 4), (1, 3), (3, 4), (3, 5), (4, 6)]),
            input_nodes=[0],
            output_nodes=[5, 6],
            measurements={
                0: Measurement.XY(0.1),  # XY
                1: Measurement.YZ(0.2),  # YZ
                2: Measurement.XY(0.3),  # XY
                3: Measurement.XY(0.4),  # XY
                4: Measurement.Y,  # Y
            },
        )

        flow = PauliFlow(
            og,
            correction_function={
                0: frozenset({2, 6}),
                1: frozenset({1, 3, 4, 6}),
                2: frozenset({3, 4, 5}),
                3: frozenset({5}),
                4: frozenset({6}),
            },
            partial_order_layers=(frozenset({5, 6}), frozenset({3}), frozenset({1, 2}), frozenset({0, 4})),
        )

        flow.check_well_formed()
        assert flow.is_focused()

        pexp_dag = PauliExponentialDAG.from_focused_flow(flow)

        pexp_dag_ref = PauliExponentialDAG(
            pauli_exponentials={
                0: PauliExponential(ANGLE_PI * 0.1 / 2, PauliString({6: Axis.X})),
                1: PauliExponential(ANGLE_PI * 0.2 / 2, PauliString({6: Axis.Y, 5: Axis.Z})),
                2: PauliExponential(ANGLE_PI * 0.3 / 2, PauliString({5: Axis.Y, 6: Axis.Z}, Sign.MINUS)),
                3: PauliExponential(ANGLE_PI * 0.4 / 2, PauliString({5: Axis.X})),
                4: PauliExponential(
                    0, PauliString({6: Axis.X})
                ),  # The angle is 0 (interpreted from the Pauli measurement).
            },
            partial_order_layers=flow.partial_order_layers,
            output_nodes=flow.og.output_nodes,
        )

        assert pexp_dag == pexp_dag_ref


class TestCliffordMap:
    @pytest.mark.parametrize(
        ("cm", "tab_ref"),
        [
            (
                CliffordMap(
                    x_map={0: PauliString({0: Axis.Z}), 1: PauliString({1: Axis.Y})},
                    z_map={0: PauliString({0: Axis.X, 1: Axis.Y}, Sign.MINUS), 1: PauliString({0: Axis.Z, 1: Axis.Z})},
                    input_nodes=[0, 1],
                    output_nodes=[0, 1],
                ),
                MatGF2([[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 1, 0, 1, 1], [0, 0, 1, 1, 0]]),
            ),
            (
                CliffordMap(
                    x_map={0: PauliString({0: Axis.Z}), 1: PauliString({1: Axis.X}), 2: PauliString({2: Axis.Y})},
                    z_map={
                        0: PauliString({0: Axis.X, 1: Axis.X}),
                        1: PauliString({0: Axis.Z, 1: Axis.Z}),
                        2: PauliString({2: Axis.Z}),
                    },
                    input_nodes=[0, 1, 2],
                    output_nodes=[0, 1, 2],
                ),
                MatGF2(
                    [
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 1, 0],
                        [1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                    ]
                ),
            ),
        ],
    )
    def test_to_tableau(self, cm: CliffordMap, tab_ref: MatGF2) -> None:
        tab = cm.to_tableau()
        assert np.all(tab == tab_ref)

    # The CliffordMap test cases were generated using conversion tools in the graphix-stim-compiler plugin and stim.Tableau.random.
    @pytest.mark.parametrize(
        ("qc_ref", "cm"),
        [
            (
                Circuit(width=1, instr=[H(0)]),
                CliffordMap(
                    x_map={0: PauliString(axes={0: Axis.Z}, sign=Sign.PLUS)},
                    z_map={0: PauliString(axes={0: Axis.X}, sign=Sign.PLUS)},
                    input_nodes=[0],
                    output_nodes=[0],
                ),
            ),
            (
                Circuit(width=1, instr=[S(0)]),
                CliffordMap(
                    x_map={0: PauliString(axes={0: Axis.Y}, sign=Sign.PLUS)},
                    z_map={0: PauliString(axes={0: Axis.Z}, sign=Sign.PLUS)},
                    input_nodes=[0],
                    output_nodes=[0],
                ),
            ),
            (
                Circuit(width=2, instr=[CNOT(1, 0)]),
                CliffordMap(
                    x_map={
                        0: PauliString(axes={0: Axis.X, 1: Axis.X}, sign=Sign.PLUS),
                        1: PauliString(axes={1: Axis.X}, sign=Sign.PLUS),
                    },
                    z_map={
                        0: PauliString(axes={0: Axis.Z}, sign=Sign.PLUS),
                        1: PauliString(axes={0: Axis.Z, 1: Axis.Z}, sign=Sign.PLUS),
                    },
                    input_nodes=[0, 1],
                    output_nodes=[0, 1],
                ),
            ),
            (
                Circuit(width=3, instr=[CNOT(1, 0), H(0), H(1), CNOT(2, 1), S(1), CNOT(2, 0), H(2), S(2)]),
                CliffordMap(
                    x_map={
                        0: PauliString(axes={0: Axis.Z, 1: Axis.Z}, sign=Sign.PLUS),
                        1: PauliString(axes={1: Axis.Z}, sign=Sign.PLUS),
                        2: PauliString(axes={2: Axis.Z}, sign=Sign.PLUS),
                    },
                    z_map={
                        0: PauliString(axes={0: Axis.X, 2: Axis.Z}, sign=Sign.PLUS),
                        1: PauliString(axes={0: Axis.X, 1: Axis.Y}, sign=Sign.PLUS),
                        2: PauliString(axes={0: Axis.Z, 1: Axis.Z, 2: Axis.Y}, sign=Sign.PLUS),
                    },
                    input_nodes=[0, 1, 2],
                    output_nodes=[0, 1, 2],
                ),
            ),
            (
                Circuit(width=1, instr=[S(0), S(0), S(0)]),
                CliffordMap(
                    x_map={0: PauliString(axes={0: Axis.Y}, sign=Sign.MINUS)},
                    z_map={0: PauliString(axes={0: Axis.Z}, sign=Sign.PLUS)},
                    input_nodes=[0],
                    output_nodes=[0],
                ),
            ),
            (
                Circuit(width=2, instr=[CNOT(1, 0), H(0), S(0), S(0), H(0), S(1), S(1)]),
                CliffordMap(
                    x_map={
                        0: PauliString(axes={0: Axis.X, 1: Axis.X}, sign=Sign.MINUS),
                        1: PauliString(axes={1: Axis.X}, sign=Sign.MINUS),
                    },
                    z_map={
                        0: PauliString(axes={0: Axis.Z}, sign=Sign.MINUS),
                        1: PauliString(axes={0: Axis.Z, 1: Axis.Z}, sign=Sign.MINUS),
                    },
                    input_nodes=[0, 1],
                    output_nodes=[0, 1],
                ),
            ),
            (
                Circuit(
                    width=3,
                    instr=[
                        S(0),
                        H(2),
                        CNOT(2, 0),
                        H(1),
                        H(2),
                        CNOT(0, 1),
                        CNOT(0, 2),
                        S(2),
                        CNOT(2, 1),
                        H(2),
                        CNOT(1, 2),
                        H(2),
                        H(1),
                        S(1),
                        S(1),
                        H(1),
                        S(0),
                        S(0),
                        S(1),
                        S(1),
                    ],
                ),
                CliffordMap(
                    x_map={
                        0: PauliString(axes={0: Axis.Y, 1: Axis.Z, 2: Axis.X}, sign=Sign.PLUS),
                        1: PauliString(axes={1: Axis.Z, 2: Axis.X}, sign=Sign.MINUS),
                        2: PauliString(axes={0: Axis.Y, 1: Axis.Z}, sign=Sign.PLUS),
                    },
                    z_map={
                        0: PauliString(axes={0: Axis.Z, 1: Axis.X, 2: Axis.Z}, sign=Sign.MINUS),
                        1: PauliString(axes={0: Axis.X, 1: Axis.X, 2: Axis.X}, sign=Sign.PLUS),
                        2: PauliString(axes={1: Axis.Y, 2: Axis.Y}, sign=Sign.PLUS),
                    },
                    input_nodes=[0, 1, 2],
                    output_nodes=[0, 1, 2],
                ),
            ),
            (
                Circuit(
                    width=4,
                    instr=[
                        CNOT(0, 3),
                        CNOT(3, 0),
                        CNOT(0, 3),
                        S(0),
                        H(0),
                        S(0),
                        H(2),
                        CNOT(2, 0),
                        H(1),
                        H(3),
                        CNOT(0, 1),
                        CNOT(0, 3),
                        S(1),
                        S(2),
                        S(3),
                        CNOT(2, 1),
                        CNOT(3, 1),
                        S(3),
                        H(3),
                        CNOT(1, 2),
                        CNOT(1, 3),
                        CNOT(2, 3),
                        CNOT(3, 2),
                        CNOT(2, 3),
                        S(2),
                        S(3),
                        H(3),
                        S(3),
                        H(0),
                        S(0),
                        S(0),
                        H(0),
                        S(0),
                        S(0),
                        S(1),
                        S(1),
                        S(2),
                        S(2),
                    ],
                ),
                CliffordMap(
                    x_map={
                        0: PauliString(axes={1: Axis.Y, 2: Axis.X, 3: Axis.Y}, sign=Sign.PLUS),
                        1: PauliString(axes={1: Axis.Z, 2: Axis.Z, 3: Axis.Y}, sign=Sign.PLUS),
                        2: PauliString(axes={0: Axis.Z, 1: Axis.Y, 2: Axis.X}, sign=Sign.MINUS),
                        3: PauliString(axes={0: Axis.X, 1: Axis.Y, 2: Axis.Z, 3: Axis.X}, sign=Sign.PLUS),
                    },
                    z_map={
                        0: PauliString(axes={0: Axis.X, 1: Axis.Z, 3: Axis.Y}, sign=Sign.PLUS),
                        1: PauliString(axes={0: Axis.X, 1: Axis.Y, 2: Axis.Y, 3: Axis.Z}, sign=Sign.MINUS),
                        2: PauliString(axes={1: Axis.Y, 2: Axis.Z, 3: Axis.X}, sign=Sign.MINUS),
                        3: PauliString(axes={0: Axis.Y, 1: Axis.Z, 2: Axis.X, 3: Axis.X}, sign=Sign.PLUS),
                    },
                    input_nodes=[0, 1, 2, 3],
                    output_nodes=[0, 1, 2, 3],
                ),
            ),
        ],
    )
    def test_cm_berg_pass(self, qc_ref: Circuit, cm: CliffordMap, fx_rng: Generator) -> None:

        qc = Circuit(qc_ref.width)
        cm_berg_pass(cm, qc)

        s_test = qc.simulate_statevector(rng=fx_rng).statevec
        s_ref = qc_ref.simulate_statevector(rng=fx_rng).statevec

        assert s_test.isclose(s_ref)


class TestExtraction:
    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_extract_rnd_circuit(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 2
        depth = 2
        circuit_ref = rand_circuit(nqubits, depth, rng, use_ccx=False)
        pattern = circuit_ref.transpile().pattern

        circuit = pattern.extract_opengraph().extract_circuit()

        s_ref = circuit.simulate_statevector(rng=rng).statevec
        s_test = circuit_ref.simulate_statevector(rng=rng).statevec
        assert s_ref.isclose(s_test)

    @pytest.mark.parametrize(
        "test_case",
        [
            OpenGraph(
                graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
                input_nodes=[1, 2],
                output_nodes=[5, 6],
                measurements={
                    1: Measurement.XY(0.1),
                    2: Measurement.XY(0.2),
                    3: Measurement.XY(0.3),
                    4: Measurement.XY(0.4),
                },
            ),
            OpenGraph(
                graph=nx.Graph([(1, 4), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)]),
                input_nodes=[1, 2, 3],
                output_nodes=[4, 5, 6],
                measurements={
                    1: Measurement.XY(0.1),
                    2: Measurement.XY(0.2),
                    3: Measurement.XY(0.3),
                },
            ),
            OpenGraph(
                graph=nx.Graph([(0, 1), (0, 2), (0, 4), (1, 5), (2, 4), (2, 5), (3, 5)]),
                input_nodes=[0, 1],
                output_nodes=[4, 5],
                measurements={
                    0: Measurement.XY(0.1),
                    1: Measurement.XY(0.1),
                    2: Measurement.XZ(0.2),
                    3: Measurement.YZ(0.3),
                },
            ),
            OpenGraph(
                graph=nx.Graph([(0, 1), (1, 2), (1, 4), (2, 3)]),
                input_nodes=[0],
                output_nodes=[4],
                measurements={
                    0: Measurement.XY(0.1),  # XY
                    1: Measurement.X,  # X
                    2: Measurement.XY(0.1),  # XY
                    3: Measurement.X,  # X
                },
            ),
            OpenGraph(
                graph=nx.Graph([(0, 1), (1, 2)]),
                input_nodes=[0],
                output_nodes=[2],
                measurements={
                    0: Measurement.XY(0.1),  # XY
                    1: Measurement.Y,  # Y
                },
            ),
            OpenGraph(
                graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)]),
                input_nodes=[0, 1],
                output_nodes=[7, 6],
                measurements={
                    0: Measurement.XY(0.1),  # XY
                    1: Measurement.XY(0.1),  # XY
                    2: Measurement.X,  # X
                    3: Measurement.XY(0.1),  # XY
                    4: Measurement.X,  # X
                    5: Measurement.Y,  # Y
                },
            ),
        ],
    )
    def test_extract_og(self, test_case: OpenGraph[Measurement], fx_rng: Generator) -> None:
        pattern = test_case.to_pattern()
        circuit = test_case.extract_circuit()

        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_ref = pattern.simulate_pattern(rng=fx_rng)
        assert state.isclose(state_ref)

    @pytest.mark.parametrize("infer_pauli", [True, False])
    def test_extract_og_infer_pauli(self, infer_pauli: bool, fx_rng: Generator) -> None:
        og: OpenGraph[Measurement] = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
            input_nodes=[0],
            output_nodes=[5],
            measurements={
                0: Measurement.XY(0),
                1: Measurement.XY(0),
                2: Measurement.XY(0),
                3: Measurement.XY(0),
                4: Measurement.XY(0),
            },
        )
        pattern = og.to_pattern()
        if infer_pauli:
            og = og.infer_pauli_measurements()

        circuit = og.extract_circuit()

        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_ref = pattern.simulate_pattern(rng=fx_rng)
        assert state.isclose(state_ref)

    def test_extract_og_gflow(self, fx_rng: Generator) -> None:
        og = OpenGraph(
            graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
            input_nodes=[1, 2],
            output_nodes=[5, 6],
            measurements={
                1: Measurement.XY(0.1),
                2: Measurement.XY(0.2),
                3: Measurement.XY(0.3),
                4: Measurement.XY(0.4),
            },
        )
        pattern = og.to_pattern()
        circuit = og.extract_gflow().extract_circuit().to_circuit()

        state = circuit.simulate_statevector(rng=fx_rng).statevec
        state_ref = pattern.simulate_pattern(rng=fx_rng)
        assert state.isclose(state_ref)

    @pytest.mark.parametrize("test_case", [0.2, 0.5, 1.0])
    def test_parametric_angles(self, test_case: float, fx_rng: Generator) -> None:
        alpha = Placeholder("alpha")
        alpha_val = test_case
        og = OpenGraph(
            graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
            input_nodes=[1, 2],
            output_nodes=[5, 6],
            measurements={
                1: Measurement.XY(0.1),
                2: Measurement.XY(alpha),
                3: Measurement.XY(0.3),
                4: Measurement.XY(alpha),
            },
        )

        # Substitute parameter at the level of the extracted circuit
        qc1 = og.extract_circuit()
        s1 = qc1.subs(alpha, alpha_val).simulate_statevector(rng=fx_rng).statevec

        # Substitute parameter at the level of the open graph object
        # Calling `infer_pauli_measurements` is not necessary for the test to pass
        # (and it should not be), but it suppresses the warnings.
        qc2 = og.subs(alpha, alpha_val).infer_pauli_measurements().extract_circuit()
        s2 = qc2.simulate_statevector(rng=fx_rng).statevec

        assert s1.isclose(s2)


def test_extend_input() -> None:
    og = OpenGraph(
        graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
        input_nodes=[1, 2],
        output_nodes=[5, 6],
        measurements={
            1: Measurement.XY(0.1),
            2: Measurement.XY(0.2),
            3: Measurement.XY(0.3),
            4: Measurement.XY(0.4),
        },
    )

    og_ref = OpenGraph(
        graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6), (1, 8), (2, 7)]),
        input_nodes=[8, 7],
        output_nodes=[5, 6],
        measurements={
            1: Measurement.XY(0.1),
            2: Measurement.XY(0.2),
            3: Measurement.XY(0.3),
            4: Measurement.XY(0.4),
            7: Measurement.X,
            8: Measurement.X,
        },
    )

    og_ext, ancillary_inputs_map = extend_input(og)

    assert og_ext.isclose(og_ref)
    assert ancillary_inputs_map == {1: 8, 2: 7}

    flow = og_ext.infer_pauli_measurements().extract_pauli_flow()
    assert flow.is_focused()
