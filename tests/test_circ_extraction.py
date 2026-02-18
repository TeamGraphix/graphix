from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import pytest

from graphix.circ_ext.compilation import LadderPass
from graphix.circ_ext.extraction import PauliExponential, PauliExponentialDAG, PauliString, extend_input
from graphix.flow.core import PauliFlow
from graphix.fundamentals import ANGLE_PI
from graphix.instruction import CNOT, RX, RY, RZ, H
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.sim.base_backend import NodeIndex
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from numpy.random import Generator


class TestPauliString:
    def test_add_circuit(self, fx_rng: Generator) -> None:
        angle = 0.3 * ANGLE_PI
        angle_rz = -2 * angle
        x_nodes = {1}
        z_nodes = {4, 2}
        pauli_string = PauliString(x_nodes=x_nodes, z_nodes=z_nodes)

        pexp = PauliExponential(angle, pauli_string)

        qc = Circuit(4)
        outputs_mapping = NodeIndex()
        outputs_mapping.extend([2, 1, 3, 4])

        LadderPass.add_pexp(pexp, outputs_mapping, qc)  # `qc` is modified in place

        qc_ref = Circuit(width=4, instr=[H(1), CNOT(3, 1), CNOT(0, 3), RZ(0, angle_rz), CNOT(0, 3), CNOT(3, 1), H(1)])

        state = qc.simulate_statevector(rng=fx_rng).statevec
        state_ref = qc_ref.simulate_statevector(rng=fx_rng).statevec

        assert state.isclose(state_ref)


class PauliExpTestCase(NamedTuple):
    p_exp: PauliExponentialDAG
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
                        0: PauliExponential(alpha / 2, PauliString(z_nodes={1}, negative_sign=True)),
                    },
                    partial_order_layers=[{1}, {0}],
                    output_nodes=[1],
                ),
                Circuit(width=1, instr=[RZ(0, alpha)]),
            ),
            PauliExpTestCase(
                PauliExponentialDAG(
                    pauli_exponentials={
                        0: PauliExponential(alpha / 2, PauliString(x_nodes={1}, negative_sign=True)),
                    },
                    partial_order_layers=[{1}, {0}],
                    output_nodes=[1],
                ),
                Circuit(width=1, instr=[RX(0, alpha)]),
            ),
            PauliExpTestCase(
                PauliExponentialDAG(
                    pauli_exponentials={
                        0: PauliExponential(alpha / 2, PauliString(y_nodes={1}, negative_sign=True)),
                    },
                    partial_order_layers=[{1}, {0}],
                    output_nodes=[1],
                ),
                Circuit(width=1, instr=[RY(0, alpha)]),
            ),
            PauliExpTestCase(
                PauliExponentialDAG(
                    pauli_exponentials={
                        0: PauliExponential(ANGLE_PI / 4, PauliString(z_nodes={3}, negative_sign=True)),
                        1: PauliExponential(ANGLE_PI / 4, PauliString(x_nodes={3}, negative_sign=True)),
                        2: PauliExponential(ANGLE_PI / 4, PauliString(z_nodes={3}, negative_sign=True)),
                    },
                    partial_order_layers=[{3}, {2}, {1}, {0}],
                    output_nodes=[3],
                ),
                Circuit(width=1, instr=[H(0)]),
            ),
            PauliExpTestCase(
                PauliExponentialDAG(
                    pauli_exponentials={
                        0: PauliExponential(ANGLE_PI / 4, PauliString(x_nodes={3})),
                        1: PauliExponential(ANGLE_PI / 4, PauliString(z_nodes={5})),
                        2: PauliExponential(ANGLE_PI / 4, PauliString(x_nodes={3}, z_nodes={5}, negative_sign=True)),
                    },
                    partial_order_layers=[{5, 3}, {2}, {0, 1}],
                    output_nodes=[5, 3],  # Node 5 -> qubit 0 (control), node 3 -> qubit 1 (target)
                ),
                Circuit(width=2, instr=[CNOT(1, 0)]),
            ),
        ],
    )
    def test_to_circuit(self, test_case: PauliExpTestCase) -> None:
        qc = LadderPass.add_to_circuit(test_case.p_exp)
        state = qc.simulate_statevector().statevec
        state_ref = test_case.qc.simulate_statevector().statevec
        assert state.isclose(state_ref)

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
                0: PauliExponential(ANGLE_PI * 0.1 / 2, PauliString(x_nodes=frozenset({6}))),
                1: PauliExponential(ANGLE_PI * 0.2 / 2, PauliString(y_nodes=frozenset({6}), z_nodes=frozenset({5}))),
                2: PauliExponential(
                    ANGLE_PI * 0.3 / 2, PauliString(y_nodes=frozenset({5}), z_nodes=frozenset({6}), negative_sign=True)
                ),
                3: PauliExponential(ANGLE_PI * 0.4 / 2, PauliString(x_nodes=frozenset({5}))),
                4: PauliExponential(
                    0, PauliString(x_nodes=frozenset({6}))
                ),  # The angle is 0 (interpreted from the Pauli measurement).
            },
            partial_order_layers=flow.partial_order_layers,
            output_nodes=flow.og.output_nodes,
        )

        assert pexp_dag == pexp_dag_ref


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
            7: Measurement.XY(0),
            8: Measurement.XY(0),
        },
    )

    og_ext, ancillary_inputs_map = extend_input(og)

    assert og_ext.isclose(og_ref)
    assert ancillary_inputs_map == {1: 8, 2: 7}

    flow = og_ext.extract_pauli_flow()
    assert flow.is_focused()
