from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
import pytest

from graphix.command import E, M, N, X, Z
from graphix.flow.core import CausalFlow, GFlow, PauliFlow, XZCorrections
from graphix.fundamentals import AbstractMeasurement, AbstractPlanarMeasurement, Axis, Plane
from graphix.measurements import Measurement
from graphix.opengraph_ import OpenGraph
from graphix.pattern import Pattern
from graphix.states import PlanarState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.random import Generator


def generate_causal_flow_0() -> CausalFlow[Plane]:
    """Generate causal flow on linear open graph.

    Open graph structure:

        [0]-1-2-(3)

    Causal flow:
        c(0) = 1, c(1) = 2, c(2) = 3
        {3} > {2} > {1} > {0}
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
        input_nodes=[0],
        output_nodes=[3],
        measurements=dict.fromkeys(range(3), Plane.XY),
    )
    return CausalFlow(
        og=og,
        correction_function={0: {1}, 1: {2}, 2: {3}},
        partial_order_layers=[{3}, {2}, {1}, {0}],
    )


def generate_causal_flow_1() -> CausalFlow[Measurement]:
    """Generate causal flow on H-shaped open graph.

    Open graph structure:

        [0]-2-(4)
            |
        [1]-3-(5)

    Causal flow:
        c(0) = 2, c(1) = 3, c(2) = 4, c(3) = 5
        {4, 5} > {2, 3} > {0, 1}
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (2, 3), (1, 3), (2, 4), (3, 5)]),
        input_nodes=[0, 1],
        output_nodes=[4, 5],
        measurements=dict.fromkeys(range(4), Measurement(angle=0, plane=Plane.XY)),
    )
    return CausalFlow(
        og=og,
        correction_function={0: {2}, 1: {3}, 2: {4}, 3: {5}},
        partial_order_layers=[{4, 5}, {2, 3}, {0, 1}],
    )


def generate_gflow_0() -> GFlow[Measurement]:
    """Generate gflow on H-shaped open graph.

    Open graph structure:

        [0]-2-(4)
            |
        [1]-3-(5)

    GFlow:
        g(0) = {2, 5}, g(1) = {3, 4}, g(2) = {4}, g(3) = {5}
        {4, 5} > {0, 1, 2, 3}

    Notes
    -----
    This is the same open graph as in `:func: generate_causal_flow_1` but now we consider a gflow which has lower depth than the causal flow.
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (2, 3), (1, 3), (2, 4), (3, 5)]),
        input_nodes=[0, 1],
        output_nodes=[4, 5],
        measurements=dict.fromkeys(range(4), Measurement(angle=0, plane=Plane.XY)),
    )
    return GFlow(
        og=og,
        correction_function={0: {2, 5}, 1: {3, 4}, 2: {4}, 3: {5}},
        partial_order_layers=[{4, 5}, {0, 1, 2, 3}],
    )


def generate_gflow_1() -> GFlow[Plane]:
    r"""Generate gflow on open graph without causal flow.

    Open graph structure:

        1
         \
         (4)-[0]-(3)
         /
        2

    GFlow:
        g(0) = {3}, g(1) = {1}, g(2) = {2, 3, 4}
        {3, 4} > {1} > {0, 2}
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 3), (0, 4), (1, 4), (2, 4)]),
        input_nodes=[0],
        output_nodes=[3, 4],
        measurements={0: Plane.XY, 1: Plane.YZ, 2: Plane.XZ},
    )
    return GFlow(
        og=og,
        correction_function={0: {3}, 1: {1}, 2: {2, 3, 4}},
        partial_order_layers=[{3, 4}, {1}, {0, 2}],
    )


def generate_gflow_2() -> GFlow[Plane]:
    r"""Generate gflow on open graph without causal flow.

    Open graph structure:

        [0]-(3)
           X
        [1]-(4)
           X
        [2]-(5)

    GFlow:
        g(0) = {4, 5}, g(1) = {3, 4, 5}, g(2) = {3, 4}
        {3, 4, 5} > {0, 1, 2}
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 3), (0, 4), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5)]),
        input_nodes=[0, 1, 2],
        output_nodes=[3, 4, 5],
        measurements=dict.fromkeys(range(3), Plane.XY),
    )
    return GFlow(
        og=og,
        correction_function={0: {4, 5}, 1: {3, 4, 5}, 2: {3, 4}},
        partial_order_layers=[{3, 4}, {1}, {0, 2}],
    )


def generate_pauli_flow_0() -> PauliFlow[Axis]:
    """Generate Pauli flow on linear open graph.

    Open graph structure:

        [0]-1-2-(3)

    Pauli flow:
        p(0) = {1, 3}, p(1) = {2}, p(2) = {3}
        {3} > {0, 1, 2}
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
        input_nodes=[0],
        output_nodes=[3],
        measurements=dict.fromkeys(range(3), Axis.X),
    )
    return PauliFlow(
        og=og,
        correction_function={0: {1, 3}, 1: {2}, 2: {3}},
        partial_order_layers=[{3}, {0, 1, 2}],
    )


def generate_pauli_flow_1() -> PauliFlow[Measurement]:
    """Generate Pauli flow on double-H-shaped open graph.

    Open graph structure:

        [0]-2-4-(6)
            | |
        [1]-3-5-(7)

    Pauli flow:
        p(0) = {2, 5, 7}, p(1) = {3, 4}, p(2) = {4, 7}, p(3) = {5, 6, 7},
        p(4) = {6}, p(5) = 7
        {6, 7} > {3} > {0, 1, 2, 4, 5}
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)]),
        input_nodes=[0, 1],
        output_nodes=[6, 7],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XY),  # XY
            2: Measurement(0.0, Plane.XY),  # X
            3: Measurement(0.1, Plane.XY),  # XY
            4: Measurement(0.0, Plane.XY),  # X
            5: Measurement(0.5, Plane.XY),  # Y
        },
    )
    return PauliFlow(
        og=og,
        correction_function={0: {2, 5, 7}, 1: {3, 4}, 2: {4, 7}, 3: {5, 6, 7}, 4: {6}, 5: {7}},
        partial_order_layers=[{6, 7}, {3}, {0, 1, 2, 4, 5}],
    )


class XZCorrectionsTestCase(NamedTuple):
    flow: CausalFlow[AbstractPlanarMeasurement] | GFlow[AbstractPlanarMeasurement] | PauliFlow[AbstractMeasurement]
    x_corr: Mapping[int, set[int]]
    z_corr: Mapping[int, set[int]]
    pattern: Pattern | None
    # Patterns can only be extracted from `Measurement`-type objects. If `flow` is of parametric type, we set `pattern = None`.


def prepare_test_xzcorrections() -> list[XZCorrectionsTestCase]:
    test_cases: list[XZCorrectionsTestCase] = []

    test_cases.extend(
        (
            XZCorrectionsTestCase(
                flow=generate_causal_flow_0(), x_corr={0: {1}, 1: {2}, 2: {3}}, z_corr={0: {2}, 1: {3}}, pattern=None
            ),
            XZCorrectionsTestCase(
                flow=generate_causal_flow_1(),
                x_corr={0: {2}, 1: {3}, 2: {4}, 3: {5}},
                z_corr={0: {3, 4}, 1: {2, 5}},
                pattern=Pattern(
                    input_nodes=[0, 1],
                    cmds=[
                        N(2),
                        N(3),
                        N(4),
                        N(5),
                        E((0, 2)),
                        E((2, 3)),
                        E((2, 4)),
                        E((3, 1)),
                        E((3, 5)),
                        M(0),
                        Z(3, {0}),
                        Z(4, {0}),
                        X(2, {0}),
                        M(1),
                        Z(2, {1}),
                        Z(5, {1}),
                        X(3, {1}),
                        M(2),
                        X(4, {2}),
                        M(3),
                        X(5, {3}),
                    ],
                    output_nodes=[4, 5],
                ),
            ),
            XZCorrectionsTestCase(
                flow=generate_gflow_0(),
                x_corr={0: {2, 5}, 1: {3, 4}, 2: {4}, 3: {5}},
                z_corr={0: {4}, 1: {5}},
                pattern=Pattern(
                    input_nodes=[0, 1],
                    cmds=[
                        N(2),
                        N(3),
                        N(4),
                        N(5),
                        E((0, 2)),
                        E((2, 3)),
                        E((2, 4)),
                        E((3, 1)),
                        E((3, 5)),
                        M(0),
                        Z(3, {0}),
                        Z(4, {0}),
                        X(2, {0}),
                        M(1),
                        Z(2, {1}),
                        Z(5, {1}),
                        X(3, {1}),
                        M(2),
                        X(4, {2}),
                        M(3),
                        X(5, {3}),
                    ],
                    output_nodes=[4, 5],
                ),
            ),
            XZCorrectionsTestCase(
                flow=generate_gflow_1(),
                x_corr={0: {3}, 2: {3, 4}},
                z_corr={1: {4}, 2: {1, 4}},
                pattern=None,
            ),
            XZCorrectionsTestCase(
                flow=generate_gflow_2(), x_corr={0: {4, 5}, 1: {3, 4, 5}, 2: {3, 4}}, z_corr={}, pattern=None
            ),
            XZCorrectionsTestCase(
                flow=generate_pauli_flow_0(),
                x_corr={0: {3}, 2: {3}},
                z_corr={1: {3}},
                pattern=None,
            ),
            XZCorrectionsTestCase(
                flow=generate_pauli_flow_1(),
                x_corr={0: {7}, 1: {3}, 2: {7}, 3: {6, 7}, 4: {6}, 5: {7}},
                z_corr={0: {7}, 1: {6}, 2: {6}, 3: {7}},
                pattern=Pattern(
                    input_nodes=[0, 1],
                    cmds=[
                        N(2),
                        N(3),
                        N(4),
                        N(5),
                        N(6),
                        N(7),
                        E((0, 2)),
                        E((2, 3)),
                        E((2, 4)),
                        E((1, 3)),
                        E((3, 5)),
                        E((4, 5)),
                        E((4, 6)),
                        E((5, 7)),
                        M(0, angle=0.1),
                        Z(3, {0}),
                        Z(4, {0}),
                        X(2, {0}),
                        M(1, angle=0.1),
                        Z(2, {1}),
                        Z(5, {1}),
                        X(3, {1}),
                        M(2),
                        Z(5, {2}),
                        Z(6, {2}),
                        X(4, {2}),
                        M(3, angle=0.1),
                        Z(4, {3}),
                        Z(7, {3}),
                        X(5, {3}),
                        M(4),
                        X(6, {4}),
                        M(5, angle=0.5),
                        X(7, {5}),
                    ],
                    output_nodes=[6, 7],
                ),
            ),
        )
    )

    return test_cases


class TestFlowPatternConversion:
    """Bundle for unit tests of the flow to XZ-corrections to pattern methods.

    The module `tests.test_opengraph.py` provides an additional (more comprehensive) suite of unit tests on this transformation.
    """

    @pytest.mark.parametrize("test_case", prepare_test_xzcorrections())
    def test_flow_to_corrections(self, test_case: XZCorrectionsTestCase) -> None:
        flow = test_case.flow
        corrections = flow.to_corrections()
        assert corrections.z_corrections == test_case.z_corr
        assert corrections.x_corrections == test_case.x_corr

    @pytest.mark.parametrize("test_case", prepare_test_xzcorrections())
    def test_corrections_to_pattern(self, test_case: XZCorrectionsTestCase, fx_rng: Generator) -> None:
        if test_case.pattern is not None:
            pattern = test_case.flow.to_corrections().to_pattern()  # type: ignore[misc]
            n_shots = 2
            results = []

            for plane in {Plane.XY, Plane.XZ, Plane.YZ}:
                alpha = 2 * np.pi * fx_rng.random()
                state_ref = test_case.pattern.simulate_pattern(input_state=PlanarState(plane, alpha))

                for _ in range(n_shots):
                    state = pattern.simulate_pattern(input_state=PlanarState(plane, alpha))
                    results.append(np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())))

            avg = sum(results) / (n_shots * 3)

            assert avg == pytest.approx(1)


class TestXZCorrections:
    """Bundle for unit tests of :class:`XZCorrections`."""

    # See `:func: generate_causal_flow_0`
    def test_order_0(self) -> None:
        corrections = generate_causal_flow_0().to_corrections()

        assert corrections.generate_total_measurement_order() == [0, 1, 2]
        assert corrections.is_compatible([0, 1, 2])  # Correct order
        assert not corrections.is_compatible([1, 0, 2])  # Wrong order
        assert not corrections.is_compatible([1, 2])  # Incomplete order
        assert not corrections.is_compatible([0, 1, 2, 3])  # Contains outputs

        assert nx.utils.graphs_equal(corrections.extract_dag(), nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 3)]))

    # See `:func: generate_causal_flow_1`
    def test_order_1(self) -> None:
        og = OpenGraph(
            graph=nx.Graph([(0, 2), (2, 3), (1, 3), (2, 4), (3, 5)]),
            input_nodes=[0, 1],
            output_nodes=[4, 5],
            measurements=dict.fromkeys(range(4), Measurement(angle=0, plane=Plane.XY)),
        )

        corrections = XZCorrections.from_measured_nodes_mapping(
            og=og, x_corrections={0: {2}, 1: {3}, 2: {4}, 3: {5}}, z_corrections={0: {3, 4}, 1: {2, 5}}
        )

        assert corrections.is_compatible([0, 1, 2, 3])
        assert corrections.is_compatible([1, 0, 2, 3])
        assert corrections.is_compatible([1, 0, 3, 2])
        assert not corrections.is_compatible([0, 2, 1, 3])  # Wrong order
        assert not corrections.is_compatible([1, 0, 3])  # Incomplete order
        assert not corrections.is_compatible([0, 1, 1, 2, 3])  # Duplicates
        assert not corrections.is_compatible([0, 1, 2, 3, 4, 5])  # Contains outputs

        assert nx.utils.graphs_equal(
            corrections.extract_dag(), nx.DiGraph([(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 5), (2, 4), (3, 5)])
        )

    # Incomplete corrections
    def test_order_2(self) -> None:
        og = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2), (1, 3)]),
            input_nodes=[0],
            output_nodes=[2, 3],
            measurements=dict.fromkeys(range(2), Measurement(angle=0, plane=Plane.XY)),
        )

        corrections = XZCorrections.from_measured_nodes_mapping(og=og, x_corrections={1: {0}})

        assert corrections.partial_order_layers == [{2, 3}, {0}, {1}]
        assert corrections.is_compatible([1, 0])
        assert not corrections.is_compatible([0, 1])  # Wrong order
        assert not corrections.is_compatible([0])  # Incomplete order
        assert not corrections.is_compatible([0, 0, 1])  # Duplicates
        assert not corrections.is_compatible([1, 0, 2, 3])  # Contains outputs

        assert nx.utils.graphs_equal(corrections.extract_dag(), nx.DiGraph([(1, 0)]))

    # OG without outputs
    def test_order_3(self) -> None:
        og = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2)]),
            input_nodes=[0],
            output_nodes=[],
            measurements=dict.fromkeys(range(3), Measurement(angle=0, plane=Plane.XY)),
        )

        corrections = XZCorrections.from_measured_nodes_mapping(
            og=og, x_corrections={0: {1, 2}}, z_corrections={0: {1}}
        )

        assert corrections.partial_order_layers == [{1, 2}, {0}]
        assert corrections.is_compatible([0, 1, 2])
        assert not corrections.is_compatible([2, 0, 1])  # Wrong order
        assert not corrections.is_compatible([0, 1])  # Incomplete order
        assert corrections.generate_total_measurement_order() in ([0, 1, 2], [0, 2, 1])
        assert nx.utils.graphs_equal(corrections.extract_dag(), nx.DiGraph([(0, 1), (0, 2)]))

    # Test exceptions
    def test_from_measured_nodes_mapping_exceptions(self) -> None:
        og = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
            input_nodes=[0],
            output_nodes=[3],
            measurements=dict.fromkeys(range(3), Measurement(angle=0, plane=Plane.XY)),
        )
        with pytest.raises(ValueError, match=r"Keys of input X-corrections contain non-measured nodes."):
            XZCorrections.from_measured_nodes_mapping(og=og, x_corrections={3: {1, 2}})

        with pytest.raises(ValueError, match=r"Keys of input Z-corrections contain non-measured nodes."):
            XZCorrections.from_measured_nodes_mapping(og=og, z_corrections={3: {1, 2}})

        with pytest.raises(
            ValueError,
            match=r"Input XZ-corrections are not runnable since the induced directed graph contains closed loops.",
        ):
            XZCorrections.from_measured_nodes_mapping(og=og, x_corrections={0: {1}, 1: {2}}, z_corrections={2: {0}})

        with pytest.raises(
            ValueError, match=r"Values of input mapping contain labels which are not nodes of the input open graph."
        ):
            XZCorrections.from_measured_nodes_mapping(og=og, x_corrections={0: {4}})
