from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
import pytest

from graphix.command import E, M, N, X, Z
from graphix.flow.core import (
    CausalFlow,
    GFlow,
    PauliFlow,
    XZCorrections,
)
from graphix.flow.exceptions import (
    FlowError,
    FlowGenericError,
    FlowGenericErrorReason,
    FlowPropositionError,
    FlowPropositionErrorReason,
    FlowPropositionOrderError,
    FlowPropositionOrderErrorReason,
    PartialOrderError,
    PartialOrderErrorReason,
    PartialOrderLayerError,
    PartialOrderLayerErrorReason,
    XZCorrectionsError,
    XZCorrectionsGenericError,
    XZCorrectionsGenericErrorReason,
    XZCorrectionsOrderError,
    XZCorrectionsOrderErrorReason,
)
from graphix.fundamentals import ANGLE_PI, AbstractMeasurement, AbstractPlanarMeasurement, Axis, Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.parameter import Placeholder
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
        {4, 5} > {2, 3} > {0, 1}

    Notes
    -----
    This is the same open graph as in `:func: generate_causal_flow_1` but now we consider a gflow.
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
        partial_order_layers=[{4, 5}, {2, 3}, {0, 1}],
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
        partial_order_layers=[{3, 4, 5}, {1}, {0, 2}],
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
                        Z(4, {0}),
                        X(2, {0}),
                        X(5, {0}),
                        M(1),
                        Z(5, {1}),
                        X(3, {1}),
                        X(4, {1}),
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
        flow.check_well_formed()
        corrections = flow.to_corrections()
        corrections.check_well_formed()
        assert corrections.z_corrections == test_case.z_corr
        assert corrections.x_corrections == test_case.x_corr

    @pytest.mark.parametrize("test_case", prepare_test_xzcorrections())
    def test_corrections_to_pattern(self, test_case: XZCorrectionsTestCase, fx_rng: Generator) -> None:
        if test_case.pattern is not None:
            pattern = test_case.flow.to_corrections().to_pattern()  # type: ignore[misc]
            n_shots = 2

            for plane in {Plane.XY, Plane.XZ, Plane.YZ}:
                alpha = 2 * ANGLE_PI * fx_rng.random()
                state_ref = test_case.pattern.simulate_pattern(input_state=PlanarState(plane, alpha))

                for _ in range(n_shots):
                    state = pattern.simulate_pattern(input_state=PlanarState(plane, alpha))
                    result = np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten()))
                    assert result == pytest.approx(1)


class TestFlow:
    """Bundle for unit tests of :class:`PauliFlow` and children."""

    # Test parametric angle replacement
    def test_subs(self) -> None:
        alpha = Placeholder("alpha")
        value = 0.3
        og = OpenGraph(
            graph=nx.Graph([(0, 1)]), input_nodes=[0], output_nodes=[1], measurements={0: Measurement(alpha, Plane.XY)}
        )
        flow = og.extract_pauli_flow()

        og_ref = OpenGraph(
            graph=nx.Graph([(0, 1)]), input_nodes=[0], output_nodes=[1], measurements={0: Measurement(value, Plane.XY)}
        )
        flow_ref = og_ref.extract_pauli_flow()

        flow_test = flow.subs(alpha, value)

        assert not flow.og.isclose(flow_test.og)
        assert flow_ref.og.isclose(flow_test.og)

        assert flow_ref.correction_function == flow_test.correction_function
        assert flow_ref.partial_order_layers == flow_test.partial_order_layers

    # Test parametric angle replacement
    def test_xreplace(self) -> None:
        parametric_angles = [Placeholder("alpha") for _ in range(2)]
        value = 0.3
        og = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2)]),
            input_nodes=[0],
            output_nodes=[2],
            measurements={node: Measurement(angle, Plane.XY) for node, angle in enumerate(parametric_angles)},
        )
        flow = og.extract_pauli_flow()

        og_ref = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2)]),
            input_nodes=[0],
            output_nodes=[2],
            measurements={node: Measurement(value, Plane.XY) for node in range(2)},
        )
        flow_ref = og_ref.extract_pauli_flow()

        flow_test = flow.xreplace(dict.fromkeys(parametric_angles, value))

        assert not flow.og.isclose(flow_test.og)
        assert flow_ref.og.isclose(flow_test.og)

        assert flow_ref.correction_function == flow_test.correction_function
        assert flow_ref.partial_order_layers == flow_test.partial_order_layers


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

        corrections.check_well_formed()

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

        corrections.check_well_formed()

        assert corrections.partial_order_layers == (frozenset({2, 3}), frozenset({0}), frozenset({1}))
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

        corrections.check_well_formed()
        assert corrections.partial_order_layers == (frozenset({1, 2}), frozenset({0}))
        assert corrections.is_compatible([0, 1, 2])
        assert not corrections.is_compatible([2, 0, 1])  # Wrong order
        assert not corrections.is_compatible([0, 1])  # Incomplete order
        assert corrections.generate_total_measurement_order() in ([0, 1, 2], [0, 2, 1])
        assert nx.utils.graphs_equal(corrections.extract_dag(), nx.DiGraph([(0, 1), (0, 2)]))

    # Only output nodes
    def test_from_measured_nodes_mapping_0(self) -> None:
        og: OpenGraph[Plane] = OpenGraph(
            graph=nx.Graph([(0, 1)]),
            input_nodes=[],
            output_nodes=[0, 1],
            measurements={},
        )

        corrections = XZCorrections.from_measured_nodes_mapping(og=og)

        corrections.check_well_formed()
        assert corrections.x_corrections == {}
        assert corrections.z_corrections == {}
        assert corrections.partial_order_layers == (frozenset({0, 1}),)

    # Empty corrections
    def test_from_measured_nodes_mapping_1(self) -> None:
        og: OpenGraph[Plane] = OpenGraph(
            graph=nx.Graph([(0, 1)]),
            input_nodes=[],
            output_nodes=[1],
            measurements={0: Plane.XY},
        )

        corrections = XZCorrections.from_measured_nodes_mapping(og=og)

        corrections.check_well_formed()
        assert corrections.x_corrections == {}
        assert corrections.z_corrections == {}
        assert corrections.partial_order_layers == (frozenset({1}), frozenset({0}))

    def test_from_measured_nodes_mapping_2(self) -> None:
        og = OpenGraph(
            graph=nx.Graph([(0, 1), (2, 3)]),
            input_nodes=[],
            output_nodes=[3],
            measurements=dict.fromkeys([0, 1, 2], Measurement(angle=0, plane=Plane.XY)),
        )
        x_corrections = {0: {1}, 2: {3}}

        corrections = XZCorrections.from_measured_nodes_mapping(og=og, x_corrections=x_corrections)

        corrections.check_well_formed()
        assert all(corrections.partial_order_layers)  # No empty sets
        assert all(
            sum(1 for layer in corrections.partial_order_layers if node in layer) == 1 for node in og.graph.nodes
        )

    # All output nodes in corrections
    def test_from_measured_nodes_mapping_3(self) -> None:
        og = OpenGraph(
            graph=nx.Graph([(0, 1), (2, 3)]),
            input_nodes=[],
            output_nodes=[1, 3],
            measurements=dict.fromkeys([0, 2], Measurement(angle=0, plane=Plane.XY)),
        )
        x_corrections = {0: {1}, 2: {3}}

        corrections = XZCorrections.from_measured_nodes_mapping(og=og, x_corrections=x_corrections)

        corrections.check_well_formed()
        assert corrections.partial_order_layers == (frozenset({1, 3}), frozenset({0, 2}))

    # Some output nodes in corrections
    def test_from_measured_nodes_mapping_4(self) -> None:
        og = OpenGraph(
            graph=nx.Graph([(0, 1), (2, 3)]),
            input_nodes=[],
            output_nodes=[1, 3],
            measurements=dict.fromkeys([0, 2], Measurement(angle=0, plane=Plane.XY)),
        )
        x_corrections = {2: {3}}

        corrections = XZCorrections.from_measured_nodes_mapping(og=og, x_corrections=x_corrections)

        corrections.check_well_formed()
        assert corrections.partial_order_layers == (frozenset({1, 3}), frozenset({0, 2}))

    # No output nodes in corrections
    def test_from_measured_nodes_mapping_5(self) -> None:
        og = OpenGraph(
            graph=nx.Graph([(0, 1), (2, 3)]),
            input_nodes=[0],
            output_nodes=[1, 3],
            measurements=dict.fromkeys([0, 2], Measurement(angle=0, plane=Plane.XY)),
        )
        x_corrections = {0: {2}}

        corrections = XZCorrections.from_measured_nodes_mapping(og=og, x_corrections=x_corrections)

        corrections.check_well_formed()
        assert corrections.partial_order_layers == (frozenset({1, 3}), frozenset({2}), frozenset({0}))

    # Test exceptions
    def test_from_measured_nodes_mapping_exceptions(self) -> None:
        og = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
            input_nodes=[0],
            output_nodes=[3],
            measurements=dict.fromkeys(range(3), Measurement(angle=0, plane=Plane.XY)),
        )
        with pytest.raises(XZCorrectionsGenericError) as exc_info:
            XZCorrections.from_measured_nodes_mapping(og=og, x_corrections={3: {1, 2}})
        assert exc_info.value.reason == XZCorrectionsGenericErrorReason.IncorrectKeys

        with pytest.raises(XZCorrectionsGenericError) as exc_info:
            XZCorrections.from_measured_nodes_mapping(og=og, z_corrections={3: {1, 2}})
        assert exc_info.value.reason == XZCorrectionsGenericErrorReason.IncorrectKeys

        with pytest.raises(XZCorrectionsGenericError) as exc_info:
            XZCorrections.from_measured_nodes_mapping(og=og, x_corrections={0: {1}, 1: {2}}, z_corrections={2: {0}})
        assert exc_info.value.reason == XZCorrectionsGenericErrorReason.ClosedLoop

        with pytest.raises(XZCorrectionsGenericError) as exc_info:
            XZCorrections.from_measured_nodes_mapping(og=og, x_corrections={0: {4}})
        assert exc_info.value.reason == XZCorrectionsGenericErrorReason.IncorrectValues

    # Test parametric angle replacement
    def test_subs(self) -> None:
        alpha = Placeholder("alpha")
        value = 0.3
        og = OpenGraph(
            graph=nx.Graph([(0, 1)]), input_nodes=[0], output_nodes=[1], measurements={0: Measurement(alpha, Plane.XY)}
        )
        xzcorr = og.extract_causal_flow().to_corrections()

        og_ref = OpenGraph(
            graph=nx.Graph([(0, 1)]), input_nodes=[0], output_nodes=[1], measurements={0: Measurement(value, Plane.XY)}
        )
        xzcorr_ref = og_ref.extract_causal_flow().to_corrections()

        xzcorr_test = xzcorr.subs(alpha, value)

        assert not xzcorr.og.isclose(xzcorr_test.og)
        assert xzcorr_ref.og.isclose(xzcorr_test.og)

        assert xzcorr_ref.x_corrections == xzcorr_test.x_corrections
        assert xzcorr_ref.z_corrections == xzcorr_test.z_corrections
        assert xzcorr_ref.partial_order_layers == xzcorr_test.partial_order_layers

    # Test parametric angle replacement
    def test_xreplace(self) -> None:
        parametric_angles = [Placeholder("alpha") for _ in range(2)]
        value = 0.3
        og = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2)]),
            input_nodes=[0],
            output_nodes=[2],
            measurements={node: Measurement(angle, Plane.XY) for node, angle in enumerate(parametric_angles)},
        )
        xzcorr = og.extract_causal_flow().to_corrections()

        og_ref = OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2)]),
            input_nodes=[0],
            output_nodes=[2],
            measurements={node: Measurement(value, Plane.XY) for node in range(2)},
        )
        xzcorr_ref = og_ref.extract_causal_flow().to_corrections()

        xzcorr_test = xzcorr.xreplace(dict.fromkeys(parametric_angles, value))

        assert not xzcorr.og.isclose(xzcorr_test.og)
        assert xzcorr_ref.og.isclose(xzcorr_test.og)

        assert xzcorr_ref.x_corrections == xzcorr_test.x_corrections
        assert xzcorr_ref.z_corrections == xzcorr_test.z_corrections
        assert xzcorr_ref.partial_order_layers == xzcorr_test.partial_order_layers


class IncorrectFlowTestCase(NamedTuple):
    flow: PauliFlow[AbstractMeasurement]
    exception: FlowError


class TestIncorrectFlows:
    """Bundle for unit tests of :func:`PauliFlow.check_well_formed` (and children) on incorrect flows. Correct flows are extensively tested in `tests.test_opengraph.py`."""

    og_c = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
        input_nodes=[0],
        output_nodes=[3],
        measurements=dict.fromkeys(range(3), Plane.XY),
    )
    og_g = OpenGraph(
        graph=nx.Graph([(0, 3), (0, 4), (1, 4), (2, 4)]),
        input_nodes=[0],
        output_nodes=[3, 4],
        measurements={0: Plane.XY, 1: Plane.YZ, 2: Plane.XZ},
    )
    og_p = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
        input_nodes=[0],
        output_nodes=[3],
        measurements={0: Plane.XY, 1: Axis.X, 2: Plane.XY},
    )

    @pytest.mark.parametrize(
        "test_case",
        [
            # Correct flow on an open graph with XZ measurements.
            IncorrectFlowTestCase(
                CausalFlow(
                    og=OpenGraph(
                        graph=nx.Graph([(0, 1)]),
                        input_nodes=[0],
                        output_nodes=[1],
                        measurements={0: Plane.XZ},
                    ),
                    correction_function={0: {1}},
                    partial_order_layers=[{1}, {0}],
                ),
                FlowGenericError(FlowGenericErrorReason.XYPlane),
            ),
            # Incomplete correction function
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2}},
                    partial_order_layers=[{3}, {2}, {1}, {0}],
                ),
                FlowGenericError(FlowGenericErrorReason.IncorrectCorrectionFunctionDomain),
            ),
            # Extra node in correction function image
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2}, 2: {4}},
                    partial_order_layers=[{3}, {2}, {1}, {0}],
                ),
                FlowGenericError(FlowGenericErrorReason.IncorrectCorrectionFunctionImage),
            ),
            # Empty partial order
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2}, 2: {3}},
                    partial_order_layers=[],
                ),
                PartialOrderError(PartialOrderErrorReason.Empty),
            ),
            # Incomplete partial order (first layer)
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2}, 2: {3}},
                    partial_order_layers=[{2}, {1}, {0}],
                ),
                PartialOrderLayerError(PartialOrderLayerErrorReason.FirstLayer, layer_index=0, layer={2}),
            ),
            # Empty layer
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2}, 2: {3}},
                    partial_order_layers=[{3}, {2}, {1}, set(), {0}],
                ),
                PartialOrderLayerError(PartialOrderLayerErrorReason.NthLayer, layer_index=3, layer=set()),
            ),
            # Duplicate layer
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2}, 2: {3}},
                    partial_order_layers=[{3}, {2}, {1}, {1}, {0}],
                ),
                PartialOrderLayerError(PartialOrderLayerErrorReason.NthLayer, layer_index=2, layer={1}),
            ),
            # Output node in nth layer
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2}, 2: {3}},
                    partial_order_layers=[{3}, {2}, {3}, {1}, {0}],
                ),
                PartialOrderLayerError(PartialOrderLayerErrorReason.NthLayer, layer_index=2, layer={3}),
            ),
            # Incomplete partial order (nth layer)
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2}, 2: {3}},
                    partial_order_layers=[{3}, {2}, {1}],
                ),
                PartialOrderError(PartialOrderErrorReason.IncorrectNodes),
            ),
            # C0
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2, 3}, 2: {3}},
                    partial_order_layers=[{3}, {2}, {1}, {0}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.C0, node=1, correction_set={2, 3}),
            ),
            # C1
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {2}, 2: {1}, 1: {3}},
                    partial_order_layers=[{3}, {1}, {2}, {0}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.C1, node=0, correction_set={2}),
            ),
            # C2
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2}, 2: {3}},
                    partial_order_layers=[{3}, {2}, {0, 1}],
                ),
                FlowPropositionOrderError(
                    FlowPropositionOrderErrorReason.C2, node=0, correction_set={1}, past_and_present_nodes={0, 1}
                ),
            ),
            # C3
            IncorrectFlowTestCase(
                CausalFlow(
                    og=og_c,
                    correction_function={0: {1}, 1: {2}, 2: {3}},
                    partial_order_layers=[{3}, {1}, {0, 2}],
                ),
                FlowPropositionOrderError(
                    FlowPropositionOrderErrorReason.C3, node=0, correction_set={1}, past_and_present_nodes={0, 2}
                ),
            ),
            # G1
            IncorrectFlowTestCase(
                GFlow(
                    og=og_g,
                    correction_function={0: {3}, 1: {1, 2}, 2: {2, 3, 4}},
                    partial_order_layers=[{3, 4}, {1}, {0, 2}],
                ),
                FlowPropositionOrderError(
                    FlowPropositionOrderErrorReason.G1, node=1, correction_set={1, 2}, past_and_present_nodes={0, 1, 2}
                ),
            ),
            # G2
            IncorrectFlowTestCase(
                GFlow(
                    og=og_g,
                    correction_function={0: {3}, 1: {1}, 2: {2, 3, 4}},
                    partial_order_layers=[{3, 4}, {1, 0, 2}],
                ),
                FlowPropositionOrderError(
                    FlowPropositionOrderErrorReason.G2,
                    node=2,
                    correction_set={2, 3, 4},
                    past_and_present_nodes={0, 1, 2},
                ),
            ),
            # G3
            IncorrectFlowTestCase(
                GFlow(
                    og=og_g,
                    correction_function={0: {3, 4}, 1: {1}, 2: {2, 3, 4}},
                    partial_order_layers=[{3, 4}, {1}, {2}, {0}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.G3, node=0, correction_set={3, 4}),
            ),
            # P4 (same as G3 but for Pauli flow)
            IncorrectFlowTestCase(
                PauliFlow(
                    og=og_g,
                    correction_function={0: {3, 4}, 1: {1}, 2: {2, 3, 4}},
                    partial_order_layers=[{3, 4}, {1}, {2}, {0}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.P4, node=0, correction_set={3, 4}),
            ),
            # G4
            IncorrectFlowTestCase(
                GFlow(
                    og=og_g,
                    correction_function={0: {3}, 1: {1}, 2: {3, 4}},
                    partial_order_layers=[{3, 4}, {1}, {2}, {0}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.G4, node=2, correction_set={3, 4}),
            ),
            # P5 (same as G4 but for Pauli flow)
            IncorrectFlowTestCase(
                PauliFlow(
                    og=og_g,
                    correction_function={0: {3}, 1: {1}, 2: {3, 4}},
                    partial_order_layers=[{3, 4}, {1}, {2}, {0}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.P5, node=2, correction_set={3, 4}),
            ),
            # G5
            IncorrectFlowTestCase(
                GFlow(
                    og=og_g,
                    correction_function={0: {3}, 1: set(), 2: {2, 3, 4}},
                    partial_order_layers=[{3, 4}, {1}, {2}, {0}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.G5, node=1, correction_set=set()),
            ),
            # P6 (same as G5 but for Pauli flow)
            IncorrectFlowTestCase(
                PauliFlow(
                    og=og_g,
                    correction_function={0: {3}, 1: set(), 2: {2, 3, 4}},
                    partial_order_layers=[{3, 4}, {1}, {2}, {0}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.P6, node=1, correction_set=set()),
            ),
            # P1
            IncorrectFlowTestCase(
                PauliFlow(
                    og=og_p,
                    correction_function={0: {1, 3}, 1: {2}, 2: {3}},
                    partial_order_layers=[{3}, {2, 0, 1}],
                ),
                FlowPropositionOrderError(
                    FlowPropositionOrderErrorReason.P1, node=1, correction_set={2}, past_and_present_nodes={0, 1, 2}
                ),
            ),
            # P2
            IncorrectFlowTestCase(
                PauliFlow(
                    og=og_p,
                    correction_function={0: {1, 3}, 1: {3}, 2: {3}},
                    partial_order_layers=[{3}, {2, 0, 1}],
                ),
                FlowPropositionOrderError(
                    FlowPropositionOrderErrorReason.P2, node=1, correction_set={3}, past_and_present_nodes={0, 1, 2}
                ),
            ),
            # P3
            IncorrectFlowTestCase(
                PauliFlow(
                    og=OpenGraph(
                        graph=nx.Graph([(0, 1), (1, 2)]),
                        input_nodes=[0],
                        output_nodes=[2],
                        measurements=dict.fromkeys(range(2), Measurement(0.5, Plane.XY)),
                    ),
                    correction_function={0: {1}, 1: {2}},
                    partial_order_layers=[{2}, {0, 1}],
                ),
                FlowPropositionOrderError(
                    FlowPropositionOrderErrorReason.P3, node=0, correction_set={1}, past_and_present_nodes={0, 1}
                ),  # Past and present nodes measured along Y.
            ),
            # P7
            IncorrectFlowTestCase(
                PauliFlow(
                    og=og_p,
                    correction_function={0: {1, 3}, 1: {3}, 2: {3}},
                    partial_order_layers=[{3}, {2}, {0, 1}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.P7, node=1, correction_set={3}),
            ),
            # P8
            IncorrectFlowTestCase(
                PauliFlow(
                    og=OpenGraph(
                        graph=nx.Graph([(0, 1)]),
                        input_nodes=[0],
                        output_nodes=[1],
                        measurements={0: Measurement(0, Plane.XZ)},
                    ),
                    correction_function={0: {1}},
                    partial_order_layers=[{1}, {0}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.P8, node=0, correction_set={1}),
            ),
            # P9
            IncorrectFlowTestCase(
                PauliFlow(
                    og=OpenGraph(
                        graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
                        input_nodes=[0],
                        output_nodes=[3],
                        measurements={0: Plane.XY, 1: Axis.Y, 2: Plane.XY},
                    ),
                    correction_function={0: {2, 3}, 1: {1, 2}, 2: {3}},
                    partial_order_layers=[{3}, {2}, {0}, {1}],
                ),
                FlowPropositionError(FlowPropositionErrorReason.P9, node=1, correction_set={1, 2}),
            ),
        ],
    )
    def test_flow_well_formed(self, test_case: IncorrectFlowTestCase) -> None:
        with pytest.raises(FlowError) as exc_info:
            test_case.flow.check_well_formed()

        for field in fields(exc_info.value):
            attr = field.name
            assert getattr(exc_info.value, attr) == getattr(test_case.exception, attr)


class IncorrectXZCTestCase(NamedTuple):
    xzcorr: XZCorrections[AbstractMeasurement]
    exception: XZCorrectionsError


class TestIncorrectXZC:
    """Bundle for unit tests of :func:`XZCorrections.check_well_formed` on incorrect instances. Correct instances are extensively tested in class:`TestXZCorrections`."""

    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
        input_nodes=[0],
        output_nodes=[3],
        measurements=dict.fromkeys(range(3), Plane.XY),
    )

    @pytest.mark.parametrize(
        "test_case",
        [
            # Empty partial order
            IncorrectXZCTestCase(
                XZCorrections(
                    og=og,
                    x_corrections={},
                    z_corrections={},
                    partial_order_layers=[],
                ),
                PartialOrderError(PartialOrderErrorReason.Empty),
            ),
            # Non-measured node in corrections dictionary keys
            IncorrectXZCTestCase(
                XZCorrections(
                    og=og,
                    x_corrections={},
                    z_corrections={3: {1, 2}},
                    partial_order_layers=[{3}, {2}, {1}, {0}],
                ),
                XZCorrectionsGenericError(XZCorrectionsGenericErrorReason.IncorrectKeys),
            ),
            # First layer contains non-output nodes
            IncorrectXZCTestCase(
                XZCorrections(
                    og=og,
                    x_corrections={},
                    z_corrections={},
                    partial_order_layers=[{3, 2}, {1}, {0}],
                ),
                PartialOrderLayerError(PartialOrderLayerErrorReason.FirstLayer, layer_index=0, layer={2, 3}),
            ),
            # Duplicate nodes in partial order
            IncorrectXZCTestCase(
                XZCorrections(
                    og=og,
                    x_corrections={},
                    z_corrections={},
                    partial_order_layers=[{3}, {2, 1}, {0, 1}],
                ),
                PartialOrderLayerError(
                    PartialOrderLayerErrorReason.NthLayer,
                    layer_index=1,
                    layer={1, 2},
                ),
            ),
            # Output nodes in nth layer
            IncorrectXZCTestCase(
                XZCorrections(
                    og=og,
                    x_corrections={},
                    z_corrections={},
                    partial_order_layers=[{3}, {2}, {0, 1, 3}],
                ),
                PartialOrderLayerError(
                    PartialOrderLayerErrorReason.NthLayer,
                    layer_index=2,
                    layer={0, 1, 3},
                ),
            ),
            # Closed loop
            IncorrectXZCTestCase(
                XZCorrections(
                    og=og,
                    x_corrections={0: {1}, 1: {2}},
                    z_corrections={2: {0}},
                    partial_order_layers=[{3}, {2}, {1}, {0}],
                ),
                XZCorrectionsOrderError(
                    XZCorrectionsOrderErrorReason.Z,
                    node=2,
                    correction_set={0},
                    past_and_present_nodes={0, 1, 2},
                ),
            ),
            # Self-correcting node
            IncorrectXZCTestCase(
                XZCorrections(
                    og=og,
                    x_corrections={0: {1}, 1: {1}},
                    z_corrections={0: {1}},
                    partial_order_layers=[{3}, {2}, {1}, {0}],
                ),
                XZCorrectionsOrderError(
                    XZCorrectionsOrderErrorReason.X,
                    node=1,
                    correction_set={1},
                    past_and_present_nodes={0, 1},
                ),
            ),
        ],
    )
    def test_xzc_well_formed(self, test_case: IncorrectXZCTestCase) -> None:
        with pytest.raises(XZCorrectionsError) as exc_info:
            test_case.xzcorr.check_well_formed()

        for field in fields(exc_info.value):
            attr = field.name
            assert getattr(exc_info.value, attr) == getattr(test_case.exception, attr)
