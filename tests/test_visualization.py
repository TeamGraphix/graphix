from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from graphix import Circuit, Pattern, Plane, XZCorrections, command, visualization
from graphix.fundamentals import ANGLE_PI
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph, OpenGraphError
from graphix.pattern import DrawPatternAnnotations
from graphix.visualization import Colored, GraphVisualizer, _edge_intersects_node

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure
    from numpy.random import Generator

    from graphix.fundamentals import Angle


def example_og() -> OpenGraph[Measurement]:
    return OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (1, 3), (4, 6)]),
        input_nodes=(0, 3, 6),
        output_nodes=(2, 5, 8),
        measurements=dict.fromkeys((0, 1, 3, 4, 6, 7), Measurement.XY(angle=0)),
    )


def example_flow(rng: Generator) -> Pattern:
    graph: nx.Graph[int] = nx.Graph([(0, 3), (1, 4), (2, 5), (1, 3), (2, 4), (3, 6), (4, 7), (5, 8)])
    inputs = [1, 0, 2]  # non-trivial order to check order is conserved.
    outputs = [7, 6, 8]
    angles = (2 * rng.random(6)).tolist()
    measurements = {node: Measurement.XY(angle) for node, angle in enumerate(angles)}

    pattern = OpenGraph(graph=graph, input_nodes=inputs, output_nodes=outputs, measurements=measurements).to_pattern()
    pattern.standardize()

    assert pattern.input_nodes == inputs
    assert pattern.output_nodes == outputs
    return pattern


def example_gflow(rng: Generator) -> Pattern:
    graph: nx.Graph[int] = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 6), (1, 6)])
    inputs = [3, 1, 5]
    outputs = [4, 2, 6]
    angles = dict(zip([1, 3, 5], (2 * rng.random(3)).tolist(), strict=True))
    measurements = {node: Measurement.XY(angle) for node, angle in angles.items()}

    pattern = OpenGraph(graph=graph, input_nodes=inputs, output_nodes=outputs, measurements=measurements).to_pattern()
    pattern.standardize()

    assert pattern.input_nodes == inputs
    assert pattern.output_nodes == outputs
    return pattern


def example_pflow(rng: Generator) -> Pattern:
    """Create a graph which has pflow but no gflow.

    Parameters
    ----------
    rng : :class:`numpy.random.Generator`
        See graphix.tests.conftest.py

    Returns
    -------
    Pattern: :class:`graphix.pattern.Pattern`
    """
    graph: nx.Graph[int] = nx.Graph(
        [(0, 2), (1, 4), (2, 3), (3, 4), (2, 5), (3, 6), (4, 7), (5, 6), (6, 7), (5, 8), (7, 9)]
    )
    inputs = [1, 0]
    outputs = [9, 8]

    # Heuristic mixture of Pauli and non-Pauli angles ensuring there's no gflow but there's pflow.
    meas_angles: dict[int, Angle] = {
        **dict.fromkeys(range(4), 0),
        **dict(zip(range(4, 8), (2 * rng.random(4)).tolist(), strict=True)),
    }
    measurements = {i: Measurement.XY(angle).to_pauli_or_bloch() for i, angle in meas_angles.items()}

    og = OpenGraph(graph=graph, input_nodes=inputs, output_nodes=outputs, measurements=measurements)
    try:
        og.to_bloch().to_gflow()
        pytest.fail("example graph shouldn't have gflow")
    except OpenGraphError:
        og.to_pauliflow()  # example graph has Pauli flow

    pattern = og.to_pattern()
    pattern.standardize()
    assert og.input_nodes == pattern.input_nodes
    assert og.output_nodes == pattern.output_nodes
    return pattern


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.parametrize("example", [example_flow, example_gflow, example_pflow])
@pytest.mark.parametrize("flow_from_pattern", [False, True])
@pytest.mark.parametrize("measurement_labels", [False, True])
@pytest.mark.parametrize("pauli_measurements", [False, True])
@pytest.mark.parametrize("local_clifford", [False, True])
def test_draw_pattern_flow(
    example: Callable[[Generator], Pattern],
    flow_from_pattern: bool,
    local_clifford: bool,
    pauli_measurements: bool,
    measurement_labels: bool,
    fx_rng: Generator,
) -> None:
    pattern = example(fx_rng)
    pattern.draw(
        flow_from_pattern=flow_from_pattern,
        pauli_measurements=pauli_measurements,
        measurement_labels=measurement_labels,
        local_clifford=local_clifford,
        node_distance=(0.7, 0.6),
    )
    plt.close()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.parametrize("example", [example_flow, example_gflow, example_pflow])
@pytest.mark.parametrize("measurement_labels", [False, True])
@pytest.mark.parametrize("pauli_measurements", [False, True])
@pytest.mark.parametrize("local_clifford", [False, True])
def test_draw_pattern_xzcorrections(
    example: Callable[[Generator], Pattern],
    local_clifford: bool,
    pauli_measurements: bool,
    measurement_labels: bool,
    fx_rng: Generator,
) -> None:
    pattern = example(fx_rng)
    pattern.draw(
        annotations=DrawPatternAnnotations.XZCorrections,
        pauli_measurements=pauli_measurements,
        measurement_labels=measurement_labels,
        local_clifford=local_clifford,
        node_distance=(0.7, 0.6),
    )
    plt.close()


def example_hadamard() -> Pattern:
    circuit = Circuit(1)
    circuit.h(0)
    return circuit.transpile().pattern


def example_local_clifford() -> Pattern:
    pattern = example_hadamard()
    pattern.infer_pauli_measurements()
    pattern.remove_pauli_measurements()
    return pattern


def test_draw_pattern_xzcorrections_save() -> None:
    pattern = example_hadamard()
    with TemporaryDirectory() as dirname:
        filename = Path(dirname) / "image.png"
        pattern.draw(annotations=DrawPatternAnnotations.XZCorrections, filename=filename)
        assert filename.exists()


@pytest.mark.usefixtures("mock_plot")
def test_large_node_number() -> None:
    pattern = Pattern(input_nodes=[100])
    pattern.draw()


def test_edge_intersects_node_equals() -> None:
    assert not _edge_intersects_node((0, 0), (0, 0), (0, 0))


@pytest.mark.usefixtures("mock_plot")
def test_custom_corrections() -> None:
    pattern = Pattern(
        input_nodes=[0, 1, 2, 3],
        cmds=[command.M(0), command.M(1), command.X(2, {0}), command.Z(2, {0}), command.Z(3, {1})],
    )
    pattern.draw(annotations=DrawPatternAnnotations.XZCorrections)


@pytest.mark.usefixtures("mock_plot")
def test_og() -> None:
    pattern = Pattern(
        input_nodes=[0, 1, 2, 3],
        cmds=[command.M(0), command.M(1), command.X(2, {0}), command.Z(2, {0}), command.Z(3, {1})],
    )
    pattern.draw(annotations=None)


@pytest.mark.usefixtures("mock_plot")
def test_non_determinist() -> None:
    pattern = Pattern(
        input_nodes=[0],
        cmds=[command.N(1), command.E((0, 1)), command.M(0)],
    )
    with pytest.warns(
        UserWarning,
        match="The pattern is not consistent with a flow. An attempt to be extract the flow from the underlying open graph will be made.",
    ):
        pattern.draw()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.parametrize("annotations", [None, DrawPatternAnnotations.Flow, DrawPatternAnnotations.XZCorrections])
def test_empty(annotations: DrawPatternAnnotations | None) -> None:
    pattern = Pattern()
    pattern.draw(annotations=annotations)


# Compare with baseline/test_draw_graph_reference.png
# Update baseline by running: pytest --mpl-generate-path=tests/baseline
@pytest.mark.usefixtures("mock_plot")
@pytest.mark.mpl_image_compare
def test_og_draw() -> Figure:
    og = example_og()
    og.draw(legend=False)
    return plt.gcf()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.mpl_image_compare
def test_causal_flow_draw() -> Figure:
    og = example_og()
    og.downcast_bloch().to_causalflow().draw(legend=False)
    return plt.gcf()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.mpl_image_compare
def test_gflow_draw() -> Figure:
    og = example_og()
    og.downcast_bloch().to_gflow().draw(legend=False)
    return plt.gcf()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.mpl_image_compare
def test_pauli_flow_draw() -> Figure:
    og = example_og()
    og.infer_pauli_measurements().to_pauliflow().draw(legend=False)
    return plt.gcf()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.mpl_image_compare
def test_xzcorr_draw() -> Figure:
    og = example_og()
    og.downcast_bloch().to_causalflow().to_xzcorrections().draw(legend=False)
    return plt.gcf()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.parametrize("flow_from_pattern_and_to_bloch", [False, True])
@pytest.mark.mpl_image_compare
def test_draw_graph_reference(flow_from_pattern_and_to_bloch: bool) -> Figure:
    circuit = Circuit(3)
    circuit.cnot(0, 1)
    circuit.cnot(2, 1)
    circuit.rx(0, ANGLE_PI / 3)
    circuit.x(2)
    circuit.cnot(2, 1)
    pattern = circuit.transpile().pattern
    if flow_from_pattern_and_to_bloch:
        pattern.blochify()
    else:
        pattern.infer_pauli_measurements()
        pattern.remove_pauli_measurements()
    pattern.standardize()
    pattern.draw(
        flow_from_pattern=flow_from_pattern_and_to_bloch, node_distance=(1, 1), measurement_labels=True, legend=False
    )
    return plt.gcf()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.parametrize("infer_pauli_measurements", [False, True])
@pytest.mark.mpl_image_compare
def test_legend_pauli_measurements(infer_pauli_measurements: bool) -> Figure:
    # See https://github.com/TeamGraphix/graphix/issues/554
    circuit = Circuit(1)
    circuit.h(0)
    pattern = circuit.transpile().pattern
    if infer_pauli_measurements:
        pattern = pattern.infer_pauli_measurements()
    pattern.draw()
    return plt.gcf()


def x_corrections_only() -> XZCorrections[Plane]:
    graph: nx.Graph[int] = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)

    og = OpenGraph(graph, input_nodes=[], output_nodes=[], measurements=dict.fromkeys([0, 1], Plane.XY))

    return XZCorrections.from_measured_nodes_mapping(og, {0: {1}}, {})


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.mpl_image_compare
def test_legend_x_corrections_only() -> Figure:
    # Related to https://github.com/TeamGraphix/graphix/issues/554
    # Draw X-corrections on open graph without inputs, outputs, edges.
    # Only "X corrections" label should be visible in legend.
    xz = x_corrections_only()
    xz.draw()
    return plt.gcf()


@pytest.mark.parametrize("flow_from_pattern", [False, True])
@pytest.mark.mpl_image_compare
@pytest.mark.usefixtures("mock_plot")
def test_draw_graph_reference_pauli_flow(flow_from_pattern: bool) -> Figure:
    circuit = Circuit(2)
    circuit.rzz(0, 1, 0.3)
    pattern = circuit.transpile().pattern.infer_pauli_measurements()
    pattern.draw(flow_from_pattern=flow_from_pattern, node_distance=(1, 1), measurement_labels=True, legend=False)
    return plt.gcf()


def test_corrections_must_have_distinct_colors() -> None:
    old_z_c = visualization.Z_C
    visualization.Z_C = visualization.X_C
    try:
        xz = x_corrections_only()
        with pytest.raises(
            RuntimeError,
            match=r"X, Z, and X-and-Z corrections must have different arrow colors to display the legend correctly.",
        ):
            xz.draw()
    finally:
        visualization.Z_C = old_z_c


@pytest.mark.usefixtures("mock_plot")
def test_unexpected_arrow_paths() -> None:
    og = OpenGraph(graph=nx.Graph([(0, 1)]), input_nodes=(), output_nodes=[1], measurements={0: Plane.XY})
    visualizer = GraphVisualizer(og=og, pos={}, edge_paths={}, arrow_paths={(0, 1): Colored([], "")})
    with pytest.raises(RuntimeError, match="Unexpected arrow paths with source None"):
        visualizer._draw_legend()
