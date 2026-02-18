from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from graphix import Circuit, Pattern, command, visualization
from graphix.fundamentals import ANGLE_PI, Axis, Sign
from graphix.measurements import Measurement, PauliMeasurement
from graphix.opengraph import OpenGraph, OpenGraphError
from graphix.visualization import GraphVisualizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure
    from numpy.random import Generator

    from graphix.fundamentals import Angle


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
        og.to_bloch().extract_gflow()
        pytest.fail("example graph shouldn't have gflow")
    except OpenGraphError:
        og.extract_pauli_flow()  # example graph has Pauli flow

    pattern = og.to_pattern()
    pattern.standardize()
    assert og.input_nodes == pattern.input_nodes
    assert og.output_nodes == pattern.output_nodes
    return pattern


def test_place_causal_flow() -> None:
    circuit = Circuit(1)
    circuit.h(0)
    pattern = circuit.transpile().pattern
    og = pattern.extract_opengraph().to_bloch()
    local_clifford = pattern.extract_clifford()
    vis = visualization.GraphVisualizer(og, local_clifford)
    causal_flow = og.extract_causal_flow()
    pos = vis.place_causal_flow(causal_flow)
    assert pos is not None


@pytest.fixture
def mock_plot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.parametrize("example", [example_flow, example_gflow, example_pflow])
@pytest.mark.parametrize("flow_from_pattern", [False, True])
def test_draw_graph(example: Callable[[Generator], Pattern], flow_from_pattern: bool, fx_rng: Generator) -> None:
    pattern = example(fx_rng)
    pattern.draw_graph(
        flow_from_pattern=flow_from_pattern,
        node_distance=(0.7, 0.6),
    )


def example_hadamard() -> Pattern:
    circuit = Circuit(1)
    circuit.h(0)
    return circuit.transpile().pattern


def example_local_clifford() -> Pattern:
    pattern = example_hadamard()
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    return pattern


@pytest.mark.usefixtures("mock_plot")
def test_draw_graph_show_local_clifford() -> None:
    pattern = example_local_clifford()
    pattern.standardize()
    pattern.draw_graph(
        show_local_clifford=True,
        node_distance=(0.7, 0.6),
    )


@pytest.mark.usefixtures("mock_plot")
def test_draw_graph_show_measurements_basic(fx_rng: Generator) -> None:
    pattern = example_pflow(fx_rng)
    pattern.draw_graph(
        show_measurements=True,
        node_distance=(0.7, 0.6),
    )


@pytest.mark.usefixtures("mock_plot")
def test_draw_graph_show_loop(fx_rng: Generator) -> None:
    pattern = example_pflow(fx_rng)
    pattern.draw_graph(
        show_loop=True,
        node_distance=(0.7, 0.6),
    )


def test_draw_graph_save() -> None:
    pattern = example_hadamard()
    with TemporaryDirectory() as dirname:
        filename = Path(dirname) / "image.png"
        pattern.draw_graph(filename=filename)
        assert filename.exists()


def example_visualizer() -> tuple[GraphVisualizer, Pattern]:
    pattern = example_hadamard()
    og = pattern.extract_opengraph()
    vis = GraphVisualizer(og)
    return vis, pattern


@pytest.mark.usefixtures("mock_plot")
def test_graph_visualizer_without_plane() -> None:
    vis, pattern = example_visualizer()
    vis.visualize()
    vis.visualize_from_pattern(pattern)


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.parametrize("flow_from_pattern", [False, True])
def test_draw_graph_without_flow(flow_from_pattern: bool) -> None:
    pattern = Pattern(input_nodes=[0], cmds=[command.N(1), command.E((0, 1)), command.M(0), command.M(1)])
    pattern.draw_graph(flow_from_pattern=flow_from_pattern)


@pytest.mark.usefixtures("mock_plot")
def test_large_node_number() -> None:
    pattern = Pattern(input_nodes=[100])
    pattern.draw_graph()


def test_determine_figsize_without_layers_or_pos() -> None:
    vis, _pattern = example_visualizer()
    with pytest.raises(ValueError):
        vis.determine_figsize(None, None)


def test_edge_intersects_node_equals() -> None:
    vis, _pattern = example_visualizer()
    assert not vis._edge_intersects_node((0, 0), (0, 0), (0, 0))


@pytest.mark.usefixtures("mock_plot")
def test_custom_corrections() -> None:
    pattern = Pattern(
        input_nodes=[0, 1, 2, 3],
        cmds=[command.M(0), command.M(1), command.X(2, {0}), command.Z(2, {0}), command.Z(3, {1})],
    )
    og = pattern.extract_opengraph()
    vis = GraphVisualizer(og)
    vis.visualize_from_pattern(pattern)


@pytest.mark.usefixtures("mock_plot")
def test_empty_pattern() -> None:
    pattern = Pattern()
    pattern.draw_graph()


# Compare with baseline/test_draw_graph_reference.png
# Update baseline by running: pytest --mpl-generate-path=tests/baseline
@pytest.mark.usefixtures("mock_plot")
@pytest.mark.parametrize("flow_and_not_pauli_presimulate", [False, True])
@pytest.mark.mpl_image_compare
def test_draw_graph_reference(flow_and_not_pauli_presimulate: bool) -> Figure:
    circuit = Circuit(3)
    circuit.cnot(0, 1)
    circuit.cnot(2, 1)
    circuit.rx(0, ANGLE_PI / 3)
    circuit.x(2)
    circuit.cnot(2, 1)
    pattern = circuit.transpile().pattern
    if flow_and_not_pauli_presimulate:
        # Pauli flow extraction from pattern is not implemented yet;
        # therefore, the pattern should not contain Pauli measurements
        # to have causal flow.
        pattern = pattern.to_bloch()
    else:
        pattern.remove_input_nodes()
        pattern.perform_pauli_measurements()
    pattern.standardize()
    pattern.draw_graph(
        flow_from_pattern=flow_and_not_pauli_presimulate, node_distance=(0.7, 0.6), show_measurements=True
    )
    return plt.gcf()


@pytest.mark.usefixtures("mock_plot")
def test_draw_graph_show_measurements(fx_rng: Generator) -> None:
    pattern = example_flow(fx_rng)
    pattern.draw_graph(
        show_measurements=True,
        node_distance=(0.7, 0.6),
    )


@pytest.mark.usefixtures("mock_plot")
def test_draw_graph_show_measurements_pflow(fx_rng: Generator) -> None:
    pattern = example_pflow(fx_rng)
    pattern.draw_graph(
        show_measurements=True,
        node_distance=(0.7, 0.6),
    )


@pytest.mark.usefixtures("mock_plot")
def test_draw_graph_show_legend(fx_rng: Generator) -> None:
    pattern = example_flow(fx_rng)
    pattern.draw_graph(
        show_legend=True,
        node_distance=(0.7, 0.6),
    )


@pytest.mark.usefixtures("mock_plot")
def test_draw_graph_show_legend_with_corrections(fx_rng: Generator) -> None:
    pattern = example_flow(fx_rng)
    pattern.draw_graph(
        flow_from_pattern=True,
        show_legend=True,
        show_pauli_measurement=True,
        node_distance=(0.7, 0.6),
    )


def test_format_measurement_label_bloch() -> None:
    bloch_xy = Measurement.XY(0.25)
    label = GraphVisualizer._format_measurement_label(bloch_xy)
    assert label is not None
    assert "XY" in label
    assert "/" in label  # pi/4 contains "/"


def test_format_measurement_label_bloch_zero() -> None:
    bloch_zero = Measurement.XY(0)
    label = GraphVisualizer._format_measurement_label(bloch_zero)
    assert label is not None
    assert "XY" in label
    assert "0" in label


def test_format_measurement_label_bloch_xz() -> None:
    bloch_xz = Measurement.XZ(0.5)
    label = GraphVisualizer._format_measurement_label(bloch_xz)
    assert label is not None
    assert "XZ" in label


def test_format_measurement_label_pauli() -> None:
    pauli_x = Measurement.X
    label = GraphVisualizer._format_measurement_label(pauli_x)
    assert label is not None
    assert label == str(pauli_x)
    assert "X" in label


def test_format_measurement_label_pauli_minus() -> None:
    pauli_minus_z = PauliMeasurement(Axis.Z, Sign.MINUS)
    label = GraphVisualizer._format_measurement_label(pauli_minus_z)
    assert label is not None
    assert label == str(pauli_minus_z)
    assert "-Z" in label
