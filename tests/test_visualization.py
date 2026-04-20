from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from graphix import Circuit, Pattern, command, visualization
from graphix.fundamentals import ANGLE_PI
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
def test_draw_graph_show_measurement_planes(fx_rng: Generator) -> None:
    pattern = example_pflow(fx_rng)
    pattern.draw_graph(
        show_measurement_planes=True,
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
        flow_from_pattern=flow_and_not_pauli_presimulate, node_distance=(0.7, 0.6), show_measurement_planes=True
    )
    return plt.gcf()


def test_draw_edges_with_routing_skips_non_subset_edge() -> None:
    vis, _pattern = example_visualizer()
    ax = MagicMock()
    edge_path = {(0, 1): [(0.0, 0.0), (1.0, 0.0)], (1, 2): [(1.0, 0.0), (2.0, 0.0)]}
    vis.draw_edges_with_routing(ax, edge_path, edge_subset=[(0, 1)])
    assert ax.plot.call_count == 1


def test_draw_edges_with_routing_color_and_linewidth_overrides() -> None:
    vis, _pattern = example_visualizer()
    ax = MagicMock()
    edge_path = {(0, 1): [(0.0, 0.0), (1.0, 0.0)]}
    vis.draw_edges_with_routing(
        ax,
        edge_path,
        edge_colors={(0, 1): "red"},
        edge_linewidths={(0, 1): 2.5},
    )
    assert ax.plot.call_count == 1
    call_kwargs = ax.plot.call_args
    assert call_kwargs.kwargs["color"] == "red"
    assert call_kwargs.kwargs["linewidth"] == pytest.approx(2.5)


def test_draw_flow_arrows_with_subset() -> None:
    vis, _pattern = example_visualizer()
    ax = MagicMock()
    pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}
    arrow_path = {(0, 1): [(0.0, 0.0), (1.0, 0.0)], (1, 2): [(1.0, 0.0), (2.0, 0.0)]}
    vis.draw_flow_arrows(ax, pos, arrow_path, arrow_subset=[(0, 1)])
    assert ax.annotate.call_count == 0  # no self-loop, no annotate


def test_draw_flow_arrows_self_loop() -> None:
    vis, _pattern = example_visualizer()
    ax = MagicMock()
    pos = {0: (0.5, 0.5)}
    loop_path = [(0.5, 0.5), (0.7, 0.7), (0.9, 0.5), (0.7, 0.3), (0.5, 0.5)]
    arrow_path = {(0, 0): loop_path}
    vis.draw_flow_arrows(ax, pos, arrow_path, show_loop=True)
    assert ax.plot.called
    assert ax.annotate.called


def test_draw_node_labels_auto_fontsize() -> None:
    vis, _pattern = example_visualizer()
    ax = MagicMock()
    pos = {0: (0.0, 0.0), 1: (1.0, 0.0)}
    vis.draw_node_labels(ax, pos)
    assert ax.text.call_count == 2


def test_draw_node_labels_with_extra_labels() -> None:
    vis, _pattern = example_visualizer()
    ax = MagicMock()
    pos = {0: (0.0, 0.0), 1: (1.0, 0.0)}
    vis.draw_node_labels(ax, pos, extra_labels={0: "m=1"})
    calls = ax.text.call_args_list
    label_args = [call.args[2] for call in calls]
    assert any("\n" in lbl for lbl in label_args)


def test_draw_nodes_role_skips_node_not_in_pos() -> None:
    vis, _pattern = example_visualizer()
    ax = MagicMock()
    pos = {99: (0.0, 0.0)}
    vis.draw_nodes_role(ax, pos)
    assert ax.scatter.call_count == 0


def test_draw_nodes_role_pauli_measurement_lightblue() -> None:
    mock_og = MagicMock()
    mock_og.graph.nodes.return_value = [0]
    mock_og.input_nodes = []
    mock_og.output_nodes = []
    mock_og.measurements = {0: MagicMock(spec=PauliMeasurement)}

    vis = GraphVisualizer(og=mock_og)
    ax = MagicMock()
    pos = {0: (0.0, 0.0)}

    vis.draw_nodes_role(ax, pos, show_pauli_measurement=True)
    call_kwargs = ax.scatter.call_args_list[0].kwargs
    assert call_kwargs["facecolors"] == "lightblue"


def test_draw_nodes_role_node_alpha_override() -> None:
    mock_og = MagicMock()
    mock_og.graph.nodes.return_value = [0]
    mock_og.input_nodes = []
    mock_og.output_nodes = []
    mock_og.measurements = {}

    vis = GraphVisualizer(og=mock_og)
    ax = MagicMock()
    pos = {0: (0.0, 0.0)}

    vis.draw_nodes_role(ax, pos, node_alpha={0: 0.3})
    call_kwargs = ax.scatter.call_args_list[0].kwargs
    assert call_kwargs["alpha"] == pytest.approx(0.3)


def test_draw_nodes_role_node_linewidths_override() -> None:
    mock_og = MagicMock()
    mock_og.graph.nodes.return_value = [0]
    mock_og.input_nodes = []
    mock_og.output_nodes = []
    mock_og.measurements = {}

    vis = GraphVisualizer(og=mock_og)
    ax = MagicMock()
    pos = {0: (0.0, 0.0)}

    vis.draw_nodes_role(ax, pos, node_linewidths={0: 4.0})
    call_kwargs = ax.scatter.call_args_list[0].kwargs
    assert call_kwargs["linewidths"] == pytest.approx(4.0)


def test_draw_layer_separators_empty_l_k() -> None:
    vis, _pattern = example_visualizer()
    ax = MagicMock()
    pos = {0: (0.0, 0.0)}
    vis.draw_layer_separators(ax, pos, l_k={})
    ax.axvline.assert_not_called()
