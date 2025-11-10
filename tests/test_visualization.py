from __future__ import annotations

from math import pi
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

from graphix import Circuit, Pattern, command, gflow, visualization
from graphix.visualization import GraphVisualizer
from tests.test_generator import example_flow, example_gflow, example_pflow

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import Generator


def test_get_pos_from_flow() -> None:
    circuit = Circuit(1)
    circuit.h(0)
    pattern = circuit.transpile().pattern
    graph = pattern.extract_graph()
    vin = pattern.input_nodes if pattern.input_nodes is not None else []
    vout = pattern.output_nodes
    meas_planes = pattern.get_meas_plane()
    meas_angles = pattern.get_angles()
    local_clifford = pattern.get_vops()
    vis = visualization.GraphVisualizer(graph, vin, vout, meas_planes, meas_angles, local_clifford)
    f, l_k = gflow.find_flow(graph, set(vin), set(vout), meas_planes)
    assert f is not None
    assert l_k is not None
    pos = vis.get_pos_from_flow(f, l_k)
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
    graph = pattern.extract_graph()
    vis = GraphVisualizer(graph, pattern.input_nodes, pattern.output_nodes)
    return vis, pattern


@pytest.mark.usefixtures("mock_plot")
def test_graph_visualizer_without_plane() -> None:
    vis, pattern = example_visualizer()
    vis.visualize()
    vis.visualize_from_pattern(pattern)


@pytest.mark.usefixtures("mock_plot")
def test_draw_graph_without_flow() -> None:
    pattern = Pattern(input_nodes=[0], cmds=[command.N(1), command.E((0, 1)), command.M(0)])
    pattern.draw_graph()


@pytest.mark.usefixtures("mock_plot")
def test_large_node_number() -> None:
    pattern = Pattern(input_nodes=[100])
    pattern.draw_graph()


def test_get_figsize_without_layers_or_pos() -> None:
    vis, _pattern = example_visualizer()
    with pytest.raises(ValueError):
        vis.get_figsize(None, None)


def test_edge_intersects_node_equals() -> None:
    vis, _pattern = example_visualizer()
    assert not vis._edge_intersects_node((0, 0), (0, 0), (0, 0))


@pytest.mark.usefixtures("mock_plot")
def test_custom_corrections() -> None:
    pattern = Pattern(
        input_nodes=[0, 1, 2, 3],
        cmds=[command.M(0), command.M(1), command.X(2, {0}), command.Z(2, {0}), command.Z(3, {1})],
    )
    graph = pattern.extract_graph()
    vis = GraphVisualizer(graph, pattern.input_nodes, pattern.output_nodes)
    vis.visualize_from_pattern(pattern)

# Compare with baseline/test_draw_graph_reference.png
# Update baseline by running: pytest --mpl-generate-path=tests/baseline
@pytest.mark.usefixtures("mock_plot")
@pytest.mark.mpl_image_compare
def test_draw_graph_reference() -> plt.Figure:
    circuit = Circuit(3)
    circuit.cnot(0, 1)
    circuit.cnot(2, 1)
    circuit.rx(0, pi / 3)
    circuit.x(2)
    circuit.cnot(2, 1)
    pattern = circuit.transpile().pattern
    pattern.perform_pauli_measurements(leave_input=True)
    pattern.draw_graph(node_distance=(0.7, 0.6))
    return plt.gcf()
