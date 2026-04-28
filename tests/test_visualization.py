from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from graphix import Circuit, Pattern, command
from graphix.fundamentals import ANGLE_PI
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph, OpenGraphError
from graphix.pattern import DrawAnnotations
from graphix.visualization import _edge_intersects_node

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
        og.to_bloch().extract_gflow()
        pytest.fail("example graph shouldn't have gflow")
    except OpenGraphError:
        og.extract_pauli_flow()  # example graph has Pauli flow

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
@pytest.mark.parametrize("show_local_clifford", [False, True])
def test_draw_pattern_flow(
    example: Callable[[Generator], Pattern],
    flow_from_pattern: bool,
    show_local_clifford: bool,
    pauli_measurements: bool,
    measurement_labels: bool,
    fx_rng: Generator,
) -> None:
    pattern = example(fx_rng)
    pattern.draw(
        flow_from_pattern=flow_from_pattern,
        pauli_measurements=pauli_measurements,
        measurement_labels=measurement_labels,
        show_local_clifford=show_local_clifford,
        node_distance=(0.7, 0.6),
    )
    plt.close()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.parametrize("example", [example_flow, example_gflow, example_pflow])
@pytest.mark.parametrize("measurement_labels", [False, True])
@pytest.mark.parametrize("pauli_measurements", [False, True])
@pytest.mark.parametrize("show_local_clifford", [False, True])
def test_draw_pattern_xzcorrections(
    example: Callable[[Generator], Pattern],
    show_local_clifford: bool,
    pauli_measurements: bool,
    measurement_labels: bool,
    fx_rng: Generator,
) -> None:
    pattern = example(fx_rng)
    pattern.draw(
        annotations=DrawAnnotations.XZCorrections,
        pauli_measurements=pauli_measurements,
        measurement_labels=measurement_labels,
        show_local_clifford=show_local_clifford,
        node_distance=(0.7, 0.6),
    )
    plt.close()


def example_hadamard() -> Pattern:
    circuit = Circuit(1)
    circuit.h(0)
    return circuit.transpile().pattern


def example_local_clifford() -> Pattern:
    pattern = example_hadamard()
    pattern.remove_input_nodes()
    pattern = pattern.infer_pauli_measurements()
    pattern.perform_pauli_measurements()
    return pattern


def test_draw_pattern_xzcorrections_save() -> None:
    pattern = example_hadamard()
    with TemporaryDirectory() as dirname:
        filename = Path(dirname) / "image.png"
        pattern.draw(annotations=DrawAnnotations.XZCorrections, filename=filename)
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
    pattern.draw(annotations=DrawAnnotations.XZCorrections)


@pytest.mark.usefixtures("mock_plot")
def test_og() -> None:
    pattern = Pattern(
        input_nodes=[0, 1, 2, 3],
        cmds=[command.M(0), command.M(1), command.X(2, {0}), command.Z(2, {0}), command.Z(3, {1})],
    )
    pattern.draw(annotations=None)


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
    og.downcast_bloch().extract_causal_flow().draw(legend=False)
    return plt.gcf()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.mpl_image_compare
def test_gflow_draw() -> Figure:
    og = example_og()
    og.downcast_bloch().extract_gflow().draw(legend=False)
    return plt.gcf()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.mpl_image_compare
def test_pauli_flow_draw() -> Figure:
    og = example_og()
    og.infer_pauli_measurements().extract_pauli_flow().draw(legend=False)
    return plt.gcf()


@pytest.mark.usefixtures("mock_plot")
@pytest.mark.mpl_image_compare
def test_xzcorr_draw() -> Figure:
    og = example_og()
    og.downcast_bloch().extract_causal_flow().to_corrections().draw(legend=False)
    return plt.gcf()


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
        pattern = pattern.infer_pauli_measurements()
        pattern.perform_pauli_measurements()
    pattern.standardize()
    pattern.draw(
        flow_from_pattern=flow_and_not_pauli_presimulate, node_distance=(1, 1), measurement_labels=True, legend=False
    )
    return plt.gcf()
