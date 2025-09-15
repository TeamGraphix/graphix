from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
import pytest

from graphix.find_pflow import (
    OpenGraphIndex,
    _find_pflow_simple,
    _get_pflow_matrices,
    _get_reduced_adj,
    _get_topological_generations,
    find_pflow,
)
from graphix.fundamentals import Plane
from graphix.generator import _pflow2pattern
from graphix.linalg import MatGF2
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.parameter import Placeholder
from graphix.random_objects import rand_circuit
from graphix.states import PlanarState
from tests.conftest import fx_rng

if TYPE_CHECKING:
    from numpy.random import Generator
    from pytest_benchmark import BenchmarkFixture


class OpenGraphTestCase(NamedTuple):
    ogi: OpenGraphIndex
    radj: MatGF2 | None
    flow_demand_mat: MatGF2 | None
    order_demand_mat: MatGF2 | None
    has_pflow: bool


class DAGTestCase(NamedTuple):
    adj_mat: MatGF2
    generations: list[list[int]] | None


def get_og_rndcircuit(depth: int, n_qubits: int, n_inputs: int | None = None) -> OpenGraph:
    """Return an open graph from a random circuit.

    Parameters
    ----------
    depth : int
        Circuit depth of the random circuits for generating open graphs.
    n_qubits : int
        Number of qubits in the random circuits for generating open graphs. It controls the number of outputs.
    n_inputs : int | None
        Optional (default to `None`). Maximum number of inputs in the returned open graph. The returned open graph is the open graph generated from the random circuit where `n_qubits - n_inputs` nodes have been removed from the input-nodes set. This operation does not change the flow properties of the graph.

    Returns
    -------
    OpenGraph
        Open graph with causal flow.
    """
    circuit = rand_circuit(n_qubits, depth, fx_rng._fixture_function())
    pattern = circuit.transpile().pattern
    _, edges = pattern.get_graph()
    graph: nx.Graph[int] = nx.Graph(edges)

    angles = pattern.get_angles()
    planes = pattern.get_meas_plane()
    meas = {node: Measurement(angle, planes[node]) for node, angle in angles.items()}

    og = OpenGraph(
        inside=graph,
        inputs=pattern.input_nodes,
        outputs=pattern.output_nodes,
        measurements=meas,
    )

    if n_inputs is not None:
        ni_remove = max(0, n_qubits - n_inputs)
        for i in range(ni_remove):
            og.inputs.remove(i)

    return og


def get_og_dense(ni: int, no: int, m: int) -> OpenGraph:
    """Return a dense open graph with causal, gflow and pflow.

    Parameters
    ----------
    ni : int
        Number of input nodes (must be equal or smaller than `no` ).
    no : int
        Number of output nodes (must be larger than 1).
    m : int
        Number of total nodes (it must satisfy `m - 2*no > 0`).

    Returns
    -------
    OpenGraph
        Open graph with causal and gflow.

    Notes
    -----
    Adapted from Fig. 1 in Houshmand et al., Phys. Rev. A, 98 (2018) (arXiv:1705.01535)
    """
    if no <= 1:
        raise ValueError("Number of outputs must be larger than 1 (no > 1).")
    if m - 2 * no <= 0:
        raise ValueError("Total number of nodes must be larger than twice the number of outputs (m - 2no > 0).")

    inputs = list(range(no))  # we remove inputs afterwards
    outputs = list(range(no, 2 * no))
    edges = [(i, o) for i, o in zip(inputs[:-2], outputs[:-2])]
    edges.extend((node, node + 1) for node in range(2 * no - 1, m - 1))
    edges.append((inputs[-2], m - 1))

    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(range(m))
    graph.add_edges_from(edges)
    graph_c = nx.complement(graph)

    meas = {node: Measurement(Placeholder("Angle"), Plane.XY) for node in range(m) if node not in set(outputs)}

    og = OpenGraph(
        inside=graph_c,
        inputs=inputs,
        outputs=outputs,
        measurements=meas,
    )  # This open graph corresponds to the example in the reference. Now we remove nodes from the set of inputs, since this operation preserves the flow properties.

    ni_remove = max(0, len(og.inputs) - ni)
    for i in og.inputs[ni_remove:]:
        og.inputs.remove(i)
    return og


def prepare_test_og() -> list[OpenGraphTestCase]:
    test_cases: list[OpenGraphTestCase] = []

    # Trivial open graph with pflow and nI = nO
    def get_og_0() -> OpenGraph:
        """Return an open graph with Pauli flow and equal number of outputs and inputs.

        The returned graph has the following structure:

        [0]-1-(2)
        """
        graph: nx.Graph[int] = nx.Graph([(0, 1), (1, 2)])
        inputs = [0]
        outputs = [2]
        meas = {
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.5, Plane.YZ),  # Y
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_0()),
            radj=MatGF2([[1, 0], [0, 1]]),
            flow_demand_mat=MatGF2([[1, 0], [1, 1]]),
            order_demand_mat=MatGF2([[0, 0], [0, 0]]),
            has_pflow=True,
        )
    )

    # Non-trivial open graph without pflow and nI = nO
    def get_og_1() -> OpenGraph:
        """Return an open graph without Pauli flow and equal number of outputs and inputs.

        The returned graph has the following structure:

        [0]-2-4-(6)
            | |
        [1]-3-5-(7)
        """
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)])
        inputs = [1, 0]
        outputs = [6, 7]
        meas = {
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XZ),  # XZ
            2: Measurement(0.5, Plane.XZ),  # X
            3: Measurement(0.5, Plane.YZ),  # Y
            4: Measurement(0.5, Plane.YZ),  # Y
            5: Measurement(0.1, Plane.YZ),  # YZ
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_1()),
            radj=MatGF2(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1],
                ]
            ),
            flow_demand_mat=MatGF2(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [1, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                ]
            ),
            order_demand_mat=MatGF2(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1],
                ]
            ),
            has_pflow=False,
        )
    )

    # Non-trivial open graph with pflow and nI = nO
    def get_og_2() -> OpenGraph:
        """Return an open graph with Pauli flow and equal number of outputs and inputs.

        The returned graph has the following structure:

        [0]-2-4-(6)
            | |
        [1]-3-5-(7)
        """
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)])
        inputs = [0, 1]
        outputs = [6, 7]
        meas = {
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XY),  # XY
            2: Measurement(0.0, Plane.XY),  # X
            3: Measurement(0.1, Plane.XY),  # XY
            4: Measurement(0.0, Plane.XY),  # X
            5: Measurement(0.5, Plane.XY),  # Y
        }
        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_2()),
            radj=MatGF2(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1],
                ]
            ),
            flow_demand_mat=MatGF2(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0, 1],
                ]
            ),
            order_demand_mat=MatGF2(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            has_pflow=True,
        )
    )

    # Non-trivial open graph with pflow and nI != nO
    def get_og_3() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs.

        Example from Fig. 1 in Mitosek and Backens, 2024 (arXiv:2410.23439).
        """
        graph: nx.Graph[int] = nx.Graph(
            [(0, 2), (2, 4), (3, 4), (4, 6), (1, 4), (1, 6), (2, 3), (3, 5), (2, 6), (3, 6)]
        )
        inputs = [0]
        outputs = [5, 6]
        meas = {
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XZ),  # XZ
            2: Measurement(0.5, Plane.YZ),  # Y
            3: Measurement(0.1, Plane.XY),  # XY
            4: Measurement(0, Plane.XZ),  # Z
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_3()),
            radj=MatGF2(
                [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 1]]
            ),
            flow_demand_mat=MatGF2(
                [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0]]
            ),
            order_demand_mat=MatGF2(
                [[0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
            ),
            has_pflow=True,
        )
    )

    # The following tests check the final result only, not the intermediate steps.

    # Non-trivial open graph with pflow and nI != nO
    def get_og_4() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 3), (2, 3), (2, 6), (3, 4), (4, 7), (4, 5), (7, 8)])
        inputs = [0, 1]
        outputs = [5, 6, 8]
        meas = {
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.0, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.5, Plane.XY),
            7: Measurement(0, Plane.XY),
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_4()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=True,
        )
    )

    # Non-trivial open graph with pflow and nI != nO
    def get_og_5() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 2), (2, 3), (3, 4)])
        inputs = [0, 1]
        outputs = [1, 3, 4]
        meas = {0: Measurement(0.1, Plane.XY), 2: Measurement(0.5, Plane.YZ)}

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_5()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=True,
        )
    )

    # Non-trivial open graph with pflow and nI != nO
    def get_og_6() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7)])
        inputs = [1]
        outputs = [6, 2, 7]
        meas = {
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.YZ),
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_6()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=True,
        )
    )

    # Disconnected open graph with pflow and nI != nO
    def get_og_7() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (4, 6)])
        inputs: list[int] = []
        outputs = [1, 3, 4]
        meas = {0: Measurement(0.5, Plane.XZ), 2: Measurement(0, Plane.YZ), 6: Measurement(0.2, Plane.XY)}

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_7()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=True,
        )
    )

    # Non-trivial open graph without pflow and nI != nO
    def get_og_8() -> OpenGraph:
        """Return an open graph without Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph(
            [(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7), (5, 6), (6, 7)]
        )
        inputs = [1]
        outputs = [6, 2, 7]
        meas = {
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.XY),
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_8()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=False,
        )
    )

    # Disconnected open graph without pflow and nI != nO
    def get_og_9() -> OpenGraph:
        """Return an open graph without Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (4, 6)])
        inputs = [0]
        outputs = [1, 3, 4]
        meas = {0: Measurement(0.1, Plane.XZ), 2: Measurement(0, Plane.YZ), 6: Measurement(0.2, Plane.XY)}

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_9()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=False,
        )
    )

    # Non-trivial open graph without pflow and nI != nO
    def get_og_10() -> OpenGraph:
        """Return a graph constructed by adding a disconnected input to graph_6. The resulting graph does not have pflow."""
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7)])
        graph.add_node(8)
        inputs = [1, 8]
        outputs = [6, 2, 7]
        meas = {
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.YZ),
            8: Measurement(0.1, Plane.XY),
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_10()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=False,
        )
    )

    # Open graph with only Pauli measurements, without pflow and nI != nO
    def get_og_11() -> OpenGraph:
        """Return an open graph without Pauli flow and unequal number of outputs and inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 3), (2, 3), (2, 6), (3, 4), (4, 7), (4, 5), (7, 8)])
        inputs = [0, 1]
        outputs = [5, 6, 8]
        meas = {
            0: Measurement(0, Plane.XY),
            1: Measurement(0, Plane.XY),
            2: Measurement(0, Plane.XZ),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.5, Plane.XY),
            7: Measurement(0, Plane.YZ),
        }

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_11()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=False,
        )
    )

    # Open graph with only Pauli measurements, with pflow and nI != nO
    def get_og_12() -> OpenGraph:
        """Return an open graph with Pauli flow and unequal number of outputs and inputs. Even though all nodes are Pauli-measured, open graph has flow because none of them are inputs."""
        graph: nx.Graph[int] = nx.Graph([(0, 2), (1, 3), (2, 3), (2, 6), (3, 4), (4, 7), (4, 5), (7, 8)])
        outputs = [5, 6, 8]
        meas = {
            0: Measurement(0, Plane.XZ),
            1: Measurement(0, Plane.XZ),
            2: Measurement(0, Plane.XZ),
            3: Measurement(0, Plane.XZ),
            4: Measurement(0, Plane.XZ),
            7: Measurement(0, Plane.XZ),
        }

        return OpenGraph(inside=graph, inputs=[], outputs=outputs, measurements=meas)

    test_cases.append(
        OpenGraphTestCase(
            ogi=OpenGraphIndex(get_og_12()),
            radj=None,
            flow_demand_mat=None,
            order_demand_mat=None,
            has_pflow=True,
        )
    )

    return test_cases


def prepare_benchmark_og() -> list[OpenGraphTestCase]:
    test_cases: list[OpenGraphTestCase] = []

    # Open graph from random circuit
    test_cases.extend(
        (
            OpenGraphTestCase(
                ogi=OpenGraphIndex(get_og_rndcircuit(depth=20, n_qubits=7, n_inputs=1)),
                radj=None,
                flow_demand_mat=None,
                order_demand_mat=None,
                has_pflow=True,
            ),
            OpenGraphTestCase(
                ogi=OpenGraphIndex(get_og_dense(ni=3, no=6, m=400)),
                radj=None,
                flow_demand_mat=None,
                order_demand_mat=None,
                has_pflow=True,
            ),
        )
    )
    return test_cases


def prepare_test_dag() -> list[DAGTestCase]:
    test_cases: list[DAGTestCase] = []

    # Simple DAG
    test_cases.extend(
        (  # Simple DAG
            DAGTestCase(
                adj_mat=MatGF2([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]]), generations=[[0], [1, 2], [3]]
            ),
            # Graph with loop
            DAGTestCase(adj_mat=MatGF2([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 0]]), generations=None),
            # Disconnected graph
            DAGTestCase(
                adj_mat=MatGF2([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]),
                generations=[[0, 2], [1, 3, 4]],
            ),
        )
    )

    return test_cases


class TestPflow:
    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_get_reduced_adj(self, test_case: OpenGraphTestCase) -> None:
        if test_case.radj is not None:
            ogi = test_case.ogi
            radj = _get_reduced_adj(ogi)
            assert radj == test_case.radj

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_get_pflow_matrices(self, test_case: OpenGraphTestCase) -> None:
        if test_case.flow_demand_mat is not None and test_case.order_demand_mat is not None:
            ogi = test_case.ogi
            flow_demand_matrix, order_demand_matrix = _get_pflow_matrices(ogi)

            assert flow_demand_matrix == test_case.flow_demand_mat
            assert order_demand_matrix == test_case.order_demand_mat

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_find_pflow_simple(self, test_case: OpenGraphTestCase) -> None:
        if test_case.flow_demand_mat is not None:
            ogi = test_case.ogi
            if len(ogi.og.outputs) == len(ogi.og.inputs):
                pflow_algebraic = _find_pflow_simple(ogi)

                if not test_case.has_pflow:
                    assert pflow_algebraic is None
                else:
                    assert pflow_algebraic is not None
                    correction_matrix, _ = pflow_algebraic
                    ident = MatGF2(np.eye(len(ogi.non_outputs), dtype=np.int_))
                    assert test_case.flow_demand_mat @ correction_matrix == ident

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_find_pflow_determinism(self, test_case: OpenGraphTestCase, fx_rng: Generator) -> None:
        og = test_case.ogi.og

        pflow = find_pflow(og)

        if not test_case.has_pflow:
            assert pflow is None
        else:
            assert pflow is not None

            pattern = _pflow2pattern(
                graph=og.inside,
                inputs=set(og.inputs),
                meas_planes={i: m.plane for i, m in og.measurements.items()},
                angles={i: m.angle for i, m in og.measurements.items()},
                p=pflow[0],
                l_k=pflow[1],
            )
            pattern.reorder_output_nodes(og.outputs)

            alpha = 2 * np.pi * fx_rng.random()
            state_ref = pattern.simulate_pattern(input_state=PlanarState(Plane.XY, alpha))

            n_shots = 5
            results = []
            for _ in range(n_shots):
                state = pattern.simulate_pattern(input_state=PlanarState(Plane.XY, alpha))
                results.append(np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())))

            avg = sum(results) / n_shots

            assert avg == pytest.approx(1)

    @pytest.mark.parametrize("test_case", prepare_benchmark_og())
    def test_benchmark_pflow(self, benchmark: BenchmarkFixture, test_case: OpenGraphTestCase) -> None:
        og = test_case.ogi.og

        pflow = benchmark(find_pflow, og)

        if not test_case.has_pflow:
            assert pflow is None
        else:
            assert pflow is not None

    @pytest.mark.parametrize("test_case", prepare_test_dag())
    def test_get_topological_generations(self, test_case: DAGTestCase) -> None:
        adj_mat = test_case.adj_mat
        generations_ref = test_case.generations

        assert generations_ref == _get_topological_generations(adj_mat)
