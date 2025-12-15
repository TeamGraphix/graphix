"""Unit tests for the algebraic flow finding algorithm (for generalised or Pauli flow).

This module tests the following:
    - Computation of the reduced adjacency matrix.
    - Computation of the flow-demand and order-demand matrices. We check this routine for open graphs with Pauli-angle measurements, intepreted as planes or as axes.
    - Computation of the correction matrix.
    - Computation of topological generations on small DAGs.

The second part of the flow-finding algorithm (namely, verifying if the correction matrix is compatible with a DAG) is not done in this test module. For a complete test on the flow-finding algorithms see `tests.test_opengraph.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
import pytest

from graphix._linalg import MatGF2
from graphix.flow._find_gpflow import (
    AlgebraicOpenGraph,
    PlanarAlgebraicOpenGraph,
    _try_ordering_matrix_to_topological_generations,
    compute_correction_matrix,
)
from graphix.fundamentals import Axis, Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph

if TYPE_CHECKING:
    from graphix.fundamentals import AbstractMeasurement, AbstractPlanarMeasurement


class AlgebraicOpenGraphTestCase(NamedTuple):
    aog: AlgebraicOpenGraph[AbstractMeasurement] | PlanarAlgebraicOpenGraph[AbstractPlanarMeasurement]
    radj: MatGF2
    flow_demand_mat: MatGF2
    order_demand_mat: MatGF2
    has_corr_mat: bool


class DAGTestCase(NamedTuple):
    adj_mat: MatGF2
    generations: tuple[frozenset[int], ...] | None


def prepare_test_og() -> list[AlgebraicOpenGraphTestCase]:
    test_cases: list[AlgebraicOpenGraphTestCase] = []

    # Trivial open graph with pflow and nI = nO
    def get_og_0() -> OpenGraph[Plane | Axis]:
        """Return an open graph with Pauli flow and equal number of outputs and inputs.

        The returned graph has the following structure:

        [0]-1-(2)
        """
        return OpenGraph(
            graph=nx.Graph([(0, 1), (1, 2)]), input_nodes=[0], output_nodes=[2], measurements={0: Plane.XY, 1: Axis.Y}
        )

    test_cases.append(
        AlgebraicOpenGraphTestCase(
            aog=AlgebraicOpenGraph(get_og_0()),
            radj=MatGF2([[1, 0], [0, 1]]),
            flow_demand_mat=MatGF2([[1, 0], [1, 1]]),
            order_demand_mat=MatGF2([[0, 0], [0, 0]]),
            has_corr_mat=True,
        )
    )

    # Non-trivial open graph with pflow and nI = nO
    def get_og_1() -> OpenGraph[Measurement]:
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
        return OpenGraph(graph=graph, input_nodes=inputs, output_nodes=outputs, measurements=meas)

    test_cases.extend(
        (
            AlgebraicOpenGraphTestCase(
                aog=AlgebraicOpenGraph(get_og_1()),
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
                has_corr_mat=True,
            ),
            # Same open graph but we interpret the measurements on Pauli axes as planar measurements, therefore, there flow-demand and order demand matrices are different.
            AlgebraicOpenGraphTestCase(
                aog=PlanarAlgebraicOpenGraph(get_og_1()),
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
                        [0, 1, 1, 0, 0, 1],
                    ]
                ),
                order_demand_mat=MatGF2(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                    ]
                ),
                has_corr_mat=True,
            ),
        )
    )

    # Non-trivial open graph with pflow and nI != nO
    def get_og_2() -> OpenGraph[Measurement]:
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

        return OpenGraph(graph=graph, input_nodes=inputs, output_nodes=outputs, measurements=meas)

    test_cases.extend(
        (
            AlgebraicOpenGraphTestCase(
                aog=AlgebraicOpenGraph(get_og_2()),
                radj=MatGF2(
                    [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 1]]
                ),
                flow_demand_mat=MatGF2(
                    [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0]]
                ),
                order_demand_mat=MatGF2(
                    [[0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
                ),
                has_corr_mat=True,
            ),
            # Same open graph but we interpret the measurements on Pauli axes as planar measurements, therefore, there flow-demand and order demand matrices are different.
            # The new flow-demand matrix is not invertible, therefore the open graph does not have gflow.
            AlgebraicOpenGraphTestCase(
                aog=PlanarAlgebraicOpenGraph(get_og_2()),
                radj=MatGF2(
                    [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 1]]
                ),
                flow_demand_mat=MatGF2(
                    [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0]]
                ),
                order_demand_mat=MatGF2(
                    [[0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 0, 1]]
                ),
                has_corr_mat=False,
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
                adj_mat=MatGF2([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]]),
                generations=(frozenset({0}), frozenset({1, 2}), frozenset({3})),
            ),
            # Graph with loop
            DAGTestCase(adj_mat=MatGF2([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 0]]), generations=None),
            # Disconnected graph
            DAGTestCase(
                adj_mat=MatGF2([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]),
                generations=(frozenset({0, 2}), frozenset({1, 3, 4})),
            ),
        )
    )

    return test_cases


class TestAlgebraicFlow:
    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_compute_reduced_adj(self, test_case: AlgebraicOpenGraphTestCase) -> None:
        aog = test_case.aog
        radj = aog._compute_reduced_adj()
        assert np.all(radj == test_case.radj)

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_og_matrices(self, test_case: AlgebraicOpenGraphTestCase) -> None:
        aog = test_case.aog
        assert np.all(aog.flow_demand_matrix == test_case.flow_demand_mat)
        assert np.all(aog.order_demand_matrix == test_case.order_demand_mat)

    @pytest.mark.parametrize("test_case", prepare_test_og())
    def test_correction_matrix(self, test_case: AlgebraicOpenGraphTestCase) -> None:
        aog = test_case.aog
        corr_matrix = compute_correction_matrix(aog)

        ident = MatGF2(np.eye(len(aog.non_outputs), dtype=np.uint8))
        if test_case.has_corr_mat:
            assert corr_matrix is not None
            assert np.all(
                (test_case.flow_demand_mat @ corr_matrix.c_matrix) % 2 == ident
            )  # Test with numpy matrix product.
        else:
            assert corr_matrix is None

    @pytest.mark.parametrize("test_case", prepare_test_dag())
    def test_try_ordering_matrix_to_topological_generations(self, test_case: DAGTestCase) -> None:
        adj_mat = test_case.adj_mat
        generations_ref = test_case.generations

        assert generations_ref == _try_ordering_matrix_to_topological_generations(adj_mat)
