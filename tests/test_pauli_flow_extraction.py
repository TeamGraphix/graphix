r"""Tests for Pauli-flow extraction from a pattern / XZ-corrections.

Correctness criterion
---------------------
A reconstructed Pauli flow ``pf`` generates the original pattern if and only if
``pf.check_well_formed()`` succeeds *and* ``pf.to_corrections()`` reproduces the pattern's
X- and Z-corrections exactly. The latter "round-trip" property is the decisive check: it
guarantees that the flow generates *this* pattern (and not merely some Pauli flow of the
underlying open graph, which need not be unique). The tests below verify this on the three
worked examples of the issue, on a Pauli-measured open graph, and on a randomized family of
open graphs that admit a Pauli flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest

from graphix import Measurement, OpenGraph, Pattern
from graphix.command import E, M, N, X, Z
from graphix.flow.core import PauliFlow, XZCorrections
from graphix.flow.exceptions import FlowGenericError, FlowGenericErrorReason
from graphix.opengraph import OpenGraphError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from collections.abc import Set as AbstractSet

    from numpy.random import Generator


def _norm(corrections: Mapping[int, AbstractSet[int]]) -> dict[int, frozenset[int]]:
    """Drop empty correction sets to compare correction dictionaries up to empty entries."""
    return {k: frozenset(v) for k, v in corrections.items() if v}


def _assert_round_trip(pattern: Pattern) -> None:
    xz = pattern.extract_xzcorrections()
    pf = xz.to_pauli_flow()
    # `to_pauli_flow` no longer runs `check_well_formed` in production; the
    # well-formedness is asserted here, in the test-suite, instead.
    assert pf.is_well_formed()
    rt = pf.to_corrections()
    assert _norm(rt.x_corrections) == _norm(xz.x_corrections)
    assert _norm(rt.z_corrections) == _norm(xz.z_corrections)


def _correction_function(pattern: Pattern) -> dict[int, set[int]]:
    pf = pattern.extract_pauli_flow()
    return {k: set(v) for k, v in pf.correction_function.items()}


def _causal_pattern() -> Pattern:
    return Pattern(input_nodes=[0], cmds=[N(1), E((0, 1)), M(0, Measurement.XY(0)), X(1, {0})], output_nodes=[1])


def _gflow_pattern() -> Pattern:
    return Pattern(
        input_nodes=[0],
        cmds=[
            N(1), N(2), N(3), E((0, 1)), E((0, 2)), E((1, 2)), E((1, 3)),
            M(0, Measurement.XY(0.1)), X(2, {0}), X(3, {0}),
            M(1, Measurement.XZ(0.2)), Z(2, {1}), Z(3, {1}), X(2, {1}),
        ],
        output_nodes=[2, 3],
    )  # fmt: skip


def _pauli_pattern() -> Pattern:
    return Pattern(
        input_nodes=[0],
        cmds=[
            N(1), N(2), N(3), E((0, 1)), E((1, 2)), E((2, 3)),
            M(0, Measurement.X), X(3, {0}),
            M(1, Measurement.X), Z(3, {1}),
            M(2, Measurement.X), X(3, {2}),
        ],
        output_nodes=[3],
    )  # fmt: skip


def test_extract_pauli_flow_causal_example() -> None:
    pattern = _causal_pattern()
    assert _correction_function(pattern) == {0: {1}}
    _assert_round_trip(pattern)


def test_extract_pauli_flow_gflow_example() -> None:
    pattern = _gflow_pattern()
    assert _correction_function(pattern) == {0: {2, 3}, 1: {1, 2}}
    _assert_round_trip(pattern)


def test_extract_pauli_flow_pauli_example() -> None:
    # The flow must include the anachronical correction (node 1 in p(0)) that does not
    # appear in the pattern, in order to satisfy the X-axis proposition (P7).
    pattern = _pauli_pattern()
    assert _correction_function(pattern) == {0: {1, 3}, 1: {2}, 2: {3}}
    _assert_round_trip(pattern)


def test_extract_pauli_flow_pauli_opengraph() -> None:
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (2, 4), (3, 4), (4, 6), (1, 4), (1, 6), (2, 3), (3, 5), (2, 6), (3, 6)]),
        input_nodes=[0],
        output_nodes=[5, 6],
        measurements={
            0: Measurement.XY(0.1),
            1: Measurement.XZ(0.1),
            2: Measurement.Y,
            3: Measurement.XY(0.1),
            4: Measurement.Z,
        },
    )
    _assert_round_trip(og.to_pattern())


def test_extract_pauli_flow_output_zcorrection() -> None:
    # Regression: a Z-correction whose future target is an *output* node imposes a real GF(2)
    # equation in the reconstruction -- it cannot be dropped (e.g. by skipping future nodes that are
    # not measured in a non-Pauli plane) without silently breaking the Z-correction round-trip.
    # Distilled from a randomized open graph that exercises this case.
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 4), (0, 5), (0, 7), (1, 3), (1, 4), (2, 4), (2, 5), (3, 7), (4, 6), (6, 7)]),
        input_nodes=[],
        output_nodes=[3, 0, 5],
        measurements={
            1: Measurement.X,
            2: Measurement.X,
            4: Measurement.YZ(0.3),
            6: Measurement.Z,
            7: Measurement.Y,
        },
    )
    _assert_round_trip(og.to_pattern())


_MEASUREMENTS: list[Callable[[Generator], Measurement]] = [
    lambda r: Measurement.XY(float(r.random())),
    lambda r: Measurement.XZ(float(r.random())),
    lambda r: Measurement.YZ(float(r.random())),
    lambda _r: Measurement.X,
    lambda _r: Measurement.Y,
    lambda _r: Measurement.Z,
]


@pytest.mark.parametrize("seed", range(400))
def test_extract_pauli_flow_randomized_round_trip(seed: int) -> None:
    # For each seed, draw a random open graph. Seeds with no edges, or whose open graph does not
    # admit a Pauli flow (so `to_pattern` raises `OpenGraphError`), are skipped; the rest are
    # converted to a pattern and round-tripped. Passing the seed as a parameter keeps each case
    # independently reproducible and easy to debug when one fails.
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 10))
    graph = nx.gnp_random_graph(n, 0.45, seed=seed)
    if graph.number_of_edges() == 0:
        pytest.skip("empty graph")
    nodes = list(graph.nodes())
    rng.shuffle(nodes)
    n_out = int(rng.integers(1, max(2, n // 2)))
    n_in = int(rng.integers(0, max(1, n // 2)))
    outputs = nodes[:n_out]
    inputs = nodes[n_out : n_out + n_in]
    measurements = {m: _MEASUREMENTS[int(rng.integers(0, len(_MEASUREMENTS)))](rng) for m in nodes if m not in outputs}
    try:
        pattern = OpenGraph(
            graph=graph, input_nodes=inputs, output_nodes=outputs, measurements=measurements
        ).to_pattern()
    except OpenGraphError:
        # The only documented raise condition of `OpenGraph.to_pattern`: no flow -> not a test case.
        pytest.skip("open graph does not admit a flow")
    _assert_round_trip(pattern)


def test_extract_pauli_flow_pins_the_pattern_specific_flow() -> None:
    r"""Reconstruction must return the Pauli flow implemented by *this* pattern, not just any Pauli flow of the underlying open graph.

    A Pauli flow on an open graph is not unique when Pauli-measured nodes admit several distinct
    anachronical-correction patterns. ``OpenGraph.find_pauli_flow`` returns *some* maximally
    delayed Pauli flow (chosen by the underlying algorithm); a trivial implementation of
    ``XZCorrections.to_pauli_flow`` that delegates to it -- and ignores the XZ-corrections of
    the pattern entirely -- would therefore pass every existing round-trip test that happens
    to feed it patterns whose flow already coincides with that algorithmic choice.

    This test pins down a small open graph that admits two well-formed Pauli flows whose
    ``to_corrections()`` outputs differ, builds the XZ-corrections of the *non-default* one,
    and asserts that the reconstruction returns the chosen flow (and not the one
    ``find_pauli_flow`` would have returned on the bare open graph).
    """
    og: OpenGraph[Measurement] = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (2, 3)]),
        input_nodes=[],
        output_nodes=[3],
        measurements={0: Measurement.X, 1: Measurement.X, 2: Measurement.X},
    )
    # Layer order: outputs first, then by measurement order (last layer measured first).
    layers: list[set[int]] = [{3}, {2}, {1}, {0}]

    # Two distinct, well-formed Pauli flows on the same open graph.
    pf_with_anachronical = PauliFlow(og, {0: {1}, 1: {2}, 2: {3}}, layers)
    pf_with_future = PauliFlow(og, {0: {1, 3}, 1: {2}, 2: {3}}, layers)
    assert pf_with_anachronical.is_well_formed()
    assert pf_with_future.is_well_formed()

    xz_with_anachronical = pf_with_anachronical.to_corrections()
    xz_with_future = pf_with_future.to_corrections()
    # The corrections genuinely differ: the future-style flow X-corrects node 3 from
    # node 0, the anachronical-style flow does not.
    assert _norm(xz_with_anachronical.x_corrections) != _norm(xz_with_future.x_corrections)

    # `find_pauli_flow` is allowed to return either valid flow (or another); what matters
    # is that the trivial implementation `pf = self.og.find_pauli_flow()` is independent of
    # the XZ-corrections we feed in -- so it cannot get both round-trips right.
    trivial_choice = og.find_pauli_flow()
    assert trivial_choice is not None
    trivial_cf = {k: set(v) for k, v in trivial_choice.correction_function.items()}

    # The real reconstruction must use the XZ-corrections to disambiguate.
    rebuilt_anachronical = xz_with_anachronical.to_pauli_flow()
    rebuilt_future = xz_with_future.to_pauli_flow()
    assert dict(rebuilt_anachronical.correction_function) == {0: {1}, 1: {2}, 2: {3}}
    assert dict(rebuilt_future.correction_function) == {0: {1, 3}, 1: {2}, 2: {3}}

    # And the round-trip on each must still recover the original XZ-corrections exactly.
    rt_anachronical = rebuilt_anachronical.to_corrections()
    rt_future = rebuilt_future.to_corrections()
    assert _norm(rt_anachronical.x_corrections) == _norm(xz_with_anachronical.x_corrections)
    assert _norm(rt_anachronical.z_corrections) == _norm(xz_with_anachronical.z_corrections)
    assert _norm(rt_future.x_corrections) == _norm(xz_with_future.x_corrections)
    assert _norm(rt_future.z_corrections) == _norm(xz_with_future.z_corrections)

    # Discriminator assertion: at least one of the two reconstructions must disagree with
    # the trivial ``find_pauli_flow`` choice (the two flows differ from each other, so the
    # trivial impl -- which returns the same flow regardless -- cannot match both).
    assert (
        dict(rebuilt_anachronical.correction_function) != trivial_cf
        or dict(rebuilt_future.correction_function) != trivial_cf
    )


def test_to_pauli_flow_empty_pattern() -> None:
    # Regression for the production manifestation of #531: an empty pattern has a trivial
    # Pauli flow, so `to_pauli_flow` must not raise. The well-formedness sanity check is no
    # longer run systematically in production (it lives in the test-suite); `check_well_formed`'s
    # own behaviour on an empty partial order is tracked separately in #531.
    pf = Pattern().extract_xzcorrections().to_pauli_flow()
    assert dict(pf.correction_function) == {}


@pytest.mark.parametrize(
    ("measurements", "inputs", "outputs", "edges", "extra_nodes", "layers"),
    [
        # A measured input node pinned into its own correction set (Z/XZ/YZ) admits no Pauli flow,
        # because the correction set's image cannot contain an input node.
        ({0: Measurement.Z}, [0], [1], [(0, 1)], [], [{1}, {0}]),
        ({0: Measurement.XZ(0.1)}, [0], [1], [(0, 1)], [], [{1}, {0}]),
        ({0: Measurement.YZ(0.1)}, [0], [1], [(0, 1)], [], [{1}, {0}]),
        # An isolated node measured in the XY plane cannot satisfy proposition P4 (it must lie in
        # the odd neighbourhood of its correction set), so the GF(2) system has no solution.
        ({0: Measurement.XY(0.1), 1: Measurement.XY(0.1)}, [], [2], [(1, 2)], [0], [{2}, {1}, {0}]),
    ],
    ids=["z-input", "xz-input", "yz-input", "isolated-xy"],
)
def test_to_pauli_flow_raises_when_no_flow_exists(
    measurements: dict[int, Measurement],
    inputs: list[int],
    outputs: list[int],
    edges: list[tuple[int, int]],
    extra_nodes: list[int],
    layers: list[set[int]],
) -> None:
    graph: nx.Graph[int] = nx.Graph(edges)
    graph.add_nodes_from(extra_nodes)
    og = OpenGraph(graph=graph, input_nodes=inputs, output_nodes=outputs, measurements=measurements)
    with pytest.raises(FlowGenericError) as exc_info:
        XZCorrections(og, {}, {}, layers).to_pauli_flow()
    assert exc_info.value.reason == FlowGenericErrorReason.NoPauliFlow
    # The rendered message names the failure so it is actionable in a traceback.
    assert "No Pauli flow" in str(exc_info.value)
