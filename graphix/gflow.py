"""Flow finding algorithm.

For a given underlying graph (G, I, O, meas_plane), this method finds a (generalized) flow [NJP 9, 250 (2007)]
in polynomincal time.
In particular, this outputs gflow with minimum depth, maximally delayed gflow.

Ref: Mhalla and Perdrix, International Colloquium on Automata,
Languages, and Programming (Springer, 2008), pp. 857-868.
Ref: Backens et al., Quantum 5, 421 (2021).

"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    import networkx as nx

    from graphix.fundamentals import PauliPlane, Plane

# MEMO: We could add layer inference


class Flow(NamedTuple):
    """Flow function and layer."""

    ffunc: dict[int, int]
    layer: dict[int, int]


class GFlow(NamedTuple):
    """Generalized flow function and layer."""

    ffunc: dict[int, set[int]]
    layer: dict[int, int]


PauliFlow = GFlow


def odd_neighbor(graph: nx.Graph[int], vertices: AbstractSet[int]) -> set[int]:
    """Return the set containing the odd neighbor of a set of vertices.

    Parameters
    ----------
    graph : :class:`networkx.Graph`
        Underlying graph
    vertices : set
        set of nodes indices to find odd neighbors

    Returns
    -------
    odd_neighbors : set
        set of indices for odd neighbor of set `vertices`.
    """
    odd_neighbors: set[int] = set()
    for vertex in vertices:
        odd_neighbors.symmetric_difference_update(graph.neighbors(vertex))
    return odd_neighbors


def find_flow(graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int]) -> Flow | None:
    """Causal flow finding algorithm.

    For open graph g with input, output, and measurement planes, this returns causal flow.
    For more detail of causal flow, see Danos and Kashefi, PRA 74, 052310 (2006).

    Original algorithm by Mhalla and Perdrix,
    International Colloquium on Automata, Languages, and Programming (2008),
    pp. 857-868.

    Parameters
    ----------

    Returns
    -------
    """
    raise NotImplementedError


def find_gflow(
    graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int], meas_planes: Mapping[int, Plane]
) -> GFlow | None:
    """Maximally delayed gflow finding algorithm.

    For open graph g with input, output, and measurement planes, this returns maximally delayed gflow.

    gflow consist of function g(i) where i is the qubit labels,
    and strict partial ordering < or layers labels l_k where each element
    specify the order of qubits to be measured to maintain determinism in MBQC.
    In practice, we must measure qubits in order specified in array l_k (increasing order
    of l_k from 1), and for each measurements of qubit i we must perform corrections on
    qubits in g(i), depending on the measurement outcome.

    For more details of gflow, see Browne et al., NJP 9, 250 (2007).
    We use the extended gflow finding algorithm in Backens et al., Quantum 5, 421 (2021).

    Parameters
    ----------

    Returns
    -------
    """
    raise NotImplementedError


def find_pauliflow(
    graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int], meas_pplanes: Mapping[int, PauliPlane]
) -> PauliFlow | None:
    """Maximally delayed Pauli flow finding algorithm.

    For open graph g with input, output, measurement planes and measurement angles, this returns maximally delayed Pauli flow.

    Pauli flow consist of function p(i) where i is the qubit labels,
    and strict partial ordering < or layers labels l_k where each element
    specify the order of qubits to be measured to maintain determinism in MBQC.
    In practice, we must measure qubits in order specified in array l_k (increasing order
    of l_k from 1), and for each measurements of qubit i we must perform corrections on
    qubits in p(i), depending on the measurement outcome.

    For more details of Pauli flow and the finding algorithm used in this method,
    see Simmons et al., EPTCS 343, 2021, pp. 50-101 (arXiv:2109.05654).

    Parameters
    ----------

    Returns
    -------
    """
    raise NotImplementedError


def verify_flow(flow: Flow, graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int]) -> bool:
    """Check whether the flow is valid.

    Parameters
    ----------

    Returns
    -------
    """
    raise NotImplementedError


def verify_gflow(
    gflow: GFlow, graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int], meas_planes: Mapping[int, Plane]
) -> bool:
    """Check whether the gflow is valid.

    Parameters
    ----------

    Returns
    -------
    """
    raise NotImplementedError


def verify_pauliflow(
    pflow: PauliFlow,
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas_pplanes: Mapping[int, PauliPlane],
) -> bool:
    """Check whether the Pauliflow is valid.

    Parameters
    ----------

    Returns
    -------
    """
    raise NotImplementedError
