"""Flow finding algorithm.

For a given underlying graph (G, I, O, meas_plane), this method finds a (generalized) flow [NJP 9, 250 (2007)]
in polynomincal time.
In particular, this outputs gflow with minimum depth, maximally delayed gflow.

Ref: Mhalla and Perdrix, International Colloquium on Automata,
Languages, and Programming (Springer, 2008), pp. 857-868.
Ref: Backens et al., Quantum 5, 421 (2021).

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    import networkx as nx

    from graphix.fundamentals import PauliPlane, Plane

Flow = dict[int, int]
GFlow = dict[int, set[int]]
Layer = dict[int, int]

# TODO: Update docstring


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


def find_flow(graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int]) -> tuple[Flow, Layer] | None:
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
) -> tuple[GFlow, Layer] | None:
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
    graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int], meas_planes: Mapping[int, PauliPlane]
) -> tuple[GFlow, Layer] | None:
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


def verify_flow(graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int], flow: Mapping[int, int]) -> bool:
    """Check whether the flow is valid.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        Graph (incl. input and output)
    flow: dict[int, set]
        flow function. flow[i] is the set of qubits to be corrected for the measurement of qubit i.
    meas_planes: dict[int, str]
        optional: measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.

    Returns
    -------
    valid_flow: bool
        True if the flow is valid. False otherwise.
    """
    raise NotImplementedError


def verify_gflow(
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    gflow: Mapping[int, AbstractSet[int]],
    meas_planes: Mapping[int, Plane],
) -> bool:
    """Check whether the gflow is valid.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        Graph (incl. input and output)
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output
    gflow: dict[int, set]
        gflow function. gflow[i] is the set of qubits to be corrected for the measurement of qubit i.
        .. seealso:: :func:`find_gflow`
    meas_planes: dict[int, str]
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.

    Returns
    -------
    valid_gflow: bool
        True if the gflow is valid. False otherwise.
    """
    raise NotImplementedError


def verify_pauliflow(
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    pauliflow: Mapping[int, AbstractSet[int]],
    meas_planes: Mapping[int, PauliPlane],
) -> bool:
    """Check whether the Pauliflow is valid.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        Graph (incl. input and output)
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output
    pauliflow: dict[int, set]
        Pauli flow function. pauliflow[i] is the set of qubits to be corrected for the measurement of qubit i.
    meas_planes: dict[int, Plane]
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
    meas_angles: dict[int, float]
        measurement angles for each qubits. meas_angles[i] is the measurement angle for qubit i.

    Returns
    -------
    valid_pauliflow: bool
        True if the Pauliflow is valid. False otherwise.
    """
    raise NotImplementedError
