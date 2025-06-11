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

from typing_extensions import assert_never

from graphix.fundamentals import Axis, Plane
from graphix.measurements import PauliMeasurement

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    import networkx as nx


def find_gflow(
    graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int], meas_planes: Mapping[int, Plane]
) -> tuple[dict[int, set[int]], dict[int, int]]:
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
    graph: :class:`networkx.Graph`
        Graph (incl. input and output)
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output
    meas_planes: dict
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.

    Returns
    -------
    g: dict
        gflow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """
    raise NotImplementedError


def find_flow(
    graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int]
) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Causal flow finding algorithm.

    For open graph g with input, output, and measurement planes, this returns causal flow.
    For more detail of causal flow, see Danos and Kashefi, PRA 74, 052310 (2006).

    Original algorithm by Mhalla and Perdrix,
    International Colloquium on Automata, Languages, and Programming (2008),
    pp. 857-868.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        Graph (incl. input and output)
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output

    Returns
    -------
    f: list of nodes
        causal flow function. f[i] is the qubit to be measured after qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """
    raise NotImplementedError


def find_pauliflow(
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas_planes: Mapping[int, Plane],
    meas_angles: Mapping[int, float],
) -> tuple[dict[int, set[int]], dict[int, int]]:
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
    graph: :class:`networkx.Graph`
        Graph (incl. input and output)
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output
    meas_planes: dict
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
    meas_angles: dict
        measurement angles for each qubits. meas_angles[i] is the measurement angle for qubit i.

    Returns
    -------
    p: dict
        Pauli flow function. p[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by  Pauli flow algorithm. l_k[d] is a node set of depth d.
    """
    raise NotImplementedError


def find_odd_neighbor(graph: nx.Graph[int], vertices: AbstractSet[int]) -> set[int]:
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
        neighbors = set(graph.neighbors(vertex))
        odd_neighbors ^= neighbors
    return odd_neighbors


def get_layers(l_k: Mapping[int, int]) -> tuple[int, dict[int, set[int]]]:
    """Get components of each layer.

    Parameters
    ----------
    l_k: dict
        layers obtained by flow or gflow algorithms

    Returns
    -------
    d: int
        minimum depth of graph
    layers: dict of set
        components of each layer
    """
    d = min(l_k.values())
    layers: dict[int, set[int]] = {k: set() for k in range(d + 1)}
    for i, val in l_k.items():
        layers[val] |= {i}
    return d, layers


def verify_flow(
    graph: nx.Graph[int], iset: AbstractSet[int], oset: AbstractSet[int], flow: Mapping[int, AbstractSet[int]]
) -> bool:
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
    meas_planes: Mapping[int, Plane],
    meas_angles: Mapping[int, float],
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


def get_pauli_nodes(
    meas_planes: Mapping[int, Plane], meas_angles: Mapping[int, float]
) -> tuple[set[int], set[int], set[int]]:
    """Get sets of nodes measured in X, Y, Z basis.

    Parameters
    ----------
    meas_planes: dict[int, Plane]
        measurement planes for each node.
    meas_angles: dict[int, float]
        measurement angles for each node.

    Returns
    -------
    l_x: set
        set of nodes measured in X basis.
    l_y: set
        set of nodes measured in Y basis.
    l_z: set
        set of nodes measured in Z basis.
    """
    l_x, l_y, l_z = set(), set(), set()
    for node, plane in meas_planes.items():
        pm = PauliMeasurement.try_from(plane, meas_angles[node])
        if pm is None:
            continue
        if pm.axis == Axis.X:
            l_x |= {node}
        elif pm.axis == Axis.Y:
            l_y |= {node}
        elif pm.axis == Axis.Z:
            l_z |= {node}
        else:
            assert_never(pm.axis)
    return l_x, l_y, l_z
