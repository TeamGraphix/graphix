"""Functions to visualize the resource state of MBQC pattern."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, reveal_type

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from graphix import gflow
from graphix.fundamentals import Plane
from graphix.measurements import PauliMeasurement

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
    from collections.abc import Set as AbstractSet
    from pathlib import Path
    from typing import TypeAlias, TypeVar

    import numpy.typing as npt

    # MEMO: Potential circular import
    from graphix.clifford import Clifford
    from graphix.parameter import ExpressionOrFloat
    from graphix.pattern import Pattern

    _Edge: TypeAlias = tuple[int, int]
    _Point: TypeAlias = tuple[float, float]

    _HashableT = TypeVar("_HashableT", bound=Hashable)  # reusable node type variable


class GraphVisualizer:
    """A class for visualizing MBQC graphs with flow or gflow structure.

    Attributes
    ----------
    g : :class:`networkx.Graph`
        The graph to be visualized
    v_in : list
        list of input nodes
    v_out : list
        list of output nodes
    meas_planes : dict
        dict specifying the measurement planes for each node, except output nodes.
    meas_angles : dict
        dict specifying the measurement angles for each node, except output nodes.
    local_clifford : dict
        dict specifying the local clifford for each node.

    """

    def __init__(
        self,
        g: nx.Graph[int],
        v_in: Collection[int],
        v_out: Collection[int],
        meas_plane: Mapping[int, Plane] | None = None,
        meas_angles: Mapping[int, ExpressionOrFloat] | None = None,
        local_clifford: Mapping[int, Clifford] | None = None,
    ):
        """
        Construct a graph visualizer.

        Parameters
        ----------
        g : :class:`networkx.Graph`
            NetworkX graph instance
        v_in : list
            list of input nodes
        v_out : list
            list of output nodes
        meas_plane : dict
            dict specifying the measurement planes for each node, except output nodes.
            if None, all measurements are assumed to be in XY-plane.
        meas_angles : dict
            dict specifying the measurement angles for each node, except output nodes.
        local_clifford : dict
            dict specifying the local clifford for each node.
        """
        self.graph = g
        self.v_in = v_in
        self.v_out = v_out
        if meas_plane is None:
            self.meas_planes = dict.fromkeys(g.nodes - set(v_out), Plane.XY)
        else:
            self.meas_planes = dict(meas_plane)
        self.meas_angles = meas_angles
        self.local_clifford = local_clifford

    def visualize(
        self,
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurement_planes: bool = False,
        show_loop: bool = True,
        node_distance: tuple[float, float] = (1, 1),
        figsize: tuple[int, int] | None = None,
        filename: Path | None = None,
    ) -> None:
        """
        Visualize the graph with flow or gflow structure.

        If there exists a flow structure, then the graph is visualized with the flow structure.
        If flow structure is not found and there exists a gflow structure, then the graph is visualized
        with the gflow structure.
        If neither flow nor gflow structure is found, then the graph is visualized without any structure.

        Parameters
        ----------
        show_pauli_measurement : bool
            If True, the nodes with Pauli measurement angles are colored light blue.
        show_local_clifford : bool
            If True, indexes of the local Clifford operator are displayed adjacent to the nodes.
        show_measurement_planes : bool
            If True, the measurement planes are displayed adjacent to the nodes.
        show_loop : bool
            whether or not to show loops for graphs with gflow. defaulted to True.
        node_distance : tuple
            Distance multiplication factor between nodes for x and y directions.
        figsize : tuple
            Figure size of the plot.
        filename : Path | None
            If not None, filename of the png file to save the plot. If None, the plot is not saved.
            Default in None.
        """
        f, l_k = gflow.find_flow(self.graph, set(self.v_in), set(self.v_out), meas_planes=self.meas_planes)  # try flow
        if f is not None and l_k is not None:
            print("Flow detected in the graph.")
            pos = self.get_pos_from_flow(f, l_k)

            def get_paths(
                pos: Mapping[int, _Point],
            ) -> tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]:
                return self.get_edge_path(f, pos)
        else:
            g, l_k = gflow.find_gflow(self.graph, set(self.v_in), set(self.v_out), self.meas_planes)  # try gflow
            if g is not None and l_k is not None:
                print("Gflow detected in the graph. (flow not detected)")
                pos = self.get_pos_from_gflow(g, l_k)

                def get_paths(
                    pos: Mapping[int, _Point],
                ) -> tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]:
                    return self.get_edge_path(g, pos)
            else:
                print("No flow or gflow detected in the graph.")
                pos = self.get_pos_wo_structure()

                def get_paths(
                    pos: Mapping[int, _Point],
                ) -> tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]:
                    return (self.get_edge_path_wo_structure(pos), None)

        self.visualize_graph(
            pos,
            get_paths,
            l_k,
            None,
            show_pauli_measurement,
            show_local_clifford,
            show_measurement_planes,
            show_loop,
            node_distance,
            figsize,
            filename,
        )

    def visualize_from_pattern(
        self,
        pattern: Pattern,
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurement_planes: bool = False,
        show_loop: bool = True,
        node_distance: tuple[float, float] = (1, 1),
        figsize: tuple[int, int] | None = None,
        filename: Path | None = None,
    ) -> None:
        """
        Visualize the graph with flow or gflow structure found from the given pattern.

        If pattern sequence is consistent with flow structure, then the graph is visualized with the flow structure.
        If it is not consistent with flow structure and consistent with gflow structure, then the graph is visualized
        with the gflow structure. If neither flow nor gflow structure is found, then the graph is visualized with all correction flows.

        Parameters
        ----------
        pattern : Pattern
            pattern to be visualized
        show_pauli_measurement : bool
            If True, the nodes with Pauli measurement angles are colored light blue.
        show_local_clifford : bool
            If True, indexes of the local Clifford operator are displayed adjacent to the nodes.
        show_measurement_planes : bool
            If True, the measurement planes are displayed adjacent to the nodes.
        show_loop : bool
            whether or not to show loops for graphs with gflow. defaulted to True.
        node_distance : tuple
            Distance multiplication factor between nodes for x and y directions.
        figsize : tuple
            Figure size of the plot.
        filename : Path | None
            If not None, filename of the png file to save the plot. If None, the plot is not saved.
            Default in None.
        """
        f, l_k = gflow.flow_from_pattern(pattern)  # try flow
        if f is not None and l_k is not None:
            print("The pattern is consistent with flow structure.")
            pos = self.get_pos_from_flow(f, l_k)

            def get_paths(
                pos: Mapping[int, _Point],
            ) -> tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]:
                return self.get_edge_path(f, pos)

            corrections: tuple[Mapping[int, AbstractSet[int]], Mapping[int, AbstractSet[int]]] | None = None
        else:
            g, l_k = gflow.gflow_from_pattern(pattern)  # try gflow
            if g is not None and l_k is not None:
                print("The pattern is consistent with gflow structure. (not with flow)")
                pos = self.get_pos_from_gflow(g, l_k)

                def get_paths(
                    pos: Mapping[int, _Point],
                ) -> tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]:
                    return self.get_edge_path(g, pos)

                corrections = None
            else:
                print("The pattern is not consistent with flow or gflow structure.")
                depth, layers = pattern.get_layers()
                unfolded_layers = {element: key for key, value_set in layers.items() for element in value_set}
                for output in pattern.output_nodes:
                    unfolded_layers[output] = depth + 1
                xflow, zflow = gflow.get_corrections_from_pattern(pattern)
                xzflow: dict[int, set[int]] = deepcopy(xflow)
                for key, value in zflow.items():
                    if key in xzflow:
                        xzflow[key] |= value
                    else:
                        xzflow[key] = set(value)  # copy
                pos = self.get_pos_all_correction(unfolded_layers)

                def get_paths(
                    pos: Mapping[int, _Point],
                ) -> tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]:
                    return self.get_edge_path(xzflow, pos)

                corrections = xflow, zflow
        self.visualize_graph(
            pos,
            get_paths,
            l_k,
            corrections,
            show_pauli_measurement,
            show_local_clifford,
            show_measurement_planes,
            show_loop,
            node_distance,
            figsize,
            filename,
        )

    @staticmethod
    def _shorten_path(path: Sequence[_Point]) -> list[_Point]:
        """Shorten the last edge not to hide arrow under the node."""
        new_path = list(path)
        last = np.array(new_path[-1])
        second_last = np.array(new_path[-2])
        last_edge: _Point = tuple(last - (last - second_last) / np.linalg.norm(last - second_last) * 0.2)
        new_path[-1] = last_edge
        return new_path

    def _draw_labels(self, pos: Mapping[int, _Point]) -> None:
        fontsize = 12
        if max(self.graph.nodes(), default=0) >= 100:
            fontsize = int(fontsize * 2 / len(str(max(self.graph.nodes()))))
        nx.draw_networkx_labels(self.graph, pos, font_size=fontsize)

    def __draw_nodes_role(self, pos: Mapping[int, _Point], show_pauli_measurement: bool = False) -> None:
        """
        Draw the nodes with different colors based on their role (input, output, or other).

        Parameters
        ----------
        pos : Mapping[int, tuple[float, float]]
            dictionary of node positions.
        show_pauli_measurement : bool
            If True, the nodes with Pauli measurement angles are colored light blue.
        """
        for node in self.graph.nodes():
            color = "black"  # default color for 'other' nodes
            inner_color = "white"
            if node in self.v_in:
                color = "red"
            if node in self.v_out:
                inner_color = "lightgray"
            elif (
                show_pauli_measurement
                and self.meas_angles is not None
                and PauliMeasurement.try_from(Plane.XY, self.meas_angles[node]) is not None
            ):
                # Pauli nodes are checked with Plane.XY by default,
                # because the actual plane does not change whether the
                # node is Pauli or not, and the current API allows
                # self.meas_plane to be None while self.meas_angles is
                # defined.
                inner_color = "lightblue"
            plt.scatter(
                *pos[node], edgecolor=color, facecolor=inner_color, s=350, zorder=2
            )  # Draw the nodes manually with scatter()

    def visualize_graph(
        self,
        pos: Mapping[int, _Point],
        get_paths: Callable[
            [Mapping[int, _Point]], tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]
        ],
        l_k: Mapping[int, int] | None,
        corrections: tuple[Mapping[int, AbstractSet[int]], Mapping[int, AbstractSet[int]]] | None,
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurement_planes: bool = False,
        show_loop: bool = True,
        node_distance: tuple[float, float] = (1, 1),
        figsize: _Point | None = None,
        filename: Path | None = None,
    ) -> None:
        """
        Visualizes the graph.

        Nodes are colored based on their role (input, output, or other) and edges are depicted as arrows
        or dashed lines depending on whether they are in the flow mapping. Vertical dashed lines separate
        different layers of the graph. This function does not return anything but plots the graph
        using matplotlib's pyplot.

        Parameters
        ----------
        pos: Mapping[int, _Point]
            Node positions.
        get_paths: Callable[
            [Mapping[int, _Point]], tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]
        ]
            Given scaled node positions, return the mapping of edge paths and the mapping of arrow paths.
        l_k: Mapping[int, int] | None
            Layer mapping if any.
        corrections: tuple[Mapping[int, AbstractSet[int]], Mapping[int, AbstractSet[int]]] | None
            X and Z corrections if any.
        show_pauli_measurement : bool
            If True, the nodes with Pauli measurement angles are colored light blue.
        show_local_clifford : bool
            If True, indexes of the local Clifford operator are displayed adjacent to the nodes.
        show_measurement_planes : bool
            If True, the measurement planes are displayed adjacent to the nodes.
        show_loop : bool
            whether or not to show loops for graphs with gflow. defaulted to True.
        node_distance : tuple
            Distance multiplication factor between nodes for x and y directions.
        figsize : tuple
            Figure size of the plot.
        filename : Path | None
            If not None, filename of the png file to save the plot. If None, the plot is not saved.
            Default in None.
        """
        if figsize is None:
            figsize = self.get_figsize(l_k, pos, node_distance=node_distance)

        pos = {k: (v[0] * node_distance[0], v[1] * node_distance[1]) for k, v in pos.items()}

        edge_path, arrow_path = get_paths(pos)

        if corrections is not None:
            # add some padding to the right for the legend
            figsize = (figsize[0] + 3.0, figsize[1])

        plt.figure(figsize=figsize)

        for edge, path in edge_path.items():
            if len(path) == 2:
                nx.draw_networkx_edges(self.graph, pos, edgelist=[edge], style="dashed", alpha=0.7)
            else:
                curve = self._bezier_curve_linspace(path)
                plt.plot(curve[:, 0], curve[:, 1], "k--", linewidth=1, alpha=0.7)

        if arrow_path is not None:
            for arrow, path in arrow_path.items():
                if corrections is None:
                    color = "k"
                else:
                    xflow, zflow = corrections
                    if arrow[1] not in xflow.get(arrow[0], set()):
                        color = "tab:green"
                    elif arrow[1] not in zflow.get(arrow[0], set()):
                        color = "tab:red"
                    else:
                        color = "tab:brown"
                if arrow[0] == arrow[1]:  # self loop
                    if show_loop:
                        curve = self._bezier_curve_linspace(path)
                        plt.plot(curve[:, 0], curve[:, 1], c="k", linewidth=1)
                        plt.annotate(
                            "",
                            xy=curve[-1],
                            xytext=curve[-2],
                            arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
                        )
                elif len(path) == 2:  # straight line
                    nx.draw_networkx_edges(
                        self.graph, pos, edgelist=[arrow], edge_color=color, arrowstyle="->", arrows=True
                    )
                else:
                    new_path = GraphVisualizer._shorten_path(path)
                    curve = self._bezier_curve_linspace(new_path)
                    plt.plot(curve[:, 0], curve[:, 1], c=color, linewidth=1)
                    plt.annotate(
                        "",
                        xy=curve[-1],
                        xytext=curve[-2],
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
                    )

        self.__draw_nodes_role(pos, show_pauli_measurement)

        if show_local_clifford:
            self.__draw_local_clifford(pos)

        if show_measurement_planes:
            self.__draw_measurement_planes(pos)

        self._draw_labels(pos)

        if corrections is not None:
            # legend for arrow colors
            plt.plot([], [], "k--", alpha=0.7, label="graph edge")
            plt.plot([], [], color="tab:red", label="xflow")
            plt.plot([], [], color="tab:green", label="zflow")
            plt.plot([], [], color="tab:brown", label="xflow and zflow")
            plt.legend(loc="upper right", fontsize=10)

        x_min = min((pos[node][0] for node in self.graph.nodes()), default=0)  # Get the minimum x coordinate
        x_max = max((pos[node][0] for node in self.graph.nodes()), default=0)  # Get the maximum x coordinate
        y_min = min((pos[node][1] for node in self.graph.nodes()), default=0)  # Get the minimum y coordinate
        y_max = max((pos[node][1] for node in self.graph.nodes()), default=0)  # Get the maximum y coordinate

        if l_k is not None and l_k:
            # Draw the vertical lines to separate different layers
            for layer in range(min(l_k.values()), max(l_k.values())):
                plt.axvline(
                    x=(layer + 0.5) * node_distance[0], color="gray", linestyle="--", alpha=0.5
                )  # Draw line between layers
            for layer in range(min(l_k.values()), max(l_k.values()) + 1):
                plt.text(
                    layer * node_distance[0], y_min - 0.5, f"L: {max(l_k.values()) - layer}", ha="center", va="top"
                )  # Add layer label at bottom

        plt.xlim(
            x_min - 0.5 * node_distance[0], x_max + 0.5 * node_distance[0]
        )  # Add some padding to the left and right
        plt.ylim(y_min - 1, y_max + 0.5)  # Add some padding to the top and bottom

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def __draw_local_clifford(self, pos: Mapping[int, _Point]) -> None:
        if self.local_clifford is not None:
            for node in self.local_clifford:
                x, y = pos[node] + np.array([0.2, 0.2])
                plt.text(x, y, f"{self.local_clifford[node]}", fontsize=10, zorder=3)

    def __draw_measurement_planes(self, pos: Mapping[int, _Point]) -> None:
        for node in self.meas_planes:
            x, y = pos[node] + np.array([0.22, -0.2])
            plt.text(x, y, f"{self.meas_planes[node].name}", fontsize=9, zorder=3)

    def get_figsize(
        self,
        l_k: Mapping[int, int] | None,
        pos: Mapping[int, _Point] | None = None,
        node_distance: tuple[float, float] = (1, 1),
    ) -> _Point:
        """
        Return the figure size of the graph.

        Parameters
        ----------
        l_k : dict
            Layer mapping.
        pos : dict
            dictionary of node positions.
        node_distance : tuple
            Distance multiplication factor between nodes for x and y directions.

        Returns
        -------
        figsize : tuple
            figure size of the graph.
        """
        if l_k is None:
            if pos is None:
                raise ValueError("Figure size can only be computed given a layer mapping (l_k) or node positions (pos)")
            width = len({pos[node][0] for node in self.graph.nodes()}) * 0.8
        else:
            width = (max(l_k.values(), default=0) + 1) * 0.8
        height = len({pos[node][1] for node in self.graph.nodes()}) if pos is not None else len(self.v_out)
        return (width * node_distance[0], height * node_distance[1])

    def get_edge_path(
        self, flow: Mapping[int, int | set[int]], pos: Mapping[int, _Point]
    ) -> tuple[dict[_Edge, list[_Point]], dict[_Edge, list[_Point]]]:
        """
        Return the path of edges and gflow arrows.

        Parameters
        ----------
        flow : dict
            flow mapping (including gflow or any correction flow)
        pos : dict
            dictionary of node positions.

        Returns
        -------
        edge_path : dict
            dictionary of edge paths.
        arrow_path : dict
            dictionary of arrow paths.
        """
        edge_path = self.get_edge_path_wo_structure(pos)
        edge_set = set(self.graph.edges())
        arrow_path: dict[_Edge, list[_Point]] = {}
        flow_arrows = {(k, v) for k, values in flow.items() for v in ((values,) if isinstance(values, int) else values)}

        for arrow in flow_arrows:
            if arrow[0] == arrow[1]:  # Self loop

                def _point_from_node(pos: Sequence[float], dist: float, angle: float) -> _Point:
                    """Return a point at a given distance and angle from ``pos``.

                    Parameters
                    ----------
                    pos : Sequence[float]
                        Coordinate of the node.
                    dist : float
                        Distance from ``pos``.
                    angle : float
                        Angle in degrees measured counter-clockwise from the
                        positive x-axis.

                    Returns
                    -------
                    _Point
                        The new ``[x, y]`` coordinate.
                    """
                    angle = np.deg2rad(angle)
                    return (pos[0] + dist * np.cos(angle), pos[1] + dist * np.sin(angle))

                bezier_path = [
                    _point_from_node(pos[arrow[0]], 0.2, 170),
                    _point_from_node(pos[arrow[0]], 0.35, 170),
                    _point_from_node(pos[arrow[0]], 0.4, 155),
                    _point_from_node(pos[arrow[0]], 0.45, 140),
                    _point_from_node(pos[arrow[0]], 0.35, 110),
                    _point_from_node(pos[arrow[0]], 0.3, 110),
                    _point_from_node(pos[arrow[0]], 0.17, 95),
                ]
            else:
                bezier_path = [pos[arrow[0]], pos[arrow[1]]]
                if arrow in edge_set or (arrow[1], arrow[0]) in edge_set:
                    mid_point = (
                        0.5 * (pos[arrow[0]][0] + pos[arrow[1]][0]),
                        0.5 * (pos[arrow[0]][1] + pos[arrow[1]][1]),
                    )
                    if self._edge_intersects_node(pos[arrow[0]], pos[arrow[1]], mid_point, buffer=0.05):
                        ctrl_point = self._control_point(pos[arrow[0]], pos[arrow[1]], mid_point, distance=0.2)
                        bezier_path.insert(1, ctrl_point)
                bezier_path = self._find_bezier_path(arrow, bezier_path, pos)

            arrow_path[arrow] = bezier_path

        return edge_path, arrow_path

    def _find_bezier_path(self, arrow: _Edge, bezier_path: Iterable[_Point], pos: Mapping[int, _Point]) -> list[_Point]:
        bezier_path = list(bezier_path)
        max_iter = 5
        iteration = 0
        nodes = set(self.graph.nodes())
        while True:
            iteration += 1
            intersect = False
            if iteration > max_iter:
                break
            ctrl_points: list[tuple[int, _Point]] = []
            for i in range(len(bezier_path) - 1):
                start = bezier_path[i]
                end = bezier_path[i + 1]
                for node in set(nodes):
                    if node != arrow[0] and node != arrow[1] and self._edge_intersects_node(start, end, pos[node]):
                        intersect = True
                        ctrl_points.append(
                            (
                                i,
                                self._control_point(
                                    bezier_path[0], bezier_path[-1], pos[node], distance=0.6 / iteration
                                ),
                            )
                        )
                        nodes -= {node}
            if not intersect:
                break
            for i, (index, ctrl_point) in enumerate(ctrl_points):
                bezier_path.insert(index + i + 1, ctrl_point)
        return self._check_path(bezier_path, pos[arrow[1]])

    def get_edge_path_wo_structure(self, pos: Mapping[int, _Point]) -> dict[_Edge, list[_Point]]:
        """
        Return the path of edges.

        Parameters
        ----------
        pos : dict
            dictionary of node positions.

        Returns
        -------
        edge_path : dict
            dictionary of edge paths.
        """
        return {edge: self._find_bezier_path(edge, [pos[edge[0]], pos[edge[1]]], pos) for edge in self.graph.edges()}

    def get_pos_from_flow(self, f: Mapping[int, set[int]], l_k: Mapping[int, int]) -> dict[int, _Point]:
        """
        Return the position of nodes based on the flow.

        Parameters
        ----------
        f : dict
            flow mapping.
        l_k : dict
            Layer mapping.

        Returns
        -------
        pos : dict
            dictionary of node positions.
        """
        values_union = set().union(*f.values())
        start_nodes = set(self.graph.nodes()) - values_union
        pos = {node: [0, 0] for node in self.graph.nodes()}
        for i, k in enumerate(start_nodes):
            pos[k][1] = i
            node = k
            while node in f:
                node = next(iter(f[node]))
                pos[node][1] = i

        if not l_k:
            return {}

        lmax = max(l_k.values())
        # Change the x coordinates of the nodes based on their layer, sort in descending order
        for node, layer in l_k.items():
            pos[node][0] = lmax - layer
        return {k: (x, y) for k, (x, y) in pos.items()}

    def get_pos_from_gflow(self, g: Mapping[int, set[int]], l_k: Mapping[int, int]) -> dict[int, _Point]:
        """
        Return the position of nodes based on the gflow.

        Parameters
        ----------
        g : dict
            gflow mapping.
        l_k : dict
            Layer mapping.

        Returns
        -------
        pos : dict
            dictionary of node positions.
        """
        g_edges: list[_Edge] = []

        for node, node_list in g.items():
            g_edges.extend((node, n) for n in node_list)

        g_prime = self.graph.copy()
        g_prime.add_nodes_from(self.graph.nodes())
        g_prime.add_edges_from(g_edges)

        l_max = max(l_k.values())
        l_reverse = {v: l_max - l for v, l in l_k.items()}

        _set_node_attributes(g_prime, l_reverse, "subset")

        reveal_type(nx.multipartite_layout)
        pos = nx.multipartite_layout(g_prime)

        for node, layer in l_k.items():
            pos[node][0] = l_max - layer

        vert = list({pos[node][1] for node in self.graph.nodes()})
        vert.sort()
        for node in self.graph.nodes():
            pos[node][1] = vert.index(pos[node][1])

        return pos

    def get_pos_wo_structure(self) -> dict[int, _Point]:
        """
        Return the position of nodes based on the graph.

        Returns
        -------
        pos : dict
            dictionary of node positions.

        Returns
        -------
        pos : dict
            dictionary of node positions.
        """
        layers: dict[int, int] = {}
        connected_components = list(nx.connected_components(self.graph))

        for component in connected_components:
            subgraph = self.graph.subgraph(component)
            initial_pos: dict[int, tuple[int, int]] = dict.fromkeys(component, (0, 0))

            if len(set(self.v_out) & set(component)) == 0 and len(set(self.v_in) & set(component)) == 0:
                pos = nx.spring_layout(subgraph)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                layers.update((node, k) for k, node in enumerate(order[::-1]))

            elif len(set(self.v_out) & set(component)) > 0 and len(set(self.v_in) & set(component)) == 0:
                fixed_nodes = list(set(self.v_out) & set(component))
                for i, node in enumerate(fixed_nodes):
                    initial_pos[node] = (10, i)
                    layers[node] = 0
                pos = nx.spring_layout(subgraph, pos=initial_pos, fixed=fixed_nodes)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                order = [node for node in order if node not in fixed_nodes]
                nv = len(self.v_out)
                for i, node in enumerate(order[::-1]):
                    k = i // nv + 1
                    layers[node] = k

            elif len(set(self.v_out) & set(component)) == 0 and len(set(self.v_in) & set(component)) > 0:
                fixed_nodes = list(set(self.v_in) & set(component))
                for i, node in enumerate(fixed_nodes):
                    initial_pos[node] = (-10, i)
                pos = nx.spring_layout(subgraph, pos=initial_pos, fixed=fixed_nodes)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                order = [node for node in order if node not in fixed_nodes]
                nv = len(self.v_in)
                for i, node in enumerate(order[::-1]):
                    k = i // nv
                    layers[node] = k
                layer_input = 0 if layers == {} else max(layers.values()) + 1
                for node in fixed_nodes:
                    layers[node] = layer_input

            else:
                for i, node in enumerate(list(set(self.v_out) & set(component))):
                    initial_pos[node] = (10, i)
                    layers[node] = 0
                for i, node in enumerate(list(set(self.v_in) & set(component))):
                    initial_pos[node] = (-10, i)
                fixed_nodes = list(set(self.v_out) & set(component)) + list(set(self.v_in) & set(component))
                pos = nx.spring_layout(subgraph, pos=initial_pos, fixed=fixed_nodes)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                order = [node for node in order if node not in fixed_nodes]
                nv = len(self.v_out)
                for i, node in enumerate(order[::-1]):
                    k = i // nv + 1
                    layers[node] = k
                layer_input = max(layers.values()) + 1
                for node in set(self.v_in) & set(component) - set(self.v_out):
                    layers[node] = layer_input

        g_prime = self.graph.copy()
        g_prime.add_nodes_from(self.graph.nodes())
        g_prime.add_edges_from(self.graph.edges())
        l_max = max(layers.values())
        l_reverse = {v: l_max - l for v, l in layers.items()}
        _set_node_attributes(g_prime, l_reverse, "subset")
        pos = nx.multipartite_layout(g_prime)
        for node, layer in layers.items():
            pos[node][0] = l_max - layer
        vert = list({pos[node][1] for node in self.graph.nodes()})
        vert.sort()
        for node in self.graph.nodes():
            pos[node][1] = vert.index(pos[node][1])
        return pos

    def get_pos_all_correction(self, layers: Mapping[int, int]) -> dict[int, _Point]:
        """
        Return the position of nodes based on the pattern.

        Parameters
        ----------
        layers : dict
            Layer mapping obtained from the measurement order of the pattern.

        Returns
        -------
        pos : dict
            dictionary of node positions.
        """
        g_prime = self.graph.copy()
        g_prime.add_nodes_from(self.graph.nodes())
        g_prime.add_edges_from(self.graph.edges())
        _set_node_attributes(g_prime, layers, "subset")
        layout = nx.multipartite_layout(g_prime)
        vert = list({layout[node][1] for node in self.graph.nodes()})
        vert.sort()
        return {node: (layers[node], vert.index(layout[node][1])) for node in self.graph.nodes()}

    @staticmethod
    def _edge_intersects_node(
        start: _Point,
        end: _Point,
        node_pos: _Point,
        buffer: float = 0.2,
    ) -> bool:
        """Determine if an edge intersects a node."""
        start_array = np.array(start)
        end_array = np.array(end)
        if np.all(start_array == end_array):
            return False
        node_pos_array = np.array(node_pos)
        # Vector from start to end
        line_vec = end_array - start_array
        # Vector from start to node_pos
        point_vec = node_pos_array - start_array
        t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)

        if t < 0.0 or t > 1.0:
            return False
        # Find the projection point
        projection = start_array + t * line_vec
        distance = np.linalg.norm(projection - node_pos)

        return bool(distance < buffer)

    @staticmethod
    def _control_point(
        start: _Point,
        end: _Point,
        node_pos: _Point,
        distance: float = 0.6,
    ) -> _Point:
        """Generate a control point to bend the edge around a node."""
        node_pos_array = np.array(node_pos)
        edge_vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        # Rotate the edge vector 90 degrees or -90 degrees according to the node position
        cross = np.cross(edge_vector, node_pos_array - np.array(start))
        if cross > 0:
            dir_vector = np.array([edge_vector[1], -edge_vector[0]])  # Rotate the edge vector 90 degrees
        else:
            dir_vector = np.array([-edge_vector[1], edge_vector[0]])
        dir_vector /= np.linalg.norm(dir_vector)  # Normalize the vector
        u, v = node_pos_array + distance * dir_vector
        return u, v

    @staticmethod
    def _bezier_curve(bezier_path: Sequence[_Point], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Generate a bezier curve from a list of points."""
        n = len(bezier_path) - 1  # order of the curve
        curve = np.zeros((len(t), 2))
        for i, point in enumerate(bezier_path):
            curve += np.outer(math.comb(n, i) * ((1 - t) ** (n - i)) * (t**i), np.array(point))
        return curve

    @staticmethod
    def _bezier_curve_linspace(bezier_path: Sequence[_Point]) -> npt.NDArray[np.float64]:
        t = np.linspace(0, 1, 100, dtype=np.float64)
        return GraphVisualizer._bezier_curve(bezier_path, t)

    @staticmethod
    def _check_path(path: Iterable[_Point], target_node_pos: _Point | None = None) -> list[_Point]:
        """If there is an acute angle in the path, merge points."""
        path = np.array(path)
        acute = True
        max_iter = 100
        it = 0
        while acute:
            if it > max_iter:
                break
            for i in range(len(path) - 2):
                v1 = path[i + 1] - path[i]
                v2 = path[i + 2] - path[i + 1]
                if (v1 == 0).all() or (v2 == 0).all():
                    path = np.delete(path, i + 1, 0)
                    break
                if np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) < np.cos(3 * np.pi / 4):
                    if i == len(path) - 3:
                        path = np.delete(path, i + 1, 0)
                        break
                    mean = (path[i + 1] + path[i + 2]) / 2
                    path = np.delete(path, i + 1, 0)
                    path = np.delete(path, i + 1, 0)
                    path = np.insert(path, i + 1, mean, 0)
                    break
                it += 1
            else:
                acute = False
        new_path: list[_Point] = path.tolist()
        if target_node_pos is not None:
            for point in new_path[:-1]:
                if np.linalg.norm(np.array(point) - np.array(target_node_pos)) < 0.2:
                    new_path.remove(point)
        return new_path


def _set_node_attributes(graph: nx.Graph[_HashableT], attrs: Mapping[_HashableT, object], name: str) -> None:
    nx.set_node_attributes(graph, attrs, name=name)  # type: ignore[arg-type]
