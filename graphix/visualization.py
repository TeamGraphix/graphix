"""Functions to visualize the resource state of MBQC pattern."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from graphix.flow.exceptions import FlowError
from graphix.measurements import BlochMeasurement, Measurement, PauliMeasurement

# OpenGraph is needed for dataclass
from graphix.opengraph import OpenGraph  # noqa: TC001
from graphix.optimization import StandardizedPattern
from graphix.pretty_print import OutputFormat, angle_to_str

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
    from collections.abc import Set as AbstractSet
    from pathlib import Path
    from typing import TypeAlias, TypeVar

    import numpy.typing as npt

    from graphix.clifford import Clifford
    from graphix.flow.core import CausalFlow, PauliFlow
    from graphix.fundamentals import AbstractMeasurement, AbstractPlanarMeasurement
    from graphix.pattern import Pattern

    _Edge: TypeAlias = tuple[int, int]
    _Point: TypeAlias = tuple[float, float]

    _HashableT = TypeVar("_HashableT", bound=Hashable)  # reusable node type variable


@dataclass(frozen=True)
class GraphVisualizer:
    """A class for visualizing MBQC graphs with flow or gflow structure.

    Attributes
    ----------
    og: OpenGraph
        The open graph to be visualized
    local_clifford : dict
        dict specifying the local clifford for each node.

    """

    og: OpenGraph[Measurement]
    local_clifford: Mapping[int, Clifford] | None = None

    def visualize(
        self,
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurements: bool = False,
        show_legend: bool = False,
        show_loop: bool = True,
        node_distance: tuple[float, float] = (1, 1),
        figsize: tuple[int, int] | None = None,
        filename: Path | None = None,
    ) -> None:
        """Visualize the graph with flow or gflow structure.

        If there exists a flow structure, then the graph is visualized with the flow structure.
        If flow structure is not found and there exists a gflow structure, then the graph is visualized
        with the gflow structure.
        If neither flow nor gflow structure is found, then the graph is visualized without any structure.

        Parameters
        ----------
        show_pauli_measurement : bool
            If True, Pauli-measured nodes are filled with blue instead of black.
        show_local_clifford : bool
            If True, indexes of the local Clifford operator are displayed adjacent to the nodes.
        show_measurements : bool
            If True, measurement labels are displayed adjacent to the nodes.
        show_legend : bool
            If True, a legend is displayed indicating node types and edge meanings.
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
        try:
            bloch_graph = self.og.downcast_bloch()
        except TypeError:
            bloch_graph = None
        causal_flow = None if bloch_graph is None else bloch_graph.find_causal_flow()
        if causal_flow is not None:
            print("Flow detected in the graph.")
            pos = self.place_causal_flow(causal_flow)
            cf = causal_flow.correction_function
            l_k = {
                node: layer_idx for layer_idx, layer in enumerate(causal_flow.partial_order_layers) for node in layer
            }

            def place_paths(
                pos: Mapping[int, _Point],
            ) -> tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]:
                return self.place_edge_paths(cf, pos)

        else:
            pauli_flow = self.og.find_pauli_flow()
            if pauli_flow is not None:
                print("Pauli flow detected in the graph (causal flow not detected)")
                pos = self.place_pauli_flow(pauli_flow)
                cf = pauli_flow.correction_function
                l_k = {
                    node: layer_idx for layer_idx, layer in enumerate(pauli_flow.partial_order_layers) for node in layer
                }

                def place_paths(
                    pos: Mapping[int, _Point],
                ) -> tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]:
                    return self.place_edge_paths(cf, pos)

            else:
                print("No causal flow, gflow, or Pauli flow detected in the graph.")
                pos = self.place_without_structure()
                l_k = None

                def place_paths(
                    pos: Mapping[int, _Point],
                ) -> tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]:
                    return (self.place_edge_paths_without_structure(pos), None)

        self.visualize_graph(
            pos,
            place_paths,
            l_k,
            None,
            show_pauli_measurement,
            show_local_clifford,
            show_measurements,
            show_legend,
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
        show_measurements: bool = False,
        show_legend: bool = False,
        show_loop: bool = True,
        node_distance: tuple[float, float] = (1, 1),
        figsize: tuple[int, int] | None = None,
        filename: Path | None = None,
    ) -> None:
        """Visualize the graph with flow or gflow structure found from the given pattern.

        If pattern sequence is consistent with flow structure, then the graph is visualized with the flow structure.
        If it is not consistent with flow structure and consistent with gflow structure, then the graph is visualized
        with the gflow structure. If neither flow nor gflow structure is found, then the graph is visualized with all correction flows.

        Parameters
        ----------
        pattern : Pattern
            pattern to be visualized
        show_pauli_measurement : bool
            If True, Pauli-measured nodes are filled with blue instead of black.
        show_local_clifford : bool
            If True, indexes of the local Clifford operator are displayed adjacent to the nodes.
        show_measurements : bool
            If True, measurement labels are displayed adjacent to the nodes.
        show_legend : bool
            If True, a legend is displayed indicating node types and edge meanings.
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
        pattern_std = StandardizedPattern.from_pattern(pattern)
        cf: Mapping[int, AbstractSet[int]]
        corrections: tuple[Mapping[int, AbstractSet[int]], Mapping[int, AbstractSet[int]]] | None

        try:
            causal_flow = pattern_std.extract_causal_flow()
        except FlowError:
            try:
                g_flow = pattern_std.extract_gflow()
            except (FlowError, TypeError):
                print("The pattern is not consistent with flow or gflow structure.")
                po_layers = pattern.extract_partial_order_layers()
                unfolded_layers = {node: layer_idx for layer_idx, layer in enumerate(po_layers[::-1]) for node in layer}
                xzc = pattern.extract_xzcorrections()
                xflow, zflow = xzc.x_corrections, xzc.z_corrections
                xzflow = dict(xflow)
                for key, value in zflow.items():
                    if key in xzflow:
                        xzflow[key] |= value
                    else:
                        xzflow[key] = set(value)  # copy
                pos = self.place_all_corrections(unfolded_layers)
                cf = xzflow
                l_k = None
                corrections = xflow, zflow
            else:
                print("The pattern is consistent with gflow structure. (not with flow)")
                pos = self.place_pauli_flow(g_flow)
                cf = g_flow.correction_function
                l_k = {node: layer_idx for layer_idx, layer in enumerate(g_flow.partial_order_layers) for node in layer}
                corrections = None
        else:
            print("The pattern is consistent with causal flow structure.")
            pos = self.place_causal_flow(causal_flow)
            cf = causal_flow.correction_function
            l_k = {
                node: layer_idx for layer_idx, layer in enumerate(causal_flow.partial_order_layers) for node in layer
            }
            corrections = None

        def place_paths(
            pos: Mapping[int, _Point],
        ) -> tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]:
            return self.place_edge_paths(cf, pos)

        self.visualize_graph(
            pos,
            place_paths,
            l_k,
            corrections,
            show_pauli_measurement,
            show_local_clifford,
            show_measurements,
            show_legend,
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

    def _draw_labels(self, pos: Mapping[int, _Point], font_color: Mapping[int, str] | str = "black") -> None:
        """Draw node number labels with appropriate text color.

        Parameters
        ----------
        pos : Mapping[int, tuple[float, float]]
            Dictionary of node positions.
        font_color : Mapping[int, str] | str
            Font color for node labels. Can be a single color string or a mapping from node to color.
        """
        fontsize = 12
        if max(self.og.graph.nodes(), default=0) >= 100:
            fontsize = int(fontsize * 2 / len(str(max(self.og.graph.nodes()))))
        nx.draw_networkx_labels(self.og.graph, pos, font_size=fontsize, font_color=font_color)  # type: ignore[arg-type]

    def __draw_nodes_role(self, pos: Mapping[int, _Point], show_pauli_measurement: bool = False) -> dict[int, str]:
        """Draw the nodes with shapes and fills following MBQC literature conventions.

        Input nodes are drawn as squares, measured (non-output) nodes as filled circles,
        and output nodes as empty circles. Pauli-measured nodes are optionally distinguished
        with a blue fill.

        Parameters
        ----------
        pos : Mapping[int, tuple[float, float]]
            Dictionary of node positions.
        show_pauli_measurement : bool
            If True, Pauli-measured nodes are filled with blue instead of black.

        Returns
        -------
        dict[int, str]
            Mapping from node index to font color for label rendering.
        """
        font_colors: dict[int, str] = {}

        for node in self.og.graph.nodes():
            marker = "s" if node in self.og.input_nodes else "o"
            is_pauli = node in self.og.measurements and isinstance(self.og.measurements[node], PauliMeasurement)

            if node in self.og.output_nodes:
                facecolor = "white"
            elif show_pauli_measurement and is_pauli:
                facecolor = "#4292c6"
            else:
                facecolor = "black"

            font_colors[node] = "white" if facecolor == "black" else "black"

            plt.scatter(
                *pos[node],
                marker=marker,
                edgecolor="black",
                facecolor=facecolor,
                s=350,
                zorder=2,
                linewidths=1.5,
            )

        return font_colors

    def visualize_graph(
        self,
        pos: Mapping[int, _Point],
        place_paths: Callable[
            [Mapping[int, _Point]], tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]
        ],
        l_k: Mapping[int, int] | None,
        corrections: tuple[Mapping[int, AbstractSet[int]], Mapping[int, AbstractSet[int]]] | None,
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurements: bool = False,
        show_legend: bool = False,
        show_loop: bool = True,
        node_distance: tuple[float, float] = (1, 1),
        figsize: _Point | None = None,
        filename: Path | None = None,
    ) -> None:
        """Visualize the graph.

        Nodes are drawn following MBQC literature conventions: inputs as squares,
        measured nodes as filled circles, and outputs as empty circles. Graph edges
        are solid lines and flow arrows indicate corrections. A horizontal arrow
        below the graph indicates the measurement order.

        Parameters
        ----------
        pos: Mapping[int, _Point]
            Node positions.
        place_paths: Callable[
            [Mapping[int, _Point]], tuple[Mapping[_Edge, Sequence[_Point]], Mapping[_Edge, Sequence[_Point]] | None]
        ]
            Given scaled node positions, return the mapping of edge paths and the mapping of arrow paths.
        l_k: Mapping[int, int] | None
            Layer mapping if any.
        corrections: tuple[Mapping[int, AbstractSet[int]], Mapping[int, AbstractSet[int]]] | None
            X and Z corrections if any.
        show_pauli_measurement : bool
            If True, Pauli-measured nodes are filled with blue instead of black.
        show_local_clifford : bool
            If True, indexes of the local Clifford operator are displayed adjacent to the nodes.
        show_measurements : bool
            If True, measurement labels are displayed adjacent to the nodes.
        show_legend : bool
            If True, a legend is displayed indicating node types and edge meanings.
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
            figsize = self.determine_figsize(l_k, pos, node_distance=node_distance)

        pos = {k: (v[0] * node_distance[0], v[1] * node_distance[1]) for k, v in pos.items()}

        edge_path, arrow_path = place_paths(pos)

        if show_legend or corrections is not None:
            # add some padding to the right for the legend
            figsize = (figsize[0] + 3.0, figsize[1])

        plt.figure(figsize=figsize)

        for edge, path in edge_path.items():
            if len(path) == 2:
                nx.draw_networkx_edges(
                    self.og.graph, pos, edgelist=[edge], style="dashed", edge_color="gray", alpha=0.6
                )
            else:
                curve = self._bezier_curve_linspace(path)
                plt.plot(curve[:, 0], curve[:, 1], color="gray", linewidth=1, alpha=0.6, linestyle="dashed")

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
                        self.og.graph, pos, edgelist=[arrow], edge_color=color, arrowstyle="->", arrows=True
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

        font_colors = self.__draw_nodes_role(pos, show_pauli_measurement)

        if show_local_clifford:
            self.__draw_local_clifford(pos)

        if show_measurements:
            self.__draw_measurement_labels(pos)

        self._draw_labels(pos, font_colors)

        if show_legend:
            self.__draw_legend(show_pauli_measurement, corrections, arrow_path is not None)
        elif corrections is not None:
            # backward-compatible minimal legend for correction arrows
            plt.plot([], [], color="gray", alpha=0.6, linestyle="dashed", label="graph edge")
            plt.plot([], [], color="tab:red", label="xflow")
            plt.plot([], [], color="tab:green", label="zflow")
            plt.plot([], [], color="tab:brown", label="xflow and zflow")
            plt.legend(loc="center left", fontsize=10, bbox_to_anchor=(1, 0.5))

        x_min = min((pos[node][0] for node in self.og.graph.nodes()), default=0)  # Get the minimum x coordinate
        x_max = max((pos[node][0] for node in self.og.graph.nodes()), default=0)  # Get the maximum x coordinate
        y_min = min((pos[node][1] for node in self.og.graph.nodes()), default=0)  # Get the minimum y coordinate
        y_max = max((pos[node][1] for node in self.og.graph.nodes()), default=0)  # Get the maximum y coordinate

        has_layers = l_k is not None and len(l_k) > 0
        if has_layers and l_k is not None:
            l_min_val = min(l_k.values())
            l_max_val = max(l_k.values())
            # Draw layer labels below nodes
            for layer in range(l_min_val, l_max_val + 1):
                plt.text(
                    layer * node_distance[0],
                    y_min - 0.4,
                    f"L{l_max_val - layer}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="gray",
                )
            # Draw horizontal arrow indicating measurement order
            if l_max_val > l_min_val:
                arrow_y = y_min - 0.7
                plt.annotate(
                    "",
                    xy=(l_max_val * node_distance[0] + 0.3, arrow_y),
                    xytext=(l_min_val * node_distance[0] - 0.3, arrow_y),
                    arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.2},
                )
                mid_x = (l_min_val + l_max_val) / 2 * node_distance[0]
                plt.text(mid_x, arrow_y - 0.2, "Measurement order", ha="center", va="top", fontsize=8, color="gray")

        plt.xlim(
            x_min - 0.5 * node_distance[0], x_max + 0.5 * node_distance[0]
        )  # Add some padding to the left and right
        bottom_margin = 1.3 if has_layers else 1
        plt.ylim(y_min - bottom_margin, y_max + 0.5)

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches="tight")

    def __draw_local_clifford(self, pos: Mapping[int, _Point]) -> None:
        if self.local_clifford is not None:
            for node in self.local_clifford:
                x, y = pos[node] + np.array([0.2, 0.2])
                plt.text(x, y, f"{self.local_clifford[node]}", fontsize=10, zorder=3)

    @staticmethod
    def __draw_legend(
        show_pauli_measurement: bool,
        corrections: tuple[Mapping[int, AbstractSet[int]], Mapping[int, AbstractSet[int]]] | None,
        has_arrows: bool,
    ) -> None:
        """Draw a legend indicating node types and edge meanings.

        Parameters
        ----------
        show_pauli_measurement : bool
            Whether Pauli-measured nodes are visually distinct.
        corrections : tuple or None
            X and Z corrections if any, to determine arrow legend entries.
        has_arrows : bool
            Whether flow arrows are present in the graph.
        """
        elements: list[Line2D] = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=10,
                label="Input",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=10,
                label="Measured",
            ),
        ]
        if show_pauli_measurement:
            elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#4292c6",
                    markeredgecolor="black",
                    markersize=10,
                    label="Pauli-measured",
                )
            )
        elements.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=10,
                    label="Output",
                ),
                Line2D([0], [0], color="gray", linewidth=1, alpha=0.6, linestyle="dashed", label="Graph edge"),
            ]
        )

        if corrections is not None:
            elements.extend(
                [
                    Line2D([0], [0], color="tab:red", linewidth=1, label="X-correction"),
                    Line2D([0], [0], color="tab:green", linewidth=1, label="Z-correction"),
                    Line2D([0], [0], color="tab:brown", linewidth=1, label="X & Z-correction"),
                ]
            )
        elif has_arrows:
            elements.append(Line2D([0], [0], color="black", linewidth=1, label="Flow"))

        plt.legend(handles=elements, loc="center left", fontsize=9, bbox_to_anchor=(1, 0.5))

    def __draw_measurement_labels(self, pos: Mapping[int, _Point]) -> None:
        """Draw measurement labels next to measured nodes.

        Labels are rendered with a white background to ensure legibility over graph edges.

        Parameters
        ----------
        pos : Mapping[int, tuple[float, float]]
            Dictionary of node positions.
        """
        for node, meas in self.og.measurements.items():
            label = self._format_measurement_label(meas)
            if label is not None:
                x, y = pos[node]
                plt.text(
                    x + 0.18,
                    y - 0.2,
                    label,
                    fontsize=8,
                    zorder=3,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                )

    @staticmethod
    def _format_measurement_label(meas: Measurement) -> str | None:
        """Format a measurement label for display.

        Parameters
        ----------
        meas : Measurement
            The measurement to format.

        Returns
        -------
        str | None
            Formatted label string, or None if nothing to show.
        """
        if isinstance(meas, PauliMeasurement):
            return str(meas)
        if isinstance(meas, BlochMeasurement):
            if isinstance(meas.angle, (int, float)):
                angle_str = angle_to_str(meas.angle, OutputFormat.Unicode)
            else:
                angle_str = str(meas.angle)
            return f"{meas.plane.name}({angle_str})"
        return None

    def determine_figsize(
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
            width = len({pos[node][0] for node in self.og.graph.nodes()}) * 0.8
        else:
            width = (max(l_k.values(), default=0) + 1) * 0.8
        height = len({pos[node][1] for node in self.og.graph.nodes()}) if pos is not None else len(self.og.output_nodes)
        return (width * node_distance[0], height * node_distance[1])

    def place_edge_paths(
        self, flow: Mapping[int, AbstractSet[int]], pos: Mapping[int, _Point]
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
        edge_path = self.place_edge_paths_without_structure(pos)
        edge_set = set(self.og.graph.edges())
        arrow_path: dict[_Edge, list[_Point]] = {}
        flow_arrows = {(k, v) for k, values in flow.items() for v in values}

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
        nodes = set(self.og.graph.nodes())
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

    def place_edge_paths_without_structure(self, pos: Mapping[int, _Point]) -> dict[_Edge, list[_Point]]:
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
        return {edge: self._find_bezier_path(edge, [pos[edge[0]], pos[edge[1]]], pos) for edge in self.og.graph.edges()}

    def place_causal_flow(self, flow: CausalFlow[AbstractPlanarMeasurement]) -> dict[int, _Point]:
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
        f = flow.correction_function
        values_union = set().union(*f.values())
        start_nodes = set(self.og.graph.nodes()) - values_union
        pos = {node: [0, 0] for node in self.og.graph.nodes()}
        for i, k in enumerate(start_nodes):
            pos[k][1] = i
            node = k
            while node in f:
                node = next(iter(f[node]))
                pos[node][1] = i

        layers = flow.partial_order_layers
        lmax = len(layers) - 1
        # Change the x coordinates of the nodes based on their layer, sort in descending order
        for layer_idx, layer in enumerate(layers):
            for node in layer:
                pos[node][0] = lmax - layer_idx
        return {k: (x, y) for k, (x, y) in pos.items()}

    def place_pauli_flow(self, flow: PauliFlow[AbstractMeasurement]) -> dict[int, _Point]:
        """
        Return the position of nodes based on the Pauli flow.

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

        Notes
        -----
        This method accepts gflows, as gflows are particular cases of Pauli flows.
        """
        g_edges: list[_Edge] = []

        g = flow.correction_function

        for node, node_list in g.items():
            g_edges.extend((node, n) for n in node_list)

        g_prime = self.og.graph.copy()
        g_prime.add_nodes_from(self.og.graph.nodes())
        g_prime.add_edges_from(g_edges)

        layers = flow.partial_order_layers
        l_max = len(layers) - 1

        l_reverse = {node: l_max - layer_idx for layer_idx, layer in enumerate(layers) for node in layer}
        _set_node_attributes(g_prime, l_reverse, "subset")
        pos = nx.multipartite_layout(g_prime)

        vert = list({pos[node][1] for node in self.og.graph.nodes()})
        vert.sort()
        index = {y: i for i, y in enumerate(vert)}
        return {
            node: (l_max - layer_idx, index[pos[node][1]]) for layer_idx, layer in enumerate(layers) for node in layer
        }

    def place_without_structure(self) -> dict[int, _Point]:
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
        connected_components = list(nx.connected_components(self.og.graph))

        for component in connected_components:
            subgraph = self.og.graph.subgraph(component)
            initial_pos: dict[int, tuple[int, int]] = dict.fromkeys(component, (0, 0))

            if (
                len(set(self.og.output_nodes) & set(component)) == 0
                and len(set(self.og.input_nodes) & set(component)) == 0
            ):
                pos = nx.spring_layout(subgraph)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                layers.update((node, k) for k, node in enumerate(order[::-1]))

            elif (
                len(set(self.og.output_nodes) & set(component)) > 0
                and len(set(self.og.input_nodes) & set(component)) == 0
            ):
                fixed_nodes = list(set(self.og.output_nodes) & set(component))
                for i, node in enumerate(fixed_nodes):
                    initial_pos[node] = (10, i)
                    layers[node] = 0
                pos = nx.spring_layout(subgraph, pos=initial_pos, fixed=fixed_nodes)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                order = [node for node in order if node not in fixed_nodes]
                nv = len(self.og.output_nodes)
                for i, node in enumerate(order[::-1]):
                    k = i // nv + 1
                    layers[node] = k

            elif (
                len(set(self.og.output_nodes) & set(component)) == 0
                and len(set(self.og.input_nodes) & set(component)) > 0
            ):
                fixed_nodes = list(set(self.og.input_nodes) & set(component))
                for i, node in enumerate(fixed_nodes):
                    initial_pos[node] = (-10, i)
                pos = nx.spring_layout(subgraph, pos=initial_pos, fixed=fixed_nodes)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                order = [node for node in order if node not in fixed_nodes]
                nv = len(self.og.input_nodes)
                for i, node in enumerate(order[::-1]):
                    k = i // nv
                    layers[node] = k
                layer_input = 0 if layers == {} else max(layers.values()) + 1
                for node in fixed_nodes:
                    layers[node] = layer_input

            else:
                for i, node in enumerate(list(set(self.og.output_nodes) & set(component))):
                    initial_pos[node] = (10, i)
                    layers[node] = 0
                for i, node in enumerate(list(set(self.og.input_nodes) & set(component))):
                    initial_pos[node] = (-10, i)
                fixed_nodes = list(set(self.og.output_nodes) & set(component)) + list(
                    set(self.og.input_nodes) & set(component)
                )
                pos = nx.spring_layout(subgraph, pos=initial_pos, fixed=fixed_nodes)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                order = [node for node in order if node not in fixed_nodes]
                nv = len(self.og.output_nodes)
                for i, node in enumerate(order[::-1]):
                    k = i // nv + 1
                    layers[node] = k
                layer_input = max(layers.values()) + 1
                for node in set(self.og.input_nodes) & set(component) - set(self.og.output_nodes):
                    layers[node] = layer_input

        g_prime = self.og.graph.copy()
        g_prime.add_nodes_from(self.og.graph.nodes())
        g_prime.add_edges_from(self.og.graph.edges())
        l_max = max(layers.values())
        l_reverse = {v: l_max - l for v, l in layers.items()}
        _set_node_attributes(g_prime, l_reverse, "subset")
        pos = nx.multipartite_layout(g_prime)
        vert = list({pos[node][1] for node in self.og.graph.nodes()})
        vert.sort()
        index = {y: i for i, y in enumerate(vert)}
        return {node: (l_max - layers[node], index[pos[node][1]]) for node in self.og.graph.nodes()}

    def place_all_corrections(self, layers: Mapping[int, int]) -> dict[int, _Point]:
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
        g_prime = self.og.graph.copy()
        g_prime.add_nodes_from(self.og.graph.nodes())
        g_prime.add_edges_from(self.og.graph.edges())
        _set_node_attributes(g_prime, layers, "subset")
        layout = nx.multipartite_layout(g_prime)
        vert = list({layout[node][1] for node in self.og.graph.nodes()})
        vert.sort()
        index = {y: i for i, y in enumerate(vert)}
        return {node: (layers[node], index[layout[node][1]]) for node in self.og.graph.nodes()}

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
