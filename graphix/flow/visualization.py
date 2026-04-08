"""Functions to visualize PauliFlow and XZCorrections objects."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import matplotlib.transforms as mtransforms
import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from graphix.measurements import PauliMeasurement

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from collections.abc import Set as AbstractSet
    from pathlib import Path
    from typing import TypeAlias

    from graphix.clifford import Clifford
    from graphix.flow.core import CausalFlow, PauliFlow, XZCorrections
    from graphix.fundamentals import AbstractMeasurement, AbstractPlanarMeasurement

    _Point: TypeAlias = tuple[float, float]
    _Edge: TypeAlias = tuple[int, int]
    _Path: TypeAlias = list[_Point]
    Color: TypeAlias = str | tuple[float, float, float] | tuple[float, float, float, float]

_T = TypeVar("_T")


@dataclass(frozen=True)
class Colored(Generic[_T]):
    value: _T
    color: Color


@dataclass
class PlotLims:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass(frozen=True)
class GraphVisualizer:
    obj: PauliFlow[AbstractMeasurement] | XZCorrections[AbstractMeasurement]
    show_pauli_measurement: bool = True
    show_measurement_labels: bool = False
    node_labels: bool | Mapping[int, str] = True
    local_clifford: Mapping[int, Clifford] | None = None
    node_distance: tuple[float, float] = (1, 1)
    figsize: tuple[int, int] | None = None
    filename: Path | None = None

    ###################
    # Style constants #
    ###################

    # Nodes
    DEFAULT_NODE_EC: ClassVar[Color] = "black"
    DEFAULT_NODE_FC: ClassVar[Color] = "white"
    INPUT_NODE_EC: ClassVar[Color] = "red"
    OUTPUT_NODE_FC: ClassVar[Color] = "lightgray"
    PAULI_NODE_FC: ClassVar[Color] = "lightblue"

    NODE_LABEL_FS: ClassVar[float] = 12

    # Edge style

    # Correction arrows
    FLOW_C: ClassVar[Color] = "black"
    X_C: ClassVar[Color] = "tab:red"
    Z_C: ClassVar[Color] = "tab:green"
    XZ_C: ClassVar[Color] = "tab:brown"

    # Layers
    LAYER_C: ClassVar[Color] = "gray"
    LAYER_FS: ClassVar[float] = 10

    # Other labels
    LABEL_MEAS_FS: ClassVar[float] = 9.5
    LC_MEAS_FS: ClassVar[float] = 9.5

    def visualize(self) -> None:

        pos = self._compute_positions()
        edge_paths = self._compute_edge_paths(pos)
        arrow_paths = self._compute_arrow_paths(pos)
        plot_lims = self._determine_plot_lims(pos)

        figsize = self.figsize or self._determine_figsize(pos)
        plt.figure(figsize=figsize)

        self._draw_edges(pos, edge_paths)
        self._draw_arrows(pos, arrow_paths)
        self._draw_nodes(pos)
        self._draw_layers(plot_lims)
        if self.show_measurement_labels:
            self._draw_measurements_labels(pos)
        self._draw_local_clifford(pos)  # Skipped if `self.local_clifford` is `None`.

        self._set_plot_lims(plot_lims)

        plt.plot()

    def _determine_figsize(self, pos: Mapping[int, _Point]) -> _Point:
        n_layers = len(self.obj.partial_order_layers)
        width = n_layers * 0.8
        height = len({pos[node][1] for node in self.obj.og.graph.nodes()})
        return (width * self.node_distance[0], height * self.node_distance[1])

    def _determine_plot_lims(self, pos: Mapping[int, _Point]) -> PlotLims:
        nodes = self.obj.og.graph.nodes()
        xmin = min((pos[node][0] for node in nodes), default=0)
        xmax = max((pos[node][0] for node in nodes), default=0)
        ymin = min((pos[node][1] for node in nodes), default=0)
        ymax = max((pos[node][1] for node in nodes), default=0)

        return PlotLims(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    @staticmethod
    def _set_plot_lims(plot_lims: PlotLims) -> None:
        offset = 0.7
        plt.xlim(plot_lims.xmin - offset, plot_lims.xmax + offset)
        plt.ylim(plot_lims.ymin - offset, plot_lims.ymax + offset)

    def _compute_positions(self) -> dict[int, _Point]:

        # We can't use functools.singledispatch here.
        # If we annotate the dispatch argument with CausalFlow[AbstractPlanarMeasurement]
        # compilation will fail because generic types are only known statically.
        # If we don't specify the generic type (we don't need to), mypy will complain.

        # Circumvent import loop
        from graphix.flow.core import CausalFlow  # noqa: PLC0415
        pos = (
            _compute_positions_causal_flow(self.obj)
            if isinstance(self.obj, CausalFlow)
            else _compute_positions(self.obj)
        )
        if self.node_distance != (1, 1):
            pos = {k: (v[0] * self.node_distance[0], v[1] * self.node_distance[1]) for k, v in pos.items()}
        return pos

    def _compute_edge_paths(self, pos: Mapping[int, _Point]) -> dict[_Edge, _Path]:
        edges = self.obj.og.graph.edges()
        return {edge: _find_bezier_path(edge, [pos[edge[0]], pos[edge[1]]], pos) for edge in edges}

    def _compute_arrow_paths(self, pos: Mapping[int, _Point]) -> dict[_Edge, Colored[_Path]]:
        correction_arrows: set[Colored[_Edge]]

        # Circumvent import loop
        from graphix.flow.core import PauliFlow, XZCorrections  # noqa: PLC0415

        if isinstance(self.obj, PauliFlow):
            correction_arrows = {
                Colored((k, v), self.FLOW_C) for k, values in self.obj.correction_function.items() for v in values
            }
        elif isinstance(self.obj, XZCorrections):
            correction_arrows = set()
            colors = (self.X_C, self.Z_C, self.XZ_C)
            for measured_node in self.obj.og.measurements:
                x_corr = self.obj.x_corrections.get(measured_node, set())
                z_corr = self.obj.z_corrections.get(measured_node, set())
                xz_corr = x_corr & z_corr

                corrs = (x_corr - xz_corr, z_corr - xz_corr, xz_corr)
                for corr_nodes, color in zip(corrs, colors, strict=True):
                    correction_arrows.update(Colored((measured_node, corr_node), color) for corr_node in corr_nodes)
        else:
            raise TypeError("Expected code to be unreachable")

        return _compute_arrow_paths(pos, self.obj.og.graph.edges(), correction_arrows)

    def _draw_nodes(self, pos: Mapping[int, _Point]) -> None:
        og = self.obj.og
        fc = self.DEFAULT_NODE_FC
        ec = self.DEFAULT_NODE_FC
        for node in og.graph.nodes():
            fc = self.DEFAULT_NODE_FC
            ec = self.DEFAULT_NODE_EC
            if node in og.input_nodes:
                ec = self.INPUT_NODE_EC
            if node in og.output_nodes:
                fc = self.OUTPUT_NODE_FC
            elif self.show_pauli_measurement and isinstance(og.measurements[node], PauliMeasurement):
                fc = self.PAULI_NODE_FC
            plt.scatter(*pos[node], edgecolor=ec, facecolor=fc, s=350, zorder=2)

        labels = dict(self.node_labels) if isinstance(self.node_labels, Mapping) else None
        fontsize = self.NODE_LABEL_FS
        if max(og.graph.nodes(), default=0) >= 100:
            fontsize = int(fontsize * 2 / len(str(max(og.graph.nodes()))))
        nx.draw_networkx_labels(og.graph, pos, labels=labels, font_size=fontsize)

    def _draw_edges(self, pos: Mapping[int, _Point], edge_paths: Mapping[_Edge, _Path]) -> None:
        for edge, path in edge_paths.items():
            if len(path) == 2:
                nx.draw_networkx_edges(self.obj.og.graph, pos, edgelist=[edge], style="dashed", alpha=0.7)
            else:
                curve = _bezier_curve_linspace(path)
                plt.plot(curve[:, 0], curve[:, 1], "k--", linewidth=1, alpha=0.7)

    def _draw_arrows(self, pos: Mapping[int, _Point], arrow_paths: Mapping[_Edge, Colored[_Path]]) -> None:
        for arrow, colored_path in arrow_paths.items():
            path = colored_path.value
            color = colored_path.color

            if len(path) == 2:  # straight line
                nx.draw_networkx_edges(
                    self.obj.og.graph, pos, edgelist=[arrow], edge_color=color, arrowstyle="->", arrows=True
                )
            else:
                new_path = _shorten_path(path) if arrow[0] != arrow[1] else path
                curve = _bezier_curve_linspace(new_path)
                plt.plot(curve[:, 0], curve[:, 1], c=color, linewidth=1)
                plt.annotate(
                    "",
                    xy=curve[-1],
                    xytext=curve[-2],
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
                )

    def _draw_layers(self, plot_lims: PlotLims) -> None:
        nlayers = len(self.obj.partial_order_layers)

        fig, ax = plt.gcf(), plt.gca()
        base = ax.transData

        # Add a fixed vertical offset (e.g. -15 points)
        # This ensures that the layer is always at the same distance of the nodes, regardless of the ylims.
        offset = mtransforms.ScaledTranslation(0, -15 / 72, fig.dpi_scale_trans)

        for layer in range(nlayers - 1):
            plt.axvline(
                x=(layer + 0.5) * self.node_distance[0],
                color=self.LAYER_C,
                linestyle=":",
                linewidth=0.9,
                alpha=0.8,
            )
            plt.text(
                layer * self.node_distance[0],
                plot_lims.ymin,
                str(nlayers - 1 - layer),
                ha="center",
                va="top",
                fontsize=self.LAYER_FS,
                color=self.LAYER_C,
                transform=base + offset,
            )

        # Add last label (layer 0)
        plt.text(
            (nlayers - 1) * self.node_distance[0],
            plot_lims.ymin,
            str(0),
            ha="center",
            va="top",
            fontsize=self.LAYER_FS,
            color=self.LAYER_C,
            transform=base + offset,
        )

        # Draw horizontal arrow indicating measurement order with "Layer" label below
        offset = mtransforms.ScaledTranslation(0, -27 / 72, fig.dpi_scale_trans)
        if nlayers > 1:
            plt.annotate(
                "",
                xy=((nlayers - 1) * self.node_distance[0] + 0.3, plot_lims.ymin),
                xytext=(-0.3, plot_lims.ymin),
                xycoords=base + offset,
                textcoords=base + offset,
                arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.2},
            )

        offset = mtransforms.ScaledTranslation(0, -30 / 72, fig.dpi_scale_trans)
        mid_x = (nlayers - 1) / 2 * self.node_distance[0]
        plt.text(
            mid_x,
            plot_lims.ymin,
            "Layer",
            ha="center",
            va="top",
            fontsize=self.LAYER_FS,
            color=self.LAYER_C,
            transform=base + offset,
        )

        # Update plot_lims to take into account label
        trans = base + offset
        _, ydisp = trans.transform((0, plot_lims.ymin))
        plot_lims.ymin = base.inverted().transform((0, ydisp))[1]

    def _draw_measurements_labels(self, pos: Mapping[int, _Point]) -> None:
        for node, meas in self.obj.og.measurements.items():
            x, y = pos[node] + np.array([0.22, -0.2])
            label = meas.to_plane_or_axis().name

            plt.text(x, y, label, fontsize=self.LABEL_MEAS_FS, zorder=3)

    def _draw_local_clifford(self, pos: Mapping[int, _Point]) -> None:
        if self.local_clifford is not None:
            for node in self.local_clifford:
                x, y = pos[node] + np.array([0.2, 0.2])
                plt.text(x, y, f"{self.local_clifford[node]}", fontsize=self.LC_MEAS_FS, zorder=3)


def _compute_positions(
    obj: PauliFlow[AbstractMeasurement] | XZCorrections[AbstractMeasurement],
) -> dict[int, tuple[float, float]]:
    graph = obj.og.graph
    pol = obj.partial_order_layers

    layers = dict(enumerate(pol[::-1]))
    pos = nx.multipartite_layout(graph, subset_key=layers)

    # TODO: Try to improve
    l_max = len(pol) - 1
    vert = list({pos[node][1] for node in graph.nodes()})
    vert.sort()
    index = {y: i for i, y in enumerate(vert)}
    return {node: (l_max - layer_idx, index[pos[node][1]]) for layer_idx, layer in enumerate(pol) for node in layer}


def _compute_positions_causal_flow(obj: CausalFlow[AbstractPlanarMeasurement]) -> dict[int, tuple[float, float]]:
    og = obj.og
    cf = obj.correction_function
    pol = obj.partial_order_layers

    values_union = set().union(*cf.values())
    start_nodes = set(og.graph.nodes()) - values_union
    pos = {node: [0, 0] for node in og.graph.nodes()}
    for i, k in enumerate(start_nodes):
        pos[k][1] = i
        node = k
        while node in cf:
            node = next(iter(cf[node]))
            pos[node][1] = i

    lmax = len(pol) - 1
    # Change the x coordinates of the nodes based on their layer, sort in descending order
    for layer_idx, layer in enumerate(pol):
        for node in layer:
            pos[node][0] = lmax - layer_idx
    return {k: (x, y) for k, (x, y) in pos.items()}


def _compute_arrow_paths(
    pos: Mapping[int, _Point], edges: AbstractSet[_Edge], correction_arrows: AbstractSet[Colored[_Edge]]
) -> dict[_Edge, Colored[_Path]]:
    arrow_paths: dict[_Edge, Colored[_Path]] = {}

    for colored_arrow in correction_arrows:
        arrow, color = colored_arrow.value, colored_arrow.color
        if arrow[0] == arrow[1]:  # Self loop
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
            if arrow in edges or (arrow[1], arrow[0]) in edges:
                mid_point = (
                    0.5 * (pos[arrow[0]][0] + pos[arrow[1]][0]),
                    0.5 * (pos[arrow[0]][1] + pos[arrow[1]][1]),
                )
                if _edge_intersects_node(pos[arrow[0]], pos[arrow[1]], mid_point, buffer=0.05):
                    ctrl_point = _control_point(pos[arrow[0]], pos[arrow[1]], mid_point, distance=0.2)
                    bezier_path.insert(1, ctrl_point)
            bezier_path = _find_bezier_path(arrow, bezier_path, pos)

        arrow_paths[arrow] = Colored(bezier_path, color)

    return arrow_paths


def _find_bezier_path(arrow: _Edge, bezier_path: Iterable[_Point], pos: Mapping[int, _Point]) -> list[_Point]:
    bezier_path = list(bezier_path)
    max_iter = 5
    iteration = 0
    nodes = set(pos.keys())
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
                if node != arrow[0] and node != arrow[1] and _edge_intersects_node(start, end, pos[node]):
                    intersect = True
                    ctrl_points.append(
                        (
                            i,
                            _control_point(bezier_path[0], bezier_path[-1], pos[node], distance=0.6 / iteration),
                        )
                    )
                    nodes -= {node}
        if not intersect:
            break
        for i, (index, ctrl_point) in enumerate(ctrl_points):
            bezier_path.insert(index + i + 1, ctrl_point)
    return _check_path(bezier_path, pos[arrow[1]])


def _bezier_curve(bezier_path: Sequence[_Point], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Generate a bezier curve from a list of points."""
    n = len(bezier_path) - 1  # order of the curve
    curve = np.zeros((len(t), 2))
    for i, point in enumerate(bezier_path):
        curve += np.outer(math.comb(n, i) * ((1 - t) ** (n - i)) * (t**i), np.array(point))
    return curve


def _bezier_curve_linspace(bezier_path: Sequence[_Point]) -> npt.NDArray[np.float64]:
    t = np.linspace(0, 1, 100, dtype=np.float64)
    return _bezier_curve(bezier_path, t)


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


def _shorten_path(path: Sequence[_Point]) -> list[_Point]:
    """Shorten the last edge not to hide arrow under the node."""
    new_path = list(path)
    last = np.array(new_path[-1])
    second_last = np.array(new_path[-2])
    last_edge: _Point = tuple(last - (last - second_last) / np.linalg.norm(last - second_last) * 0.2)
    new_path[-1] = last_edge
    return new_path


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
    cross = _cross2d(edge_vector, node_pos_array - np.array(start))
    if cross > 0:
        dir_vector = np.array([edge_vector[1], -edge_vector[0]])  # Rotate the edge vector 90 degrees
    else:
        dir_vector = np.array([-edge_vector[1], edge_vector[0]])
    dir_vector /= np.linalg.norm(dir_vector)  # Normalize the vector
    u, v = node_pos_array + distance * dir_vector
    return u, v


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


def _cross2d(
    arr1: np.ndarray[tuple[int], np.dtype[np.float64]], arr2: np.ndarray[tuple[int], np.dtype[np.float64]]
) -> np.float64:
    """Cross-product for 2D vectors.

    `np.cross()` is deprecated for 2D vectors since numpy 2.
    See https://github.com/numpy/numpy/issues/26620 .

    The cross-product of two 2D vectors is the determinant of the
    2×2 matrix formed by placing them as columns.  Equivalently, it
    is the z-component of the 3D cross-product when the vectors are
    extended with a zero z-coordinate.
    """
    if arr1.shape != (2,) or arr2.shape != (2,):
        raise ValueError("Expected 2D vectors of shape (2,)")
    product: np.float64 = arr1[0] * arr2[1] - arr1[1] * arr2[0]
    return product


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
