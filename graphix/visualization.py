"""Functions to visualize PauliFlow and XZCorrections objects."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

import matplotlib.transforms as mtransforms
import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from typing_extensions import assert_never

from graphix.measurements import PauliMeasurement

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from collections.abc import Set as AbstractSet
    from pathlib import Path
    from typing import TypeAlias

    # Unpack introduced in Python 3.12
    from typing_extensions import Unpack

    from graphix.clifford import Clifford
    from graphix.flow.core import CausalFlow, PauliFlow, XZCorrections
    from graphix.fundamentals import AbstractMeasurement, AbstractPlanarMeasurement
    from graphix.opengraph import OpenGraph

    _Point: TypeAlias = tuple[float, float]
    _Edge: TypeAlias = tuple[int, int]
    _Path: TypeAlias = list[_Point]
    Color: TypeAlias = str | tuple[float, float, float] | tuple[float, float, float, float]

_T = TypeVar("_T")

###################
# Style constants #
###################

# Nodes
DEFAULT_NODE_EC = "black"
DEFAULT_NODE_FC = "white"
OUTPUT_NODE_FC = "lightgray"
PAULI_NODE_FC = "lightblue"

INPUT_NODE_MARKER = "s"
DEFAULT_NODE_MARKER = "o"

NODE_LABEL_FS = 12

# Edge style
EDGE_C = (0, 0, 0, 0.7)

# Correction arrows
FLOW_C = "black"
X_C = "tab:red"
Z_C = "tab:green"
XZ_C = "tab:brown"

# Layers
LAYER_C = "gray"
LAYER_FS = 10

# Other labels
LABEL_MEAS_FS = 9.5
LC_MEAS_FS = 9.5


@dataclass(frozen=True)
class Colored(Generic[_T]):
    """A generic container pairing a value with a color for visualization.

    Parameters
    ----------
    value : _T
        The value to be colored. Can be any type.
    color : Color
        The color to associate with the value, used during visualization.
    """

    value: _T
    color: Color


@dataclass(frozen=True)
class PlotLims:
    """Dataclass to wrap the plot limits in axis coordinates."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float


class _Source(Enum):
    """Enumeration to indicate the possible sources of a ``GraphVisualizer``."""

    XZCorr = auto()
    Flow = auto()
    OG = auto()


class DrawKwargs(TypedDict, total=False):
    """Common keyword arguments for graph visualization methods.

    The keys correspond to the fields of :class:`VisualizationOptions`.
    """

    pauli_measurements: bool
    measurement_labels: bool
    node_labels: bool | Mapping[int, str]
    local_clifford: Mapping[int, Clifford] | None
    node_distance: tuple[float, float]
    legend: bool
    figsize: tuple[int, int] | None
    filename: Path | None


@dataclass(frozen=True)
class VisualizationOptions:
    """Options controlling graph visualization.

    Parameters
    ----------
    pauli_measurements : bool, default=True
        If ``True``, Pauli-measured nodes are highlighted with distinct coloring.
    measurement_labels : bool, default=False
        If ``True``, measurement labels (planes and axis) are displayed in the visualization.
    node_labels : bool | Mapping[int, str], default=True
        If ``True``, display numeric node labels. If a mapping, use custom labels
        for nodes specified in the mapping.
    local_clifford : Mapping[int, Clifford] | None, default=None
        Mapping of node identifiers to local Clifford operators. If provided,
        operators are displayed on their corresponding nodes.
    node_distance : tuple[float, float], default=(1, 1)
        Scaling factors (x_scale, y_scale) applied to node positions.
    legend : bool, default=True
        If ``True``, legend is shown.
    figsize : tuple[int, int] | None, default=None
        Figure dimensions (width, height) in inches. If ``None``, dimensions are
        determined automatically based on graph structure.
    filename : Path | None, default=None
        File path to save the visualization. If ``None``, figure is displayed but not saved.
    """

    pauli_measurements: bool = True
    measurement_labels: bool = False
    node_labels: bool | Mapping[int, str] = True
    local_clifford: Mapping[int, Clifford] | None = None
    node_distance: tuple[float, float] = (1, 1)
    legend: bool = True
    figsize: tuple[int, int] | None = None
    filename: Path | None = None


@dataclass(frozen=True)
class GraphVisualizer:
    """Visualizer for flows and XZ-correction structures.

    Attributes
    ----------
    og : OpenGraph[AbstractMeasurement]
        The open graph of the object being rendered.
    pos : Mapping[int, _Point]
        Mapping of node identifiers to (x, y) coordinates.
    edge_paths : Mapping[_Edge, _Path]
        Bezier curve paths for graph edges.
    arrow_paths : Mapping[_Edge, Colored[_Path]]
        Colored bezier curve paths for correction dependency arrows.
    n_layers : int
        Number of measurement layers in the partial order (determines horizontal extent).
    options : VisualizationOptions, default=VisualizationOptions()
        Options controlling graph visualization.
    _source : _Source | None, default=None
        Internal metadata indicating the source of the graph (e.g., ``OpenGraph``, ``PauliFlow`` or
        ``XZCorrections``).

    Notes
    -----
    Instantiate this class via factory methods rather than direct construction:
    - :meth:`from_opengraph` : Create from ``OpenGraph`` instance.
    - :meth:`from_flow` : Create from ``PauliFlow`` instance.
    - :meth:`from_xzcorrections` : Create from ``XZCorrections`` instance.

    Call :meth:`visualize` to generate and display the graph visualization.
    """

    og: OpenGraph[AbstractMeasurement]
    pos: Mapping[int, _Point]
    edge_paths: Mapping[_Edge, _Path]
    arrow_paths: Mapping[_Edge, Colored[_Path]] | None = None
    n_layers: int | None = None
    options: VisualizationOptions = dataclasses.field(default_factory=VisualizationOptions)
    _source: _Source | None = None

    @staticmethod
    def from_opengraph(
        og: OpenGraph[AbstractMeasurement],
        **kwargs: Unpack[DrawKwargs],
    ) -> GraphVisualizer:
        """Create a ``GraphVisualizer`` from an ``OpenGraph`` instance.

        Parameters
        ----------
        og : OpenGraph[AbstractMeasurement]
        options: Unpack[DrawKwargs]
            Options controlling graph visualization. See :class:`VisualizationOptions`.

        Returns
        -------
        GraphVisualizer
        """
        options = VisualizationOptions(**kwargs)
        pos = _compute_positions_opengraph(og)
        pos = _scale_positions(pos, options.node_distance)
        edge_paths = _compute_edge_paths(og, pos)

        return GraphVisualizer(
            og=og,
            pos=pos,
            edge_paths=edge_paths,
            options=options,
            _source=_Source.OG,
        )

    @staticmethod
    def from_flow(
        flow: PauliFlow[AbstractMeasurement],
        **kwargs: Unpack[DrawKwargs],
    ) -> GraphVisualizer:
        """Create a ``GraphVisualizer`` from a ``PauliFlow`` instance.

        Parameters
        ----------
        flow : PauliFlow[AbstractMeasurement]
        options: Unpack[DrawKwargs]
            Options controlling graph visualization. See :class:`VisualizationOptions`.

        Returns
        -------
        GraphVisualizer
        """
        options = VisualizationOptions(**kwargs)

        # We can't use functools.singledispatch here.
        # If we annotate the dispatch argument with CausalFlow[AbstractPlanarMeasurement]
        # compilation will fail because generic types are only known statically.
        # If we don't specify the generic type (we don't need to), mypy will complain.

        # Circumvent import loop
        from graphix.flow.core import CausalFlow  # noqa: PLC0415

        pos = (
            _compute_positions_causal_flow(flow)
            if isinstance(flow, CausalFlow)
            else _compute_positions_partial_order(flow)
        )
        pos = _scale_positions(pos, options.node_distance)
        edge_paths = _compute_edge_paths(flow.og, pos)
        corrections = _format_corrections_flow(flow)
        arrow_paths = _compute_arrow_paths(pos, flow.og.graph.edges(), corrections)
        n_layers = len(flow.partial_order_layers)

        return GraphVisualizer(
            og=flow.og,
            pos=pos,
            edge_paths=edge_paths,
            arrow_paths=arrow_paths,
            n_layers=n_layers,
            options=options,
            _source=_Source.Flow,
        )

    @staticmethod
    def from_xzcorrections(
        xz_corr: XZCorrections[AbstractMeasurement],
        **kwargs: Unpack[DrawKwargs],
    ) -> GraphVisualizer:
        """Create a ``GraphVisualizer`` from an ``XZCorrections`` instance.

        Parameters
        ----------
        xz_corr : XZCorrections[AbstractMeasurement]
        options: Unpack[DrawKwargs]
            Options controlling graph visualization. See :class:`VisualizationOptions`.

        Returns
        -------
        GraphVisualizer
        """
        options = VisualizationOptions(**kwargs)
        pos = _compute_positions_partial_order(xz_corr)
        pos = _scale_positions(pos, options.node_distance)
        edge_paths = _compute_edge_paths(xz_corr.og, pos)
        corrections = _format_corrections_xz(xz_corr)
        arrow_paths = _compute_arrow_paths(pos, xz_corr.og.graph.edges(), corrections)
        n_layers = len(xz_corr.partial_order_layers)

        return GraphVisualizer(
            og=xz_corr.og,
            pos=pos,
            edge_paths=edge_paths,
            arrow_paths=arrow_paths,
            n_layers=n_layers,
            options=options,
            _source=_Source.XZCorr,
        )

    def visualize(self) -> None:
        """Generate and display the complete graph visualization.

        The figure is displayed via ``plt.show()`` or saved when ``self.options.filename`` has a non-null value.
        """
        plot_lims = self._determine_plot_lims()
        figsize = self.options.figsize or self._determine_figsize()
        plt.figure(figsize=figsize)

        self._draw_edges()

        if self.arrow_paths is not None:
            self._draw_arrows()

        self._draw_nodes()

        if self.n_layers:
            plot_lims = self._draw_layers(plot_lims)

        if self.options.measurement_labels:
            self._draw_measurements_labels()

        if self.options.local_clifford is not None:
            self._draw_local_clifford()

        self._set_plot_lims(plot_lims)
        if self.options.legend:
            self._draw_legend()

        if self.options.filename is None:
            plt.show()
        else:
            plt.savefig(self.options.filename, bbox_inches="tight")

    def _determine_figsize(self) -> _Point:
        """Determine figure size based on node positions and node distance.

        Returns
        -------
        _Point
        """
        x_pos: set[float] = set()
        y_pos: set[float] = set()
        for node in self.og.graph.nodes():
            x_pos.add(self.pos[node][0])
            y_pos.add(self.pos[node][1])

        width = len(x_pos) * 0.8
        height = len(y_pos)
        return (width * self.options.node_distance[0], height * self.options.node_distance[1])

    def _determine_plot_lims(self) -> PlotLims:
        """Determine plot axis limits based on node positions.

        Returns
        -------
        PlotLims
        """
        nodes = self.og.graph.nodes()
        xmin = min((self.pos[node][0] for node in nodes), default=0)
        xmax = max((self.pos[node][0] for node in nodes), default=0)
        ymin = min((self.pos[node][1] for node in nodes), default=0)
        ymax = max((self.pos[node][1] for node in nodes), default=0)

        return PlotLims(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    @staticmethod
    def _set_plot_lims(plot_lims: PlotLims) -> None:
        """Adjust plot limits.

        Parameters
        ----------
        plot_lims : PlotLims
            Current plot limits in axis coordinates.
        """
        offset = 0.7
        plt.xlim(plot_lims.xmin - offset, plot_lims.xmax + offset)
        plt.ylim(plot_lims.ymin - offset, plot_lims.ymax + offset)

    def _draw_nodes(self) -> None:
        """Draw graph nodes with style indicating their role and measurement type.

        Notes
        -----
        Node styles are assigned as follows: and shapes
        - Input nodes: input shape and default color.
        - Output nodes: default shape and output-colored fill.
        - Pauli measurement nodes (if ``self.show_pauli_measurement=True``): default shape and Pauli-measured-colored fill.
        - Other nodes: default shape and color.

        See module-level Style constants for actual color values.

        Node labels are drawn with adaptive font sizing: font size scales inversely
        with the number of digits in the largest node index to prevent label overflow
        in graphs with many nodes (100+).

        Node labels are obtained from the ``self.options.node_labels`` mapping if provided; otherwise
        ``networkx`` automatically generates numeric labels from node identifiers.
        """
        for node in self.og.graph.nodes():
            fc = DEFAULT_NODE_FC
            ec = DEFAULT_NODE_EC
            marker = DEFAULT_NODE_MARKER
            if node in self.og.input_nodes:
                marker = INPUT_NODE_MARKER
            if node in self.og.output_nodes:
                fc = OUTPUT_NODE_FC
            elif self.options.pauli_measurements and isinstance(self.og.measurements[node], PauliMeasurement):
                fc = PAULI_NODE_FC
            plt.scatter(*self.pos[node], edgecolor=ec, facecolor=fc, s=350, zorder=2, marker=marker)

        labels = dict(self.options.node_labels) if isinstance(self.options.node_labels, Mapping) else None
        fontsize = NODE_LABEL_FS
        if max(self.og.graph.nodes(), default=0) >= 100:
            fontsize = int(fontsize * 2 / len(str(max(self.og.graph.nodes()))))
        nx.draw_networkx_labels(self.og.graph, self.pos, labels=labels, font_size=fontsize)

    def _draw_edges(self) -> None:
        """Draw open graph's edges on the plot."""
        for edge, path in self.edge_paths.items():
            if len(path) == 2:
                nx.draw_networkx_edges(self.og.graph, self.pos, edgelist=[edge], style="dashed", edge_color=EDGE_C)
            else:
                curve = _bezier_curve_linspace(path)
                plt.plot(curve[:, 0], curve[:, 1], "--", color=EDGE_C, linewidth=1)

    def _draw_arrows(self) -> None:
        """Draw correction arrows on the plot."""
        assert self.arrow_paths is not None
        for arrow, colored_path in self.arrow_paths.items():
            path = colored_path.value
            color = colored_path.color

            if len(path) == 2:  # straight line
                nx.draw_networkx_edges(
                    self.og.graph, self.pos, edgelist=[arrow], edge_color=color, arrowstyle="->", arrows=True
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

    def _draw_layers(self, plot_lims: PlotLims) -> PlotLims:
        """Draw partial order layer separators and labels on the plot.

        Parameters
        ----------
        plot_lims : PlotLims
            Current plot limits in axis coordinates.

        Returns
        -------
        PlotLims
            Updated plot limits with ``ymin`` adjusted to account for the layer labels
            and measurement order indicator below the plot.

        Notes
        -----
        This method adds the following visual elements:
        1. Vertical dashed lines separating layers.
        2. Layer index labels positioned at the bottom, numbered in descending order
            from ``n_layers - 1`` to ``0``.
        3. A horizontal arrow indicating measurement order direction.
        4. A "Layer" label below the arrow.

        All text elements use fixed point offsets (in display coordinates) to ensure
        consistent spacing relative to the plot, independent of y-axis scaling.
        """
        assert self.n_layers is not None
        fig, ax = plt.gcf(), plt.gca()
        base = ax.transData

        # Add a fixed vertical offset (e.g. -15 points)
        # This ensures that the layer is always at the same distance of the nodes, regardless of the ylims.
        offset = mtransforms.ScaledTranslation(0, -15 / 72, fig.dpi_scale_trans)

        for layer in range(self.n_layers - 1):
            plt.axvline(
                x=(layer + 0.5) * self.options.node_distance[0],
                color=LAYER_C,
                linestyle=":",
                linewidth=0.9,
                alpha=0.8,
            )
            plt.text(
                layer * self.options.node_distance[0],
                plot_lims.ymin,
                str(self.n_layers - 1 - layer),
                ha="center",
                va="top",
                fontsize=LAYER_FS,
                color=LAYER_C,
                transform=base + offset,
            )

        # Add last label (layer 0)
        plt.text(
            (self.n_layers - 1) * self.options.node_distance[0],
            plot_lims.ymin,
            str(0),
            ha="center",
            va="top",
            fontsize=LAYER_FS,
            color=LAYER_C,
            transform=base + offset,
        )

        # Draw horizontal arrow indicating measurement order with "Layer" label below
        offset = mtransforms.ScaledTranslation(0, -27 / 72, fig.dpi_scale_trans)
        if self.n_layers > 1:
            plt.annotate(
                "",
                xy=((self.n_layers - 1) * self.options.node_distance[0] + 0.3, plot_lims.ymin),
                xytext=(-0.3, plot_lims.ymin),
                xycoords=base + offset,
                textcoords=base + offset,
                arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.2},
            )

        offset = mtransforms.ScaledTranslation(0, -30 / 72, fig.dpi_scale_trans)
        mid_x = (self.n_layers - 1) / 2 * self.options.node_distance[0]
        plt.text(
            mid_x,
            plot_lims.ymin,
            "Layer",
            ha="center",
            va="top",
            fontsize=LAYER_FS,
            color=LAYER_C,
            transform=base + offset,
        )

        # Update plot_lims to take into account label
        trans = base + offset
        _, ydisp = trans.transform((0, plot_lims.ymin))
        return replace(plot_lims, ymin=base.inverted().transform((0, ydisp))[1])

    def _draw_measurements_labels(self) -> None:
        """Add text labels indicating measurement planes and axes."""
        for node, meas in self.og.measurements.items():
            x, y = self.pos[node] + np.array([0.22, -0.2])
            label = meas.to_plane_or_axis().name

            plt.text(x, y, label, fontsize=LABEL_MEAS_FS, zorder=3)

    def _draw_local_clifford(self) -> None:
        """Add text labels indicating Clifford commands."""
        assert self.options.local_clifford is not None
        for node in self.options.local_clifford:
            x, y = self.pos[node] + np.array([0.2, 0.2])
            plt.text(x, y, f"{self.options.local_clifford[node]}", fontsize=LC_MEAS_FS, zorder=3)

    def _draw_legend(self) -> None:
        """Add legend to plot.

        Legend is customized depending on the object being plotted (flow or XZ-corections) indicated in ``self._source``.
        """
        plt.scatter(
            [],
            [],
            edgecolor=DEFAULT_NODE_EC,
            facecolor=DEFAULT_NODE_FC,
            s=150,
            zorder=2,
            label="Input nodes",
            marker=INPUT_NODE_MARKER,
        )
        plt.scatter(
            [],
            [],
            edgecolor=DEFAULT_NODE_EC,
            facecolor=OUTPUT_NODE_FC,
            s=150,
            zorder=2,
            label="Output nodes",
            marker=DEFAULT_NODE_MARKER,
        )
        if self.options.pauli_measurements:
            plt.scatter(
                [],
                [],
                edgecolor=DEFAULT_NODE_EC,
                facecolor=PAULI_NODE_FC,
                s=150,
                zorder=2,
                label="Pauli-measured nodes",
                marker=DEFAULT_NODE_MARKER,
            )
        plt.plot([], [], "--", c=EDGE_C, label="Graph edge")

        assert self._source is not None
        match self._source:
            case _Source.Flow:
                plt.plot([], [], color=FLOW_C, label="Correction function")
            case _Source.XZCorr:
                plt.plot([], [], color=X_C, label="X corrections")
                plt.plot([], [], color=Z_C, label="Z corrections")
                plt.plot([], [], color=XZ_C, label="X and Z corrections")
            case _Source.OG:
                pass
            case _:
                assert_never(self._source)

        plt.legend(loc="center left", fontsize=10, bbox_to_anchor=(1, 0.5))


def _compute_positions_opengraph(og: OpenGraph[AbstractMeasurement]) -> dict[int, _Point]:
    """Compute node positions for an open graph without partial order.

    Parameters
    ----------
    obj : OpenGraph[AbstractMeasurement]

    Returns
    -------
    dict[int, _Point]
        Dictionary mapping node identifiers to (x, y) coordinates for visualization.
        X-coordinates represent the layer in the partial order (higher x = earlier layer).
        Y-coordinates represent the vertical position within start node chains.
    """
    layers: dict[int, int] = {}
    connected_components = list(nx.connected_components(og.graph))

    n_outputs = len(og.output_nodes)
    n_inputs = len(og.input_nodes)

    oset = set(og.output_nodes)
    iset = set(og.input_nodes)

    def update_layers(
        subgraph: nx.Graph[int],
        initial_pos: dict[int, tuple[int, int]],
        fixed_nodes: Sequence[int],
        n: int,
        offset: int,
    ) -> None:
        pos = nx.spring_layout(subgraph, pos=initial_pos, fixed=fixed_nodes)
        # order the nodes based on the x-coordinate
        order = sorted(pos, key=lambda x: pos[x][0])
        order = [node for node in order if node not in fixed_nodes]
        for i, node in enumerate(order[::-1]):
            k = i // n + offset
            layers[node] = k

    for component in connected_components:
        subgraph = og.graph.subgraph(component)
        comp_set = set(component)
        initial_pos: dict[int, tuple[int, int]] = dict.fromkeys(component, (0, 0))

        n_comp_o = len(oset & comp_set)
        n_comp_i = len(iset & comp_set)

        if n_comp_o == 0 and n_comp_i == 0:
            pos = nx.spring_layout(subgraph)
            # order the nodes based on the x-coordinate
            order = sorted(pos, key=lambda x: pos[x][0])
            layers.update((node, k) for k, node in enumerate(order[::-1]))

        elif n_comp_o > 0 and n_comp_i == 0:
            fixed_nodes = list(oset & comp_set)
            update_layers(subgraph, initial_pos, fixed_nodes, n=n_outputs, offset=1)
            for i, node in enumerate(fixed_nodes):
                initial_pos[node] = (10, i)
                layers[node] = 0

        elif n_comp_o == 0 and n_comp_i > 0:
            fixed_nodes = list(iset & comp_set)
            update_layers(subgraph, initial_pos, fixed_nodes, n=n_inputs, offset=0)
            for i, node in enumerate(fixed_nodes):
                initial_pos[node] = (-10, i)
            layer_input = 0 if layers == {} else max(layers.values()) + 1
            for node in fixed_nodes:
                layers[node] = layer_input

        else:
            o_comp_lst = list(oset & comp_set)
            i_comp_lst = list(iset & comp_set)
            for i, node in enumerate(o_comp_lst):
                initial_pos[node] = (10, i)
                layers[node] = 0
            for i, node in enumerate(i_comp_lst):
                initial_pos[node] = (-10, i)

            fixed_nodes = o_comp_lst + i_comp_lst
            update_layers(subgraph, initial_pos, fixed_nodes, n=n_outputs, offset=1)
            layer_input = max(layers.values()) + 1
            for node in iset & comp_set - oset:
                layers[node] = layer_input

    g_prime = og.graph.copy()
    g_prime.add_nodes_from(og.graph.nodes())
    g_prime.add_edges_from(og.graph.edges())
    l_max = max(layers.values())
    l_reverse = {v: l_max - l for v, l in layers.items()}
    nx.set_node_attributes(g_prime, l_reverse, name="subset")  # type: ignore[arg-type]
    pos = nx.multipartite_layout(g_prime)
    vert = list({pos[node][1] for node in og.graph.nodes()})
    vert.sort()
    index = {y: i for i, y in enumerate(vert)}
    return {node: (l_max - layers[node], index[pos[node][1]]) for node in og.graph.nodes()}


def _compute_positions_partial_order(
    obj: PauliFlow[AbstractMeasurement] | XZCorrections[AbstractMeasurement],
) -> dict[int, _Point]:
    """Compute node positions for objects with a partial order.

    Parameters
    ----------
    obj : PauliFlow[AbstractMeasurement] | XZCorrections[AbstractMeasurement]

    Returns
    -------
    dict[int, _Point]
        Dictionary mapping node identifiers to (x, y) coordinates for visualization.
        X-coordinates represent the layer in the partial order (higher x = earlier layer).
        Y-coordinates represent the vertical position within start node chains.
    """
    graph = obj.og.graph
    pol = obj.partial_order_layers
    layers = dict(enumerate(pol[::-1]))
    pos = nx.multipartite_layout(graph, subset_key=layers)

    l_max = len(pol) - 1
    vert = sorted({pos[node][1] for node in graph.nodes()})
    index = {y: i for i, y in enumerate(vert)}
    return {node: (l_max - layer_idx, index[pos[node][1]]) for layer_idx, layer in enumerate(pol) for node in layer}


def _compute_positions_causal_flow(obj: CausalFlow[AbstractPlanarMeasurement]) -> dict[int, _Point]:
    """Compute node positions for causal flow graph layout.

    Parameters
    ----------
    obj : CausalFlow[AbstractPlanarMeasurement]

    Returns
    -------
    dict[int, _Point]
        Dictionary mapping node identifiers to (x, y) coordinates for visualization.
        X-coordinates represent the layer in the partial order (higher x = earlier layer).
        Y-coordinates represent the vertical position within start node chains.
    """
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


def _scale_positions(pos: dict[int, _Point], node_distance: tuple[float, float]) -> dict[int, _Point]:
    """Scale node positions by a distance factor.

    Parameters
    ----------
    pos : dict[int, _Point]
        Mapping of node identifiers to their (x, y) coordinates.
    node_distance : tuple[float, float]
        Scaling factors as (x_scale, y_scale) to apply to each position.

    Returns
    -------
    dict[int, _Point]
        Dictionary mapping node identifiers to scaled (x, y) coordinates.
        If node_distance is (1, 1), returns the original positions as a dict.
    """
    if node_distance != (1, 1):
        pos = {k: (v[0] * node_distance[0], v[1] * node_distance[1]) for k, v in pos.items()}
    return pos


def _compute_edge_paths(og: OpenGraph[AbstractMeasurement], pos: Mapping[int, _Point]) -> dict[_Edge, _Path]:
    """Compute bezier curve paths for all edges in the graph.

    Parameters
    ----------
    og : OpenGraph[AbstractMeasurement]
        The open graph containing nodes and edges to be rendered.
    pos : Mapping[int, _Point]
        Mapping of node identifiers to their (x, y) coordinates.

    Returns
    -------
    dict[_Edge, _Path]
        Dictionary mapping edges to their computed bezier curve paths.

    See Also
    --------
    _find_bezier_path : Computes individual bezier curve paths between node pairs.
    """
    edges = og.graph.edges()
    return {edge: _find_bezier_path(edge, [pos[edge[0]], pos[edge[1]]], pos) for edge in edges}


def _format_corrections_flow(flow: PauliFlow[AbstractMeasurement]) -> set[Colored[_Edge]]:
    """Assign the standard flow color (FLOW_C) to correction function edges (pairs of measured and corrected nodes) in the flow graph.

    Parameters
    ----------
    flow : PauliFlow[AbstractMeasurement]

    Returns
    -------
    set[Colored[_Edge]]
        Set of colored edges.

    Notes
    -----
    See module-level Style constants for actual color values.
    """
    return {Colored((k, v), FLOW_C) for k, values in flow.correction_function.items() for v in values}


def _format_corrections_xz(xz_corr: XZCorrections[AbstractMeasurement]) -> set[Colored[_Edge]]:
    """Assign colors to X and Z correction edges (pairs of measured and corrected nodes) based on correction type.

    Parameters
    ----------
    xz_corr : XZCorrections[AbstractMeasurement]

    Returns
    -------
    set[Colored[_Edge]]
        Set of colored edges representing corrections. Edge colors are assigned as:
        - ``X_C`` : edges in X-corrections only
        - ``Z_C`` : edges in Z-corrections only
        - ``XZ_C`` : edges in both X-corrections and Z-corrections

    Notes
    -----
    See module-level Style constants for actual color values.
    """
    correction_arrows: set[Colored[_Edge]] = set()
    colors = (X_C, Z_C, XZ_C)
    for measured_node in xz_corr.og.measurements:
        x_corr = xz_corr.x_corrections.get(measured_node, set())
        z_corr = xz_corr.z_corrections.get(measured_node, set())
        x_and_z_corr = x_corr & z_corr
        corrs = (x_corr - x_and_z_corr, z_corr - x_and_z_corr, x_and_z_corr)
        for corr_nodes, color in zip(corrs, colors, strict=True):
            correction_arrows.update(Colored((measured_node, corr_node), color) for corr_node in corr_nodes)

    return correction_arrows


def _compute_arrow_paths(
    pos: Mapping[int, _Point], edges: AbstractSet[_Edge], correction_arrows: AbstractSet[Colored[_Edge]]
) -> dict[_Edge, Colored[_Path]]:
    """Compute bezier curve paths for all correction arrows in the graph.

    Parameters
    ----------
    pos : Mapping[int, _Point]
        Mapping of node identifiers to their (x, y) coordinates.
    edges : AbstractSet[_Edge]
        Open graph's edges.
    correction_arrows: AbstractSet[Colored[_Edge]]
        Set of colored edges representing corrections.

    Returns
    -------
    dict[_Edge, _Path]
        Dictionary mapping correction edges to their colored computed bezier curve paths.

    See Also
    --------
    _find_bezier_path : Computes individual bezier curve paths between node pairs.
    """
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
    """Compute a bezier curve path between two nodes, avoiding intersections with other nodes.

    Parameters
    ----------
    arrow : _Edge
        The edge as a (source_node, target_node) pair for which to compute the path.
    bezier_path : Iterable[_Point]
        Initial bezier path as a sequence of (x, y) coordinates.
    pos : Mapping[int, _Point]
        Mapping of all node identifiers to their (x, y) coordinates.

    Returns
    -------
    list[_Point]
        Refined bezier curve path as a list of (x, y) coordinates. The path is
        iteratively adjusted to avoid intersections with intermediate nodes while
        preserving the start and end points.
    """
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
