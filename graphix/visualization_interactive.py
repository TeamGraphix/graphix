"""Interactive visualization for MBQC patterns."""

from __future__ import annotations

import sys
import traceback
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from matplotlib.widgets import Button, Slider

from graphix.clifford import Clifford
from graphix.command import CommandKind
from graphix.opengraph import OpenGraph
from graphix.pretty_print import OutputFormat, command_to_str
from graphix.sim.statevec import StatevectorBackend
from graphix.visualization import GraphVisualizer

if TYPE_CHECKING:
    from graphix.pattern import Pattern


class InteractiveGraphVisualizer:
    """Interactive visualization tool for MBQC patterns.

    Attributes
    ----------
    pattern : Pattern
        The MBQC pattern to visualize.
    node_distance : tuple[float, float]
        Scale factors (x, y) for the node positions.
    enable_simulation : bool
        If True, simulates the state vector and measurement outcomes.
    marker_fill_ratio : float
        Fraction of the inter-node spacing used by each marker (0-1).
    label_size_ratio : float
        Label font size as a fraction of the marker diameter.
    max_label_fontsize : int
        Upper bound for label font size in points.
    min_inches_per_node : float
        Minimum vertical inches per node for adaptive figure height.
    active_node_color : str
        Border colour for active (current-step) nodes.
    measured_node_color : str
        Fill colour for already-measured nodes.
    """

    def __init__(
        self,
        pattern: Pattern,
        node_distance: tuple[float, float] = (1, 1),
        enable_simulation: bool = True,
        *,
        marker_fill_ratio: float = 0.80,
        label_size_ratio: float = 0.55,
        max_label_fontsize: int = 12,
        min_inches_per_node: float = 0.3,
        active_node_color: str = "#2060cc",
        measured_node_color: str = "lightgray",
    ) -> None:
        """Construct an interactive visualizer.

        Parameters
        ----------
        pattern : Pattern
            The MBQC pattern to visualize.
        node_distance : tuple[float, float], optional
            Scale factors (x, y) for node positions. Defaults to (1, 1).
        enable_simulation : bool, optional
            If True, enables state vector simulation. Defaults to True.
        marker_fill_ratio : float, optional
            Fraction of the inter-node spacing used by marker diameter.
            Defaults to 0.80.
        label_size_ratio : float, optional
            Label font size as a fraction of the marker diameter in points.
            Defaults to 0.55.
        max_label_fontsize : int, optional
            Upper bound for label font size.  Prevents text from overflowing
            the marker in sparse graphs. Defaults to 12.
        min_inches_per_node : float, optional
            Minimum vertical inches allocated per node when computing the
            adaptive figure height. Defaults to 0.3.
        active_node_color : str, optional
            Border colour for active nodes. Defaults to ``"#2060cc"``.
        measured_node_color : str, optional
            Fill colour for measured nodes. Defaults to ``"lightgray"``.
        """
        self.pattern = pattern
        self.node_positions: dict[int, tuple[float, float]] = {}
        self.node_distance = node_distance
        self.enable_simulation = enable_simulation
        self.marker_fill_ratio = marker_fill_ratio
        self.label_size_ratio = label_size_ratio
        self.max_label_fontsize = max_label_fontsize
        self.min_inches_per_node = min_inches_per_node
        self.active_node_color = active_node_color
        self.measured_node_color = measured_node_color

        # Prepare graph layout reusing GraphVisualizer
        self._prepare_layout()

        # Figure height and width adapts to graph density like GraphVisualizer
        ax_h_frac = 0.80  # height fraction of ax_graph in figure
        fig_width, needed_height = self._graph_visualizer.determine_figsize(self._l_k, pos=self.node_positions)
        fig_height = max(7, needed_height / ax_h_frac)

        # Compute node marker size and label font size ONCE from the known
        # figure geometry.  This avoids instability when the user resizes the
        # window, because the values are fixed at construction time.
        ax_height_inches = fig_height * ax_h_frac
        if self.node_positions:
            ys = [p[1] for p in self.node_positions.values()]
            y_data_span = max(ys) - min(ys) + 1
        else:
            y_data_span = 1

        y_margin = y_data_span * 0.08 + 0.5  # mirrors _draw_graph margin
        y_range = y_data_span + 2 * y_margin
        points_per_unit = (ax_height_inches / y_range) * 72  # 72 pt/inch
        marker_diameter = self.marker_fill_ratio * points_per_unit
        self.node_size: int = min(400, max(30, int(marker_diameter**2)))
        self.label_fontsize: int = min(self.max_label_fontsize, max(6, int(marker_diameter * self.label_size_ratio)))

        self.fig = plt.figure(figsize=(fig_width, fig_height))

        # Grid layout (Bottom-heavy for controls, max space for graph)
        self.ax_graph = self.fig.add_axes((0.02, 0.20, 0.96, 0.78))
        self.ax_commands = self.fig.add_axes((0.10, 0.10, 0.80, 0.08))

        # Slider and buttons Centred at the very bottom
        self.ax_prev = self.fig.add_axes((0.35, 0.03, 0.04, 0.04))
        self.ax_slider = self.fig.add_axes((0.40, 0.03, 0.20, 0.04))
        self.ax_next = self.fig.add_axes((0.61, 0.03, 0.04, 0.04))

        # Turn off axes frame for command list and graph
        self.ax_commands.axis("off")
        self.ax_graph.axis("off")

        # Interaction state
        self.current_step = 0
        self.total_steps = len(pattern)
        self.command_window_size = 7  # Increased for horizontal viewing

        # Widget placeholders
        self.slider: Slider | None = None
        self.btn_prev: Button | None = None
        self.btn_next: Button | None = None

    def _prepare_layout(self) -> None:
        """Compute node positions by reusing :class:`GraphVisualizer` layout.

        Builds the full graph from the pattern commands, delegates layout
        computation to :meth:`GraphVisualizer.get_layout`, and normalizes
        the resulting positions to fit the interactive panel area.
        The flow-based layout is always preserved.
        """
        # Build the full graph from all commands
        g: Any = __import__("networkx").Graph()
        measurements: dict[int, Any] = {}
        for cmd in self.pattern:
            if cmd.kind == CommandKind.N:
                g.add_node(cmd.node)
            elif cmd.kind == CommandKind.E:
                g.add_edge(cmd.nodes[0], cmd.nodes[1])
            elif cmd.kind == CommandKind.M:
                measurements[cmd.node] = cmd.measurement

        # Delegate layout to GraphVisualizer (shares flow-detection logic)
        og = OpenGraph(g, self.pattern.input_nodes, self.pattern.output_nodes, measurements)
        og = og.infer_pauli_measurements()

        vis = GraphVisualizer(og)
        pos_mapping, self._place_paths, self._l_k = vis.get_layout()
        self.node_positions = dict(pos_mapping)

        # Apply user-provided scaling
        self.node_positions = {
            k: (v[0] * self.node_distance[0], v[1] * self.node_distance[1]) for k, v in self.node_positions.items()
        }
        # Store the visualizer for reuse in drawing helpers
        self._graph_visualizer = vis

    def visualize(self) -> None:
        """Launch the interactive visualization window."""
        # Initial draw
        self._draw_command_list()
        self._draw_graph()
        self._update(0)

        # Step slider (horizontal, bottom centered, without label text)
        self.slider = Slider(self.ax_slider, "", 0, self.total_steps, valinit=0, valstep=1, color="lightblue")
        self.slider.valtext.set_visible(False)
        self.slider.on_changed(self._update)

        # Buttons config
        self.btn_prev = Button(self.ax_prev, "<")
        self.btn_prev.on_clicked(self._prev_step)

        self.btn_next = Button(self.ax_next, ">")
        self.btn_next.on_clicked(self._next_step)

        # Key events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Pick events for command list
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

        plt.show()

    def _draw_command_list(self) -> None:
        self.ax_commands.clear()
        self.ax_commands.axis("off")

        # Use current step as center of the visible window
        half_window = self.command_window_size // 2
        start = max(0, self.current_step - half_window)
        end = min(self.total_steps, self.current_step + half_window + 1)

        cmds: Any = self.pattern[start:end]  # type: ignore[index]

        # Horizontal layout calculation
        num_visible = len(cmds)
        if num_visible == 0:
            return

        spacing = 1.0 / self.command_window_size

        for i, cmd in enumerate(cmds):
            abs_idx = start + i
            text_str = command_to_str(cmd, OutputFormat.Unicode)

            color = "gray"
            weight = "normal"
            fontsize = 11

            if abs_idx == self.current_step:
                color = "black"
                weight = "bold"
                fontsize = 13
            elif abs_idx > self.current_step:
                color = "lightgray"

            # Horizontal placement, relative to center (0.5)
            offset = abs_idx - self.current_step
            x_pos = 0.5 + offset * spacing

            if 0.0 <= x_pos <= 1.0:
                text_obj = self.ax_commands.text(
                    x_pos,
                    0.5,
                    text_str,
                    color=color,
                    weight=weight,
                    fontsize=fontsize,
                    transform=self.ax_commands.transAxes,
                    ha="center",
                    va="center",
                    picker=True,
                )
                text_obj.index = abs_idx  # type: ignore[attr-defined]

    def _update_graph_state(
        self, step: int
    ) -> tuple[set[int], set[int], list[tuple[int, int]], dict[int, set[str]], dict[int, int]]:
        """Calculate the graph state by simulating the pattern up to *step*.

        Parameters
        ----------
        step : int
            The command index up to which the pattern is executed.

        Returns
        -------
        active_nodes : set[int]
            Nodes that have been initialised but not yet measured.
        measured_nodes : set[int]
            Nodes that have been measured.
        active_edges : list[tuple[int, int]]
            Edges currently present in the graph (both endpoints active).
        corrections : dict[int, set[str]]
            Accumulated byproduct corrections per node (``"X"`` and/or ``"Z"``).
        results : dict[int, int]
            Measurement outcomes keyed by node (only populated when
            *enable_simulation* is ``True``).
        """
        active_nodes = set()
        measured_nodes = set()
        active_edges = []
        corrections: dict[int, set[str]] = {}
        results: dict[int, int] = {}

        if self.enable_simulation:
            backend = StatevectorBackend()

            # Prerun input nodes (standard MBQC initialization)
            for node in self.pattern.input_nodes:
                backend.add_nodes([node])

            rng = np.random.default_rng(42)  # Fixed seed for determinism

            for i in range(step):
                cmd = self.pattern[i]
                if cmd.kind == CommandKind.N:
                    backend.add_nodes([cmd.node], data=cmd.state)
                elif cmd.kind == CommandKind.E:
                    backend.entangle_nodes(cmd.nodes)
                elif cmd.kind == CommandKind.M:
                    # Adaptive measurement (feedforward)
                    s_signal = sum(results.get(j, 0) for j in cmd.s_domain) if cmd.s_domain else 0
                    t_signal = sum(results.get(j, 0) for j in cmd.t_domain) if cmd.t_domain else 0

                    clifford = Clifford.I
                    if s_signal % 2 == 1:
                        clifford = Clifford.X @ clifford
                    if t_signal % 2 == 1:
                        clifford = Clifford.Z @ clifford

                    measurement = cmd.measurement.clifford(clifford)
                    result = backend.measure(cmd.node, measurement, rng=rng)
                    results[cmd.node] = result
                elif cmd.kind == CommandKind.X:
                    if cmd.node not in corrections:
                        corrections[cmd.node] = set()
                    corrections[cmd.node].add("X")
                    backend.correct_byproduct(cmd)
                elif cmd.kind == CommandKind.Z:
                    if cmd.node not in corrections:
                        corrections[cmd.node] = set()
                    corrections[cmd.node].add("Z")
                    backend.correct_byproduct(cmd)

        # ---- Topological tracking (independent of simulation) ----
        current_active: set[int] = set(self.pattern.input_nodes)
        current_edges: set[tuple[int, int]] = set()
        current_measured: set[int] = set()

        for i in range(step):
            cmd = self.pattern[i]
            if cmd.kind == CommandKind.N:
                current_active.add(cmd.node)
            elif cmd.kind == CommandKind.E:
                u, v = cmd.nodes
                if u in current_active and v in current_active:
                    current_edges.add((min(u, v), max(u, v)))
            elif cmd.kind == CommandKind.M and cmd.node in current_active:
                current_active.remove(cmd.node)
                current_measured.add(cmd.node)
                current_edges = {e for e in current_edges if cmd.node not in e}

        active_nodes = current_active
        measured_nodes = current_measured
        active_edges = list(current_edges)

        return active_nodes, measured_nodes, active_edges, corrections, results

    def _draw_graph(self) -> None:
        """Draw nodes and edges onto the graph axes.

        Delegates to :class:`GraphVisualizer` for edge and node rendering,
        passing per-node colour overrides to distinguish measured (grey) from
        active (blue border) nodes.  Labels are drawn locally because they
        include dynamic content (measurement results, corrections).
        """
        try:
            self.ax_graph.clear()

            active_nodes, measured_nodes, active_edges, corrections, results = self._update_graph_state(
                self.current_step
            )

            # Highlight logic
            highlight_nodes: set[int] = set()
            highlight_edges: set[tuple[int, int]] = set()
            if self.current_step > 0:
                last_cmd = self.pattern[self.current_step - 1]
                if last_cmd.kind in {CommandKind.N, CommandKind.M, CommandKind.C, CommandKind.X, CommandKind.Z}:
                    highlight_nodes.add(last_cmd.node)  # type: ignore[union-attr]
                elif last_cmd.kind == CommandKind.E:
                    highlight_nodes.update(last_cmd.nodes)
                    highlight_edges.add(last_cmd.nodes)

            # Axis limits (set before drawing nodes so geometry is known)
            xs = [p[0] for p in self.node_positions.values()]
            ys = [p[1] for p in self.node_positions.values()]
            x_margin = (max(xs) - min(xs)) * 0.08 + 0.5
            y_margin = (max(ys) - min(ys)) * 0.08 + 0.5
            self.ax_graph.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
            self.ax_graph.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

            # Layer separators
            if self._l_k is not None:
                self._graph_visualizer.draw_layer_separators(
                    self.ax_graph, self.node_positions, self._l_k, self.node_distance
                )

            # Edges and arrows
            edge_path, arrow_path = self._place_paths(self.node_positions)

            edge_colors: dict[tuple[int, int], str] = {}
            edge_linewidths: dict[tuple[int, int], float] = {}
            for edge in highlight_edges:
                e_sorted = (min(edge), max(edge))
                edge_colors[e_sorted] = "black"
                edge_linewidths[e_sorted] = 2.0

            if arrow_path is not None:
                self._graph_visualizer.draw_edges_with_routing(
                    self.ax_graph,
                    edge_path,
                    edge_subset=None,
                    edge_colors=edge_colors,
                    edge_linewidths=edge_linewidths,
                )
                self._graph_visualizer.draw_flow_arrows(
                    self.ax_graph, self.node_positions, arrow_path, arrow_subset=active_edges
                )
            else:
                self._graph_visualizer.draw_edges_with_routing(
                    self.ax_graph,
                    edge_path,
                    edge_subset=None,
                    edge_colors=edge_colors,
                    edge_linewidths=edge_linewidths,
                )

            # Nodes
            node_facecolors: dict[int, str] = {}
            node_edgecolors: dict[int, str] = {}
            node_alpha: dict[int, float] = {}
            node_linewidths: dict[int, float] = {}

            # Don't fade out or override colors to keep exact GraphVisualizer look
            for node in highlight_nodes:
                if node not in self._graph_visualizer.og.input_nodes:
                    node_edgecolors[node] = "black"
                node_linewidths[node] = 2.0

            self._graph_visualizer.draw_nodes_role(
                self.ax_graph,
                self.node_positions,
                node_facecolors=node_facecolors,
                node_edgecolors=node_edgecolors,
                node_alpha=node_alpha,
                node_linewidths=node_linewidths,
                node_size=self.node_size,
            )

            # Labels

            # Show "XY", "XZ" etc for non-measured output via the underlying graph logic
            # Offset values logic matches static visualization.py exactly
            for node in self.node_positions:
                if node not in measured_nodes and node in self._graph_visualizer.og.measurements:
                    meas = self._graph_visualizer.og.measurements[node]
                    plane = meas.to_plane_or_axis().name
                    if isinstance(plane, str):
                        x, y = self.node_positions[node]
                        self.ax_graph.text(x + 0.22, y - 0.2, plane, fontsize=self.label_fontsize - 2, zorder=3)

            for node in measured_nodes:
                if node in results:
                    x, y = self.node_positions[node]
                    self.ax_graph.text(
                        x + 0.22, y - 0.2, f"m={results[node]}", fontsize=self.label_fontsize - 2, zorder=3
                    )
            for node in active_nodes:
                if node in corrections:
                    lbl = "".join(sorted(corrections[node]))
                    if lbl:
                        x, y = self.node_positions[node]
                        self.ax_graph.text(x + 0.22, y - 0.2, lbl, fontsize=self.label_fontsize - 2, zorder=3)

            self._graph_visualizer.draw_node_labels(
                self.ax_graph, self.node_positions, extra_labels=None, fontsize=self.label_fontsize
            )

            self.ax_graph.axis("off")

        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print(f"Error drawing graph: {e}", file=sys.stderr)

    def _update(self, val: float) -> None:
        step = int(val)
        if step != self.current_step:
            self.current_step = step
            self._draw_command_list()
            self._draw_graph()
            self.fig.canvas.draw_idle()

    def _prev_step(self, _event: Any) -> None:
        if self.current_step > 0 and self.slider is not None:
            self.slider.set_val(self.current_step - 1)

    def _next_step(self, _event: Any) -> None:
        if self.current_step < self.total_steps and self.slider is not None:
            self.slider.set_val(self.current_step + 1)

    def _on_key(self, event: Any) -> None:
        if event.key == "right":
            self._next_step(None)
        elif event.key == "left":
            self._prev_step(None)

    def _on_pick(self, event: Any) -> None:
        if isinstance(event.artist, Text):
            idx = getattr(event.artist, "index", None)
            if idx is not None and self.slider is not None:
                self.slider.set_val(idx + 1)  # Jump to state AFTER the clicked command
