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
        ax_h_frac = 0.80
        fig_width, needed_height = self._graph_visualizer.determine_figsize(self._l_k, pos=self.node_positions)
        fig_height = max(7.0, needed_height / ax_h_frac)

        # Enforce minimum window width for controls
        fig_width = max(4.0, fig_width)

        self.fig = plt.figure(figsize=(fig_width, fig_height))

        # Dynamically scale command strip capacity down on narrow graphs
        self.command_window_size = max(1, min(7, int(fig_width / 1.8)))

        # Ensure command_window_size is always an odd number so the current command is perfectly centered.
        if self.command_window_size % 2 == 0:
            self.command_window_size -= 1

        self.node_size = 350
        self.label_fontsize = 10

        # Axes layout fractions [left, bottom, width, height]
        self.ax_graph = self.fig.add_axes((0.00, 0.15, 1.0, 0.85))
        self.ax_commands = self.fig.add_axes((0.10, 0.05, 0.80, 0.08))

        self.ax_prev = self.fig.add_axes((0.32, 0.015, 0.04, 0.04))
        self.ax_slider = self.fig.add_axes((0.40, 0.015, 0.20, 0.04))
        self.ax_next = self.fig.add_axes((0.64, 0.015, 0.04, 0.04))

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
        # Initial state simulation
        state = self._update_graph_state(0)

        # Initial draw
        self._draw_command_list(state[4])  # pass results dict
        self._draw_graph(state)
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

    def _draw_command_list(self, results: dict[int, int]) -> None:
        self.ax_commands.clear()
        self.ax_commands.axis("off")

        # Use current step as center of the visible window
        half_window = self.command_window_size // 2
        start = max(0, self.current_step - half_window)
        end = min(self.total_steps, self.current_step + half_window + 1)

        def _get_props(abs_idx: int, cmd: Any) -> tuple[str, str, str, str, int, float]:
            text_str = command_to_str(cmd, OutputFormat.Unicode)
            meas_str = ""
            if cmd.kind == CommandKind.M and abs_idx <= self.current_step and cmd.node in results:
                meas_str = f"m={results[cmd.node]}"

            color = "gray"
            weight = "normal"
            fontsize = 11
            alpha = 1.0

            if abs_idx == self.current_step:
                color = "black"
                weight = "bold"
                fontsize = 13
                alpha = 1.0
            elif abs_idx < self.current_step:
                color = "black"
                alpha = 0.4
            elif abs_idx > self.current_step:
                color = "lightgray"
                alpha = 0.7

            return text_str, meas_str, color, weight, fontsize, alpha

        artists: dict[int, Any] = {}

        # Handle out-of-bounds slider focus
        focus_idx = min(self.current_step, end - 1)
        if focus_idx < start:
            return

        cmd = self.pattern[focus_idx]
        txt, meas_str, color, weight, fsize, alpha = _get_props(focus_idx, cmd)
        artists[focus_idx] = self.ax_commands.text(
            0.5, 0.5, txt,
            color=color, weight=weight, fontsize=fsize, alpha=alpha,
            transform=self.ax_commands.transAxes,
            ha="center", va="center", picker=True,
            clip_on=True,
        )
        artists[focus_idx].index = focus_idx  # type: ignore[attr-defined]

        if meas_str:
            self.ax_commands.annotate(
                meas_str,
                xy=(0.5, 1.0), xycoords=artists[focus_idx],
                xytext=(0, 2), textcoords="offset points",
                color=color, fontsize=10, alpha=alpha,
                ha="center", va="bottom",
                annotation_clip=True, clip_on=True,
            )

        # Draw past commands
        prev_idx = focus_idx
        for abs_idx in range(focus_idx - 1, start - 1, -1):
            cmd = self.pattern[abs_idx]
            txt, meas_str, color, weight, fsize, alpha = _get_props(abs_idx, cmd)

            artists[abs_idx] = self.ax_commands.annotate(
                txt,
                xy=(0, 0.5), xycoords=artists[prev_idx],
                xytext=(-4, 0), textcoords="offset points",
                color=color, weight=weight, fontsize=fsize, alpha=alpha,
                ha="right", va="center", picker=True,
                annotation_clip=True, clip_on=True,
            )
            artists[abs_idx].index = abs_idx  # type: ignore[attr-defined]

            if meas_str:
                self.ax_commands.annotate(
                    meas_str,
                    xy=(0.5, 1.0), xycoords=artists[abs_idx],
                    xytext=(0, 2), textcoords="offset points",
                    color=color, fontsize=9, alpha=alpha,
                    ha="center", va="bottom",
                    annotation_clip=True, clip_on=True,
                )

            prev_idx = abs_idx

        # Draw future commands
        prev_idx = focus_idx
        for abs_idx in range(focus_idx + 1, end):
            cmd = self.pattern[abs_idx]
            txt, meas_str, color, weight, fsize, alpha = _get_props(abs_idx, cmd)

            artists[abs_idx] = self.ax_commands.annotate(
                txt,
                xy=(1, 0.5), xycoords=artists[prev_idx],
                xytext=(4, 0), textcoords="offset points",
                color=color, weight=weight, fontsize=fsize, alpha=alpha,
                ha="left", va="center", picker=True,
                annotation_clip=True, clip_on=True,
            )
            artists[abs_idx].index = abs_idx  # type: ignore[attr-defined]

            if meas_str:
                self.ax_commands.annotate(
                    meas_str,
                    xy=(0.5, 1.0), xycoords=artists[abs_idx],
                    xytext=(0, 2), textcoords="offset points",
                    color=color, fontsize=9, alpha=alpha,
                    ha="center", va="bottom",
                    annotation_clip=True, clip_on=True,
                )

            prev_idx = abs_idx

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

            for i in range(min(step + 1, len(self.pattern))):
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

        for i in range(min(step + 1, len(self.pattern))):
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

        active_nodes = current_active
        measured_nodes = current_measured
        active_edges = list(current_edges)

        return active_nodes, measured_nodes, active_edges, corrections, results

    def _draw_graph(
        self, state: tuple[set[int], set[int], list[tuple[int, int]], dict[int, set[str]], dict[int, int]]
    ) -> None:
        """Draw nodes and edges onto the graph axes.

        Delegates to :class:`GraphVisualizer` for edge and node rendering,
        passing per-node colour overrides to distinguish measured (grey) from
        active (blue border) nodes.  Labels are drawn locally because they
        include dynamic content (measurement results, corrections).
        """
        try:
            self.ax_graph.clear()

            active_nodes, measured_nodes, active_edges, corrections, results = state

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

            # Axis limits
            xs = [p[0] for p in self.node_positions.values()]
            ys = [p[1] for p in self.node_positions.values()]
            if xs and ys:
                self.ax_graph.set_xlim(min(xs) - 0.1 * self.node_distance[0], max(xs) + 0.1 * self.node_distance[0])
                self.ax_graph.set_ylim(min(ys) - 0.4, max(ys) + 0.4)

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
                    edge_subset=active_edges,
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
                    edge_subset=active_edges,
                    edge_colors=edge_colors,
                    edge_linewidths=edge_linewidths,
                )

            # Nodes
            node_facecolors: dict[int, str] = {}
            node_edgecolors: dict[int, str] = {}
            node_alpha: dict[int, float] = {}
            node_linewidths: dict[int, float] = {}
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
            label_offset_pts = (14, -10)

            # Show "XY", "XZ" etc for non-measured nodes
            for node in self.node_positions:
                if node not in measured_nodes and node in self._graph_visualizer.og.measurements:
                    meas = self._graph_visualizer.og.measurements[node]
                    plane = meas.to_plane_or_axis().name
                    if isinstance(plane, str):
                        xy = self.node_positions[node]
                        self.ax_graph.annotate(
                            plane,
                            xy=xy,
                            xytext=label_offset_pts,
                            textcoords="offset points",
                            fontsize=9,
                            zorder=3,
                        )

            for node in measured_nodes:
                if node in results:
                    xy = self.node_positions[node]
                    self.ax_graph.annotate(
                        f"m={results[node]}",
                        xy=xy,
                        xytext=label_offset_pts,
                        textcoords="offset points",
                        fontsize=9,
                        zorder=3,
                    )

            for node in active_nodes:
                if node in corrections:
                    lbl = "".join(sorted(corrections[node]))
                    if lbl:
                        xy = self.node_positions[node]
                        self.ax_graph.annotate(
                            lbl,
                            xy=xy,
                            xytext=label_offset_pts,
                            textcoords="offset points",
                            fontsize=9,
                            zorder=3,
                        )

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

            # Fetch state once per tick to feed both visual layers
            state = self._update_graph_state(self.current_step)
            results = state[4]

            self._draw_command_list(results)
            self._draw_graph(state)
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
