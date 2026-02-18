"""Interactive visualization for MBQC patterns."""

from __future__ import annotations

import sys
import traceback
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import networkx as nx
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
    """

    def __init__(
        self,
        pattern: Pattern,
        node_distance: tuple[float, float] = (1, 1),
        enable_simulation: bool = True,
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
        """
        self.pattern = pattern
        self.node_positions: dict[int, tuple[float, float]] = {}
        self.node_distance = node_distance
        self.enable_simulation = enable_simulation

        # Prepare graph layout reusing GraphVisualizer
        self._prepare_layout()

        # Figure setup - tighter layout to reduce whitespace
        self.fig = plt.figure(figsize=(14, 7))

        # Grid layout: command list (~28%), graph (~67%), bottom strip for controls
        self.ax_commands = self.fig.add_axes((0.02, 0.15, 0.27, 0.80))
        self.ax_graph = self.fig.add_axes((0.32, 0.15, 0.65, 0.80))
        self.ax_prev = self.fig.add_axes((0.30, 0.04, 0.03, 0.03))
        self.ax_slider = self.fig.add_axes((0.34, 0.04, 0.55, 0.03))
        self.ax_next = self.fig.add_axes((0.90, 0.04, 0.03, 0.03))

        # Turn off axes frame for command list and graph
        self.ax_commands.axis("off")
        self.ax_graph.axis("off")

        # Interaction state
        self.current_step = 0
        self.total_steps = len(pattern)

        # Widget placeholders
        self.slider: Slider | None = None
        self.btn_prev: Button | None = None
        self.btn_next: Button | None = None

    def _prepare_layout(self) -> None:
        """Compute node positions by reusing :class:`GraphVisualizer` layout.

        Builds the full graph from the pattern commands, delegates layout
        computation to :meth:`GraphVisualizer.get_layout`, and normalizes
        the resulting positions to fit the interactive panel area.  If the
        flow-based layout is too narrow for comfortable display (e.g. a
        deep Pauli-flow graph), a spring-layout fallback is used.
        """
        # Build the full graph from all commands
        g: Any = nx.Graph()
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
        pos_mapping, _, _ = vis.get_layout()
        self.node_positions = dict(pos_mapping)

        # Check if the layout is too narrow for the interactive panel
        x_coords = [p[0] for p in self.node_positions.values()]
        y_coords = [p[1] for p in self.node_positions.values()]
        if x_coords and y_coords:
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            aspect = x_range / max(y_range, 1e-6)
            if aspect < 0.3 and len(self.node_positions) > 5:
                # Layout is too narrow (tall vertical strip) -- use spring layout
                pos_spring = nx.spring_layout(g, seed=42)
                self.node_positions = {n: (float(p[0]), float(p[1])) for n, p in pos_spring.items()}

        # Apply user-provided scaling
        self.node_positions = {
            k: (v[0] * self.node_distance[0], v[1] * self.node_distance[1]) for k, v in self.node_positions.items()
        }

        # Normalize to [0, 1] range so positions fill the available axes area
        # regardless of the data's original aspect ratio.
        self._normalize_positions()

        # Store the visualizer for reuse in drawing helpers
        self._graph_visualizer = vis

    def _normalize_positions(self) -> None:
        """Normalize node positions into the ``[margin, 1-margin]`` range.

        This ensures the graph fills the interactive axes area uniformly,
        avoiding the distortion caused by ``set_aspect("equal")`` when the
        data's x/y ranges differ significantly.
        """
        if not self.node_positions:
            return

        xs = [p[0] for p in self.node_positions.values()]
        ys = [p[1] for p in self.node_positions.values()]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        margin = 0.08
        lo = margin
        hi = 1.0 - margin

        self.node_positions = {
            k: (lo + (v[0] - x_min) / x_range * (hi - lo), lo + (v[1] - y_min) / y_range * (hi - lo))
            for k, v in self.node_positions.items()
        }

    def visualize(self) -> None:
        """Launch the interactive visualization window."""
        # Initial draw
        self._draw_command_list()
        self._draw_graph()
        self._update(0)

        # Slider config
        self.slider = Slider(self.ax_slider, "Step", 0, self.total_steps, valinit=0, valstep=1, color="lightblue")
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
        self.ax_commands.set_title(f"Commands ({self.total_steps})", loc="left")

        # Windowing logic to show relevant commands
        window_size = 30
        start = max(0, int(self.current_step) - window_size // 2)
        end = min(self.total_steps, start + window_size)

        if end == self.total_steps:
            start = max(0, end - window_size)

        cmds: Any = self.pattern[start:end]  # type: ignore[index]

        for i, cmd in enumerate(cmds):
            abs_idx = start + i
            text_str = f"{abs_idx}: {command_to_str(cmd, OutputFormat.Unicode)}"

            color = "black"
            weight = "normal"
            if abs_idx < self.current_step:
                color = "green"
            elif abs_idx == self.current_step:
                color = "red"
                weight = "bold"

            # Position text from top to bottom
            y_pos = 1.0 - (i + 1) * (1.0 / (window_size + 2))

            text_obj = self.ax_commands.text(
                0.05,
                y_pos,
                text_str,
                color=color,
                weight=weight,
                fontsize=10,
                transform=self.ax_commands.transAxes,
                picker=True,
            )
            # Store index with artist for picking
            text_obj.index = abs_idx  # type: ignore[attr-defined]

    def _update_graph_state(
        self, step: int
    ) -> tuple[set[int], set[int], list[tuple[int, ...]], dict[int, set[str]], dict[int, int]]:
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
        active_edges : list[tuple[int, ...]]
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
        current_edges: set[tuple[int, ...]] = set()
        current_measured: set[int] = set()

        for i in range(step):
            cmd = self.pattern[i]
            if cmd.kind == CommandKind.N:
                current_active.add(cmd.node)
            elif cmd.kind == CommandKind.E:
                u, v = cmd.nodes
                if u in current_active and v in current_active:
                    current_edges.add(tuple(sorted((u, v))))
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

        Uses :meth:`GraphVisualizer.draw_edges` for edge rendering (shared
        with the static visualizer) and draws nodes with interactive-specific
        colouring: grey for measured, red border for active.
        """
        try:
            self.ax_graph.clear()

            active_nodes, measured_nodes, active_edges, corrections, results = self._update_graph_state(
                self.current_step
            )

            # ---- Edges (reuse GraphVisualizer helper if possible) ----
            for u, v in active_edges:
                if u in self.node_positions and v in self.node_positions:
                    x1, y1 = self.node_positions[u]
                    x2, y2 = self.node_positions[v]
                    self.ax_graph.plot([x1, x2], [y1, y2], color="black", alpha=0.7, zorder=1)

            # Adaptive font-size: shrink labels when node numbers are large
            fontsize = 10
            max_node = max(
                (n for ns in (active_nodes, measured_nodes) for n in ns),
                default=0,
            )
            if max_node >= 100:
                fontsize = max(7, int(fontsize * 2 / len(str(max_node))))

            # ---- Measured nodes (grey fill, black border) ----
            for node in measured_nodes:
                if node not in self.node_positions:
                    continue
                x, y = self.node_positions[node]
                self.ax_graph.scatter(x, y, edgecolors="black", facecolors="lightgray", s=350, zorder=2, linewidths=1.5)

                label_text = str(node)
                if node in results:
                    label_text += f"\nm={results[node]}"

                self.ax_graph.text(x, y, label_text, ha="center", va="center", fontsize=fontsize, zorder=3)

            # ---- Active nodes (white fill, red border) ----
            for node in active_nodes:
                if node not in self.node_positions:
                    continue
                x, y = self.node_positions[node]
                self.ax_graph.scatter(x, y, edgecolors="red", facecolors="white", s=350, zorder=2, linewidths=1.5)

                label_text = str(node)
                if node in corrections:
                    label_text += "\n" + "".join(sorted(corrections[node]))

                text_color = "blue" if node in corrections else "black"
                self.ax_graph.text(
                    x, y, label_text, ha="center", va="center", fontsize=fontsize, color=text_color, zorder=3
                )

            # Axis limits use normalized [0, 1] positions - no set_aspect("equal")
            self.ax_graph.set_xlim(-0.02, 1.02)
            self.ax_graph.set_ylim(-0.02, 1.02)
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
