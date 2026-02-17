"""Interactive visualization for MBQC patterns."""

from __future__ import annotations

import sys
import traceback
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle
from matplotlib.text import Text
from matplotlib.widgets import Button, Slider

from graphix.clifford import Clifford
from graphix.command import CommandKind, MeasureUpdate
from graphix.measurements import Measurement
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

        # Prepare graph layout using Graphix's visualizer or fallbacks
        self._prepare_layout()

        # Figure setup
        self.fig = plt.figure(figsize=(15, 8))

        # Grid layout: Command list on left, Graph on right
        # Layout optimized to prevent overlap:
        # Commands: Left 2% to 30%
        # Graph: Left 40% to 98%
        self.ax_commands = self.fig.add_axes((0.02, 0.2, 0.28, 0.7))  # [left, bottom, width, height]
        self.ax_graph = self.fig.add_axes((0.4, 0.2, 0.58, 0.7))
        self.ax_slider = self.fig.add_axes((0.4, 0.05, 0.5, 0.03))
        self.ax_prev = self.fig.add_axes((0.3, 0.05, 0.04, 0.04))
        self.ax_next = self.fig.add_axes((0.92, 0.05, 0.04, 0.04))

        # Turn off axes frame for command list and graph
        self.ax_commands.axis("off")
        self.ax_graph.axis("off")  # Start hidden to avoid "square" artifact

        # Interaction state
        self.current_step = 0
        self.total_steps = len(pattern)

        # Interaction state placeholders
        self.slider: Slider | None = None
        self.btn_prev: Button | None = None
        self.btn_next: Button | None = None

    def _prepare_layout(self) -> None:
        # Build full graph to determine positions
        g: Any = nx.Graph()
        for cmd in self.pattern:
            if cmd.kind == CommandKind.N:
                g.add_node(cmd.node)
            elif cmd.kind == CommandKind.E:
                g.add_edge(cmd.nodes[0], cmd.nodes[1])

        # Use GraphVisualizer to determine positions based on flow/structure
        vis = GraphVisualizer(g, self.pattern.input_nodes, self.pattern.output_nodes)
        pos_mapping, _, _ = vis.get_layout()
        self.node_positions = dict(pos_mapping)

        # Apply scaling
        self.node_positions = {
            k: (v[0] * self.node_distance[0], v[1] * self.node_distance[1]) for k, v in self.node_positions.items()
        }

        # Determine fixed bounds for the graph to prevent autoscaling issues
        all_x = [pos[0] for pos in self.node_positions.values()]
        all_y = [pos[1] for pos in self.node_positions.values()]
        margin = 0.5
        self.x_limits = (min(all_x) - margin, max(all_x) + margin)
        self.y_limits = (min(all_y) - margin, max(all_y) + margin)

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
        """Calculate the graph state by simulating the pattern up to `step`."""
        # Prepare return containers
        active_nodes = set()
        measured_nodes = set()
        active_edges = []
        corrections: dict[int, set[str]] = {}
        results: dict[int, int] = {}

        if self.enable_simulation:
            # --- Simulation Mode ---
            backend = StatevectorBackend()

            # Prerun input nodes (standard MBQC initialization)
            input_nodes = self.pattern.input_nodes
            for node in input_nodes:
                backend.add_nodes([node])

            rng = np.random.default_rng(42)  # Fixed seed for determinism

            # Re-execute commands up to current step
            for i in range(step):
                cmd = self.pattern[i]
                if cmd.kind == CommandKind.N:
                    backend.add_nodes([cmd.node], data=cmd.state)
                elif cmd.kind == CommandKind.E:
                    backend.entangle_nodes(cmd.nodes)
                elif cmd.kind == CommandKind.M:
                    # --- Adaptive Measurement Logic (Feedforward) ---
                    # Calculate s and t signals from previous measurement results
                    s_signal = sum(results.get(j, 0) for j in cmd.s_domain) if cmd.s_domain else 0
                    t_signal = sum(results.get(j, 0) for j in cmd.t_domain) if cmd.t_domain else 0

                    s_bool = s_signal % 2 == 1
                    t_bool = t_signal % 2 == 1

                    # Compute the updated angle and plane based on signals
                    measure_update = MeasureUpdate.compute(cmd.plane, s_bool, t_bool, Clifford.I)

                    new_angle = cmd.angle * measure_update.coeff + measure_update.add_term
                    new_plane = measure_update.new_plane

                    # Execute measurement on the backend using the adapted measurement
                    measurement = Measurement(new_angle, new_plane)
                    result = backend.measure(cmd.node, measurement, rng=rng)
                    results[cmd.node] = result
                elif cmd.kind == CommandKind.X:
                    # Accumulate X corrections
                    if cmd.node not in corrections:
                        corrections[cmd.node] = set()
                    corrections[cmd.node].add("X")
                    backend.correct_byproduct(cmd)
                elif cmd.kind == CommandKind.Z:
                    if cmd.node not in corrections:
                        corrections[cmd.node] = set()
                    corrections[cmd.node].add("Z")
                    backend.correct_byproduct(cmd)

        # --- Common Logic (Topological Tracking) ---
        # We track nodes/edges based on command history regardless of simulation
        # This ensures visualization works even if simulation is disabled

        # Reset tracking
        current_active_nodes = set(self.pattern.input_nodes)  # Start with input nodes
        current_edges = set()
        current_measured_nodes = set()  # Track measured nodes for topological view

        for i in range(step):
            cmd = self.pattern[i]
            if cmd.kind == CommandKind.N:
                current_active_nodes.add(cmd.node)
            elif cmd.kind == CommandKind.E:
                u, v = cmd.nodes
                # Only add edge if both nodes are currently active (not yet measured)
                if u in current_active_nodes and v in current_active_nodes:
                    current_edges.add(tuple(sorted((u, v))))
            elif cmd.kind == CommandKind.M and cmd.node in current_active_nodes:
                current_active_nodes.remove(cmd.node)
                current_measured_nodes.add(cmd.node)
                # Remove connected edges involving the measured node
                current_edges = {e for e in current_edges if cmd.node not in e}

            # Corrections are visualization-only metadata, handled in simulation block or ignored

        active_nodes = current_active_nodes
        measured_nodes = current_measured_nodes
        active_edges = list(current_edges)

        return active_nodes, measured_nodes, active_edges, corrections, results

    def _draw_graph(self) -> None:
        try:
            self.ax_graph.clear()

            # Get current state from simulation
            active_nodes, measured_nodes, active_edges, corrections, results = self._update_graph_state(
                self.current_step
            )

            # Draw edges
            for u, v in active_edges:
                x1, y1 = self.node_positions[u]
                x2, y2 = self.node_positions[v]
                self.ax_graph.plot([x1, x2], [y1, y2], color="black", zorder=1)

            # Draw nodes
            # 1. Measured nodes (grey, with result text)
            for node in measured_nodes:
                if node in self.node_positions:
                    x, y = self.node_positions[node]
                    circle = Circle((x, y), 0.1, color="lightgray", zorder=2)
                    self.ax_graph.add_patch(circle)

                    label_text = str(node)
                    # Show measurement outcome if available
                    if node in results:
                        label_text += f"\n={results[node]}"

                    self.ax_graph.text(x, y, label_text, ha="center", va="center", fontsize=9, zorder=3)

            # 2. Active nodes (white with colored edge, with correction text)
            for node in active_nodes:
                if node in self.node_positions:
                    x, y = self.node_positions[node]
                    circle = Circle((x, y), 0.1, edgecolor="red", facecolor="white", linewidth=1.5, zorder=2)
                    self.ax_graph.add_patch(circle)

                    label_text = str(node)
                    # Show accumulated internal corrections
                    if node in corrections:
                        label_text += "\n" + "".join(sorted(corrections[node]))

                    color = "black"
                    if node in corrections:
                        color = "blue"  # Highlight corrected nodes

                    self.ax_graph.text(x, y, label_text, ha="center", va="center", fontsize=9, color=color, zorder=3)

            # Set aspect close to equal and hide axes
            self.ax_graph.set_aspect("equal")
            self.ax_graph.set_xlim(self.x_limits)
            self.ax_graph.set_ylim(self.y_limits)
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
