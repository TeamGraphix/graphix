"""Interactive visualization for MBQC patterns."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.widgets import Button, Slider

from graphix.clifford import Clifford
from graphix.command import CommandKind, MeasureUpdate
from graphix.measurements import Measurement
from graphix.pattern import Pattern
from graphix.pretty_print import OutputFormat, command_to_str
from graphix.sim.statevec import StatevectorBackend
from graphix.visualization import GraphVisualizer

if TYPE_CHECKING:
    from collections.abc import Collection


class InteractiveGraphVisualizer:
    """
    Interactive visualization tool for MBQC patterns.

    This visualizer provides a matplotlib-based GUI to step through the execution
    of an MBQC pattern. It displays the sequence of commands and the corresponding
    state of the graph state, including real-time simulation of measurement outcomes.

    Attributes
    ----------
    pattern : Pattern
        The MBQC pattern to visualize.
    node_distance : tuple[float, float]
        Scale factors (x, y) for the node positions in the graph layout.
    """

    def __init__(self, pattern: Pattern, node_distance: tuple[float, float] = (1, 1)) -> None:
        """
        Initialize the interactive visualizer.

        Parameters
        ----------
        pattern : Pattern
            The MBQC pattern to visualize.
        node_distance : tuple[float, float], optional
            Scale factors for x and y coordinates of the graph nodes. Defaults to (1, 1).
        """
        self.pattern = pattern
        self.node_distance = node_distance

        # Prepare graph layout using Graphix's visualizer or fallbacks
        self._prepare_layout()

        # Figure setup
        self.fig = plt.figure(figsize=(15, 8))

        # Grid layout: Command list on left, Graph on right
        # Layout optimized to prevent overlap:
        # Commands: Left 2% to 30%
        # Graph: Left 40% to 98%
        self.ax_commands = self.fig.add_axes([0.02, 0.2, 0.28, 0.7])  # [left, bottom, width, height]
        self.ax_graph = self.fig.add_axes([0.4, 0.2, 0.58, 0.7])
        self.ax_slider = self.fig.add_axes([0.4, 0.05, 0.5, 0.03])
        self.ax_prev = self.fig.add_axes([0.3, 0.05, 0.04, 0.04])
        self.ax_next = self.fig.add_axes([0.92, 0.05, 0.04, 0.04])

        # Turn off axes frame for command list and graph
        self.ax_commands.axis("off")
        self.ax_graph.axis("off")  # Start hidden to avoid "square" artifact

        # Interaction state
        self.current_step = 0
        self.total_steps = len(pattern)

    # ... (other methods) ...

    def _draw_graph(self) -> None:
        """Render the graph state on the right panel."""
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
                    circle = plt.Circle((x, y), 0.1, color="lightgray", zorder=2)
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
                    circle = plt.Circle(
                        (x, y), 0.1, edgecolor="red", facecolor="white", linewidth=1.5, zorder=2
                    )
                    self.ax_graph.add_patch(circle)
                    
                    label_text = str(node)
                    # Show accumulated internal corrections
                    if node in corrections:
                        label_text += "\n" + "".join(sorted(corrections[node]))
                    
                    color = "black"
                    if node in corrections:
                        color = "blue"  # Highlight corrected nodes

                    self.ax_graph.text(
                        x, y, label_text, ha="center", va="center", fontsize=9, color=color, zorder=3
                    )

            # Set aspect close to equal and hide axes
            self.ax_graph.set_aspect("equal")
            self.ax_graph.set_xlim(self.x_limits)
            self.ax_graph.set_ylim(self.y_limits)
            self.ax_graph.axis("off")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error drawing graph: {e}", file=sys.stderr)
        # Matplotlib widgets placeholders (initialized in visualize)
        self.slider: Slider | None = None
        self.btn_prev: Button | None = None
        self.btn_next: Button | None = None

    def _prepare_layout(self) -> None:
        """Calculate node positions for the graph."""
        # Build full graph to determine positions
        g = nx.Graph()
        for cmd in self.pattern:
            if cmd.kind == CommandKind.N:
                g.add_node(cmd.node)
            elif cmd.kind == CommandKind.E:
                g.add_edge(cmd.nodes[0], cmd.nodes[1])

        # Use GraphVisualizer to determine positions based on flow/structure
        vis = GraphVisualizer(g, self.pattern.input_nodes, self.pattern.output_nodes)

        # Try to find flow/gflow for better layout, fallback to spring layout
        try:
            from graphix.optimization import StandardizedPattern

            pattern_std = StandardizedPattern.from_pattern(self.pattern)
            try:
                flow = pattern_std.extract_causal_flow()
                self.node_positions = vis.place_flow(flow)
            except Exception:
                try:
                    gflow = pattern_std.extract_gflow()
                    self.node_positions = vis.place_gflow(gflow)
                except Exception:
                    self.node_positions = vis.place_without_structure()
        except Exception:
            self.node_positions = vis.place_without_structure()

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
        self.slider = Slider(
            self.ax_slider, "Step", 0, self.total_steps, valinit=0, valstep=1, color="lightblue"
        )
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
        """Render the list of commands in the left panel."""
        self.ax_commands.clear()
        self.ax_commands.axis("off")
        self.ax_commands.set_title(f"Commands ({self.total_steps})", loc="left")

        # Windowing logic to show relevant commands
        window_size = 30
        start = max(0, int(self.current_step) - window_size // 2)
        end = min(self.total_steps, start + window_size)

        if end == self.total_steps:
            start = max(0, end - window_size)

        cmds = self.pattern[start:end]

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
            text_obj.index = abs_idx

    def _update_graph_state(
        self, step: int
    ) -> tuple[set, set, set, dict[int, set[str]], dict[int, int]]:
        """
        Calculate the state of the graph by simulating the pattern up to `step`.

        This method performs a full re-simulation using `StatevectorBackend`
        to ensure deterministic measurement outcomes and correct adaptive behavior.

        Parameters
        ----------
        step : int
            Current execution step.

        Returns
        -------
        active_nodes : set
            Nodes currently in the graph (active).
        measured_nodes : set
            Nodes that have been measured.
        active_edges : set
            Edges currently in the graph.
        corrections : dict
            Accumulated Pauli corrections ('X', 'Z') for each node.
        results : dict
            Measurement outcomes (0 or 1) for measured nodes.
        """
        # Initialize sets
        active_nodes = set(self.pattern.input_nodes)
        measured_nodes = set()
        active_edges = set()
        corrections: dict[int, set[str]] = {}
        
        # Simulation setup
        backend = StatevectorBackend()
        
        # Initialize input nodes in the backend
        if self.pattern.input_nodes:
            backend.add_nodes(self.pattern.input_nodes)

        # Fixed seed for deterministic scrubbing
        rng = np.random.default_rng(42)
        results: dict[int, int] = {}

        # Replay commands
        for i in range(int(step)):
            cmd = self.pattern[i]

            if cmd.kind == CommandKind.N:
                active_nodes.add(cmd.node)
                backend.add_nodes([cmd.node], data=cmd.state)

            elif cmd.kind == CommandKind.M:
                if cmd.node in active_nodes:
                    active_nodes.remove(cmd.node)
                measured_nodes.add(cmd.node)

                # --- Adaptive Measurement Logic (Feedforward) ---
                # Calculate s and t signals from previous measurement results
                if cmd.s_domain:
                    s_signal = sum(results.get(j, 0) for j in cmd.s_domain)
                else:
                    s_signal = 0
                if cmd.t_domain:
                    t_signal = sum(results.get(j, 0) for j in cmd.t_domain)
                else:
                    t_signal = 0

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

            elif cmd.kind == CommandKind.E:
                active_edges.add(cmd.nodes)
                # Apply entanglement in simulation
                backend.entangle_nodes(cmd.nodes)

            elif cmd.kind in (CommandKind.X, CommandKind.Z):
                # Apply Pauli corrections conditionally
                do_op = True
                if cmd.domain:
                    do_op = sum(results.get(j, 0) for j in cmd.domain) % 2 == 1

                if do_op:
                    backend.correct_byproduct(cmd)
                    # Visual tracking of corrections
                    if cmd.node not in corrections:
                        corrections[cmd.node] = set()
                    corrections[cmd.node].add(cmd.kind.name)
            
            # Note: C, S, T, etc. are not explicitly visualized but exist in backend if supported.
            # StatevectorBackend handles Clifford logic internally if pattern is standardized,
            # but visualizer focuses on MBQC core set {N, M, E, X, Z}.

        return active_nodes, measured_nodes, active_edges, corrections, results

    def _draw_graph(self) -> None:
        """Render the graph state on the right panel."""
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
                circle = plt.Circle((x, y), 0.1, color="lightgray", zorder=2)
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
                circle = plt.Circle(
                    (x, y), 0.1, edgecolor="red", facecolor="white", linewidth=1.5, zorder=2
                )
                self.ax_graph.add_patch(circle)
                
                label_text = str(node)
                # Show accumulated internal corrections
                if node in corrections:
                    label_text += "\n" + "".join(sorted(corrections[node]))
                
                color = "black"
                if node in corrections:
                    color = "blue"  # Highlight corrected nodes

                self.ax_graph.text(
                    x, y, label_text, ha="center", va="center", fontsize=9, color=color, zorder=3
                )

        # Set aspect close to equal and hide axes
        self.ax_graph.set_aspect("equal")
        self.ax_graph.set_xlim(self.x_limits)
        self.ax_graph.set_ylim(self.y_limits)
        self.ax_graph.axis("off")

    def _update(self, val: float) -> None:
        """Update visualization when slider changes."""
        step = int(val)
        if step != self.current_step:
            self.current_step = step
            self._draw_command_list()
            self._draw_graph()
            self.fig.canvas.draw_idle()

    def _prev_step(self, event: object) -> None:
        """Go backward one step."""
        if self.current_step > 0:
            self.slider.set_val(self.current_step - 1)

    def _next_step(self, event: object) -> None:
        """Go forward one step."""
        if self.current_step < self.total_steps:
            self.slider.set_val(self.current_step + 1)

    def _on_key(self, event: object) -> None:
        """Handle keyboard navigation."""
        if event.key == "right":
            self._next_step(None)
        elif event.key == "left":
            self._prev_step(None)

    def _on_pick(self, event: object) -> None:
        """Handle clicks on command list items."""
        if isinstance(event.artist, plt.Text):
            idx = getattr(event.artist, "index", None)
            if idx is not None:
                self.slider.set_val(idx + 1)  # Jump to state AFTER the clicked command
