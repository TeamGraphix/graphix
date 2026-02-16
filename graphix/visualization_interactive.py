"""Interactive visualization for MBQC patterns."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.widgets import Button, Slider
from graphix.pattern import Pattern, CommandKind
from graphix.pretty_print import OutputFormat, command_to_str
from graphix.visualization import GraphVisualizer

class InteractiveGraphVisualizer:
    """
    Interactive visualizer for MBQC patterns.
    
    This tool allows users to visualize the execution of a measurement-based quantum computing 
    pattern step-by-step. It displays the command sequence and the corresponding graph state, 
    updating dynamically as the user navigates through the commands.

    Attributes
    ----------
    pattern : Pattern
        The MBQC pattern to be visualized.
    node_distance : tuple[float, float]
        Distance multiplication factor between nodes for x and y directions.
    total_steps : int
        Total number of commands in the pattern.
    current_step : int
        Current step index in the command sequence.
    """
    
    def __init__(self, pattern: Pattern, node_distance: tuple[float, float] = (1, 1)):
        """
        Construct an InteractiveGraphVisualizer.

        Parameters
        ----------
        pattern : Pattern
            The MBQC pattern to be visualized.
        node_distance : tuple[float, float], optional
            Distance scale for the graph layout, by default (1, 1).
        """
        self.pattern = pattern
        self.node_distance = node_distance
        self.total_steps = len(pattern)
        self.current_step = 0
        
        # Extract graph structure for layout
        self.nodes = pattern.input_nodes.copy() if pattern.input_nodes else []
        self.edges = []
        self.node_positions = {}
        self.vis = None # GraphVisualizer instance for layout calculation
        
        # Pre-calculate graph layout
        self._prepare_layout()

    def _prepare_layout(self) -> None:
        """
        Pre-calculate the layout using the standard GraphVisualizer.
        
        It constructs a full graph from the pattern commands and uses GraphVisualizer's
        layout algorithms (flow, gflow, or spring) to determine node positions.
        """
        # We need to construct the full graph to get the layout
        # This is a bit of a workaround because GraphVisualizer expects a graph
        G = nx.Graph()
        for cmd in self.pattern:
            if cmd.kind == CommandKind.N:
                G.add_node(cmd.node)
            elif cmd.kind == CommandKind.E:
                G.add_edge(cmd.nodes[0], cmd.nodes[1])
        
        # Use GraphVisualizer to determine positions based on flow/structure
        # We create a dummy visualizer just for the layout
        self.vis = GraphVisualizer(G, self.pattern.input_nodes, self.pattern.output_nodes)
        
        # Try to find flow/gflow for better layout
        # This logic mimics visualize_from_pattern but just gets positions
        try:
            from graphix.optimization import StandardizedPattern
            pattern_std = StandardizedPattern.from_pattern(self.pattern)
            try:
                flow = pattern_std.extract_causal_flow()
                self.node_positions = self.vis.place_flow(flow)
            except:
                try:
                    gflow = pattern_std.extract_gflow()
                    self.node_positions = self.vis.place_gflow(gflow)
                except:
                    self.node_positions = self.vis.place_without_structure()
        except Exception:
             self.node_positions = self.vis.place_without_structure()
             
        # Apply scaling
        self.node_positions = {k: (v[0] * self.node_distance[0], v[1] * self.node_distance[1]) 
                               for k, v in self.node_positions.items()}

    def visualize(self) -> None:
        """
        Launch the interactive visualization window.
        
        This method sets up the Matplotlib figure, axes, slider, and buttons,
        and starts the event loop.
        """
        self.fig = plt.figure(figsize=(15, 8))
        
        # Grid layout: Command list on left, Graph on right
        # We use a bit of manual placement to fit the slider and list nicely
        self.ax_commands = self.fig.add_axes([0.05, 0.2, 0.2, 0.7]) # [left, bottom, width, height]
        self.ax_graph = self.fig.add_axes([0.3, 0.2, 0.65, 0.7])
        self.ax_slider = self.fig.add_axes([0.3, 0.05, 0.5, 0.03])
        self.ax_prev = self.fig.add_axes([0.2, 0.05, 0.04, 0.04])
        self.ax_next = self.fig.add_axes([0.85, 0.05, 0.04, 0.04])
        
        # Turn off axes for command list
        self.ax_commands.axis('off')
        
        # Setup Slider
        self.slider = Slider(
            self.ax_slider, "Step", 0, self.total_steps, 
            valinit=0, valstep=1, color="lightblue"
        )
        self.slider.on_changed(self._update)
        
        # Setup Buttons
        self.btn_prev = Button(self.ax_prev, '<')
        self.btn_next = Button(self.ax_next, '>')
        self.btn_prev.on_clicked(self._prev_step)
        self.btn_next.on_clicked(self._next_step)
        
        # Initial Draw
        self._update(0)
        self._draw_command_list()
        
        # Events
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        plt.show()

    def _on_key(self, event) -> None:
        """Handle key press events for navigation."""
        if event.key == 'right':
            self._next_step(None)
        elif event.key == 'left':
            self._prev_step(None)

    def _draw_command_list(self) -> None:
        """
        Draw the list of commands in the left panel.
        
        Only draws a window of commands around current_step to maintain performance and readability.
        Executed commands are shown in green, the current/next command in red, and future commands in black.
        """
        self.ax_commands.clear()
        self.ax_commands.axis('off')
        self.ax_commands.set_title(f"Commands ({self.total_steps})")
        
        # Define window size
        window_size = 20
        start = max(0, int(self.current_step) - window_size // 2)
        end = min(self.total_steps, start + window_size)
        
        # Adjust start if we are near the end
        if end == self.total_steps:
            start = max(0, end - window_size)
            
        cmds = self.pattern[start:end]
        
        for i, cmd in enumerate(cmds):
            abs_idx = start + i
            text = f"{abs_idx}: {command_to_str(cmd, OutputFormat.Unicode)}"
            
            color = "black"
            weight = "normal"
            if abs_idx < self.current_step:
                color = "green" # Executed (Updated to green as per issue suggestion)
            elif abs_idx == self.current_step:
                color = "red" # Current/Next to be executed
                weight = "bold"
            else:
                color = "black" # Future
                
            # clickable text
            t = self.ax_commands.text(
                0, 1.0 - (i / window_size), text, 
                transform=self.ax_commands.transAxes,
                fontsize=10, color=color, weight=weight,
                picker=True
            )
            # Store index with artist for picking
            t.index = abs_idx

    def _update_graph_state(self, step: int) -> tuple[set, set, set, dict]:
        """
        Calculate the state of nodes and edges at a given step.

        Parameters
        ----------
        step : int
            The command index up to which the pattern is executed.

        Returns
        -------
        active_nodes : set
            Set of indices of nodes that are initialized and alive (not measured).
        measured_nodes : set
            Set of indices of nodes that have been measured.
        active_edges : set
            Set of edges (tuples) that have been created.
        corrections : dict
            Dictionary mapping node indices to sets of Pauli corrections ('X', 'Z').
        """
        active_nodes = set()
        measured_nodes = set()
        active_edges = set()
        corrections = {} # node -> set of 'X', 'Z'

        # Replay commands up to 'step'
        for i in range(int(step)):
            cmd = self.pattern[i]
            if cmd.kind == CommandKind.N:
                active_nodes.add(cmd.node)
            elif cmd.kind == CommandKind.M:
                if cmd.node in active_nodes:
                    active_nodes.remove(cmd.node)
                measured_nodes.add(cmd.node)
            elif cmd.kind == CommandKind.E:
                active_edges.add(cmd.nodes)
            elif cmd.kind == CommandKind.X:
                if cmd.node not in corrections:
                    corrections[cmd.node] = set()
                corrections[cmd.node].add('X')
            elif cmd.kind == CommandKind.Z:
                if cmd.node not in corrections:
                    corrections[cmd.node] = set()
                corrections[cmd.node].add('Z')
                
        return active_nodes, measured_nodes, active_edges, corrections

    def _draw_graph(self) -> None:
        """
        Render the graph in the right panel based on the current state.
        
        Highlights active nodes (white/red), measured nodes (grey), and active edges.
        Displays badges for Pauli corrections.
        """
        self.ax_graph.clear()
        
        active_nodes, measured_nodes, active_edges, corrections = self._update_graph_state(self.current_step)
        
        # If no nodes, just return
        if not self.node_positions:
            return

        # Draw Edges        
        for u, v in active_edges:
            if u in self.node_positions and v in self.node_positions:
                pos_u = self.node_positions[u]
                pos_v = self.node_positions[v]
                self.ax_graph.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 'k-', zorder=1)
                
        # Draw Nodes
        for node, pos in self.node_positions.items():
            # Determine visual properties
            facecolor = 'none'
            edgecolor = 'lightgray'
            linestyle = ':'
            alpha = 1.0
            label_color = 'k'
            
            if node in active_nodes:
                facecolor = 'white'
                edgecolor = 'red'
                linestyle = '-'
            elif node in measured_nodes:
                facecolor = 'lightgray'
                edgecolor = 'gray'
                linestyle = '-'
                
            self.ax_graph.scatter(*pos, s=300, c=facecolor, edgecolors=edgecolor, linestyle=linestyle, alpha=alpha, zorder=2)
            self.ax_graph.text(*pos, str(node), ha='center', va='center', color=label_color, zorder=3)
            
            # Draw corrections badges
            if node in corrections:
                corr_str = "".join(sorted(corrections[node]))
                # Offset position for badge
                badge_pos = (pos[0] + 0.1, pos[1] + 0.1) 
                self.ax_graph.text(*badge_pos, corr_str, fontsize=8, color='blue', weight='bold', zorder=4)

        self.ax_graph.set_aspect('equal')
        self.ax_graph.axis('off')

    def _update(self, val) -> None:
        """
        Update the visualization to a specific step.

        Parameters
        ----------
        val : float
            The slider value representing the step index.
        """
        self.current_step = int(val)
        self._draw_command_list()
        self._draw_graph()
        self.fig.canvas.draw_idle()

    def _prev_step(self, event) -> None:
        """Callback for 'Previous' button."""
        if self.current_step > 0:
            self.slider.set_val(self.current_step - 1)

    def _next_step(self, event) -> None:
        """Callback for 'Next' button."""
        if self.current_step < self.total_steps:
            self.slider.set_val(self.current_step + 1)
            
    def _on_pick(self, event) -> None:
        """
        Handle pick events on the command list.
        
        Clicking a command sets the current step to immediately after that command.
        """
        if isinstance(event.artist, plt.Text):
            idx = getattr(event.artist, 'index', None)
            if idx is not None:
                # Set step to idx + 1 so the clicked command is executed
                self.slider.set_val(idx + 1)
