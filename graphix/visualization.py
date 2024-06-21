from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphix.pattern import Pattern

import math
from copy import deepcopy

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from graphix import gflow


class GraphVisualizer:
    """
    A class for visualizing MBQC graphs with flow or gflow structure.

    Attributes
    ----------
    g : networkx graph
        the graph to be visualized
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
        G: nx.Graph,
        v_in: list[int],
        v_out: list[int],
        meas_plane: dict[int, str] | None = None,
        meas_angles: dict[int, float] | None = None,
        local_clifford: dict[int, int] | None = None,
    ):
        """
        Parameters
        ----------
        G : :class:`networkx.graph.Graph` object
            networkx graph
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
        self.G = G
        self.v_in = v_in
        self.v_out = v_out
        if meas_plane is None:
            self.meas_planes = {i: "XY" for i in iter(G.nodes)}
        else:
            self.meas_planes = meas_plane
        self.meas_angles = meas_angles
        self.local_clifford = local_clifford

    def visualize(
        self,
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurement_planes: bool = False,
        show_loop: bool = True,
        node_distance: tuple[int, int] = (1, 1),
        figsize: tuple[int, int] | None = None,
        save: bool = False,
        filename: str | None = None,
    ):
        """
        Visualizes the graph with flow or gflow structure.
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
        save : bool
            If True, the plot is saved as a png file.
        filename : str
            Filename of the saved plot.
        """

        f, l_k = gflow.find_flow(self.G, set(self.v_in), set(self.v_out), meas_planes=self.meas_planes)  # try flow
        if f:
            print("Flow detected in the graph.")
            self.visualize_w_flow(
                f,
                l_k,
                show_pauli_measurement,
                show_local_clifford,
                show_measurement_planes,
                node_distance,
                figsize,
                save,
                filename,
            )
        else:
            g, l_k = gflow.find_gflow(self.G, set(self.v_in), set(self.v_out), self.meas_planes)  # try gflow
            if g:
                print("Gflow detected in the graph. (flow not detected)")
                self.visualize_w_gflow(
                    g,
                    l_k,
                    show_pauli_measurement,
                    show_local_clifford,
                    show_measurement_planes,
                    show_loop,
                    node_distance,
                    figsize,
                    save,
                    filename,
                )
            else:
                print("No flow or gflow detected in the graph.")
                self.visualize_wo_structure(
                    show_pauli_measurement,
                    show_local_clifford,
                    show_measurement_planes,
                    node_distance,
                    figsize,
                    save,
                    filename,
                )

    def visualize_from_pattern(
        self,
        pattern: Pattern,
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurement_planes: bool = False,
        show_loop: bool = True,
        node_distance: tuple[int, int] = (1, 1),
        figsize: tuple[int, int] | None = None,
        save: bool = False,
        filename: str | None = None,
    ):
        """
        Visualizes the graph with flow or gflow structure found from the given pattern.
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
        save : bool
            If True, the plot is saved as a png file.
        filename : str
            Filename of the saved plot.
        """
        f, l_k = gflow.flow_from_pattern(pattern)  # try flow
        if f:
            print("The pattern is consistent with flow structure.")
            self.visualize_w_flow(
                f,
                l_k,
                show_pauli_measurement,
                show_local_clifford,
                show_measurement_planes,
                node_distance,
                figsize,
                save,
                filename,
            )
        else:
            g, l_k = gflow.gflow_from_pattern(pattern)  # try gflow
            if g:
                print("The pattern is consistent with gflow structure. (not with flow)")
                self.visualize_w_gflow(
                    g,
                    l_k,
                    show_pauli_measurement,
                    show_local_clifford,
                    show_measurement_planes,
                    show_loop,
                    node_distance,
                    figsize,
                    save,
                    filename,
                )
            else:
                print("The pattern is not consistent with flow or gflow structure.")
                depth, layers = pattern.get_layers()
                layers = {element: key for key, value_set in layers.items() for element in value_set}
                for output in pattern.output_nodes:
                    layers[output] = depth + 1
                xflow, zflow = gflow.get_corrections_from_pattern(pattern)
                self.visualize_all_correction(
                    layers,
                    xflow,
                    zflow,
                    show_pauli_measurement,
                    show_local_clifford,
                    show_measurement_planes,
                    node_distance,
                    figsize,
                    save,
                    filename,
                )

    def visualize_w_flow(
        self,
        f: dict[int, set[int]],
        l_k: dict[int, int],
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurement_planes: bool = False,
        node_distance: tuple[int, int] = (1, 1),
        figsize: tuple[int, int] | None = None,
        save: bool = False,
        filename: str | None = None,
    ):
        """
        visualizes the graph with flow structure.

        Nodes are colored based on their role (input, output, or other) and edges are depicted as arrows
        or dashed lines depending on whether they are in the flow mapping. Vertical dashed lines separate
        different layers of the graph. This function does not return anything but plots the graph
        using matplotlib's pyplot.

        Parameters
        ----------
        f : dict
            flow mapping.
        l_k : dict
            Layer mapping.
        show_pauli_measurement : bool
            If True, the nodes with Pauli measurement angles are colored light blue.
        show_local_clifford : bool
            If True, indexes of the local Clifford operator are displayed adjacent to the nodes.
        show_measurement_planes : bool
            If True, the measurement planes are displayed adjacent to the nodes.
        node_distance : tuple
            Distance multiplication factor between nodes for x and y directions.
        figsize : tuple
            Figure size of the plot.
        save : bool
            If True, the plot is saved.
        filename : str
            Filename of the saved plot.
        """
        if figsize is None:
            figsize = self.get_figsize(l_k, node_distance=node_distance)
        plt.figure(figsize=figsize)
        pos = self.get_pos_from_flow(f, l_k)

        edge_path, arrow_path = self.get_edge_path(f, pos)

        for edge in edge_path.keys():
            if len(edge_path[edge]) == 2:
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], style="dashed", alpha=0.7)
            else:
                t = np.linspace(0, 1, 100)
                curve = self._bezier_curve(edge_path[edge], t)
                plt.plot(curve[:, 0], curve[:, 1], "k--", linewidth=1, alpha=0.7)

        for arrow in arrow_path.keys():
            if len(arrow_path[arrow]) == 2:
                nx.draw_networkx_edges(self.G, pos, edgelist=[arrow], edge_color="black", arrowstyle="->", arrows=True)
            else:
                path = arrow_path[arrow]
                last = np.array(path[-1])
                second_last = np.array(path[-2])
                path[-1] = list(
                    last - (last - second_last) / np.linalg.norm(last - second_last) * 0.2
                )  # Shorten the last edge not to hide arrow under the node
                t = np.linspace(0, 1, 100)
                curve = self._bezier_curve(path, t)

                plt.plot(curve[:, 0], curve[:, 1], c="k", linewidth=1)
                plt.annotate(
                    "",
                    xy=curve[-1],
                    xytext=curve[-2],
                    arrowprops=dict(arrowstyle="->", color="k", lw=1),
                )

        # Draw the nodes with different colors based on their role (input, output, or other)
        for node in self.G.nodes():
            color = "black"  # default color for 'other' nodes
            inner_color = "white"
            if node in self.v_in:
                color = "red"
            if node in self.v_out:
                inner_color = "lightgray"
            elif (
                show_pauli_measurement
                and self.meas_angles is not None
                and (
                    2 * self.meas_angles[node] == int(2 * self.meas_angles[node])
                )  # measurement angle is integer or half-integer
            ):
                inner_color = "lightblue"
            plt.scatter(
                *pos[node], edgecolor=color, facecolor=inner_color, s=350, zorder=2
            )  # Draw the nodes manually with scatter()

        if show_local_clifford and self.local_clifford is not None:
            for node in self.G.nodes():
                if node in self.local_clifford.keys():
                    plt.text(*pos[node] + np.array([0.2, 0.2]), f"{self.local_clifford[node]}", fontsize=10, zorder=3)

        if show_measurement_planes:
            for node in self.G.nodes():
                if node in self.meas_planes.keys():
                    plt.text(*pos[node] + np.array([0.22, -0.2]), f"{self.meas_planes[node]}", fontsize=9, zorder=3)

        # Draw the labels
        fontsize = 12
        if max(self.G.nodes()) >= 100:
            fontsize = fontsize * 2 / len(str(max(self.G.nodes())))
        nx.draw_networkx_labels(self.G, pos, font_size=fontsize)

        x_min = min([pos[node][0] for node in self.G.nodes()])  # Get the minimum x coordinate
        x_max = max([pos[node][0] for node in self.G.nodes()])  # Get the maximum x coordinate
        y_min = min([pos[node][1] for node in self.G.nodes()])  # Get the minimum y coordinate
        y_max = max([pos[node][1] for node in self.G.nodes()])  # Get the maximum y coordinate

        # Draw the vertical lines to separate different layers
        for layer in range(min(l_k.values()), max(l_k.values())):
            plt.axvline(
                x=(layer + 0.5) * node_distance[0], color="gray", linestyle="--", alpha=0.5
            )  # Draw line between layers
        for layer in range(min(l_k.values()), max(l_k.values()) + 1):
            plt.text(
                layer * node_distance[0], y_min - 0.5, f"l: {max(l_k.values()) - layer}", ha="center", va="top"
            )  # Add layer label at bottom

        plt.xlim(
            x_min - 0.5 * node_distance[0], x_max + 0.5 * node_distance[0]
        )  # Add some padding to the left and right
        plt.ylim(y_min - 1, y_max + 0.5)  # Add some padding to the top and bottom
        if save:
            plt.savefig(filename)
        plt.show()

    def visualize_w_gflow(
        self,
        g: dict[int, set[int]],
        l_k: dict[int, int],
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurement_planes: bool = False,
        show_loop: bool = True,
        node_distance: tuple[int, int] = (1, 1),
        figsize: tuple[int, int] | None = None,
        save: bool = False,
        filename: str | None = None,
    ):
        """
        visualizes the graph with flow structure.

        Nodes are colored based on their role (input, output, or other) and edges are depicted as arrows
        or dashed lines depending on whether they are in the flow mapping. Vertical dashed lines separate
        different layers of the graph. This function does not return anything but plots the graph
        using matplotlib's pyplot.

        Parameters
        ----------
        g : dict
            gflow mapping.
        l_k : dict
            Layer mapping.
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
        save : bool
            If True, the plot is saved as a png file.
        filename : str
            Filename of the saved plot.
        """

        pos = self.get_pos_from_gflow(g, l_k)
        pos = {k: (v[0] * node_distance[0], v[1] * node_distance[1]) for k, v in pos.items()}  # Scale the layout

        edge_path, arrow_path = self.get_edge_path(g, pos)

        if figsize is None:
            figsize = self.get_figsize(l_k, pos, node_distance=node_distance)
        plt.figure(figsize=figsize)

        for edge in edge_path.keys():
            if len(edge_path[edge]) == 2:
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], style="dashed", alpha=0.7)
            else:
                t = np.linspace(0, 1, 100)
                curve = self._bezier_curve(edge_path[edge], t)
                plt.plot(curve[:, 0], curve[:, 1], "k--", linewidth=1, alpha=0.7)

        for arrow in arrow_path.keys():
            if arrow[0] == arrow[1]:  # self loop
                if show_loop:
                    t = np.linspace(0, 1, 100)
                    curve = self._bezier_curve(arrow_path[arrow], t)
                    plt.plot(curve[:, 0], curve[:, 1], c="k", linewidth=1)
                    plt.annotate(
                        "",
                        xy=curve[-1],
                        xytext=curve[-2],
                        arrowprops=dict(arrowstyle="->", color="k", lw=1),
                    )
            elif len(arrow_path[arrow]) == 2:  # straight line
                nx.draw_networkx_edges(self.G, pos, edgelist=[arrow], edge_color="black", arrowstyle="->", arrows=True)
            else:
                path = arrow_path[arrow]
                last = np.array(path[-1])
                second_last = np.array(path[-2])
                path[-1] = list(
                    last - (last - second_last) / np.linalg.norm(last - second_last) * 0.2
                )  # Shorten the last edge not to hide arrow under the node
                t = np.linspace(0, 1, 100)
                curve = self._bezier_curve(path, t)

                plt.plot(curve[:, 0], curve[:, 1], c="k", linewidth=1)
                plt.annotate(
                    "",
                    xy=curve[-1],
                    xytext=curve[-2],
                    arrowprops=dict(arrowstyle="->", color="k", lw=1),
                )

        # Draw the nodes with different colors based on their role (input, output, or other)
        for node in self.G.nodes():
            color = "black"  # default color for 'other' nodes
            inner_color = "white"
            if node in self.v_in:
                color = "red"
            if node in self.v_out:
                inner_color = "lightgray"
            elif (
                show_pauli_measurement
                and self.meas_angles is not None
                and (
                    2 * self.meas_angles[node] == int(2 * self.meas_angles[node])
                )  # measurement angle is integer or half-integer
            ):
                inner_color = "lightblue"
            plt.scatter(
                *pos[node], edgecolor=color, facecolor=inner_color, s=350, zorder=2
            )  # Draw the nodes manually with scatter()

        if show_local_clifford and self.local_clifford is not None:
            for node in self.G.nodes():
                if node in self.local_clifford.keys():
                    plt.text(*pos[node] + np.array([0.2, 0.2]), f"{self.local_clifford[node]}", fontsize=10, zorder=3)

        if show_measurement_planes:
            for node in self.G.nodes():
                if node in self.meas_planes.keys():
                    plt.text(*pos[node] + np.array([0.22, -0.2]), f"{self.meas_planes[node]}", fontsize=9, zorder=3)

        # Draw the labels
        fontsize = 12
        if max(self.G.nodes()) >= 100:
            fontsize = fontsize * 2 / len(str(max(self.G.nodes())))
        nx.draw_networkx_labels(self.G, pos, font_size=fontsize)

        x_min = min([pos[node][0] for node in self.G.nodes()])  # Get the minimum x coordinate
        x_max = max([pos[node][0] for node in self.G.nodes()])  # Get the maximum x coordinate
        y_min = min([pos[node][1] for node in self.G.nodes()])  # Get the minimum y coordinate
        y_max = max([pos[node][1] for node in self.G.nodes()])  # Get the maximum y coordinate

        # Draw the vertical lines to separate different layers
        for layer in range(min(l_k.values()), max(l_k.values())):
            plt.axvline(
                x=(layer + 0.5) * node_distance[0], color="gray", linestyle="--", alpha=0.5
            )  # Draw line between layers
        for layer in range(min(l_k.values()), max(l_k.values()) + 1):
            plt.text(
                layer * node_distance[0], y_min - 0.5, f"l: {max(l_k.values()) - layer}", ha="center", va="top"
            )  # Add layer label at bottom

        plt.xlim(
            x_min - 0.5 * node_distance[0], x_max + 0.5 * node_distance[0]
        )  # Add some padding to the left and right
        plt.ylim(y_min - 1, y_max + 0.5)  # Add some padding to the top and bottom
        if save:
            plt.savefig(filename)
        plt.show()

    def visualize_wo_structure(
        self,
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurement_planes: bool = False,
        node_distance: tuple[int, int] = (1, 1),
        figsize: tuple[int, int] | None = None,
        save: bool = False,
        filename: str | None = None,
    ):
        """
        visualizes the graph without flow or gflow.

        Nodes are colored based on their role (input, output, or other) and edges are depicted as arrows
        or dashed lines depending on whether they are in the flow mapping. Vertical dashed lines separate
        different layers of the graph. This function does not return anything but plots the graph
        using matplotlib's pyplot.

        Parameters
        ----------
        show_pauli_measurement : bool
            If True, the nodes with Pauli measurement angles are colored light blue.
        show_local_clifford : bool
            If True, indexes of the local Clifford operator are displayed adjacent to the nodes.
        show_measurement_planes : bool
            If True, the measurement planes are displayed adjacent to the nodes.
        node_distance : tuple
            Distance multiplication factor between nodes for x and y directions.
        figsize : tuple
            Figure size of the plot.
        save : bool
            If True, the plot is saved as a png file.
        filename : str
            Filename of the saved plot.
        """
        pos = self.get_pos_wo_structure()
        pos = {k: (v[0] * node_distance[0], v[1] * node_distance[1]) for k, v in pos.items()}  # Scale the layout

        if figsize is None:
            figsize = self.get_figsize(None, pos, node_distance=node_distance)
        plt.figure(figsize=figsize)

        edge_path = self.get_edge_path_wo_structure(pos)

        for edge in edge_path.keys():
            if len(edge_path[edge]) == 2:
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], style="dashed", alpha=0.7)
            else:
                t = np.linspace(0, 1, 100)
                curve = self._bezier_curve(edge_path[edge], t)
                plt.plot(curve[:, 0], curve[:, 1], "k--", linewidth=1, alpha=0.7)

        # Draw the nodes with different colors based on their role (input, output, or other)
        for node in self.G.nodes():
            color = "black"  # default color for 'other' nodes
            inner_color = "white"
            if node in self.v_in:
                color = "red"
            if node in self.v_out:
                inner_color = "lightgray"
            elif (
                show_pauli_measurement
                and self.meas_angles is not None
                and (
                    2 * self.meas_angles[node] == int(2 * self.meas_angles[node])
                )  # measurement angle is integer or half-integer
            ):
                inner_color = "lightblue"
            plt.scatter(
                *pos[node], edgecolor=color, facecolor=inner_color, s=350, zorder=2
            )  # Draw the nodes manually with scatter()

        if show_local_clifford and self.local_clifford is not None:
            for node in self.G.nodes():
                if node in self.local_clifford.keys():
                    plt.text(*pos[node] + np.array([0.2, 0.2]), f"{self.local_clifford[node]}", fontsize=10, zorder=3)

        if show_measurement_planes:
            for node in self.G.nodes():
                if node in self.meas_planes.keys():
                    plt.text(*pos[node] + np.array([0.22, -0.2]), f"{self.meas_planes[node]}", fontsize=9, zorder=3)

        # Draw the labels
        fontsize = 12
        if max(self.G.nodes()) >= 100:
            fontsize = fontsize * 2 / len(str(max(self.G.nodes())))
        nx.draw_networkx_labels(self.G, pos, font_size=fontsize)

        x_min = min([pos[node][0] for node in self.G.nodes()])  # Get the minimum x coordinate
        x_max = max([pos[node][0] for node in self.G.nodes()])  # Get the maximum x coordinate
        y_min = min([pos[node][1] for node in self.G.nodes()])  # Get the minimum y coordinate
        y_max = max([pos[node][1] for node in self.G.nodes()])  # Get the maximum y coordinate

        plt.xlim(
            x_min - 0.5 * node_distance[0], x_max + 0.5 * node_distance[0]
        )  # Add some padding to the left and right
        plt.ylim(y_min - 0.5, y_max + 0.5)  # Add some padding to the top and bottom

        if save:
            plt.savefig(filename)
        plt.show()

    def visualize_all_correction(
        self,
        layers: dict[int, int],
        xflow: dict[int, set[int]],
        zflow: dict[int, set[int]],
        show_pauli_measurement: bool = True,
        show_local_clifford: bool = False,
        show_measurement_planes: bool = False,
        node_distance: tuple[int, int] = (1, 1),
        figsize: tuple[int, int] | None = None,
        save: bool = False,
        filename: str | None = None,
    ):
        """
        visualizes the graph of pattern with all correction flows.

        Nodes are colored based on their role (input, output, or other) and edges of graph are depicted as dashed lines.
        Xflow is depicted as red arrows and Zflow is depicted as blue arrows. The function does not return anything but plots the graph using matplotlib's pyplot.

        Parameters
        ----------
        layers : dict
            Layer mapping obtained from the measurement order of the pattern.
        xflow : dict
            Dictionary for x correction of the pattern.
        zflow : dict
            Dictionary for z correction of the pattern.
        show_pauli_measurement : bool
            If True, the nodes with Pauli measurement angles are colored light blue.
        show_local_clifford : bool
            If True, indexes of the local Clifford operator are displayed adjacent to the nodes.
        show_measurement_planes : bool
            If True, the measurement planes are displayed adjacent to the nodes.
        node_distance : tuple
            Distance multiplication factor between nodes for x and y directions.
        figsize : tuple
            Figure size of the plot.
        save : bool
            If True, the plot is saved as a png file.
        filename : str
            Filename of the saved plot.
        """
        pos = self.get_pos_all_correction(layers)
        pos = {k: (v[0] * node_distance[0], v[1] * node_distance[1]) for k, v in pos.items()}  # Scale the layout

        if figsize is None:
            figsize = self.get_figsize(layers, pos, node_distance=node_distance)
        # add some padding to the right for the legend
        figsize = (figsize[0] + 3.0, figsize[1])
        plt.figure(figsize=figsize)

        xzflow = dict()
        for key, value in deepcopy(xflow).items():
            if key in xzflow:
                xzflow[key] |= value
            else:
                xzflow[key] = value
        for key, value in deepcopy(zflow).items():
            if key in xzflow:
                xzflow[key] |= value
            else:
                xzflow[key] = value
        edge_path, arrow_path = self.get_edge_path(xzflow, pos)

        for edge in edge_path.keys():
            if len(edge_path[edge]) == 2:
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], style="dashed", alpha=0.7)
            else:
                t = np.linspace(0, 1, 100)
                curve = self._bezier_curve(edge_path[edge], t)
                plt.plot(curve[:, 0], curve[:, 1], "k--", linewidth=1, alpha=0.7)
        for arrow in arrow_path.keys():
            if arrow[1] not in xflow.get(arrow[0], set()):
                color = "tab:green"
            elif arrow[1] not in zflow.get(arrow[0], set()):
                color = "tab:red"
            else:
                color = "tab:brown"
            if len(arrow_path[arrow]) == 2:  # straight line
                nx.draw_networkx_edges(self.G, pos, edgelist=[arrow], edge_color=color, arrowstyle="->", arrows=True)
            else:
                path = arrow_path[arrow]
                last = np.array(path[-1])
                second_last = np.array(path[-2])
                path[-1] = list(
                    last - (last - second_last) / np.linalg.norm(last - second_last) * 0.2
                )  # Shorten the last edge not to hide arrow under the node

                t = np.linspace(0, 1, 100)
                curve = self._bezier_curve(path, t)

                plt.plot(curve[:, 0], curve[:, 1], c=color, linewidth=1)
                plt.annotate(
                    "",
                    xy=curve[-1],
                    xytext=curve[-2],
                    arrowprops=dict(arrowstyle="->", color=color, lw=1),
                )

        # Draw the nodes with different colors based on their role (input, output, or other)
        for node in self.G.nodes():
            color = "black"
            inner_color = "white"
            if node in self.v_in:
                color = "red"
            if node in self.v_out:
                inner_color = "lightgray"
            elif (
                show_pauli_measurement
                and self.meas_angles is not None
                and (
                    2 * self.meas_angles[node] == int(2 * self.meas_angles[node])
                )  # measurement angle is integer or half-integer
            ):
                inner_color = "lightblue"
            plt.scatter(*pos[node], edgecolor=color, facecolor=inner_color, s=350, zorder=2)

        if show_local_clifford and self.local_clifford is not None:
            for node in self.G.nodes():
                if node in self.local_clifford.keys():
                    plt.text(*pos[node] + np.array([0.2, 0.2]), f"{self.local_clifford[node]}", fontsize=10, zorder=3)

        if show_measurement_planes:
            for node in self.G.nodes():
                if node in self.meas_planes.keys():
                    plt.text(*pos[node] + np.array([0.22, -0.2]), f"{self.meas_planes[node]}", fontsize=9, zorder=3)

        # Draw the labels
        fontsize = 12
        if max(self.G.nodes()) >= 100:
            fontsize = fontsize * 2 / len(str(max(self.G.nodes())))
        nx.draw_networkx_labels(self.G, pos, font_size=fontsize)

        # legend for arrow colors
        plt.plot([], [], "k--", alpha=0.7, label="graph edge")
        plt.plot([], [], color="tab:red", label="xflow")
        plt.plot([], [], color="tab:green", label="zflow")
        plt.plot([], [], color="tab:brown", label="xflow and zflow")

        x_min = min([pos[node][0] for node in self.G.nodes()])  # Get the minimum x coordinate
        x_max = max([pos[node][0] for node in self.G.nodes()])
        y_min = min([pos[node][1] for node in self.G.nodes()])
        y_max = max([pos[node][1] for node in self.G.nodes()])

        plt.xlim(
            x_min - 0.5 * node_distance[0], x_max + 3.5 * node_distance[0]
        )  # Add some padding to the left and right
        plt.ylim(y_min - 0.5, y_max + 0.5)  # Add some padding to the top and bottom

        plt.legend(loc="upper right", fontsize=10)

        if save:
            plt.savefig(filename)
        plt.show()

    def get_figsize(
        self,
        l_k: dict[int, int],
        pos: dict[int, tuple[float, float]] | None = None,
        node_distance: tuple[int, int] = (1, 1),
    ) -> tuple[int, int]:
        """
        Returns the figure size of the graph.

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
            width = len(set([pos[node][0] for node in self.G.nodes()])) * 0.8
        else:
            width = (max(l_k.values()) + 1) * 0.8
        if pos is not None:
            height = len(set([pos[node][1] for node in self.G.nodes()]))
        else:
            height = len(self.v_out)
        figsize = (width * node_distance[0], height * node_distance[1])

        return figsize

    def get_edge_path(self, flow: dict[int, int | set[int]], pos: dict[int, tuple[float, float]]) -> dict[int, list]:
        """
        Returns the path of edges and gflow arrows.

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
        max_iter = 5
        edge_path = {}
        arrow_path = {}
        edge_set = set(self.G.edges())
        flow_arrows = {(k, v) for k, values in flow.items() for v in values}
        # set of mid-points of the edges
        # mid_points = {(0.5 * (pos[k][0] + pos[v][0]), 0.5 * (pos[k][1] + pos[v][1])) for k, v in edge_set} - set(pos[node] for node in self.G.nodes())

        for edge in edge_set:
            iteration = 0
            nodes = self.G.nodes()
            bezier_path = [pos[edge[0]], pos[edge[1]]]
            while True:
                iteration += 1
                intersect = False
                if iteration > max_iter:
                    break
                ctrl_points = []
                for i in range(len(bezier_path) - 1):
                    start = bezier_path[i]
                    end = bezier_path[i + 1]
                    for node in nodes:
                        if node != edge[0] and node != edge[1] and self._edge_intersects_node(start, end, pos[node]):
                            intersect = True
                            ctrl_points.append(
                                [
                                    i,
                                    self._control_point(
                                        bezier_path[0], bezier_path[-1], pos[node], distance=0.6 / iteration
                                    ),
                                ]
                            )
                            nodes = set(nodes) - {node}
                if not intersect:
                    break
                else:
                    for i, ctrl_point in enumerate(ctrl_points):
                        bezier_path.insert(ctrl_point[0] + i + 1, ctrl_point[1])
            bezier_path = self._check_path(bezier_path)
            edge_path[edge] = bezier_path

        for arrow in flow_arrows:
            if arrow[0] == arrow[1]:  # Self loop

                def _point_from_node(pos, dist, angle):
                    angle = np.deg2rad(angle)
                    return [pos[0] + dist * np.cos(angle), pos[1] + dist * np.sin(angle)]

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
                iteration = 0
                nodes = self.G.nodes()
                bezier_path = [pos[arrow[0]], pos[arrow[1]]]
                if arrow in edge_set or (arrow[1], arrow[0]) in edge_set:
                    mid_point = (
                        0.5 * (pos[arrow[0]][0] + pos[arrow[1]][0]),
                        0.5 * (pos[arrow[0]][1] + pos[arrow[1]][1]),
                    )
                    if self._edge_intersects_node(pos[arrow[0]], pos[arrow[1]], mid_point, buffer=0.05):
                        ctrl_point = self._control_point(pos[arrow[0]], pos[arrow[1]], mid_point, distance=0.2)
                        bezier_path.insert(1, ctrl_point)
                while True:
                    iteration += 1
                    intersect = False
                    if iteration > max_iter:
                        break
                    ctrl_points = []
                    for i in range(len(bezier_path) - 1):
                        start = bezier_path[i]
                        end = bezier_path[i + 1]
                        for node in nodes:
                            if (
                                node != arrow[0]
                                and node != arrow[1]
                                and self._edge_intersects_node(start, end, pos[node])
                            ):
                                intersect = True
                                ctrl_points.append(
                                    [
                                        i,
                                        self._control_point(start, end, pos[node], distance=0.6 / iteration),
                                    ]
                                )
                    if not intersect:
                        break
                    else:
                        for i, ctrl_point in enumerate(ctrl_points):
                            bezier_path.insert(ctrl_point[0] + i + 1, ctrl_point[1])
                bezier_path = self._check_path(bezier_path, pos[arrow[1]])
            arrow_path[arrow] = bezier_path

        return edge_path, arrow_path

    def get_edge_path_wo_structure(self, pos: dict[int, tuple[float, float]]) -> dict[int, list]:
        """
        Returns the path of edges.

        Parameters
        ----------
        pos : dict
            dictionary of node positions.

        Returns
        -------
        edge_path : dict
            dictionary of edge paths.
        """
        max_iter = 5
        edge_path = {}
        edge_set = set(self.G.edges())
        for edge in edge_set:
            iteration = 0
            nodes = self.G.nodes()
            bezier_path = [pos[edge[0]], pos[edge[1]]]
            while True:
                iteration += 1
                intersect = False
                if iteration > max_iter:
                    break
                ctrl_points = []
                for i in range(len(bezier_path) - 1):
                    start = bezier_path[i]
                    end = bezier_path[i + 1]
                    for node in nodes:
                        if node != edge[0] and node != edge[1] and self._edge_intersects_node(start, end, pos[node]):
                            intersect = True
                            ctrl_points.append(
                                [
                                    i,
                                    self._control_point(
                                        bezier_path[0], bezier_path[-1], pos[node], distance=0.6 / iteration
                                    ),
                                ]
                            )
                            nodes = set(nodes) - {node}
                if not intersect:
                    break
                else:
                    for i, ctrl_point in enumerate(ctrl_points):
                        bezier_path.insert(ctrl_point[0] + i + 1, ctrl_point[1])
            bezier_path = self._check_path(bezier_path)
            edge_path[edge] = bezier_path
        return edge_path

    def get_pos_from_flow(self, f: dict[int, int], l_k: dict[int, int]) -> dict[int, tuple[float, float]]:
        """
        Returns the position of nodes based on the flow.

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
        start_nodes = self.G.nodes() - values_union
        pos = {node: [0, 0] for node in self.G.nodes()}
        for i, k in enumerate(start_nodes):
            pos[k][1] = i
            while k in f.keys():
                k = list(f[k])[0]
                pos[k][1] = i

        lmax = max(l_k.values())
        # Change the x coordinates of the nodes based on their layer, sort in descending order
        for node, layer in l_k.items():
            pos[node][0] = lmax - layer
        pos = {k: tuple(v) for k, v in pos.items()}
        return pos

    def get_pos_from_gflow(self, g: dict[int, set[int]], l_k: dict[int, int]) -> dict[int, tuple[float, float]]:
        """
        Returns the position of nodes based on the gflow.

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

        g_edges = []

        for node, node_list in g.items():
            for n in node_list:
                g_edges.append((node, n))

        G_prime = self.G.copy()
        G_prime.add_nodes_from(self.G.nodes())
        G_prime.add_edges_from(g_edges)

        l_max = max(l_k.values())
        l_reverse = {v: l_max - l for v, l in l_k.items()}

        nx.set_node_attributes(G_prime, l_reverse, "subset")

        pos = nx.multipartite_layout(G_prime)

        for node, layer in l_k.items():
            pos[node][0] = l_max - layer

        vert = list(set([pos[node][1] for node in self.G.nodes()]))
        vert.sort()
        for node in self.G.nodes():
            pos[node][1] = vert.index(pos[node][1])

        return pos

    def get_pos_wo_structure(self) -> dict[int, tuple[float, float]]:
        """
        Returns the position of nodes based on the graph.

        Returns
        -------
        pos : dict
            dictionary of node positions.

        Returns
        -------
        pos : dict
            dictionary of node positions.
        """

        layers = dict()
        connected_components = list(nx.connected_components(self.G))

        for component in connected_components:
            subgraph = self.G.subgraph(component)
            initial_pos = {node: (0, 0) for node in component}

            if len(set(self.v_out) & set(component)) == 0 and len(set(self.v_in) & set(component)) == 0:
                pos = nx.spring_layout(subgraph)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                for k, node in enumerate(order[::-1]):
                    layers[node] = k

            elif len(set(self.v_out) & set(component)) > 0 and len(set(self.v_in) & set(component)) == 0:
                fixed_nodes = list(set(self.v_out) & set(component))
                for i, node in enumerate(fixed_nodes):
                    initial_pos[node] = (10, i)
                    layers[node] = 0
                pos = nx.spring_layout(subgraph, pos=initial_pos, fixed=fixed_nodes)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                order = [node for node in order if node not in fixed_nodes]
                Nv = len(self.v_out)
                for i, node in enumerate(order[::-1]):
                    k = i // Nv + 1
                    layers[node] = k

            elif len(set(self.v_out) & set(component)) == 0 and len(set(self.v_in) & set(component)) > 0:
                fixed_nodes = list(set(self.v_in) & set(component))
                for i, node in enumerate(fixed_nodes):
                    initial_pos[node] = (-10, i)
                pos = nx.spring_layout(subgraph, pos=initial_pos, fixed=fixed_nodes)
                # order the nodes based on the x-coordinate
                order = sorted(pos, key=lambda x: pos[x][0])
                order = [node for node in order if node not in fixed_nodes]
                Nv = len(self.v_in)
                for i, node in enumerate(order[::-1]):
                    k = i // Nv
                    layers[node] = k
                if layers == dict():
                    layer_input = 0
                else:
                    layer_input = max(layers.values()) + 1
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
                Nv = len(self.v_out)
                for i, node in enumerate(order[::-1]):
                    k = i // Nv + 1
                    layers[node] = k
                layer_input = max(layers.values()) + 1
                for node in set(self.v_in) & set(component) - set(self.v_out):
                    layers[node] = layer_input

        G_prime = self.G.copy()
        G_prime.add_nodes_from(self.G.nodes())
        G_prime.add_edges_from(self.G.edges())
        l_max = max(layers.values())
        l_reverse = {v: l_max - l for v, l in layers.items()}
        nx.set_node_attributes(G_prime, l_reverse, "subset")
        pos = nx.multipartite_layout(G_prime)
        for node, layer in layers.items():
            pos[node][0] = l_max - layer
        vert = list(set([pos[node][1] for node in self.G.nodes()]))
        vert.sort()
        for node in self.G.nodes():
            pos[node][1] = vert.index(pos[node][1])
        return pos

    def get_pos_all_correction(self, layers: dict[int, int]) -> dict[int, tuple[float, float]]:
        """
        Returns the position of nodes based on the pattern

        Parameters
        ----------
        layers : dict
            Layer mapping obtained from the measurement order of the pattern.

        Returns
        -------
        pos : dict
            dictionary of node positions.
        """

        G_prime = self.G.copy()
        G_prime.add_nodes_from(self.G.nodes())
        G_prime.add_edges_from(self.G.edges())
        nx.set_node_attributes(G_prime, layers, "subset")
        pos = nx.multipartite_layout(G_prime)
        for node, layer in layers.items():
            pos[node][0] = layer
        vert = list(set([pos[node][1] for node in self.G.nodes()]))
        vert.sort()
        for node in self.G.nodes():
            pos[node][1] = vert.index(pos[node][1])
        return pos

    @staticmethod
    def _edge_intersects_node(start, end, node_pos, buffer=0.2):
        """
        Determine if an edge intersects a node.
        """
        start = np.array(start)
        end = np.array(end)
        if np.all(start == end):
            return False
        node_pos = np.array(node_pos)
        # Vector from start to end
        line_vec = end - start
        # Vector from start to node_pos
        point_vec = node_pos - start
        t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)

        if t < 0.0 or t > 1.0:
            return False
        # Find the projection point
        projection = start + t * line_vec
        distance = np.linalg.norm(projection - node_pos)

        return distance < buffer

    @staticmethod
    def _control_point(start, end, node_pos, distance=0.6):
        """
        Generate a control point to bend the edge around a node.
        """
        edge_vector = np.array(end) - np.array(start)
        # Rotate the edge vector 90 degrees or -90 degrees according to the node position
        cross = np.cross(edge_vector, np.array(node_pos) - np.array(start))
        if cross > 0:
            dir_vector = np.array([edge_vector[1], -edge_vector[0]])  # Rotate the edge vector 90 degrees
        else:
            dir_vector = np.array([-edge_vector[1], edge_vector[0]])
        dir_vector = dir_vector / np.linalg.norm(dir_vector)  # Normalize the vector
        control = node_pos + distance * dir_vector
        return control.tolist()

    @staticmethod
    def _bezier_curve(bezier_path, t):
        """
        Generate a bezier curve from a list of points.
        """
        n = len(bezier_path) - 1  # order of the curve
        curve = np.zeros((len(t), 2))
        for i, point in enumerate(bezier_path):
            curve += np.outer(comb(n, i) * ((1 - t) ** (n - i)) * (t**i), np.array(point))
        return curve

    def _check_path(self, path, target_node_pos=None):
        """
        if there is an acute angle in the path, merge points
        """
        path = np.array(path)
        acute = True
        max_iter = 100
        iter = 0
        while acute:
            if iter > max_iter:
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
                    else:
                        mean = (path[i + 1] + path[i + 2]) / 2
                        path = np.delete(path, i + 1, 0)
                        path = np.delete(path, i + 1, 0)
                        path = np.insert(path, i + 1, mean, 0)
                        break
                iter += 1
            else:
                acute = False
        new_path = path.tolist()
        if target_node_pos is not None:
            for point in new_path[:-1]:
                if np.linalg.norm(np.array(point) - np.array(target_node_pos)) < 0.2:
                    new_path.remove(point)
        return new_path


def comb(n, r):
    """
    returns the binomial coefficient of n and r.
    """
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
