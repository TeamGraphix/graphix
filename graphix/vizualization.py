import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from scipy.special import comb
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
    """

    def __init__(self, g, v_in, v_out):
        """
        g: networkx graph
        v_in: list of input nodes
        v_out: list of output nodes
        """
        self.g = g
        self.v_in = v_in
        self.v_out = v_out

    def visualize(self, angles=None, local_clifford=None, figsize=None, save=False, filename=None):
        """
        Visualizes the graph with flow or gflow structure.
        If there exists a flow structure, then the graph is visualized with the flow structure.
        If flow structure is not found and there exists a gflow structure, then the graph is visualized
        with the gflow structure.
        If neither flow nor gflow structure is found, then the graph is visualized without any structure.

        Parameters
        ----------
        angles : dict
            Measurement angles for each nodes on the graph (unit of pi), except output nodes.
            If not None, the nodes with Pauli measurement angles are colored light blue.
        local_clifford_indicator : bool
            If True, the nindex of local clifford gates are shown by the nodes.
        figsize : tuple
            Figure size of the plot.
        save : bool
            If True, the plot is saved as a png file.
        filename : str
            Filename of the saved plot.
        """

        f, l_k = gflow.flow(self.g, set(self.v_in), set(self.v_out))
        if f:
            self.visualize_w_flow(f, l_k, angles, local_clifford, figsize, save, filename)
        else:
            g, l_k = gflow.gflow(self.g, set(self.v_in), set(self.v_out))
            if g:
                self.visualize_w_gflow(g, l_k, angles, local_clifford, figsize, save, filename)
            else:
                print("No flow or gflow found.")
                nx.draw(self.g, with_labels=True)
                plt.show()

    def visualize_w_flow(self, f, l_k, angles=None, local_clifford=None, figsize=None, save=False, filename=None):
        """
        visualizes the graph with flow structure.

        Nodes are colored based on their role (input, output, or other) and edges are depicted as arrows
        or dashed lines depending on whether they are in the flow mapping. Vertical dashed lines separate
        different layers of the graph. This function does not return anything but plots the graph
        using matplotlib's pyplot.

        Parameters
        ----------
        f : dict
            Flow mapping.
        l_k : dict
            Layer mapping.
        angles : dict
            Measurement angles for each nodes on the graph (unit of pi), except output nodes.
            If not None, the nodes with Pauli measurement angles are colored light blue.
        local_clifford_indicator : bool
            If True, the nindex of local clifford gates are shown by the nodes.
        figsize : tuple
            Figure size of the plot.
        save : bool
            If True, the plot is saved as a png file.
        filename : str
            Filename of the saved plot.
        """
        if figsize is None:
            width = (max(l_k.values()) + 1) * 0.8
            height = len(self.v_in)
            figsize = (width, height)
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.g)  # Initial layout for the nodes

        n = len(self.v_in)
        for i in range(n):
            k = self.v_in[i]
            pos[k][1] = i
            while k in f.keys():
                k = f[k]
                pos[k][1] = i

        # Change the x coordinates of the nodes based on their layer, sort in descending order
        for node, layer in l_k.items():
            pos[node] = (max(l_k.values()) - layer, pos[node][1])  # Subtracting from max for descending order

        # Draw the arrows
        for a, b in f.items():
            nx.draw_networkx_edges(self.g, pos, edgelist=[(a, b)], edge_color="black", arrowstyle="->", arrows=True)

        # Draw the dashed edges
        for edge in self.g.edges():
            if f.get(edge[0]) != edge[1] and f.get(edge[1]) != edge[0]:  # This edge is not an arrow
                intersect = False
                bezier_path = [pos[edge[0]]]
                for node in self.g.nodes():
                    if (
                        node != edge[0]
                        and node != edge[1]
                        and self.edge_intersects_node(pos[edge[0]], pos[edge[1]], pos[node])
                    ):
                        intersect = True
                        ctrl_point = self.control_point(pos[edge[0]], pos[edge[1]], pos[node])
                        bezier_path.append(ctrl_point)
                if not intersect:
                    nx.draw_networkx_edges(self.g, pos, edgelist=[edge], style="dashed", alpha=0.7)
                    continue
                bezier_path.append(pos[edge[1]])
                t = np.linspace(0, 1, 100)
                curve = self.bezier_curve(bezier_path, t)
                plt.plot(curve[:, 0], curve[:, 1], "k--", linewidth=1, alpha=0.7)

        # Draw the nodes with different colors based on their role (input, output, or other)
        for node in self.g.nodes():
            color = "black"  # default color for 'other' nodes
            inner_color = "white"
            if node in self.v_in:
                color = "red"
            if node in self.v_out:
                inner_color = "lightgray"
            elif angles is not None and (angles[node] == 0 or angles[node] == 1 / 2):
                inner_color = "lightblue"
            plt.scatter(
                *pos[node], edgecolor=color, facecolor=inner_color, s=350, zorder=2
            )  # Draw the nodes manually with scatter()

        if local_clifford is not None:
            for node in self.g.nodes():
                if node in local_clifford.keys():
                    plt.text(*pos[node] + np.array([0.2, 0.2]), f"{local_clifford[node]}", fontsize=10, zorder=3)

        # Draw the labels
        nx.draw_networkx_labels(self.g, pos)

        x_min = min([pos[node][0] for node in self.g.nodes()])  # Get the minimum x coordinate
        x_max = max([pos[node][0] for node in self.g.nodes()])  # Get the maximum x coordinate
        y_min = min([pos[node][1] for node in self.g.nodes()])  # Get the minimum y coordinate
        y_max = max([pos[node][1] for node in self.g.nodes()])  # Get the maximum y coordinate

        # Draw the vertical lines to separate different layers
        for layer in range(min(l_k.values()), max(l_k.values())):
            plt.axvline(x=layer + 0.5, color="gray", linestyle="--", alpha=0.5)  # Draw line between layers
        for layer in range(min(l_k.values()), max(l_k.values()) + 1):
            # plt.axvline(x=layer + 0.5, color='gray', linestyle='--')  # Draw line between layers
            plt.text(
                layer, y_min - 0.5, f"l: {max(l_k.values()) - layer}", ha="center", va="top"
            )  # Add layer label at bottom

        plt.xlim(x_min - 0.5, x_max + 0.5)  # Add some padding to the left and right
        plt.ylim(y_min - 1, y_max + 0.5)  # Add some padding to the top and bottom
        plt.show()

        if save:
            plt.savefig(filename)

    def visualize_w_gflow(self, g, l_k, figsize=None):
        """
        visualizes the graph with gflow structure.
        to be implemented
        """
        pass

    @staticmethod
    def edge_intersects_node(start, end, node_pos, buffer=0.2):
        """Determine if an edge intersects a node."""
        dist = np.linalg.norm(
            np.cross(np.array(end) - np.array(start), np.array(start) - np.array(node_pos))
        ) / np.linalg.norm(np.array(end) - np.array(start))
        return (
            dist < buffer
            and min(start[0], end[0]) <= node_pos[0] <= max(start[0], end[0])
            and min(start[1], end[1]) <= node_pos[1] <= max(start[1], end[1])
        )

    @staticmethod
    def control_point(start, end, node_pos, distance=0.6):
        """Generate a control point to bend the edge around a node."""
        edge_vector = np.array(end) - np.array(start)
        # Rotate the edge vector 90 degrees or -90 degrees according to the node position
        cross = np.cross(np.array(end) - np.array(start), np.array(start) - np.array(node_pos))
        if cross > 0:
            dir_vector = np.array([-edge_vector[1], edge_vector[0]])  # Rotate the edge vector 90 degrees
        else:
            dir_vector = np.array([edge_vector[1], -edge_vector[0]])
        dir_vector = dir_vector / np.linalg.norm(dir_vector)  # Normalize the vector
        control = node_pos + distance * dir_vector
        return control.tolist()

    @staticmethod
    def bezier_curve(bezier_path, t):
        n = len(bezier_path) - 1  # order of the curve
        curve = np.zeros((len(t), 2))
        for i, point in enumerate(bezier_path):
            curve += np.outer(comb(n, i) * ((1 - t) ** (n - i)) * (t**i), np.array(point))
        return curve
