import numpy as np
from matplotlib import pyplot as plt
import math
import networkx as nx
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

    def __init__(self, G, v_in, v_out, meas_plane=None):
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
        """
        self.G = G
        self.v_in = v_in
        self.v_out = v_out
        if meas_plane is None:
            self.meas_plane = {i: "XY" for i in iter(G.nodes)}
        else:
            self.meas_plane = meas_plane

    def visualize(
        self,
        angles=None,
        local_clifford=None,
        node_distance=(1, 1),
        show_loop=True,
        figsize=None,
        save=False,
        filename=None,
    ):
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
        local_clifford : dict
            Indexes of local clifford operations for each nodes.
            If not None, indexes of the local Clifford operator are displayed adjacent to the nodes.
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

        f, l_k = gflow.flow(self.G, set(self.v_in), set(self.v_out), meas_planes=self.meas_plane)
        if f:
            print("Flow found.")
            self.visualize_w_flow(f, l_k, angles, local_clifford, node_distance, figsize, save, filename)
        else:
            g, l_k = gflow.gflow(self.G, set(self.v_in), set(self.v_out), self.meas_plane)
            if g:
                print("No flow found. Gflow found.")
                self.visualize_w_gflow(
                    g, l_k, angles, local_clifford, node_distance, show_loop, figsize, save, filename
                )
            else:
                print("No flow or gflow found.")
                self.visualize_wo_structure(angles, local_clifford, node_distance, save, filename)

    def visualize_w_flow(
        self, f, l_k, angles=None, local_clifford=None, node_distance=(1, 1), figsize=None, save=False, filename=None
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
        angles : dict
            Measurement angles for each nodes on the graph (unit of pi), except output nodes.
            If not None, the nodes with Pauli measurement angles are colored light blue.
        local_clifford : dict
            Indexes of local clifford operations for each nodes.
            If not None, indexes of the local Clifford operator are displayed adjacent to the nodes.
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

        # Change the x coordinates of the nodes based on their layer, sort in descending order
        for node, layer in l_k.items():
            pos[node] = (
                (max(l_k.values()) - layer) * node_distance[0],
                pos[node][1] * node_distance[1],
            )  # Subtracting from max for descending order

        # Draw the arrows
        for a, b in f.items():
            nx.draw_networkx_edges(self.G, pos, edgelist=[(a, b)], edge_color="black", arrowstyle="->", arrows=True)

        # Draw the dashed edges
        edge_path = self.get_edge_path(f, pos)
        for edge in edge_path.keys():
            if f.get(edge[0]) != edge[1] and f.get(edge[1]) != edge[0]:  # This edge is not an arrow
                if len(edge_path[edge]) == 2:
                    nx.draw_networkx_edges(self.G, pos, edgelist=[edge], style="dashed", alpha=0.7)
                else:
                    t = np.linspace(0, 1, 100)
                    curve = self.bezier_curve(edge_path[edge], t)
                    plt.plot(curve[:, 0], curve[:, 1], "k--", linewidth=1, alpha=0.7)

        # Draw the nodes with different colors based on their role (input, output, or other)
        for node in self.G.nodes():
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
            for node in self.G.nodes():
                if node in local_clifford.keys():
                    plt.text(*pos[node] + np.array([0.2, 0.2]), f"{local_clifford[node]}", fontsize=10, zorder=3)

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
        g,
        l_k,
        angles=None,
        local_clifford=None,
        node_distance=(1, 1),
        show_loop=True,
        figsize=None,
        save=False,
        filename=None,
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
        angles : dict
            Measurement angles for each nodes on the graph (unit of pi), except output nodes.
            If not None, the nodes with Pauli measurement angles are colored light blue.
        local_clifford : dict
            Indexes of local clifford operations for each nodes.
            If not None, indexes of the local Clifford operator are displayed adjacent to the nodes.
        node_distance : tuple
            Distance multiplication factor between nodes for x and y directions.
        show_loop : bool
            whether or not to show loops for graphs with gflow. defaulted to True.
        figsize : tuple
            Figure size of the plot.
        save : bool
            If True, the plot is saved as a png file.
        filename : str
            Filename of the saved plot.
        """

        pos = self.get_pos_from_gflow(g, l_k)
        pos = {k: (v[0] * node_distance[0], v[1] * node_distance[1]) for k, v in pos.items()}  # Scale the layout

        edge_path = self.get_edge_path(g, pos)

        if figsize is None:
            figsize = self.get_figsize(l_k, pos, node_distance=node_distance)
        plt.figure(figsize=figsize)

        for edge in edge_path.keys():
            if edge[1] in g.get(edge[0], set()) or edge[0] in g.get(edge[1], set()):  # This edge is an arrow
                if edge[0] == edge[1]:  # self loop
                    if show_loop:
                        t = np.linspace(0, 1, 100)
                        curve = self.bezier_curve(edge_path[edge], t)
                        plt.plot(curve[:, 0], curve[:, 1], c="k", linewidth=1)
                        plt.annotate(
                            "",
                            xy=curve[-1],
                            xytext=curve[-2],
                            arrowprops=dict(arrowstyle="->", color="k", lw=1),
                        )
                elif len(edge_path[edge]) == 2:  # straight line
                    nx.draw_networkx_edges(
                        self.G, pos, edgelist=[edge], edge_color="black", arrowstyle="->", arrows=True
                    )
                else:
                    path = edge_path[edge]
                    last = np.array(path[-1])
                    second_last = np.array(path[-2])
                    path[-1] = list(
                        last - (last - second_last) / np.linalg.norm(last - second_last) * 0.2
                    )  # Shorten the last edge not to hide arrow under the node

                    t = np.linspace(0, 1, 100)
                    curve = self.bezier_curve(path, t)

                    plt.plot(curve[:, 0], curve[:, 1], c="k", linewidth=1)
                    plt.annotate(
                        "",
                        xy=curve[-1],
                        xytext=curve[-2],
                        arrowprops=dict(arrowstyle="->", color="k", lw=1),
                    )
            else:
                if len(edge_path[edge]) == 2:
                    nx.draw_networkx_edges(self.G, pos, edgelist=[edge], style="dashed", alpha=0.7)
                else:
                    t = np.linspace(0, 1, 100)
                    curve = self.bezier_curve(edge_path[edge], t)
                    plt.plot(curve[:, 0], curve[:, 1], "k--", linewidth=1, alpha=0.7)

        # Draw the nodes with different colors based on their role (input, output, or other)
        for node in self.G.nodes():
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
            for node in self.G.nodes():
                if node in local_clifford.keys():
                    plt.text(*pos[node] + np.array([0.2, 0.2]), f"{local_clifford[node]}", fontsize=10, zorder=3)

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

    def visualize_wo_structure(self, angles=None, local_clifford=None, node_distance=(1, 1), save=False, filename=None):
        """
        visualizes the graph without flow or gflow.

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
        angles : dict
            Measurement angles for each nodes on the graph (unit of pi), except output nodes.
            If not None, the nodes with Pauli measurement angles are colored light blue.
        local_clifford : dict
            Indexes of local clifford operations for each nodes.
            If not None, indexes of the local Clifford operator are displayed adjacent to the nodes.
        node_distance : tuple
            Distance multiplication factor between nodes for x and y directions.
        figsize : tuple
            Figure size of the plot.
        save : bool
            If True, the plot is saved as a png file.
        filename : str
            Filename of the saved plot.
        """

        scale = max(2 * np.log(len(self.G.nodes())), 5)
        plt.figure(figsize=(scale, (2 / 3) * scale))
        k_val = 2 / np.sqrt(len(self.G.nodes()))
        pos = nx.spring_layout(self.G, k=k_val)  # Layout for the nodes
        pos = {k: (v[0] * node_distance[0], v[1] * node_distance[1]) for k, v in pos.items()}  # Scale the layout

        # Draw the edges
        nx.draw_networkx_edges(self.G, pos, edge_color="black")

        # Draw the nodes with different colors based on their role (input, output, or other)
        for node in self.G.nodes():
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
            for node in self.G.nodes():
                if node in local_clifford.keys():
                    plt.text(*pos[node] + np.array([0.04, 0.04]), f"{local_clifford[node]}", fontsize=10, zorder=3)

        # Draw the labels
        fontsize = 12
        if max(self.G.nodes()) >= 100:
            fontsize = fontsize * 2 / len(str(max(self.G.nodes())))
        nx.draw_networkx_labels(self.G, pos, font_size=fontsize)

        if save:
            plt.savefig(filename)
        plt.show()

    def get_figsize(self, l_k, pos=None, node_distance=(1, 1)):
        """
        Returns the figure size of the graph.

        Parameters
        ----------
        l_k : dict
            Layer mapping.

        Returns
        -------
        figsize : tuple
            figure size of the graph.
        """
        width = (max(l_k.values()) + 1) * 0.8
        if pos is not None:
            height = len(set([pos[node][1] for node in self.G.nodes()]))
        elif len(self.v_in) > 0:
            height = len(self.v_in)
        figsize = (width * node_distance[0], height * node_distance[1])

        return figsize

    def get_edge_path(self, fg, pos):
        """
        Returns the path of edges.

        Parameters
        ----------
        fg : dict
            flow or gflow mapping.
        pos : dict
            dictionary of node positions.

        Returns
        -------
        edge_path : dict
            dictionary of edge paths.
        """
        max_iter = 5
        edge_path = {}
        if type(next(iter(fg.values()))) is set:  # fg is gflow
            edge_set1 = set(self.G.edges())
            edge_set2 = {(k, v) for k, values in fg.items() for v in values}
            edge_set = edge_set1.union(edge_set2)
        else:
            edge_set = set(self.G.edges())
        for edge in edge_set:
            if type(next(iter(fg.values()))) is not set and (fg.get(edge[0]) == edge[1] or fg.get(edge[1]) == edge[0]):
                # fg is flow and edge is an arrow
                bezier_path = [pos[edge[0]], pos[edge[1]]]
            elif edge[0] == edge[1]:  # Self loop

                def _point_from_node(pos, dist, angle):
                    angle = np.deg2rad(angle)
                    return [pos[0] + dist * np.cos(angle), pos[1] + dist * np.sin(angle)]

                bezier_path = [
                    _point_from_node(pos[edge[0]], 0.2, 170),
                    _point_from_node(pos[edge[0]], 0.35, 170),
                    _point_from_node(pos[edge[0]], 0.4, 155),
                    _point_from_node(pos[edge[0]], 0.45, 140),
                    _point_from_node(pos[edge[0]], 0.35, 110),
                    _point_from_node(pos[edge[0]], 0.3, 110),
                    _point_from_node(pos[edge[0]], 0.17, 95),
                ]
            else:
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
                            if node != edge[0] and node != edge[1] and self.edge_intersects_node(start, end, pos[node]):
                                intersect = True
                                ctrl_points.append(
                                    [
                                        i,
                                        self.control_point(
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
            bezier_path = self.check_path(bezier_path)
            edge_path[edge] = bezier_path

        return edge_path

    def get_pos_from_flow(self, f, l_k):
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
        pos = nx.spring_layout(self.G)  # Initial layout for the nodes
        n = len(self.v_in)
        for i in range(n):
            k = self.v_in[i]
            pos[k][1] = i
            while k in f.keys():
                k = f[k]
                pos[k][1] = i

        lmax = max(l_k.values())
        # Change the x coordinates of the nodes based on their layer, sort in descending order
        for node, layer in l_k.items():
            pos[node][0] = lmax - layer
        return pos

    def get_pos_from_gflow(self, g, l_k):
        """
        Returns the position of nodes based on the gflow.

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

    @staticmethod
    def edge_intersects_node(start, end, node_pos, buffer=0.2):
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
    def control_point(start, end, node_pos, distance=0.6):
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
    def bezier_curve(bezier_path, t):
        """
        Generate a bezier curve from a list of points.
        """
        n = len(bezier_path) - 1  # order of the curve
        curve = np.zeros((len(t), 2))
        for i, point in enumerate(bezier_path):
            curve += np.outer(comb(n, i) * ((1 - t) ** (n - i)) * (t**i), np.array(point))
        return curve

    def check_path(self, path):
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
        return path.tolist()


def comb(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
