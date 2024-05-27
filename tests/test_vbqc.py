import unittest
import graphix.pauli
import tests.random_circuit as rc
from copy import deepcopy
import random
import graphix.gflow
import graphix.visualization
import networkx as nx


class TestVBQC(unittest.TestCase):

    def test_graph(self) :
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize()
        # pattern.draw_graph()
        graph = nx.Graph()
        graph.add_nodes_from(pattern.get_graph()[0])
        graph.add_edges_from(pattern.get_graph()[1])

        depth, layers = pattern.get_layers()
        layers = {element: key for key, value_set in layers.items() for element in value_set}
        
        for output in pattern.output_nodes:
            layers[output] = depth + 1
        xflow, zflow = graphix.gflow.get_corrections_from_pattern(pattern)
        visualization = graphix.visualization.GraphVisualizer(
            G= graph,
            v_in=pattern.input_nodes,
            v_out=pattern.output_nodes,
            meas_plane=pattern.get_meas_plane(),
            meas_angles=pattern.get_angles(),
            local_clifford=pattern.get_vops()
        )
        
        visualization.visualize_all_correction(
            layers=layers,
            xflow=dict(),
            zflow=zflow,
            show_pauli_measurement=False,
            show_local_clifford=False,
            show_measurement_planes=False,
            node_distance=(1,1),
            figsize=None,
        )

        visualization.visualize_all_correction(
            layers=layers,
            xflow=xflow,
            zflow=dict(),
            show_pauli_measurement=False,
            show_local_clifford=False,
            show_measurement_planes=False,
            node_distance=(1,1),
            figsize=None,
        )
        visualization.visualize_all_correction(
            layers=layers,
            xflow=dict(),
            zflow=dict(),
            show_pauli_measurement=False,
            show_local_clifford=False,
            show_measurement_planes=False,
            node_distance=(1,1),
            figsize=None,
        )


        self.assertTrue(True)

    def test_stabilizer_chain(self) :
        from graphix.gflow import find_flow
        import networkx as nx
        for _ in range(10) :
            print("################################################")
            nqubits = 5
            depth = 3
            circuit = rc.get_rand_circuit(nqubits, depth)
            pattern = circuit.transpile()
            # pattern.standardize(method="global")

            graph = nx.Graph()
            nodes, edges = pattern.get_graph()
            graph.add_edges_from(edges)
            graph.add_nodes_from(nodes)

            f, _ = find_flow(
                graph=graph,
                input=set(pattern.input_nodes),
                output=set(pattern.output_nodes),
                meas_planes=pattern.get_meas_plane()
            )
            f_inv = {}
            for prev, next in f.items() :
                f[prev] = list(next)[0]
                f_inv[list(next)[0]] = prev


            # print(f)
            # print(f_inv)

            # graph = pattern.get_graph()
            measurement_db = pattern.get_measurement_db()
            byproduct_db = pattern.get_byproduct_db()


            def get_s_z(target_node, flow, graph:nx.Graph) :
                s_z = []
                for node in graph.nodes :
                    if node not in pattern.output_nodes :
                        node_after = flow[node]
                        if target_node in graph.neighbors(node_after) and node != target_node :
                            s_z.append(node)
                return s_z
            
            def get_s_x(target_node, flow_inv) :
                return [flow_inv[target_node]] if target_node not in pattern.input_nodes else []

            # for node in graph.nodes :
            #     print(node, get_s_z(target_node=node, flow=f, graph=graph), get_s_x(target_node=node, flow_inv=f_inv))

            
            def get_stabilizer_chain(nodes:list) :
                stabilizer = dict.fromkeys(sorted(graph.nodes), graphix.pauli.I)
                for i in nodes :
                    stabilizer[i] @= graphix.pauli.X
                    for j in graph.neighbors(i) :
                        stabilizer[j] @= graphix.pauli.Z

                ## Print the stabilizer
                bug = False

                # print("INPUT NODES : ", pattern.input_nodes)
                # print("OUTPUT NODES : ", pattern.output_nodes)
                # print("GRAPH NODES : ", sorted(graph.nodes))
                # print("f-1 : ", f_inv)
                for node, pauli in stabilizer.items() :
                    print(node, pauli, "" if x_dict[node]==0 else "x_dep") 

                for node, pauli in stabilizer.items() :
                    if pauli == graphix.pauli.Z and node not in pattern.input_nodes:
                        bug = True
                        buggy = node
                        print("BUGS : ")
                        print(node, pauli)
                        print(f"f-1({node}) : {f_inv[node]}")
                        print(list(graph.neighbors(node)))
                        for i in graph.neighbors(node) :
                            print("Neighbor ", i)
                            if i in pattern.input_nodes :
                                print(f"input")
                            else :
                                print(f"f-1 : {f_inv[i]}")
                            if i in pattern.output_nodes :
                                print(f"output")
                            else :
                                print(f"f : {f[i]}")
                if bug == True :
                    print(nodes)
                    print(graph.edges)
                self.assertFalse(bug)



            x_dict = {}
            for node in graph.nodes: 
                x_dict[node] = random.randint(0, 1)


            def extend_s_z(S:list, flow, graph:nx.Graph) :
                extended_S = deepcopy(S)
                extension = []
                # print("Extending nodes ", S)
                for node in S :
                    node_s_z = get_s_z(target_node=node, flow=flow, graph=graph)
                    extension.extend(node_s_z)
                    if x_dict[node] :
                        extension.extend(get_s_x(target_node=node, flow_inv=f_inv))
                return extended_S, extension

            for output_node in pattern.output_nodes :
                print("ANALYZING OUTPUT NODE ", output_node)
                S_1 = get_s_z(target_node=output_node, flow=f, graph=graph)
                # print("Has S_Z = ", S_1)
                _, extension = extend_s_z(S=S_1, flow=f, graph=graph)
                i = 1
                # print(f"Extension {i} is {S_1 + extension}")
                while len(extension) != 0 :
                    S_1.extend(extension)
                    _, extension = extend_s_z(S=extension, flow=f, graph=graph)
                    i += 1
                    # print(f"Extension {i} is \n{S_1 + extension} compared to \n{S_1}")
                    # input()

                extended_dependencies = [output_node] + S_1
                # print(extended_dependencies)
                get_stabilizer_chain(nodes=extended_dependencies)

            # def get_extended_dependencies(output_node) :
            #     """
            #     Function to apply to an output node only
            #     """
            #     S = byproduct_db[output_node]["z-domain"]
            #     # print("Studying new output node")
            #     # print(output_node, S)
            #     original, extended = extend_dependencies(S)
            #     # print(f"Now comparing old {S} and new {extended_S}")
            #     while len(extended) != 0 :
            #         # print("Need more extension for ", difference)
            #         # Current dependency set becomes the previous one 
            #         original = original + extended 
            #         # Compute new extension
            #         ext, new = extend_dependencies(extended)
            #         extended = new

            #         # print(f"Now comparing old {S} and new {extended_S}")
            #     return original + extended + [output_node]

            # def extend_dependencies(S) :
            #     # print("Compute new extension of ", S)
            #     """
            #     Function to apply on a set of measured nodes only
            #     """
            #     dependency_set = deepcopy(S)
            #     added = []
            #     for measured_node in S :
            #         measurement_parameters = measurement_db[measured_node]
            #         t_domain = measurement_parameters.t_domain
            #         added.extend(t_domain)
            #         if x_dict[measured_node] :
            #             s_domain = measurement_parameters.s_domain
            #             if len(s_domain) != 0 :
            #                 added.append(s_domain[-1])
            #         # print(measured_node, t_domain)
            #         # print(S)
            #     # print("Finished extension with ", S)
            #     return dependency_set, added

            # print(pattern.input_nodes)
            
            # x_dict = {}
            # for node in graph[0] :
            #     if node not in pattern.output_nodes :
            #         x = random.randint(0,1)
            #         x_dict[node] = 0
            #         if x :
            #             print("s_domain for ", node)


            # for output_node in byproduct_db :
            #     print("STABILIZER CHAIN FOR OUTPUT NODE ", output_node)
            #     extended_z_dependencies = get_extended_dependencies(output_node)
            #     get_stabilizer_chain(extended_z_dependencies)
            #         # print(measured_node)
