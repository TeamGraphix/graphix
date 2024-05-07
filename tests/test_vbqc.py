import unittest
import graphix.pauli
import tests.random_circuit as rc
from copy import deepcopy
import random

class TestVBQC(unittest.TestCase):

    def test_stabilizer_chain(self) :
        for _ in range(10) :
            nqubits = 5
            depth = 3
            circuit = rc.get_rand_circuit(nqubits, depth)
            pattern = circuit.transpile()
            pattern.standardize(method="global")
            graph = pattern.get_graph()
            measurement_db = pattern.get_measurement_db()
            byproduct_db = pattern.get_byproduct_db()
            print(byproduct_db)


            
            def get_stabilizer_chain(nodes:list) :
                stabilizer = dict.fromkeys(graph[0], graphix.pauli.I)
                for i in nodes :
                    stabilizer[i] @= graphix.pauli.X
                    for j in graph[0] :
                        if (i,j) in graph[1] or (j,i) in graph[1] :
                            stabilizer[j] @= graphix.pauli.Z

                ## Print the stabilizer
                bug = False

                # for node, pauli in stabilizer.items() :
                #     print(node, pauli) 

                for node, pauli in stabilizer.items() :
                    if pauli == graphix.pauli.Z and node not in pattern.input_nodes:
                        bug = True
                        buggy = node
                        print(node, pauli) 
                if bug == True :
                    print(nodes)
                    print(graph)
                    print(measurement_db[buggy].t_domain)
                    for node in measurement_db :
                        print(node, measurement_db[node].t_domain, measurement_db[node].s_domain)
                    print(byproduct_db)            
                    pattern.draw_graph()

            def get_extended_dependencies(output_node) :
                """
                Function to apply to an output node only
                """
                S = byproduct_db[output_node]["z-domain"]
                # print("Studying new output node")
                # print(output_node, S)
                original, extended = extend_dependencies(S)
                # print(f"Now comparing old {S} and new {extended_S}")
                while len(extended) != 0 :
                    # print("Need more extension for ", difference)
                    # Current dependency set becomes the previous one 
                    original = original + extended 
                    # Compute new extension
                    ext, new = extend_dependencies(extended)
                    extended = new

                    # print(f"Now comparing old {S} and new {extended_S}")
                return original + extended + [output_node]

            def extend_dependencies(S) :
                # print("Compute new extension of ", S)
                """
                Function to apply on a set of measured nodes only
                """
                dependency_set = deepcopy(S)
                added = []
                for measured_node in S :
                    measurement_parameters = measurement_db[measured_node]
                    t_domain = measurement_parameters.t_domain
                    added.extend(t_domain)
                    if x_dict[measured_node] :
                        s_domain = measurement_parameters.s_domain
                        if len(s_domain) != 0 :
                            added.append(s_domain[-1])
                    # print(measured_node, t_domain)
                    # print(S)
                # print("Finished extension with ", S)
                return dependency_set, added

            print(pattern.input_nodes)
            
            x_dict = {}
            for node in graph[0] :
                if node not in pattern.output_nodes :
                    x = random.randint(0,1)
                    x_dict[node] = 0
                    if x :
                        print("s_domain for ", node)


            for output_node in byproduct_db :
                print("STABILIZER CHAIN FOR OUTPUT NODE ", output_node)
                extended_z_dependencies = get_extended_dependencies(output_node)
                get_stabilizer_chain(extended_z_dependencies)
                    # print(measured_node)
