import dataclasses
import numpy as np
from graphix.clifford import CLIFFORD_CONJ, CLIFFORD, CLIFFORD_MUL
import graphix.ops
from graphix.states import State, PlanarState, BasicStates
import graphix.pattern
import graphix.sim.base_backend
import graphix.sim.statevec
import graphix.simulator
from graphix.pauli import Plane
import graphix.pauli
from copy import deepcopy
import typing
import networkx as nx
"""
Usage:

client = Client(pattern:Pattern, blind=False) ## For pure MBQC
sv_backend = StatevecBackend(client.pattern, meas_op = client.meas_op)

simulator = PatternSimulator(client.pattern, backend=sv_backend)
simulator.run()

"""


@dataclasses.dataclass
class TrappifiedRun :
    input_state = list
    tested_qubits = list[int]
    stabilizer = graphix.pauli

@dataclasses.dataclass
class Secret_a:
    a : dict[int, int]
    a_N : dict[int, int]

@dataclasses.dataclass
class Secrets:
    r : bool = False
    a : bool = False
    theta : bool = False

@dataclasses.dataclass
class SecretDatas:
    r : dict[int, int]
    a : Secret_a
    theta : dict[int, int]

    @staticmethod
    def from_secrets(secrets: Secrets, graph, input_nodes, output_nodes):
        node_list, edge_list = graph
        r = {}
        if secrets.r:
            # Need to generate the random bit for each measured qubit, 0 for the rest (output qubits)
            for node in node_list :
                r[node] = np.random.randint(0, 2) if node not in output_nodes else 0

        theta = {}
        if secrets.theta :
            # Create theta secret for all non-output nodes (measured qubits)
            for node in node_list :
                theta[node] = np.random.randint(0,8) if node not in output_nodes else 0    # Expressed in pi/4 units

        a = {}
        a_N = {}
        if secrets.a :
            # Create `a` secret for all
            for node in node_list :
                a[node] = np.random.randint(0,2)
            
            # After all the `a` secrets have been generated, the `a_N` value can be
            # computed from the graph topology
            for i in node_list :
                a_N_value = 0
                for j in node_list :
                    if (i,j) in edge_list or (j,i) in edge_list :
                        a_N_value ^= a[j]
                a_N[i] = a_N_value 
                    
        return SecretDatas(r, Secret_a(a, a_N), theta)

@dataclasses.dataclass
class MeasureParameters:
    plane: graphix.pauli.Plane
    angle: float
    s_domain: list[int]
    t_domain: list[int]
    vop: int

class Client:
    def __init__(self, pattern, input_state=None, secrets=Secrets()):
        self.initial_pattern = pattern

        self.input_nodes = self.initial_pattern.input_nodes.copy()
        self.output_nodes = self.initial_pattern.output_nodes.copy()
        self.graph = self.initial_pattern.get_graph()
        self.nodes_list = self.graph[0]

        # Copy the pauli-preprocessed nodes' measurement outcomes
        self.results = pattern.results.copy()
        self.measure_method = ClientMeasureMethod(self)

        self.measurement_db = pattern.get_measurement_db()
        self.byproduct_db = pattern.get_byproduct_db()

        self.secrets = SecretDatas.from_secrets(secrets, self.graph, self.input_nodes, self.output_nodes)

        pattern_without_flow = pattern.remove_flow()
        self.clean_pattern = self.remove_prepared_nodes(pattern_without_flow)

        self.state = None
        self.input_state = input_state if input_state!= None else [BasicStates.PLUS for _ in self.input_nodes]


    def remove_prepared_nodes(self, pattern) :
        clean_pattern = graphix.Pattern(self.input_nodes)
        for cmd in pattern :
            if cmd[0] == "N" :
                clean_pattern.add_auxiliary_node(node=cmd[1])
            else :
                clean_pattern.add(cmd)
        return clean_pattern

    def blind_qubits(self, backend) :
        z_rotation = lambda theta : np.array([[1, 0], [0, np.exp(1j*theta*np.pi/4)]])
        x_blind = lambda a : graphix.pauli.X if (a == 1) else graphix.pauli.I
        for node in self.nodes_list :
            index = self.node_index.index(node)
            theta = self.secrets.theta.get(node, 0)
            a = self.secrets.a.a.get(node, 0)
            self.state = backend.apply_single(state=self.state, op=x_blind(a).matrix, i=index)
            self.state = backend.apply_single(state=self.state, op=z_rotation(theta), i=index)

    def prepare_states(self, backend) :
        # First prepare inputs
        state, node_index = backend.add_nodes(input_state=self.state, node_index=[], nodes=self.input_nodes, data=self.input_state)
        
        # Then iterate over auxiliaries required to blind
        aux_nodes = []
        for node in self.nodes_list :
            if node not in self.input_nodes and node not in self.output_nodes :
                aux_nodes.append(node)
        aux_data = [BasicStates.PLUS for _ in aux_nodes]
        state, node_index = backend.add_nodes(input_state=state, node_index=node_index, nodes = aux_nodes, data=aux_data)   

        # Prepare outputs
        output_data = []
        for node in self.output_nodes :
            r_value = self.secrets.r.get(node, 0)
            a_N_value = self.secrets.a.a_N.get(node, 0)
            output_data.append(BasicStates.PLUS if r_value^a_N_value == 0 else BasicStates.MINUS)
        state, node_index = backend.add_nodes(input_state=state, node_index=node_index, nodes = self.output_nodes, data=output_data)   

        self.state, self.node_index = state, node_index
        return


    def create_test_runs(self) -> tuple[list[TrappifiedRun], dict[int, int]] :
        graph = nx.Graph()
        nodes, edges = self.graph
        graph.add_edges_from(edges)
        graph.add_nodes_from(nodes)
        coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
        
        colors = set(coloring.values())
        nodes_by_color = {c:[] for c in colors}
        for node in sorted(graph.nodes) :
            color = coloring[node]
            nodes_by_color[color].append(node)            
        runs = []
        # 1 color = 1 set of traps, but 1 test run so 1 stabilizer
        for color in colors :
            run = TrappifiedRun()
                # Build stabilizer (1 per test run)
            stabilizer = dict.fromkeys(sorted(graph.nodes), graphix.pauli.I)
            # single-qubit traps
            for node in nodes_by_color[color] :
                stabilizer[node] @= graphix.pauli.X
                for n in nx.neighbors(graph, node) :
                    stabilizer[n] @= graphix.pauli.Z
            
            states_to_prepare = []
            for index in sorted(graph.nodes) :
                if stabilizer[index] == graphix.pauli.X :
                    state = BasicStates.PLUS
                if stabilizer[index] == graphix.pauli.Y :
                    state = BasicStates.PLUS_I
                if stabilizer[index] == graphix.pauli.Z :
                    state = BasicStates.ZERO
                else :
                    state = BasicStates.PLUS
                states_to_prepare.append(state)
            run.input_state = states_to_prepare
            run.tested_qubits = nodes_by_color[color]
            run.stabilizer = stabilizer
            runs.append(run)
        return runs, coloring

    def delegate_test_run(self, backend, run:TrappifiedRun) :
        self.state, self.node_index = backend.add_nodes(input_state = None, node_index=[], nodes=sorted(self.graph[0]), data=run.input_state)
        # self.prepare_states(backend) 
        self.blind_qubits(backend)

        graph = nx.Graph()
        nodes, edges = self.graph
        graph.add_edges_from(edges)
        graph.add_nodes_from(nodes)

        new_measurement_db = dict()
        for node in self.measurement_db :
            new_measurement_db[node] = MeasureParameters(plane=graphix.pauli.Plane.XY, angle=0, s_domain=[], t_domain=[], vop=0)
        self.measurement_db = new_measurement_db
        
        
        sim = graphix.simulator.PatternSimulator(state=self.state, node_index=self.node_index, backend=backend, pattern=self.clean_pattern, measure_method=self.measure_method)
        self.state, self.node_index = sim.run()

        # returns the final state as well as the server object (Simulator)
        for single_qubit_trap in run.tested_qubits :
            if single_qubit_trap not in self.output_nodes :
                print(f"Result for trap around {int(single_qubit_trap)} : {int(self.results[single_qubit_trap])}")

        return self.state


    def delegate_pattern(self, backend):
        self.state = None
        self.prepare_states(backend)      
        self.blind_qubits(backend)

        sim = graphix.simulator.PatternSimulator(state=self.state, node_index=self.node_index, backend=backend, pattern=self.clean_pattern, measure_method=self.measure_method)
        self.state, self.node_index = sim.run()
        self.decode_output_state(backend)
        # returns the final state as well as the server object (Simulator)
        return self.state, sim


    def decode_output_state(self, backend):
        for node in self.output_nodes:
            if node in self.byproduct_db:
                index = self.node_index.index(node)
                z_decoding, x_decoding = self.decode_output(node)
                if z_decoding:
                    self.state = backend.apply_single(state=self.state, op=graphix.ops.Ops.z, i=index)
                if x_decoding:
                    self.state = backend.apply_single(state=self.state, op=graphix.ops.Ops.x, i=index)
        return 

    def get_secrets_size(self):
        secrets_size = {}
        for secret in self.secrets:
            secrets_size[secret] = len(self.secrets[secret])
        return secrets_size

    def decode_output(self, node):
        z_decoding = np.sum(self.results[z_dep] for z_dep in self.byproduct_db[node]['z-domain'])%2
        z_decoding ^= self.secrets.r.get(node, 0)
        x_decoding = np.sum(self.results[x_dep] for x_dep in self.byproduct_db[node]['x-domain'])%2
        x_decoding ^= self.secrets.a.a.get(node, 0)
        return z_decoding, x_decoding


class ClientMeasureMethod(graphix.simulator.MeasureMethod):
    def __init__(self, client: Client):
        self.__client = client


    def get_measurement_description(self, cmd, results) -> graphix.simulator.MeasurementDescription:
        node = cmd[1]

        parameters = self.__client.measurement_db[node]

        r_value = self.__client.secrets.r.get(node, 0)
        theta_value = self.__client.secrets.theta.get(node, 0)
        a_value = self.__client.secrets.a.a.get(node, 0)
        a_N_value = self.__client.secrets.a.a_N.get(node, 0)

        # extract signals for adaptive angle
        s_signal = np.sum(self.__client.results[j] for j in parameters.s_domain)
        t_signal = np.sum(self.__client.results[j] for j in parameters.t_domain)
        measure_update = graphix.pauli.MeasureUpdate.compute(
            parameters.plane, s_signal % 2 == 1, t_signal % 2 == 1, graphix.clifford.TABLE[parameters.vop]
        )
        angle=parameters.angle
        angle = angle * measure_update.coeff + measure_update.add_term
        angle = (-1)**a_value * angle + theta_value*np.pi/4 + np.pi * (r_value + a_N_value)
        # angle = angle * measure_update.coeff + measure_update.add_term
        return graphix.simulator.MeasurementDescription(measure_update.new_plane, angle)
        # return graphix.sim.base_backend.MeasurementDescription(measure_update.new_plane, angle)

    def set_measure_result(self, node, result: bool) -> None:
        if self.__client.secrets.r:
            result ^= self.__client.secrets.r[node]
        self.__client.results[node] = result

