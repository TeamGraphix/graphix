import dataclasses
import numpy as np
from graphix.clifford import CLIFFORD_CONJ, CLIFFORD, CLIFFORD_MUL
import graphix.ops
from graphix.states import State, PlanarState, BasicStates
import graphix.pattern
import graphix.sim.base_backend
from graphix.sim.base_backend import BackendState
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

## TODO : extract somewhere else
import random

class Stabilizer :
    def __init__(self, graph:nx.Graph, nodes:list[int]) -> None:
        self.graph = graph
        self.nodes = nodes
        self.chain:list[graphix.pauli.Pauli] = [graphix.pauli.I for _ in self.graph.nodes]
        self.init_chain(nodes)

        
    @property
    def size(self) -> int :
        return len(self.graph.nodes)
    
    @property
    def span(self) -> set[int] :
        span = set(self.nodes)
        for node in self.nodes :
            for neighbor in self.graph.neighbors(node):
                span.add(neighbor)
        return span
    

    def init_chain(self, nodes):
        for node in nodes :
            self.compute_product(node)

    def compute_product(self, node):
        ## Caution : here the stabilizer is re-written because there could be overlap on neighbors only
        self.chain[node] = graphix.pauli.X
        for neighbor in self.graph.neighbors(node) :
            self.chain[neighbor] = graphix.pauli.Z

    def __repr__(self) -> str:
        string = f"""
        Product stabilizer of nodes in {self.nodes}\n
        For graph with edges {self.graph.edges}\n
        """
        for node in sorted(self.graph.nodes):
            string += f"{node} {self.chain[node]}\n"

        return string


class TrappifiedCanvas :
    def __init__(self, graph:nx.Graph, traps_list:list[list[int]]) -> None:
        self.graph = graph
        self.traps_list = traps_list
        stabilizers = [Stabilizer(graph, trap_nodes) for trap_nodes in traps_list]
        self.stabilizer = self.merge_stabilizers(stabilizers)
        print(self.stabilizer)
        dummies_coins = self.generate_coins_dummies()
        self.coins = self.generate_coins_trap_qubits(coins=dummies_coins)
        self.spans = dict(zip(
            self.trap_qubits, [stabilizer.span for stabilizer in stabilizers]
        ))
        self.states = self.generate_eigenstate()

    @property
    def trap_qubits(self) :
        return [node
                for trap in self.traps_list
                for node in trap]
    @property
    def dummy_qubits(self):
        return [neighbor
                for trap in self.traps_list
                for node in trap
                for neighbor in list(self.graph.neighbors(node))]



    def merge_stabilizers(self, stabilizers:list[Stabilizer]) :
        common_stabilizer = Stabilizer(self.graph, [])
        for stabilizer in stabilizers :
            for node in stabilizer.span :
                # If the Pauli was identity, just replace it by the upcoming Pauli
                if common_stabilizer.chain[node] == graphix.pauli.I :
                    common_stabilizer.chain[node] = stabilizer.chain[node]
                else :
                # If the Pauli wasn't identity, the incoming Pauli must coincide with the previous one
                # otherwise it's dramatic
                    if common_stabilizer.chain[node] != stabilizer.chain[node] :
                        print("Huge error")
                        return
                    else :
                        # (nothing to do as they already coincide)
                        pass 
            common_stabilizer.nodes += stabilizer.nodes
        return common_stabilizer
                

    def generate_eigenstate(self) -> list[State] :
        states = []
        for node in sorted(self.stabilizer.graph.nodes) :
            operator = self.stabilizer.chain[node]
            states.append(operator.get_eigenstate(eigenvalue=self.coins[node]))
        return states

    def generate_coins_dummies(self):
        coins = dict()
        for node in self.stabilizer.graph.nodes :
            if node not in self.trap_qubits :
                coins[node]= random.randint(0,1)
                coins[node]= 0
            else :
                coins[node] = 0
        return coins

    def generate_coins_trap_qubits(self, coins):
        for node in self.trap_qubits :
            neighbors_coins = sum(coins[n] for n in self.stabilizer.graph.neighbors(node))%2
            coins[node] = neighbors_coins
        return coins


    def __repr__(self) -> str:
        text = ""
        for node in sorted(self.stabilizer.graph.nodes) :
            trap_span = ""
            if node in self.trap_qubits :
                trap_span += f"{self.spans[node]}"
            text += f"{node} {self.stabilizer.chain[node]} ^ {self.coins[node]} {trap_span} -> {self.states[node]}\n"
        return text


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

        self.backend_state = BackendState()
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
            theta = self.secrets.theta.get(node, 0)
            a = self.secrets.a.a.get(node, 0)
            self.backend_state = backend.apply_single(backendState=self.backend_state, node=node, op=x_blind(a).matrix)
            self.backend_state = backend.apply_single(backendState=self.backend_state, node=node, op=z_rotation(theta))

    def prepare_states(self, backend) :
        # First prepare inputs
        backendState =  backend.add_nodes(backendState=self.backend_state, nodes=self.input_nodes, data=self.input_state)
        
        # Then iterate over auxiliaries required to blind
        aux_nodes = []
        for node in self.nodes_list :
            if node not in self.input_nodes and node not in self.output_nodes :
                aux_nodes.append(node)
        aux_data = [BasicStates.PLUS for _ in aux_nodes]
        backendState =  backend.add_nodes(backendState=backendState, nodes = aux_nodes, data=aux_data)   

        # Prepare outputs
        output_data = []
        for node in self.output_nodes :
            r_value = self.secrets.r.get(node, 0)
            a_N_value = self.secrets.a.a_N.get(node, 0)
            output_data.append(BasicStates.PLUS if r_value^a_N_value == 0 else BasicStates.MINUS)
        backendState =  backend.add_nodes(backendState=backendState, nodes = self.output_nodes, data=output_data)   

        self.backend_state = backendState
        return


    def create_test_runs(self) -> tuple[list[TrappifiedCanvas], dict[int, int]] :
        graph = nx.Graph()
        nodes, edges = self.graph
        graph.add_edges_from(edges)
        graph.add_nodes_from(nodes)
        
        # Create the graph coloring
        coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
        colors = set(coloring.values())
        nodes_by_color = {c:[] for c in colors}
        for node in sorted(graph.nodes) :
            color = coloring[node]
            nodes_by_color[color].append(node)

        # Create the test runs : one per color            
        runs:list[TrappifiedCanvas] = []
        for color in colors :
            # 1 color = 1 test run = 1 set of traps = 1 stabilizer
            trap_qubits = nodes_by_color[color]
            isolated_traps = [[node] for node in trap_qubits]
            trappified_canvas = TrappifiedCanvas(graph, traps_list=isolated_traps)
 
            runs.append(trappified_canvas)
        return runs, coloring

    def delegate_test_run(self, backend, run:TrappifiedCanvas) :
        # The state is entirely prepared and blinded by the client before being sent to the server
        backendState =  BackendState(state=None, node_index=[])
        self.backend_state = backend.add_nodes(backendState, nodes=sorted(self.graph[0]), data=run.states)
        self.blind_qubits(backend)

        # Modify the pattern to be all X-basis measurements, no shifts/signalling updates
        for node in self.measurement_db :
            self.measurement_db[node] = MeasureParameters(plane=graphix.pauli.Plane.XY, angle=0, s_domain=[], t_domain=[], vop=0)

        sim = graphix.simulator.PatternSimulator(backend=backend, pattern=self.clean_pattern, measure_method=self.measure_method)
        self.backend_state = sim.run(backend_state=self.backend_state)
        return self.backend_state


    def delegate_pattern(self, backend):
        self.backend_state = BackendState()
        self.prepare_states(backend)      
        self.blind_qubits(backend)

        sim = graphix.simulator.PatternSimulator(backend=backend, pattern=self.clean_pattern, measure_method=self.measure_method)
        self.backend_state = sim.run(self.backend_state)
        self.decode_output_state(backend)
        # returns the final state as well as the server object (Simulator)
        return self.backend_state, sim


    def decode_output_state(self, backend):
        for node in self.output_nodes:
            if node in self.byproduct_db:
                z_decoding, x_decoding = self.decode_output(node)
                if z_decoding:
                    backend.apply_single(backendState=self.backend_state, node=node, op=graphix.ops.Ops.z)
                if x_decoding:
                    backend.apply_single(backendState=self.backend_state, node=node, op=graphix.ops.Ops.x)
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




