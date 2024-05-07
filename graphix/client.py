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
from copy import deepcopy
import typing

"""
Usage:

client = Client(pattern:Pattern, blind=False) ## For pure MBQC
sv_backend = StatevecBackend(client.pattern, meas_op = client.meas_op)

simulator = PatternSimulator(client.pattern, backend=sv_backend)
simulator.run()

"""



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
                a[node] = np.random.randint(0,2) if node in input_nodes else 0
            
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

        # Copy the pauli-preprocessed nodes' measurement outcomes
        self.results = pattern.results.copy()
        self.measure_method = ClientMeasureMethod(self)

        self.measurement_db = pattern.get_measurement_db()
        self.byproduct_db = pattern.get_byproduct_db()

        self.secrets = SecretDatas.from_secrets(secrets, pattern.get_graph(), self.input_nodes, self.output_nodes)

        pattern_without_flow = pattern.remove_flow()
        self.clean_pattern = self.remove_prepared_nodes(pattern_without_flow)

        self.input_state = input_state


    def remove_prepared_nodes(self, pattern) :
        clean_pattern = graphix.Pattern(self.input_nodes)
        for cmd in pattern :
            if cmd[0] == "N" :
                clean_pattern.add_auxiliary_node(node=cmd[1])
            else :
                clean_pattern.add(cmd)
        return clean_pattern

    def blind_qubits(self, backend) :
        z_rotation = lambda theta : np.array([[1, 0], [0, np.exp(1j*theta)]])
        for node, theta in self.secrets.theta.items() :
            index = backend.node_index.index(node)
            backend.state.evolve_single(op=z_rotation(theta), i=index)

    def prepare_states(self, backend) :
        # First prepare inputs
        backend.prepare_state(nodes=self.input_nodes,
                                data=self.input_state)
        # Then iterate over auxiliaries required to blind
        auxiliaries = []
        for node in self.secrets.theta :
            if node not in self.input_nodes :
            # if node not in self.output_nodes and node not in self.input_nodes :
                auxiliaries.append(node)
        aux_data = [BasicStates.PLUS for _ in auxiliaries]
        backend.prepare_state(nodes=auxiliaries, data=aux_data)

    def delegate_pattern(self, backend):
        self.prepare_states(backend)        
        self.blind_qubits(backend)

        sim = graphix.simulator.PatternSimulator(backend=backend, pattern=self.clean_pattern)
        sim.run()
        self.decode_output_state(backend)
        return


    def decode_output_state(self, backend):
        for node in self.output_nodes:
            if node in self.byproduct_db:
                index = backend.node_index.index(node)
                z_decoding, x_decoding = self.decode_output(node)
                if z_decoding:
                    backend.state.evolve_single(op=graphix.ops.Ops.z, i=index)
                if x_decoding:
                    backend.state.evolve_single(op=graphix.ops.Ops.x, i=index)

    def get_secrets_size(self):
        secrets_size = {}
        for secret in self.secrets:
            secrets_size[secret] = len(self.secrets[secret])
        return secrets_size

    def decode_output(self, node):
        z_decoding = 0
        x_decoding = 0
        for z_dep_node in self.byproduct_db[node]['z-domain']:
            z_decoding += self.results[z_dep_node]
        z_decoding = z_decoding % 2
        for x_dep_node in self.byproduct_db[node]['x-domain']:
            x_decoding += self.results[x_dep_node]
        x_decoding = x_decoding % 2
        return z_decoding, x_decoding



class ClientMeasureMethod(graphix.sim.base_backend.MeasureMethod):
    def __init__(self, client: Client):
        self.__client = client


    def get_measurement_description(self, cmd, results) -> graphix.sim.base_backend.MeasurementDescription:
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
        return graphix.sim.base_backend.MeasurementDescription(measure_update.new_plane, angle)

    def set_measure_result(self, cmd, result: bool) -> None:
        node = cmd[1]
        if self.__client.secrets.r:
            result ^= self.__client.secrets.r[node]
        self.__client.results[node] = result

