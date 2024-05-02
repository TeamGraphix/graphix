import dataclasses
import numpy as np
from graphix.clifford import CLIFFORD_CONJ, CLIFFORD, CLIFFORD_MUL
import graphix.ops
from graphix.states import State, PlanarState
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
    r : typing.Optional[dict[int, int]]
    a : typing.Optional[Secret_a]
    theta : typing.Optional[dict[int, int]]


@dataclasses.dataclass
class MeasureParameters:
    plane: graphix.pauli.Plane
    angle: float
    s_domain: list[int]
    t_domain: list[int]
    vop: int

class Client:
    def __init__(self, pattern, input_state=None, blind=False, secrets={}):
        self.initial_pattern = pattern

        self.input_nodes = self.initial_pattern.input_nodes.copy()
        self.output_nodes = self.initial_pattern.output_nodes.copy()
        # to be completed later
        self.auxiliary_nodes = []

        self.clean_pattern = self.remove_pattern_flow()


        # Copy the pauli-preprocessed nodes' measurement outcomes
        self.results = pattern.results.copy()
        self.measure_method = ClientMeasureMethod(self)
        self.measurement_db = {}
        self.init_measurement_db()
        self.byproduct_db = {}
        self.init_byproduct_db()
        self.backend_results = {}
        self.blind = blind
        # By default, no secrets
        self.secrets = Secrets(None, None, None)
        # Initialize the secrets
        self.init_secrets(secrets)
        self.init_inputs(input_state)


    def init_inputs(self, input_state) :
        # Initialization to all |+> states if nothing specified
        if input_state == None :
            states = [PlanarState(plane=0, angle=0) for _ in self.input_nodes]
            input_state = dict(zip(self.input_nodes, states))
        
        # The input state is modified (with secrets) before being sent to server
        sent_input = {}
        for input_node in input_state :
            initial_state = input_state[input_node]
            theta_value = 0 if not self.secrets.theta else self.secrets.theta[input_node]
            a_value = 0 if not self.secrets.a else self.secrets.a.a[input_node]
            new_angle = (-1)**a_value *initial_state.angle + theta_value*np.pi/4
            blinded_state = PlanarState(plane=initial_state.plane, angle=new_angle)
            sent_input[input_node] = blinded_state

        # gros probleme : on doit espérer que les index matchent. (ça a l'air de marcher pour le moment)
        self.input_state = list(sent_input.values())

        new_pattern = self.add_secret_angles()
        self.clean_pattern = new_pattern

    def init_secrets(self, secrets):
        if self.blind:
            node_list, edge_list = self.clean_pattern.get_graph()
            if 'r' in secrets:
                r = {}
                # Need to generate the random bit for each measured qubit, 0 for the rest (output qubits)
                for node in node_list :
                    r[node] = np.random.randint(0, 2) if node not in self.clean_pattern.output_nodes else 0
                self.secrets.r = r
                

            if 'theta' in secrets :
                theta = {}
                # Create theta secret for all non-output nodes (measured qubits)
                for node in node_list :
                    theta[node] = np.random.randint(0,8) if node not in self.clean_pattern.output_nodes else 0    # Expressed in pi/4 units
                self.secrets.theta = theta

            if 'a' in secrets :
                a = {}
                a_N = {}
                # Create `a` secret for all
                for node in node_list :
                    a[node] = np.random.randint(0,2) if node in self.clean_pattern.input_nodes else 0
                
                # After all the `a` secrets have been generated, the `a_N` value can be
                # computed from the graph topology
                for i in node_list :
                    a_N_value = 0
                    for j in node_list :
                        if (i,j) in edge_list or (j,i) in edge_list :
                            a_N_value ^= a[j]
                    a_N[i] = a_N_value 
                        
                self.secrets.a = Secret_a(a, a_N)
                
    def add_secret_angles(self) :
        """
        This function adds a secret angle to all auxiliary qubits (measured qubits that are not part of the input),
        i.e the qubits created through "N" commands originally in the |+> state
        """
        new_pattern = graphix.Pattern(self.clean_pattern.input_nodes)
        for cmd in self.clean_pattern :
            if cmd[0] == 'N' :
                node = cmd[1]
                theta_value = 0 if not self.secrets.theta else self.secrets.theta[node]
                auxiliary_state = PlanarState(plane=Plane(0), angle=theta_value*np.pi/4)
                new_pattern.add_auxiliary_node(node)
                self.input_state.append(auxiliary_state)
                self.auxiliary_nodes.append(node)
            else :
                new_pattern.add(cmd)

        return new_pattern


    def init_measurement_db(self):
        """
        Initializes the "database of measurement configurations and results" held by the customer
        according to the pattern desired
        Initial measurement outcomes are set to 0
        """
        for cmd in self.initial_pattern:
            if cmd[0] == 'M':
                node = cmd[1]
                plane = graphix.pauli.Plane[cmd[2]]
                angle = cmd[3] * np.pi
                s_domain = cmd[4]
                t_domain = cmd[5]
                if len(cmd) == 7:
                    vop = cmd[6]
                else:
                    vop = 0
                self.measurement_db[node] = MeasureParameters(plane, angle, s_domain, t_domain, vop)

    def remove_pattern_flow(self) :
        clean_pattern = graphix.pattern.Pattern(self.input_nodes)
        for cmd in self.initial_pattern :
            # by default, copy the command
            new_cmd = deepcopy(cmd) 

            # If measure, remove the s-domain and t-domain, vop
            if cmd[0] == 'M' :
                del new_cmd[2:]
            # If byproduct, remove it so it's not done by the server
            if cmd[0] != 'X' and cmd[0] != 'Z' :
                clean_pattern.add(new_cmd)
        return clean_pattern
        
    def simulate_pattern(self):
        backend = graphix.sim.statevec.StatevectorBackend(pattern=self.clean_pattern, measure_method=self.measure_method, input_state=self.input_state, prepared_nodes=self.input_nodes + self.auxiliary_nodes)
        sim = graphix.simulator.PatternSimulator(backend=backend, pattern=self.clean_pattern)
        state = sim.run()
        self.backend_results = backend.results
        self.decode_output_state(backend, state)
        return state


    def decode_output_state(self, backend, state):
        if self.blind:
            for node in self.output_nodes:
                if node in self.byproduct_db:
                    z_decoding, x_decoding = self.decode_output(node)
                    if z_decoding:
                        state.evolve_single(op=graphix.ops.Ops.z, i=backend.node_index.index(node))
                    if x_decoding:
                        state.evolve_single(op=graphix.ops.Ops.x, i=backend.node_index.index(node))

    def get_secrets_size(self):
        secrets_size = {}
        for secret in self.secrets:
            secrets_size[secret] = len(self.secrets[secret])
        return secrets_size

    def init_byproduct_db(self):
        for node in self.output_nodes:
            self.byproduct_db[node] = {
                'z-domain': [],
                'x-domain': []
            }

        for cmd in self.initial_pattern:
            if cmd[0] == 'Z' or cmd[0] == 'X':
                node = cmd[1]

                if cmd[0] == 'Z':
                    self.byproduct_db[node]['z-domain'] = cmd[2]
                if cmd[0] == 'X':
                    self.byproduct_db[node]['x-domain'] = cmd[2]

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
        r_value = 0 if not self.__client.secrets.r else self.__client.secrets.r[node]
        theta_value = 0 if not self.__client.secrets.theta else self.__client.secrets.theta[node]
        if self.__client.secrets.a :
            a_value = self.__client.secrets.a.a[node]
            a_N_value = self.__client.secrets.a.a_N[node]
        else : 
            a_value, a_N_value = 0, 0

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

