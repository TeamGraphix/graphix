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
"""
Usage:

client = Client(pattern:Pattern, blind=False) ## For pure MBQC
sv_backend = StatevecBackend(client.pattern, meas_op = client.meas_op)

simulator = PatternSimulator(client.pattern, backend=sv_backend)
simulator.run()

"""

@dataclasses.dataclass
class MeasureParameters:
    plane: graphix.pauli.Plane
    angle: float
    s_domain: list[int]
    t_domain: list[int]
    vop: int

class Client:
    def __init__(self, pattern, input_state=None, blind=False, secrets={}):
        self.pattern = pattern
        self.clean_pattern = self.remove_pattern_flow()

        """
        Database containing the "measurement configuration"
        - Node
        - Measurement parameters : plane, angle, X and Z dependencies
        - Measurement outcome
        """
        self.results = pattern.results.copy()
        self.measure_method = ClientMeasureMethod(self)
        self.measurement_db = {}
        self.init_measurement_db()
        self.byproduct_db = {}
        self.init_byproduct_db()
        self.backend_results = {}
        self.blind = blind
        # By default, no secrets
        self.r_secret = False
        self.theta_secret = False
        self.a_secret = False
        self.secrets = {}
        # Initialize the secrets
        self.init_secrets(secrets)
        self.init_inputs(input_state)


    def init_inputs(self, input_state) :
        # Initialization to all |+> states if nothing specified
        if input_state == None :
            states = [PlanarState(plane=0, angle=0) for node in self.pattern.input_nodes]
            input_state = dict(zip(self.clean_pattern.input_nodes, states))
        
        # The input state is modified (with secrets) before being sent to server
        sent_input = {}
        for input_node in input_state :
            initial_state = input_state[input_node]
            theta_value = 0 if not self.theta_secret else self.secrets['theta'][input_node]
            a_value = 0 if not self.a_secret else self.secrets['a'][input_node]
            new_angle = (-1)**a_value *initial_state.angle + theta_value*np.pi/4
            blinded_state = PlanarState(plane=initial_state.plane, angle=new_angle)
            sent_input[input_node] = blinded_state

        # gros probleme : on doit espérer que les index matchent. (ça a l'air de marcher pour le moment)
        self.input_state = list(sent_input.values())

    def init_secrets(self, secrets):
        if self.blind:
            if 'r' in secrets:
                # User wants to add an r-secret, either customized or generated on the fly
                self.r_secret = True
                self.secrets['r'] = {}
                r_size = len(secrets['r'].keys())
                # If the client entered an empty secret (need to generate it)
                if r_size == 0:
                    # Need to generate the bit for each measured qubit
                    for node in self.measurement_db:
                        self.secrets['r'][node] = np.random.randint(0, 2)

                # If the client entered a customized secret : need to check its validity
                elif self.is_valid_secret('r', secrets['r']):
                    self.secrets['r'] = secrets['r']
                    # TODO : add more rigorous test of the r-secret format
                else:
                    raise ValueError("`r` has wrong format.")
                

            if 'theta' in secrets :
                self.theta_secret = True
                self.secrets['theta'] = {}
                # Create theta secret for all non-output nodes
                for node in self.clean_pattern.non_output_nodes :
                    k = np.random.randint(0,8)
                    angle = k  # *pi/4
                    self.secrets['theta'][node] = angle
                new_pattern = self.add_secret_angles()
                self.clean_pattern = new_pattern

            if 'a' in secrets :
                self.a_secret = True
                self.secrets['a'] = {}
                node_list, edge_list = self.clean_pattern.get_graph()
                for node in node_list :
                    if node in self.clean_pattern.input_nodes :
                        a = np.random.randint(0,2)
                        self.secrets['a'][node] = a
                    else :
                        self.secrets['a'][node] = 0
                
                # After all the `a` secrets have been generated, the `a_N` value can be
                # computed from the graph topology
                self.secrets['a_N'] = {}
                for i in node_list :
                    for j in node_list :
                        self.secrets['a_N'][i] = 0
                        if (i,j) in edge_list :
                            self.secrets['a_N'][i] += 1
                        self.secrets['a_N'][i] %= 2 
                        
                
    def add_secret_angles(self) :
        """
        This function adds a secret angle to all auxiliary qubits (measured qubits that are not part of the input),
        i.e the qubits created through "N" commands originally in the |+> state
        """
        new_pattern = graphix.Pattern(self.clean_pattern.input_nodes)
        for cmd in self.clean_pattern :
            if cmd[0] == 'N' :
                node = cmd[1]
                if node not in self.clean_pattern.output_nodes :
                    angle = self.secrets['theta'][node]
                    plane = Plane(0)                    # Default : XY plane
                    new_cmd = ['N', node, plane, angle]
                else :
                    new_cmd = cmd
                new_pattern.add(new_cmd)
            else :
                new_pattern.add(cmd)

        return new_pattern


    def init_measurement_db(self):
        """
        Initializes the "database of measurement configurations and results" held by the customer
        according to the pattern desired
        Initial measurement outcomes are set to 0
        """
        for cmd in self.pattern:
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
        clean_pattern = graphix.pattern.Pattern(self.pattern.input_nodes)
        for cmd in self.pattern :
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
        backend = graphix.sim.statevec.StatevectorBackend(pattern=self.clean_pattern, measure_method=self.measure_method, input_state=self.input_state)
        sim = graphix.simulator.PatternSimulator(backend=backend, pattern=self.clean_pattern)
        state = sim.run()
        self.backend_results = backend.results
        self.decode_output_state(backend, state)
        return state


    def decode_output_state(self, backend, state):
        if self.blind:
            for node in self.pattern.output_nodes:
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
        for node in self.pattern.output_nodes:
            self.byproduct_db[node] = {
                'z-domain': [],
                'x-domain': []
            }

        for cmd in self.pattern:
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

    def get_secrets_locations(self):
        locations = {}
        for secret in self.secrets:
            secret_dict = self.secrets[secret]
            secrets_location = secret_dict.keys()
            locations[secret] = secrets_location
        return locations

    def is_valid_secret(self, secret_type, custom_secret):
        if any((i != 0 and i != 1) for i in custom_secret.values()) :
            return False
        if secret_type == 'r':
            return set(custom_secret.keys()) == set(self.measurement_db.keys())


class ClientMeasureMethod(graphix.sim.base_backend.MeasureMethod):
    def __init__(self, client: Client):
        self.__client = client

    def get_measurement_description(self, cmd, results) -> graphix.sim.base_backend.MeasurementDescription:
        node = cmd[1]
        parameters = self.__client.measurement_db[node]
        r_value = 0 if not self.__client.r_secret else self.__client.secrets['r'][node]
        theta_value = 0 if not self.__client.theta_secret else self.__client.secrets['theta'][node]
        a_value = 0 if not self.__client.a_secret else self.__client.secrets['a'][node]
        a_N_value = 0 if not self.__client.a_secret else self.__client.secrets['a_N'][node]

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
        if self.__client.r_secret:
            r_value = self.__client.secrets['r'][node]
            result = (result + r_value) % 2
        self.__client.results[node] = result

