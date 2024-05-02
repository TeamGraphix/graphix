import unittest
from random import random, randint

import graphix
import tests.random_circuit as rc
import numpy as np
from graphix.client import Client, Secrets
from graphix.sim.statevec import StatevectorBackend, Statevec
from graphix.simulator import PatternSimulator
from graphix.states import PlanarState, BasicStates


class TestClient(unittest.TestCase):

    def setUp(self):
        # set up the random numbers
        self.rng = np.random.default_rng()  # seed=422


    def test_client_input(self) :
        # Generate random pattern
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize(method="global")

        secrets = Secrets(theta=True)

        # Create a |+> state for each input node
        states = [PlanarState(plane = 0, angle = 0) for node in pattern.input_nodes]
        
        
        # Create the client with the input state
        client = Client(pattern=pattern, input_state=states, secrets=secrets)
        

        # Assert something...
        # Todo ?

    def test_r_secret_simulation(self):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        for i in range(10) :
            circuit = rc.get_rand_circuit(nqubits, depth)
            pattern = circuit.transpile()
            pattern.standardize(method="global")

            state = circuit.simulate_statevector()

            # Initialize the client
            secrets = Secrets(r=True)
            # Giving it empty will create a random secret
            client = Client(pattern=pattern, secrets=secrets)

            state_mbqc, _ = client.simulate_pattern()
            np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_theta_secret_simulation(self) :
        # Generate random pattern
        nqubits = 2
        depth = 1
        for i in range(10) :
            circuit = rc.get_rand_circuit(nqubits, depth)
            pattern = circuit.transpile()
            pattern.standardize(method="global")

            secrets = Secrets(theta=True)

            # Create a |+> state for each input node
            states = [PlanarState(plane = 0, angle = 0) for node in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets)
            
            # Blinded simulation, between the client and the server
            blinded_simulation, _ = client.simulate_pattern()

            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector()
            np.testing.assert_almost_equal(np.abs(np.dot(blinded_simulation.flatten().conjugate(), clear_simulation.flatten())), 1)
    
    def test_a_secret_simulation(self) :
        # Generate random pattern
        nqubits = 2
        depth = 1
        for i in range(10) :
            circuit = rc.get_rand_circuit(nqubits, depth)
            pattern = circuit.transpile()
            pattern.standardize(method="global")

            secrets = Secrets(a=True)

            # Create a |+> state for each input node
            states = [PlanarState(plane = 0, angle = 0) for node in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets)

            # Blinded simulation, between the client and the server
            blinded_simulation, _ = client.simulate_pattern()

            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector()
            np.testing.assert_almost_equal(np.abs(np.dot(blinded_simulation.flatten().conjugate(), clear_simulation.flatten())), 1)


    def test_r_secret_results(self):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize(method="global")

        # Initialize the client
        secrets = Secrets(r=True)
        # Giving it empty will create a random secret
        client = Client(pattern=pattern, secrets=secrets)
        _state, backend = client.simulate_pattern()

        for measured_node in client.measurement_db:
            # Compare results on the client side and on the server side : should differ by r[node]
            result = client.results[measured_node]
            client_r_secret = client.secrets.r[measured_node]
            server_result = backend.results[measured_node]
            assert result == (server_result + client_r_secret) % 2


    def test_UBQC(self) :
        # Generate random pattern
        nqubits = 2
        depth = 1
        for i in range(10) :
            circuit = rc.get_rand_circuit(nqubits, depth)
            pattern = circuit.transpile()
            pattern.standardize(method="global")

            secrets = Secrets(a=True, r=True, theta=True)

            # Create a |+> state for each input node, and associate index
            states = [BasicStates.PLUS for node in pattern.input_nodes]
            

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets)

            # Blinded simulation, between the client and the server
            blinded_simulation, _backend = client.simulate_pattern()
            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector()
            np.testing.assert_almost_equal(np.abs(np.dot(blinded_simulation.flatten().conjugate(), clear_simulation.flatten())), 1)



if __name__ == '__main__':
    unittest.main()
