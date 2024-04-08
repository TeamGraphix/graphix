import unittest
from random import random, randint

import graphix
import tests.random_circuit as rc
import numpy as np
from graphix.client import Client
from graphix.sim.statevec import StatevectorBackend, Statevec
from graphix.simulator import PatternSimulator
from graphix.states import PlanarState


class TestClient(unittest.TestCase):

    def setUp(self):
        # set up the random numbers
        self.rng = np.random.default_rng()  # seed=422

    def test_custom_secret(self):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize(method="global")

        measured_qubits = [i[1] for i in pattern.get_measurement_commands()]

        # Initialize the client
        # A client that inputs a secret values other than bits should throw an error
        r = {}
        for qubit in measured_qubits:
            r[qubit] = 666
        secrets = {'r': r}
        self.assertRaises(ValueError, Client, pattern, True, secrets)

        # TODO : do the same for `a` and `theta` secrets


    def test_secret_location(self):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize(method="global")

        measured_qubits = [i[1] for i in pattern.get_measurement_commands()]

        # Initialize the client
        r = {}
        a = {}
        theta = {}
        secrets = {
            'r': r,
            'a': a,
            'theta': theta
        }
        client = Client(pattern=pattern, blind=True, secrets=secrets)
        secret_locations = client.get_secrets_locations()

        for secret in secret_locations:

            # For the 'r' secret, a secret bit must be assigned to each measured qubit
            if secret == 'r':
                assert set(measured_qubits) == set(secret_locations['r'])

    def test_secret_size(self):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize(method="global")

        n_measured_qubits = len(pattern.get_measurement_commands())

        # Initialize the client
        r = {}
        a = {}
        theta = {}
        secrets = {
            'r': r,
            'a': a,
            'theta': theta
        }
        client = Client(pattern=pattern, blind=True, secrets=secrets)
        secrets_size = client.get_secrets_size()

        assert 'r' in secrets_size
        assert secrets_size['r'] == n_measured_qubits

    def test_r_secret_simulation(self):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize(method="global")

        state = circuit.simulate_statevector()

        # Initialize the client
        secrets = {'r': {}}
        # Giving it empty will create a random secret
        client = Client(pattern=pattern, blind=True, secrets=secrets)

        state_mbqc = client.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_client_input(self) :
        # Generate random pattern
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize(method="global")

        secrets = {'theta': {}}

        rand_angles = self.rng.random(nqubits) * 2 * np.pi
        rand_planes = self.rng.choice(np.array([i for i in graphix.pauli.Plane]), nqubits)
        # set default angles to zero in order to have a comparison with the "clear simulation"
        states = [PlanarState(plane = 0, angle = 0) for i, j in zip(rand_planes, rand_angles)]
        
        input_state = {}
        j = 0
        for i in pattern.input_nodes :
            input_state[i] = states[j]
            j += 1
        
        client = Client(pattern=pattern, input_state=input_state, blind=True, secrets=secrets)
        
        print(client.input_state)
        # Clear simulation = no secret, just simulate the circuit
        clear_simulation = circuit.simulate_statevector()
        # Blinded simulation, between the client and the server
        blinded_simulation = client.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(blinded_simulation.flatten().conjugate(), clear_simulation.flatten())), 1)



    def test_theta_secret_simulation(self):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize(method="global")

        state = circuit.simulate_statevector()

        # Initialize the client
        secrets = {'theta': {}}
        # Giving it empty will create a random secret
        client = Client(pattern=pattern, blind=True, secrets=secrets)

        state_mbqc = client.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)

    def test_r_secret_results(self):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize(method="global")

        # Initialize the client
        secrets = {'r': {}}
        # Giving it empty will create a random secret
        client = Client(pattern=pattern, blind=True, secrets=secrets)
        client.simulate_pattern()

        for measured_node in client.measurement_db:
            # Compare results on the client side and on the server side : should differ by r[node]
            result = client.results[measured_node]
            client_r_secret = client.secrets['r'][measured_node]
            server_result = client.backend_results[measured_node]
            assert result == (server_result + client_r_secret) % 2


if __name__ == '__main__':
    unittest.main()
