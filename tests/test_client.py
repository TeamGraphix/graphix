import unittest
from random import random, randint

import tests.random_circuit as rc
import numpy as np

from graphix.client import Client
from graphix.sim.statevec import StatevectorBackend
from graphix.simulator import PatternSimulator


class TestClient(unittest.TestCase):
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
