import unittest
from copy import deepcopy
from random import randint, random

import numpy as np

import graphix
import graphix.pauli
import tests.random_circuit as rc
from graphix.client import Client, ClientMeasureMethod, Secrets
from graphix.sim.statevec import Statevec, StatevectorBackend
from graphix.simulator import PatternSimulator
from graphix.states import BasicStates, PlanarState


class TestClient(unittest.TestCase):

    def setUp(self):
        # set up the random numbers
        self.rng = np.random.default_rng()  # seed=422

    def test_client_input(self):
        # Generate random pattern
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize(method="global")

        secrets = Secrets(theta=True)

        # Create a |+> state for each input node
        states = [BasicStates.PLUS for node in pattern.input_nodes]

        # Create the client with the input state
        client = Client(pattern=pattern, input_state=states, secrets=secrets)

        # Assert something...
        # Todo ?

    def test_r_secret_simulation(self):
        # Generate and standardize pattern
        nqubits = 2
        depth = 1
        for i in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth)
            pattern = circuit.transpile()
            pattern.standardize(method="global")

            state = circuit.simulate_statevector()

            backend = StatevectorBackend()
            # Initialize the client
            secrets = Secrets(r=True)
            # Giving it empty will create a random secret
            client = Client(pattern=pattern, secrets=secrets)
            state_mbqc, _ = client.delegate_pattern(backend)
            np.testing.assert_almost_equal(
                np.abs(np.dot(state_mbqc.state.psi.flatten().conjugate(), state.psi.flatten())), 1
            )

    def test_theta_secret_simulation(self):
        # Generate random pattern
        nqubits = 2
        depth = 1
        for i in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth)
            pattern = circuit.transpile()
            pattern.standardize(method="global")

            secrets = Secrets(theta=True)

            # Create a |+> state for each input node
            states = [BasicStates.PLUS for node in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets)
            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            blinded_simulation, _ = client.delegate_pattern(backend)

            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector()

            np.testing.assert_almost_equal(
                np.abs(np.dot(blinded_simulation.state.psi.flatten().conjugate(), clear_simulation.psi.flatten())), 1
            )

    def test_a_secret_simulation(self):
        # Generate random pattern
        nqubits = 2
        depth = 1
        for _ in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth)
            pattern = circuit.transpile()
            pattern.standardize(method="global")

            secrets = Secrets(a=True)

            # Create a |+> state for each input node
            states = [BasicStates.PLUS for __ in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets)
            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            blinded_simulation, _ = client.delegate_pattern(backend)

            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector()
            np.testing.assert_almost_equal(
                np.abs(np.dot(blinded_simulation.state.psi.flatten().conjugate(), clear_simulation.psi.flatten())), 1
            )

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
        backend = StatevectorBackend()
        _, server = client.delegate_pattern(backend)

        for measured_node in client.measurement_db:
            # Compare results on the client side and on the server side : should differ by r[node]
            result = client.results[measured_node]
            client_r_secret = client.secrets.r[measured_node]
            server_result = server.results[measured_node]
            assert result == (server_result + client_r_secret) % 2

    def test_qubits_preparation(self):
        nqubits = 2
        depth = 1
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        nodes = pattern.get_graph()[0]
        pattern.standardize(method="global")
        secrets = Secrets(a=True, r=True, theta=True)

        # Create a |+> state for each input node, and associate index
        states = [BasicStates.PLUS for node in pattern.input_nodes]

        # Create the client with the input state
        client = Client(pattern=pattern, input_state=states, secrets=secrets)

        backend = StatevectorBackend()
        # Blinded simulation, between the client and the server
        backend = client.prepare_states(backend)
        assert set(backend.node_index) == set(nodes)
        backend = client.blind_qubits(backend)
        assert set(backend.node_index) == set(nodes)

    def test_UBQC(self):
        # Generate random pattern
        nqubits = 2
        # TODO : work on optimization of the quantum communication
        depth = 1
        for i in range(10):
            circuit = rc.get_rand_circuit(nqubits, depth)
            pattern = circuit.transpile()
            pattern.standardize(method="global")

            secrets = Secrets(a=True, r=True, theta=True)

            # Create a |+> state for each input node, and associate index
            states = [BasicStates.PLUS for node in pattern.input_nodes]

            # Create the client with the input state
            client = Client(pattern=pattern, input_state=states, secrets=secrets)

            backend = StatevectorBackend()
            # Blinded simulation, between the client and the server
            blinded_simulation, _ = client.delegate_pattern(backend)
            # Clear simulation = no secret, just simulate the circuit defined above
            clear_simulation = circuit.simulate_statevector()
            np.testing.assert_almost_equal(
                np.abs(np.dot(blinded_simulation.state.psi.flatten().conjugate(), clear_simulation.psi.flatten())), 1
            )


if __name__ == "__main__":
    unittest.main()
