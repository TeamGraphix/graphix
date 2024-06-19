from copy import deepcopy
import unittest

import numpy as np
from graphix import Circuit
from graphix.states import BasicStates, PlanarState
from graphix.sim.statevec import Statevec, StatevectorBackend
from graphix.sim.base_backend import BackendState
import graphix.pauli
import graphix.clifford
import graphix.simulator
from tests.test_graphsim import meas_op
import random

class TestStatevec(unittest.TestCase):
    def test_remove_one_qubit(self):
        n = 10
        k = 3

        sv = Statevec(nqubit=n)
        for i in range(n):
            sv.entangle([i, (i + 1) % n])
        m_op = meas_op(np.pi / 5)
        sv.evolve(m_op, [k])
        sv2 = deepcopy(sv)

        sv.remove_qubit(k)
        sv2.ptrace([k])
        sv2.normalize()

        np.testing.assert_almost_equal(np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())), 1)

    #TODO This is a weird test!
    def test_measurement_into_each_XYZ_basis(self):
        n = 3
        k = 0
        # for measurement into |-> returns [[0, 0], ..., [0, 0]] (whose norm is zero)
        # NOTE weird choice (MINUS is orthogonal to PLUS so zero)
        for state in [BasicStates.PLUS, BasicStates.ZERO, BasicStates.ONE, BasicStates.PLUS_I, BasicStates.MINUS_I]:
            m_op = np.outer(state.get_statevector(), state.get_statevector().T.conjugate())
            # print(m_op)
            sv = Statevec(nqubit=n)
            # print(sv)
            sv.evolve(m_op, [k])
            sv.remove_qubit(k)

            sv2 = Statevec(nqubit=n - 1)
            np.testing.assert_almost_equal(np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())), 1)

    def test_measurement_into_minus_state(self):
        n = 3
        k = 0
        m_op = np.outer(BasicStates.MINUS.get_statevector(), BasicStates.MINUS.get_statevector().T.conjugate())
        sv = Statevec(nqubit=n)
        sv.evolve(m_op, [k])
        with self.assertRaises(AssertionError):
            sv.remove_qubit(k)

class TestStatevecNew(unittest.TestCase):
    def setUp(self):
        # set up the random numbers
        self.rng = np.random.default_rng()  # seed=422

        circ = Circuit(1)
        circ.h(0)
        self.hadamardpattern = circ.transpile()

    def test_clifford(self) :
        for clifford_index in range(24) :   
            state = BasicStates.PLUS
            vec = Statevec(nqubit=1, data=state)
            backend = StatevectorBackend()
            backendState = backend.add_nodes(backendState=BackendState(), nodes=[0], data=state)
            # Applies clifford gate "Z"
            clifford_cmd = ["C", 0, clifford_index]
            clifford_gate = graphix.clifford.CLIFFORD[clifford_index]

            vec.evolve_single(clifford_gate, 0)

            backendState = backend.apply_clifford(backendState=backendState, cmd=clifford_cmd)
            np.testing.assert_allclose(vec.psi, backendState.state.psi)

    def test_deterministic_measure_one(self) :
        # plus state & zero state (default), but with tossed coins
        for _ in range(10) :
            backend = StatevectorBackend()
            coins = [random.randint(0,1), random.randint(0,1)]
            expected_result = sum(coins)%2
            states = [graphix.pauli.X.get_eigenstate(eigenvalue=coins[0]), graphix.pauli.Z.get_eigenstate(eigenvalue=coins[1])] 
            nodes = range(len(states))
            backendState = backend.add_nodes(backendState=BackendState(), nodes=nodes, data=states)

            backendState = backend.entangle_nodes(backendState=backendState, edge=(nodes[0], nodes[1]))
            measurement_description = graphix.simulator.MeasurementDescription(plane=graphix.pauli.Plane.XY, angle=0)
            node_to_measure = backendState.node_index[0]
            backendState, result = backend.measure(backendState=backendState, node=node_to_measure, measurement_description=measurement_description)
            assert result == expected_result
        

    def test_deterministic_measure(self) :
        """
         Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1.
        """
        for _ in range(10) :
            # plus state (default)
            backend = StatevectorBackend()
            N_neighbors = 10
            states = [graphix.pauli.X.get_eigenstate()] + [graphix.pauli.Z.get_eigenstate() for i in range(N_neighbors)]
            nodes = range(len(states))
            backendState = backend.add_nodes(backendState=BackendState(), nodes=nodes, data=states)

            for i in range(1, N_neighbors+1) :
                backendState = backend.entangle_nodes(backendState=backendState, edge=(nodes[0], i))
                # backendState = backend.entangle_nodes(backendState=backendState, edge=(nodes[-1], i))
            measurement_description = graphix.simulator.MeasurementDescription(plane=graphix.pauli.Plane.XY, angle=0)
            node_to_measure = backendState.node_index[0]
            backendState, result = backend.measure(backendState=backendState, node=node_to_measure, measurement_description=measurement_description)
            assert result == 0
            assert backendState.node_index == list(range(1, N_neighbors+1))
            
    def test_deterministic_measure_many(self) :
        """
         Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1.
        """
        for _ in range(10) :
            # plus state (default)
            backend = StatevectorBackend()
            N_traps = 5
            N_neighbors = 5
            N_whatever = 5
            traps = [graphix.pauli.X.get_eigenstate() for _ in range(N_traps)]
            dummies = [graphix.pauli.Z.get_eigenstate() for _ in range(N_neighbors)] 
            others = [graphix.pauli.I.get_eigenstate() for _ in range(N_whatever)]
            states = traps + dummies + others
            nodes = range(len(states))
            backendState = backend.add_nodes(backendState=BackendState(), nodes=nodes, data=states)

            for dummy in nodes[N_traps: N_traps + N_neighbors] :
                for trap in nodes[:N_traps]:
                    backendState = backend.entangle_nodes(backendState=backendState, edge=(trap, dummy))
                for other in nodes[N_traps + N_neighbors:]:
                    backendState = backend.entangle_nodes(backendState=backendState, edge=(other, dummy))

            # Same measurement for all traps
            measurement_description = graphix.simulator.MeasurementDescription(plane=graphix.pauli.Plane.XY, angle=0)
            
            for trap in nodes[:N_traps] :
                node_to_measure = trap
                backendState, result = backend.measure(backendState=backendState, node=node_to_measure, measurement_description=measurement_description)
                assert result == 0

            assert backendState.node_index == list(range(N_traps, N_neighbors+N_traps+N_whatever))

    def test_deterministic_measure_with_coin(self) :
        """
         Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1.
            We add coin toss to that
        """
        for _ in range(10) :
            # plus state (default)
            backend = StatevectorBackend()
            N_neighbors = 10
            coins = [random.randint(0,1)] + [random.randint(0,1) for _ in range(N_neighbors)]
            expected_result = sum(coins)%2
            states = [graphix.pauli.X.get_eigenstate(eigenvalue=coins[0])] + [graphix.pauli.Z.get_eigenstate(eigenvalue=coins[i+1]) for i in range(N_neighbors)]
            nodes = range(len(states))
            backendState = backend.add_nodes(backendState=BackendState(), nodes=nodes, data=states)

            for i in range(1, N_neighbors+1) :
                backendState = backend.entangle_nodes(backendState=backendState, edge=(nodes[0], i))
            measurement_description = graphix.simulator.MeasurementDescription(plane=graphix.pauli.Plane.XY, angle=0)
            node_to_measure = backendState.node_index[0]
            backendState, result = backend.measure(backendState=backendState, node=node_to_measure, measurement_description=measurement_description)
            assert result == expected_result
            assert backendState.node_index == list(range(1, N_neighbors+1))

    # test initialization only
    def test_init_success(self):

        # plus state (default)
        state = BasicStates.PLUS
        vec = Statevec(nqubit=1, data=state)
        backend = StatevectorBackend()
        backendState = backend.add_nodes(backendState=BackendState(), nodes=[0], data=state)
        np.testing.assert_allclose(vec.psi, backendState.state.psi)
        # assert backend.state.Nqubit == 1
        assert backendState.node_index == [0]

        # minus state 
        state = BasicStates.MINUS
        vec = Statevec(nqubit=1, data=state)
        backend = StatevectorBackend()
        backendState = backend.add_nodes(backendState=BackendState(), nodes=[0], data=state)
        np.testing.assert_allclose(vec.psi, backendState.state.psi)
        # assert backend.state.Nqubit == 1
        assert backendState.node_index == [0]

        # random planar state
        rand_angle = self.rng.random() * 2 * np.pi
        rand_plane = self.rng.choice(np.array([i for i in graphix.pauli.Plane]))
        state = PlanarState(plane = rand_plane, angle = rand_angle)
        vec = Statevec(nqubit=1, data=state)
        backend = StatevectorBackend()
        backendState = backend.add_nodes(backendState=BackendState(), nodes=[0], data=state)
        np.testing.assert_allclose(vec.psi, backendState.state.psi)
        # assert backend.state.Nqubit == 1
        assert backendState.node_index == [0]



    
    def test_init_fail(self):
        # Fails if number of qubits doesn't match the dimension of the state asked to prepare
        # number of qubits is in len(nodes) or backend.prepare_states(nodes, data)
        # dimension is in len(data)

        rand_angle = self.rng.random(2) * 2 * np.pi
        rand_plane = self.rng.choice(np.array([i for i in graphix.pauli.Plane]), 2)

        state = PlanarState(plane = rand_plane[0], angle = rand_angle[0])
        state2 = PlanarState(plane = rand_plane[1], angle = rand_angle[1])
        backend = StatevectorBackend()
        with self.assertRaises(ValueError):
            backend.add_nodes(backendState=BackendState(), nodes=[0], data=[state, state2])






if __name__ == "__main__":
    unittest.main()
