from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphix.clifford import Clifford
from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.pauli import Pauli
from graphix.sim.statevec import Statevec, StatevectorBackend
from graphix.states import BasicStates, PlanarState
from tests.test_graphsim import meas_op

if TYPE_CHECKING:
    from numpy.random import Generator


class TestStatevec:
    def test_remove_one_qubit(self) -> None:
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

        assert np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())) == pytest.approx(1)

    @pytest.mark.parametrize(
        "state", [BasicStates.PLUS, BasicStates.ZERO, BasicStates.ONE, BasicStates.PLUS_I, BasicStates.MINUS_I]
    )
    def test_measurement_into_each_xyz_basis(self, state: BasicStates) -> None:
        n = 3
        k = 0
        # for measurement into |-> returns [[0, 0], ..., [0, 0]] (whose norm is zero)
        statevector = state.get_statevector()
        m_op = np.outer(statevector, statevector.T.conjugate())
        sv = Statevec(nqubit=n)
        sv.evolve(m_op, [k])
        sv.remove_qubit(k)

        sv2 = Statevec(nqubit=n - 1)
        assert np.abs(sv.psi.flatten().dot(sv2.psi.flatten().conj())) == pytest.approx(1)

    def test_measurement_into_minus_state(self) -> None:
        n = 3
        k = 0
        m_op = np.outer(BasicStates.MINUS.get_statevector(), BasicStates.MINUS.get_statevector().T.conjugate())
        sv = Statevec(nqubit=n)
        sv.evolve(m_op, [k])
        with pytest.raises(AssertionError):
            sv.remove_qubit(k)


class TestStatevecNew:
    # test initialization only
    def test_init_success(self, hadamardpattern, fx_rng: Generator) -> None:
        # plus state (default)
        backend = StatevectorBackend()
        backend.add_nodes(hadamardpattern.input_nodes)
        vec = Statevec(nqubit=1)
        assert np.allclose(vec.psi, backend.state.psi)
        assert len(backend.state.dims()) == 1

        # minus state
        backend = StatevectorBackend()
        backend.add_nodes(hadamardpattern.input_nodes, data=BasicStates.MINUS)
        vec = Statevec(nqubit=1, data=BasicStates.MINUS)
        assert np.allclose(vec.psi, backend.state.psi)
        assert len(backend.state.dims()) == 1

        # random planar state
        rand_angle = fx_rng.random() * 2 * np.pi
        rand_plane = fx_rng.choice(np.array(Plane))
        state = PlanarState(rand_plane, rand_angle)
        backend = StatevectorBackend()
        backend.add_nodes(hadamardpattern.input_nodes, data=state)
        vec = Statevec(nqubit=1, data=state)
        assert np.allclose(vec.psi, backend.state.psi)
        # assert backend.state.nqubit == 1
        assert len(backend.state.dims()) == 1

        # data input and Statevec input

    def test_init_fail(self, hadamardpattern, fx_rng: Generator) -> None:
        rand_angle = fx_rng.random(2) * 2 * np.pi
        rand_plane = fx_rng.choice(np.array(Plane), 2)

        state = PlanarState(rand_plane[0], rand_angle[0])
        state2 = PlanarState(rand_plane[1], rand_angle[1])
        with pytest.raises(ValueError):
            StatevectorBackend().add_nodes(hadamardpattern.input_nodes, data=[state, state2])

    def test_clifford(self) -> None:
        for clifford in Clifford:
            state = BasicStates.PLUS
            vec = Statevec(nqubit=1, data=state)
            backend = StatevectorBackend()
            backend.add_nodes(nodes=[0], data=state)
            # Applies clifford gate "Z"
            vec.evolve_single(clifford.matrix, 0)
            backend.apply_clifford(node=0, clifford=clifford)
            np.testing.assert_allclose(vec.psi, backend.state.psi)

    def test_deterministic_measure_one(self, fx_rng: Generator):
        # plus state & zero state (default), but with tossed coins
        for _ in range(10):
            backend = StatevectorBackend()
            coins = [fx_rng.choice([0, 1]), fx_rng.choice([0, 1])]
            expected_result = sum(coins) % 2
            states = [
                Pauli.X.eigenstate(coins[0]),
                Pauli.Z.eigenstate(coins[1]),
            ]
            nodes = range(len(states))
            backend.add_nodes(nodes=nodes, data=states)

            backend.entangle_nodes(edge=(nodes[0], nodes[1]))
            measurement = Measurement(0, Plane.XY)
            node_to_measure = backend.node_index[0]
            result = backend.measure(node=node_to_measure, measurement=measurement)
            assert result == expected_result

    def test_deterministic_measure(self):
        """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1."""
        for _ in range(10):
            # plus state (default)
            backend = StatevectorBackend()
            n_neighbors = 10
            states = [Pauli.X.eigenstate()] + [Pauli.Z.eigenstate() for i in range(n_neighbors)]
            nodes = range(len(states))
            backend.add_nodes(nodes=nodes, data=states)

            for i in range(1, n_neighbors + 1):
                backend.entangle_nodes(edge=(nodes[0], i))
            measurement = Measurement(0, Plane.XY)
            node_to_measure = backend.node_index[0]
            result = backend.measure(node=node_to_measure, measurement=measurement)
            assert result == 0
            assert list(backend.node_index) == list(range(1, n_neighbors + 1))

    def test_deterministic_measure_many(self):
        """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1."""
        for _ in range(10):
            # plus state (default)
            backend = StatevectorBackend()
            n_traps = 5
            n_neighbors = 5
            n_whatever = 5
            traps = [Pauli.X.eigenstate() for _ in range(n_traps)]
            dummies = [Pauli.Z.eigenstate() for _ in range(n_neighbors)]
            others = [Pauli.I.eigenstate() for _ in range(n_whatever)]
            states = traps + dummies + others
            nodes = range(len(states))
            backend.add_nodes(nodes=nodes, data=states)

            for dummy in nodes[n_traps : n_traps + n_neighbors]:
                for trap in nodes[:n_traps]:
                    backend.entangle_nodes(edge=(trap, dummy))
                for other in nodes[n_traps + n_neighbors :]:
                    backend.entangle_nodes(edge=(other, dummy))

            # Same measurement for all traps
            measurement = Measurement(0, Plane.XY)

            for trap in nodes[:n_traps]:
                node_to_measure = trap
                result = backend.measure(node=node_to_measure, measurement=measurement)
                assert result == 0

            assert list(backend.node_index) == list(range(n_traps, n_neighbors + n_traps + n_whatever))

    def test_deterministic_measure_with_coin(self, fx_rng: Generator):
        """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1.

        We add coin toss to that.
        """
        for _ in range(10):
            # plus state (default)
            backend = StatevectorBackend()
            n_neighbors = 10
            coins = [fx_rng.choice([0, 1])] + [fx_rng.choice([0, 1]) for _ in range(n_neighbors)]
            expected_result = sum(coins) % 2
            states = [Pauli.X.eigenstate(coins[0])] + [Pauli.Z.eigenstate(coins[i + 1]) for i in range(n_neighbors)]
            nodes = range(len(states))
            backend.add_nodes(nodes=nodes, data=states)

            for i in range(1, n_neighbors + 1):
                backend.entangle_nodes(edge=(nodes[0], i))
            measurement = Measurement(0, Plane.XY)
            node_to_measure = backend.node_index[0]
            result = backend.measure(node=node_to_measure, measurement=measurement)
            assert result == expected_result
            assert list(backend.node_index) == list(range(1, n_neighbors + 1))
