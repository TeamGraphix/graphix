from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest
from numpy.random import PCG64, Generator
from quimb.tensor import Tensor

import tests.random_circuit as rc
from graphix.clifford import CLIFFORD
from graphix.ops import Ops
from graphix.sim.tensornet import MBQCTensorNet, gen_str
from graphix.states import BasicStates
from graphix.transpiler import Circuit


def random_op(sites: int, dtype: type, rng: Generator) -> npt.NDArray:
    size = 2**sites
    if dtype is np.complex64:
        return rng.normal(size=(size, size)).astype(np.float32) + 1j * rng.normal(size=(size, size)).astype(np.float32)
    if dtype is np.complex128:
        return rng.normal(size=(size, size)).astype(np.float64) + 1j * rng.normal(size=(size, size)).astype(np.float64)
    return rng.normal(size=(size, size)).astype(dtype)


CZ = Ops.cz
plus = BasicStates.PLUS.get_statevector()


class TestTN:
    def test_add_node(self, fx_rng: Generator) -> None:
        node_index = fx_rng.integers(0, 1000)
        tn = MBQCTensorNet()

        tn.add_qubit(node_index)

        assert set(tn.tag_map.keys()) == {str(node_index), "Open"}
        assert (tn.tensors[0].data == plus).all()

    def test_add_nodes(self, fx_rng: Generator) -> None:
        node_index = set(fx_rng.integers(0, 1000, 20))
        tn = MBQCTensorNet()

        tn.graph_prep = "sequential"
        tn.add_qubits(node_index)

        assert set(tn.tag_map.keys()) == {str(ind) for ind in node_index} | {"Open"}
        for tensor in tn.tensor_map.values():
            assert (tensor.data == plus).all()

    def test_entangle_nodes(self) -> None:
        random_vec = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, 2)
        circuit = Circuit(2)
        pattern = circuit.transpile().pattern
        pattern.add(["E", (0, 1)])
        tn = pattern.simulate_pattern(backend="tensornetwork", graph_prep="sequential")
        dummy_index = [gen_str() for _ in range(2)]
        for qubit_index, n in enumerate(tn._dangling):
            ind = tn._dangling[n]
            tids = tn._get_tids_from_inds(ind)
            tensor = tn.tensor_map[tids.popleft()]
            tensor.reindex({ind: dummy_index[qubit_index]}, inplace=True)

        random_vec_ts = Tensor(random_vec, dummy_index, ["random_vector"])
        tn.add_tensor(random_vec_ts)
        contracted = tn.contract()
        # reference
        contracted_ref = np.einsum("abcd, c, d, ab->", CZ.reshape(2, 2, 2, 2), plus, plus, random_vec)
        assert contracted == pytest.approx(contracted_ref)

    def test_apply_one_site_operator(self, fx_rng: Generator) -> None:
        cmds = [
            ["X", 0, [15]],
            ["Z", 0, [15]],
            ["C", 0, fx_rng.integers(0, 23)],
        ]
        random_vec = fx_rng.normal(size=2)

        circuit = Circuit(1)
        pattern = circuit.transpile().pattern
        pattern.results[15] = 1  # X&Z operator will be applied.
        for cmd in cmds:
            pattern.add(cmd)
        tn = pattern.simulate_pattern(backend="tensornetwork")
        dummy_index = gen_str()
        ind = tn._dangling.pop("0")
        tensor = tn.tensor_map[tn._get_tids_from_inds(ind).popleft()]
        tensor.reindex({ind: dummy_index}, inplace=True)
        random_vec_ts = Tensor(random_vec, [dummy_index], ["random_vector"])
        tn.add_tensor(random_vec_ts)
        contracted = tn.contract()

        # reference
        ops = [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, -1.0]]),
            CLIFFORD[cmds[2][2]],
        ]
        contracted_ref = np.einsum("i,ij,jk,kl,l", random_vec, ops[2], ops[1], ops[0], plus)
        assert contracted == pytest.approx(contracted_ref)

    def test_expectation_value1(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        state = circuit.simulate_statevector().statevec
        pattern = circuit.transpile().pattern
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        assert value1 == pytest.approx(value2)

    def test_expectation_value2(self, fx_rng: Generator) -> None:
        circuit = Circuit(2)
        state = circuit.simulate_statevector().statevec
        pattern = circuit.transpile().pattern
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op2 = random_op(2, np.complex128, fx_rng)
        input_list = [0, 1]
        for qargs in itertools.permutations(input_list):
            value1 = state.expectation_value(random_op2, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op2, list(qargs))
            assert value1 == pytest.approx(value2)

    def test_expectation_value3(self, fx_rng: Generator) -> None:
        circuit = Circuit(3)
        state = circuit.simulate_statevector().statevec
        pattern = circuit.transpile().pattern
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op3 = random_op(3, np.complex128, fx_rng)
        input_list = [0, 1, 2]
        for qargs in itertools.permutations(input_list):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op3, list(qargs))
            assert value1 == pytest.approx(value2)

    def test_expectation_value3_sequential(self, fx_rng: Generator) -> None:
        circuit = Circuit(3)
        state = circuit.simulate_statevector().statevec
        pattern = circuit.transpile().pattern
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork", graph_prep="sequential")
        random_op3 = random_op(3, np.complex128, fx_rng)
        input_list = [0, 1, 2]
        for qargs in itertools.permutations(input_list):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op3, list(qargs))
            assert value1 == pytest.approx(value2)

    def test_expectation_value3_subspace1(self, fx_rng: Generator) -> None:
        circuit = Circuit(3)
        state = circuit.simulate_statevector().statevec
        pattern = circuit.transpile().pattern
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        input_list = [0, 1, 2]
        for qargs in itertools.permutations(input_list, 1):
            value1 = state.expectation_value(random_op1, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op1, list(qargs))
            assert value1 == pytest.approx(value2)

    def test_expectation_value3_subspace2(self, fx_rng: Generator) -> None:
        circuit = Circuit(3)
        state = circuit.simulate_statevector().statevec
        pattern = circuit.transpile().pattern
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op2 = random_op(2, np.complex128, fx_rng)
        input_list = [0, 1, 2]
        for qargs in itertools.permutations(input_list, 2):
            value1 = state.expectation_value(random_op2, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op2, list(qargs))
            assert value1 == pytest.approx(value2)

    def test_expectation_value3_subspace2_sequential(self, fx_rng: Generator) -> None:
        circuit = Circuit(3)
        state = circuit.simulate_statevector().statevec
        pattern = circuit.transpile().pattern
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork", graph_prep="sequential")
        random_op2 = random_op(2, np.complex128, fx_rng)
        input_list = [0, 1, 2]
        for qargs in itertools.permutations(input_list, 2):
            value1 = state.expectation_value(random_op2, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op2, list(qargs))
            assert value1 == pytest.approx(value2)

    def test_hadamard(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.h(0)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        assert value1 == pytest.approx(value2)

    def test_s(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.s(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        assert value1 == pytest.approx(value2)

    def test_x(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.x(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        assert value1 == pytest.approx(value2)

    def test_y(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.y(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        assert value1 == pytest.approx(value2)

    def test_z(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.z(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        assert value1 == pytest.approx(value2)

    def test_rx(self, fx_rng: Generator) -> None:
        theta = fx_rng.uniform() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rx(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        assert value1 == pytest.approx(value2)

    def test_ry(self, fx_rng: Generator) -> None:
        theta = fx_rng.uniform() * 2 * np.pi
        circuit = Circuit(1)
        circuit.ry(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        assert value1 == pytest.approx(value2)

    def test_rz(self, fx_rng: Generator) -> None:
        theta = fx_rng.uniform() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rz(0, theta)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        assert value1 == pytest.approx(value2)

    def test_i(self, fx_rng: Generator) -> None:
        circuit = Circuit(1)
        circuit.i(0)
        pattern = circuit.transpile().pattern
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        assert value1 == pytest.approx(value2)

    def test_cnot(self, fx_rng: Generator) -> None:
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op2 = random_op(2, np.complex128, fx_rng)
        value1 = state.expectation_value(random_op2, [0, 1])
        value2 = tn_mbqc.expectation_value(random_op2, [0, 1])
        assert value1 == pytest.approx(value2)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_ccx(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = rc.get_rand_circuit(4, 6, rng)
        circuit.ccx(0, 1, 2)
        pattern = circuit.transpile().pattern
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op3 = random_op(3, np.complex128, rng)
        value1 = state.expectation_value(random_op3, [0, 1, 2])
        value2 = tn_mbqc.expectation_value(random_op3, [0, 1, 2])
        assert value1 == pytest.approx(value2)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_with_graphtrans(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = rc.get_rand_circuit(4, 6, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op3 = random_op(3, np.complex128, rng)
        input_list = [0, 1, 2]
        for qargs in itertools.permutations(input_list):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op3, list(qargs))
            assert value1 == pytest.approx(value2)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_with_graphtrans_sequential(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = rc.get_rand_circuit(4, 6, rng)
        pattern = circuit.transpile().pattern
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork", graph_prep="sequential")
        random_op3 = random_op(3, np.complex128, rng)
        input_list = [0, 1, 2]
        for qargs in itertools.permutations(input_list):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op3, list(qargs))
            assert value1 == pytest.approx(value2)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_coef_state(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = rc.get_rand_circuit(4, 2, rng)

        pattern = circuit.standardize_and_transpile().pattern

        statevec_ref = circuit.simulate_statevector().statevec

        tn = pattern.simulate_pattern("tensornetwork")
        for number in range(len(statevec_ref.flatten())):
            coef_tn = tn.get_basis_coefficient(number)
            coef_sv = statevec_ref.flatten()[number]

            assert abs(coef_tn) == pytest.approx(abs(coef_sv))

    @pytest.mark.parametrize(("nqubits", "jumps"), itertools.product(range(2, 6), range(1, 6)))
    def test_to_statevector(self, fx_bg: PCG64, nqubits: int, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = rc.get_rand_circuit(nqubits, 3, rng)
        pattern = circuit.standardize_and_transpile().pattern
        statevec_ref = circuit.simulate_statevector().statevec

        tn = pattern.simulate_pattern("tensornetwork")
        statevec_tn = tn.to_statevector()

        inner_product = np.inner(statevec_tn, statevec_ref.flatten().conjugate())
        assert abs(inner_product) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_evolve(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        circuit = rc.get_rand_circuit(4, 6, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        state = circuit.simulate_statevector().statevec
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op3 = random_op(3, np.complex128, rng)
        random_op3_exp = random_op(3, np.complex128, rng)

        state.evolve(random_op3, [0, 1, 2])
        tn_mbqc.evolve(random_op3, [0, 1, 2], decompose=False)

        expval_tn = tn_mbqc.expectation_value(random_op3_exp, [0, 1, 2])
        expval_ref = state.expectation_value(random_op3_exp, [0, 1, 2])

        assert expval_tn == pytest.approx(expval_ref)
