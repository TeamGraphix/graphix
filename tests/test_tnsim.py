import unittest
import itertools
import numpy as np
from quimb.tensor import Tensor
from graphix.transpiler import Circuit
from graphix.pattern import Pattern
from graphix.ops import Ops, States
from graphix.sim.tensornet import TensorNetworkBackend, MBQCTensorNet, gen_str
from graphix.clifford import CLIFFORD
import tests.random_circuit as rc


def random_op(sites, dtype=np.complex64, seed=0):
    np.random.seed(seed)
    size = 2**sites
    if dtype is np.complex64:
        return np.random.randn(size, size).astype(np.float32) + 1j * np.random.randn(size, size).astype(np.float32)
    if dtype is np.complex128:
        return np.random.randn(size, size).astype(np.float64) + 1j * np.random.randn(size, size).astype(np.float64)
    return np.random.randn(size, size).astype(dtype)


CZ = Ops.cz
plus = States.plus


class TestTN(unittest.TestCase):
    def test_add_node(self):
        node_index = np.random.randint(0, 1000)
        tn = MBQCTensorNet()

        tn.add_qubit(node_index)

        np.testing.assert_equal(set(tn.tag_map.keys()), {str(node_index), "Open"})
        np.testing.assert_equal(tn.tensors[0].data, plus)

    def test_add_nodes(self):
        node_index = set(np.random.randint(0, 1000, 20))
        tn = MBQCTensorNet()

        tn.graph_prep = "sequential"
        tn.add_qubits(node_index)

        np.testing.assert_equal(set(tn.tag_map.keys()), set([str(ind) for ind in node_index]) | {"Open"})
        for tensor in tn.tensor_map.values():
            np.testing.assert_equal(tensor.data, plus)

    def test_entangle_nodes(self):
        random_vec = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, 2)
        circuit = Circuit(2)
        pattern = circuit.transpile()
        pattern.add(["E", (0, 1)])
        tn = pattern.simulate_pattern(backend="tensornetwork", graph_prep="sequential")
        dummy_index = [gen_str() for _ in range(2)]
        qubit_index = 0
        for n in tn._dangling.keys():
            ind = tn._dangling[n]
            tids = tn._get_tids_from_inds(ind)
            tensor = tn.tensor_map[tids.popleft()]
            tensor.reindex({ind: dummy_index[qubit_index]}, inplace=True)
            qubit_index += 1

        random_vec_ts = Tensor(random_vec, dummy_index, ["random_vector"])
        tn.add_tensor(random_vec_ts)
        contracted = tn.contract()
        # reference
        contracted_ref = np.einsum("abcd, c, d, ab->", CZ.reshape(2, 2, 2, 2), plus, plus, random_vec)
        np.testing.assert_almost_equal(contracted, contracted_ref)

    def test_apply_one_site_operator(self):
        cmds = [
            ["X", 0, [15]],
            ["Z", 0, [15]],
            ["C", 0, np.random.randint(0, 23)],
        ]
        random_vec = np.random.randn(2)

        circuit = Circuit(1)
        pattern = circuit.transpile()
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
        np.testing.assert_almost_equal(contracted, contracted_ref)

    def test_expectation_value1(self):
        circuit = Circuit(1)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_expectation_value2(self):
        circuit = Circuit(2)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op2 = random_op(2)
        input = [0, 1]
        for qargs in itertools.permutations(input):
            value1 = state.expectation_value(random_op2, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op2, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_expectation_value3(self):
        circuit = Circuit(3)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op3 = random_op(3)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op3, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_expectation_value3_sequential(self):
        circuit = Circuit(3)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork", graph_prep="sequential")
        random_op3 = random_op(3)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op3, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_expectation_value3_subspace1(self):
        circuit = Circuit(3)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input, 1):
            value1 = state.expectation_value(random_op1, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op1, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_expectation_value3_subspace2(self):
        circuit = Circuit(3)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op2 = random_op(2)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input, 2):
            value1 = state.expectation_value(random_op2, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op2, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_expectation_value3_subspace2_sequential(self):
        circuit = Circuit(3)
        state = circuit.simulate_statevector()
        pattern = circuit.transpile()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork", graph_prep="sequential")
        random_op2 = random_op(2)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input, 2):
            value1 = state.expectation_value(random_op2, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op2, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_hadamard(self):
        circuit = Circuit(1)
        circuit.h(0)
        pattern = circuit.transpile()
        pattern.standardize()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(
            value1,
            value2,
        )

    def test_s(self):
        circuit = Circuit(1)
        circuit.s(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_x(self):
        circuit = Circuit(1)
        circuit.x(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_y(self):
        circuit = Circuit(1)
        circuit.y(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_z(self):
        circuit = Circuit(1)
        circuit.z(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_rx(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rx(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_ry(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.ry(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_rz(self):
        theta = np.random.random() * 2 * np.pi
        circuit = Circuit(1)
        circuit.rz(0, theta)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_i(self):
        circuit = Circuit(1)
        circuit.i(0)
        pattern = circuit.transpile()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op1 = random_op(1)
        value1 = state.expectation_value(random_op1, [0])
        value2 = tn_mbqc.expectation_value(random_op1, [0])
        np.testing.assert_almost_equal(value1, value2)

    def test_cnot(self):
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        pattern = circuit.transpile()
        pattern.standardize()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op2 = random_op(2)
        value1 = state.expectation_value(random_op2, [0, 1])
        value2 = tn_mbqc.expectation_value(random_op2, [0, 1])
        np.testing.assert_almost_equal(value1, value2)

    def test_with_graphtrans(self):
        nqubits = 3
        depth = 6
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op3 = random_op(3)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op3, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_with_graphtrans_sequential(self):
        nqubits = 3
        depth = 6
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork", graph_prep="sequential")
        random_op3 = random_op(3)
        input = [0, 1, 2]
        for qargs in itertools.permutations(input):
            value1 = state.expectation_value(random_op3, list(qargs))
            value2 = tn_mbqc.expectation_value(random_op3, list(qargs))
            np.testing.assert_almost_equal(value1, value2)

    def test_coef_state(self):
        nqubits = 4
        depth = 2
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.standardize_and_transpile()

        statevec_ref = circuit.simulate_statevector()

        tn = pattern.simulate_pattern("tensornetwork")
        for number in range(len(statevec_ref.flatten())):
            self.subTest(number=number)
            coef_tn = tn.get_basis_coefficient(number)
            coef_sv = statevec_ref.flatten()[number]

            np.testing.assert_almost_equal(abs(coef_tn), abs(coef_sv))

    def test_to_statevector(self):
        nqubits_set = [i for i in range(2, 6)]
        depth = 3
        for nqubits in nqubits_set:
            self.subTest(nqubit=nqubits)
            pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
            circuit = rc.generate_gate(nqubits, depth, pairs)
            pattern = circuit.standardize_and_transpile()
            statevec_ref = circuit.simulate_statevector()

            tn = pattern.simulate_pattern("tensornetwork")
            statevec_tn = tn.to_statevector()

            inner_product = np.inner(statevec_tn, statevec_ref.flatten().conjugate())
            np.testing.assert_almost_equal(abs(inner_product), 1)

    def test_evolve(self):
        nqubits = 3
        depth = 6
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        state = circuit.simulate_statevector()
        tn_mbqc = pattern.simulate_pattern(backend="tensornetwork")
        random_op3 = random_op(3)
        random_op3_exp = random_op(3)

        state.evolve(random_op3, [0, 1, 2])
        tn_mbqc.evolve(random_op3, [0, 1, 2], decompose=False)

        expval_tn = tn_mbqc.expectation_value(random_op3_exp, [0, 1, 2])
        expval_ref = state.expectation_value(random_op3_exp, [0, 1, 2])

        np.testing.assert_almost_equal(expval_tn, expval_ref)


if __name__ == "__main__":
    unittest.main()
