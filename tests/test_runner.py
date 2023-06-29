import unittest
import numpy as np
from graphix.device_interface import PatternRunner
import tests.random_circuit as rc
from qiskit import Aer


def modify_statevector(statevector, output_qubit):
    N = round(np.log2(len(statevector)))
    new_statevector = np.zeros(2 ** len(output_qubit), dtype=complex)
    for i in range(len(statevector)):
        i_str = format(i, f"0{N}b")
        new_idx = ""
        for idx in output_qubit:
            new_idx += i_str[N - idx - 1]
        new_statevector[int(new_idx, 2)] += statevector[i]
    return new_statevector


class TestPatternRunner(unittest.TestCase):
    def test_ibmq_backend(self):
        nqubits = 5
        depth = 5
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        pattern = circuit.transpile()
        state = pattern.simulate_pattern()

        sim = Aer.get_backend("aer_simulator")
        runner = PatternRunner(pattern, backend="ibmq", save_statevector=True)
        sim_result = runner.simulate(format_result=False)
        state_qiskit = sim_result.get_statevector(runner.backend.circ)
        state_qiskit_mod = modify_statevector(state_qiskit, runner.backend.circ_output)

        np.testing.assert_almost_equal(np.abs(np.dot(state_qiskit_mod.conjugate(), state.flatten())), 1)


if __name__ == "__main__":
    unittest.main()
