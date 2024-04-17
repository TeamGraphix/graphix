import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

try:
    import qiskit
    from qiskit_aer import Aer
except ModuleNotFoundError:
    pass

import graphix
from graphix.device_interface import PatternRunner

# Bypass the import error of graphix_ibmq
gx_ibmq_mock = MagicMock()
sys.modules["graphix_ibmq.runner"] = gx_ibmq_mock


def modify_statevector(statevector, output_qubit):
    statevector = np.asarray(statevector)
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
    @unittest.skipIf(sys.modules.get("qiskit") is None, "qiskit not installed")
    def test_ibmq_backend(self):
        # circuit in qiskit
        qc = qiskit.QuantumCircuit(3)
        qc.h(0)
        qc.rx(1.23, 1)
        qc.rz(1.23, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.save_statevector()
        sim = Aer.get_backend("aer_simulator")
        new_qc = qiskit.transpile(qc, sim)
        job = sim.run(new_qc)
        result = job.result()

        # Mock
        gx_ibmq_mock.IBMQBackend().circ = qc
        gx_ibmq_mock.IBMQBackend().simulate.return_value = result
        gx_ibmq_mock.IBMQBackend().circ_output = [0, 1, 2]

        # circuit in graphix
        circuit = graphix.Circuit(3)
        circuit.h(1)
        circuit.h(2)
        circuit.rx(1, 1.23)
        circuit.rz(2, 1.23)
        circuit.cnot(0, 1)
        circuit.cnot(1, 2)

        pattern = circuit.transpile()
        state = pattern.simulate_pattern()

        runner = PatternRunner(pattern, backend="ibmq", save_statevector=True)
        sim_result = runner.simulate(format_result=False)
        state_qiskit = sim_result.get_statevector(runner.backend.circ)
        state_qiskit_mod = modify_statevector(state_qiskit, runner.backend.circ_output)

        np.testing.assert_almost_equal(np.abs(np.dot(state_qiskit_mod.conjugate(), state.flatten())), 1)


if __name__ == "__main__":
    unittest.main()
