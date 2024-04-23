from __future__ import annotations

import sys

import numpy as np
import numpy.typing as npt
import pytest

try:
    import qiskit
    from qiskit_aer import Aer
except ModuleNotFoundError:
    pass

from typing import TYPE_CHECKING

import graphix
from graphix.device_interface import PatternRunner

if TYPE_CHECKING:
    from collections.abc import Collection

    from pytest_mock import MockerFixture


def modify_statevector(statevector: npt.ArrayLike, output_qubit: Collection[int]) -> npt.NDArray:
    statevector = np.asarray(statevector)
    n = round(np.log2(len(statevector)))
    new_statevector = np.zeros(2 ** len(output_qubit), dtype=complex)
    for i in range(len(statevector)):
        i_str = format(i, f"0{n}b")
        new_idx = ""
        for idx in output_qubit:
            new_idx += i_str[n - idx - 1]
        new_statevector[int(new_idx, 2)] += statevector[i]
    return new_statevector


class TestPatternRunner:
    @pytest.mark.skipif(sys.modules.get("qiskit") is None, reason="qiskit not installed")
    def test_ibmq_backend(self, mocker: MockerFixture) -> None:
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

        runner = mocker.Mock()

        runner.IBMQBackend().circ = qc
        runner.IBMQBackend().simulate.return_value = result
        runner.IBMQBackend().circ_output = [0, 1, 2]

        sys.modules["graphix_ibmq.runner"] = runner

        # circuit in graphix
        circuit = graphix.Circuit(3)
        circuit.h(1)
        circuit.h(2)
        circuit.rx(1, 1.23)
        circuit.rz(2, 1.23)
        circuit.cnot(0, 1)
        circuit.cnot(1, 2)

        pattern = circuit.transpile().pattern
        state = pattern.simulate_pattern()

        runner = PatternRunner(pattern, backend="ibmq", save_statevector=True)
        sim_result = runner.simulate(format_result=False)
        state_qiskit = sim_result.get_statevector(runner.backend.circ)
        state_qiskit_mod = modify_statevector(state_qiskit, runner.backend.circ_output)

        assert np.abs(np.dot(state_qiskit_mod.conjugate(), state.flatten())) == pytest.approx(1)
