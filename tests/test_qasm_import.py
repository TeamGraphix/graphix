import pytest
from graphix.transpiler import pattern_from_qasm3

def test_pattern_from_qasm3_basic():
    qasm = """
    OPENQASM 3;
    qubit q[2];
    h q[0];
    cx q[0], q[1];
    """
    pattern = pattern_from_qasm3(qasm)
    # Check that the pattern has the expected number of input nodes
    assert hasattr(pattern, "input_nodes")
    assert len(pattern.input_nodes) == 2
    # Check that the pattern is iterable and has commands
    cmds = list(pattern)
    assert len(cmds) > 0

def test_pattern_from_qasm3_all_supported_gates():
    qasm = """
    OPENQASM 3;
    qubit q[3];
    h q[0];
    x q[1];
    y q[2];
    z q[0];
    s q[1];
    rx(pi/2) q[2];
    ry(pi/4) q[0];
    rz(pi) q[1];
    cx q[0], q[1];
    swap q[1], q[2];
    ccx q[0], q[1], q[2];
    """
    pattern = pattern_from_qasm3(qasm)
    assert hasattr(pattern, "input_nodes")
    assert len(pattern.input_nodes) == 3
    cmds = list(pattern)
    assert len(cmds) > 0

def test_pattern_from_qasm3_unsupported_gate_warns():
    qasm = """
    OPENQASM 3;
    qubit q[1];
    t q[0];
    """
    with pytest.warns(UserWarning):
        pattern_from_qasm3(qasm)

def test_pattern_from_qasm3_measure_warns():
    qasm = """
    OPENQASM 3;
    qubit q[1];
    h q[0];
    measure q[0];
    """
    with pytest.warns(UserWarning, match="Measurement statements are ignored"):
        pattern_from_qasm3(qasm)

def test_pattern_from_qasm3_missing_qubit_register():
    qasm = """
    OPENQASM 3;
    h q[0];
    """
    with pytest.raises(ValueError, match="Qubit register must be declared before gates."):
        pattern_from_qasm3(qasm)
