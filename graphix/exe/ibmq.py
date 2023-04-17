import numpy as np
from qiskit import IBMQ
from qiskit_ibm_provider import IBMProvider
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from graphix.clifford import CLIFFORD_TO_QISKIT

class IBMQBackend:
    """MBQC executor with IBMQ backend."""

    def __init__(self, pattern, instance, resource, shots):
        """
        Parameteres
        -----------
        pattern: :class:`graphix.pattern.Pattern` object
            MBQC pattern to be simulated.
        instance : str
            instance name of IBMQ provider.
        resource : str
            resource name of IBMQ provider.
        """
        self.pattern = pattern
        self.instance = instance
        self.resource = resource
        self.shots = shots
        self.provider = IBMProvider(instance = self.instance)
        self.backend = self.provider.get_backend(self.resource)
        self.circ = self.to_qiskit()

    def to_qiskit(self):
        """convert the MBQC pattern to the qiskit cuicuit and transpile for the designated resource.
        Returns
        -------
        circ : :class:`qiskit.circuit.quantumcircuit.QuantumCircuit` object
        """
        n = self.pattern.max_space()
        N_node = self.pattern.Nnode

        qr = QuantumRegister(n)
        cr = ClassicalRegister(N_node)
        circ = QuantumCircuit(qr, cr)

        empty_qubit = [i for i in range(n)]  # list indicating the free circuit qubits
        qubit_dict = {}  # dictionary to record the correspondance of pattern nodes and circuit qubits

        for cmd in self.pattern.seq: 

            if cmd[0] == 'N':
                circ_ind = empty_qubit[0]
                empty_qubit.pop(0)
                circ.reset(circ_ind)
                circ.h(circ_ind)
                qubit_dict[cmd[1]] = circ_ind

            if cmd[0] == 'E':
                circ.cz(qubit_dict[cmd[1][0]], qubit_dict[cmd[1][1]])

            if cmd[0] == 'M':
                circ_ind = qubit_dict[cmd[1]]
                plane = cmd[2]
                alpha = cmd[3]*np.pi
                s_list = cmd[4]
                t_list = cmd[5]

                if len(cmd) == 6:
                    if plane == 'XY':
                        # act p and h to implement non-Z-basis measurement 
                        if alpha != 0:
                            for s in s_list: # act x every time 1 comes in the s_list
                                with circ.if_test((cr[s], 1)):
                                    circ.x(circ_ind)
                            circ.p(-alpha, circ_ind) # align |+_alpha> (or |+_-alpha>) with |+>

                        for t in t_list: # act z every time 1 comes in the t_list
                            with circ.if_test((cr[t], 1)):
                                circ.z(circ_ind)
                
                        circ.h(circ_ind) # align |+> with |0>
                        
                        circ.measure(circ_ind, cmd[1]) # measure and store the result
                        empty_qubit.append(circ_ind) # liberate the circuit qubit

                elif len(cmd) == 7:
                    cid = cmd[6]
                    for op in CLIFFORD_TO_QISKIT[cid]:
                        exec(f"circ.{op}({circ_ind})")

                    if plane == 'XY':
                        # act p and h to implement non-Z-basis measurement 
                        if alpha != 0:
                            for s in s_list: # act x every time 1 comes in the s_list
                                with circ.if_test((cr[s], 1)):
                                    circ.x(circ_ind)
                            circ.p(-alpha, circ_ind) # align |+_alpha> (or |+_-alpha>) with |+>

                        for t in t_list: # act z every time 1 comes in the t_list
                            with circ.if_test((cr[t], 1)):
                                circ.z(circ_ind)
                
                        circ.h(circ_ind) # align |+> with |0>
                        
                        circ.measure(circ_ind, cmd[1]) # measure and store the result
                        empty_qubit.append(circ_ind) # liberate the circuit qubit

            if cmd[0] == 'X':
                circ_ind = qubit_dict[cmd[1]]
                s_list = cmd[2]
                for s in s_list:
                    with circ.if_test((cr[s], 1)):
                        circ.x(circ_ind)

            if cmd[0] == 'Z':
                circ_ind = qubit_dict[cmd[1]]
                s_list = cmd[2]
                for s in s_list:
                    with circ.if_test((cr[s], 1)):
                        circ.z(circ_ind)

            if cmd[0]  == 'C':
                circ_ind = qubit_dict[cmd[1]]
                cid = cmd[2]
                for op in CLIFFORD_TO_QISKIT[cid]:
                    exec(f"circ.{op}({circ_ind})")

        for node in self.pattern.output_nodes:
            circ_ind = qubit_dict[node]
            circ.measure(circ_ind, node)

        circ = transpile(circ, backend = self.backend)

        return circ