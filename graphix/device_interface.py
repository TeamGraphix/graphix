"""Quantum hardware device interface

Runs MBQC command sequence on quantum hardware.

"""


class PatternRunner:
    """MBQC pattern runner

    Executes the measurement pattern.
    """

    def __init__(self, pattern, backend="ibmq", **kwargs):
        """

        Parameters
        -----------
        pattern: :class:`graphix.pattern.Pattern` object
            MBQC pattern to be executed.
        backend_name: str, optional
            execution backend, default is 'ibmq'.
        kwargs: dict
            keyword args for specified backend.
        """
        self.pattern = pattern
        self.backend_name = backend

        if self.backend_name == "ibmq":
            try:
                from graphix_ibmq.runner import IBMQBackend
            except:
                raise ImportError(
                    "Failed to import graphix_ibmq. Please install graphix_ibmq by `pip install graphix-ibmq`."
                )
            self.backend = IBMQBackend(pattern)
            try:
                instance = kwargs.get("instance", "ibm-q/open/main")
                resource = kwargs.get("resource", None)
                save_statevector = kwargs.get("save_statevector", False)
                optimization_level = kwargs.get("optimizer_level", 1)

                self.backend.get_backend(instance, resource)
                self.backend.to_qiskit(save_statevector)
                self.backend.transpile(optimization_level)
                self.shots = kwargs.get("shots", 1024)
            except:
                save_statevector = kwargs.get("save_statevector", False)
                optimization_level = kwargs.get("optimizer_level", 1)
                self.backend.to_qiskit(save_statevector)
                self.shots = kwargs.get("shots", 1024)
        else:
            raise ValueError("unknown backend")

    def simulate(self, **kwargs):
        """Perform the simulation.

        Parameters
        ----------
        kwargs: dict
            keyword args for specified backend.

        Returns
        -------
        result :
            the simulation result,
            in the representation depending on the backend used.
        """
        if self.backend_name == "ibmq":
            shots = kwargs.get("shots", self.shots)
            noise_model = kwargs.get("noise_model", None)
            format_result = kwargs.get("format_result", True)

            result = self.backend.simulate(shots=shots, noise_model=noise_model, format_result=format_result)

        return result

    def run(self, **kwargs):
        """Perform the execution.

        Parameters
        ----------
        kwargs: dict
            keyword args for specified backend.

        Returns
        -------
        result :
            the measurement result,
            in the representation depending on the backend used.
        """
        if self.backend_name == "ibmq":
            shots = kwargs.get("shots", self.shots)
            format_result = kwargs.get("format_result", True)
            optimization_level = kwargs.get("optimizer_level", 1)

            result = self.backend.run(shots=shots, format_result=format_result, optimization_level=optimization_level)

        return result

    def retrieve_result(self, **kwargs):
        """Retrieve the execution result.

        Parameters
        ----------
        kwargs: dict
            keyword args for specified backend.

        Returns
        -------
        result :
            the measurement result,
            in the representation depending on the backend used.
        """
        if self.backend_name == "ibmq":
            job_id = kwargs.get("job_id", None)
            result = self.backend.retrieve_result(job_id)

        return result
