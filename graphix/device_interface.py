"""Quantum hardware device interface.

Runs MBQC command sequence on quantum hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphix.pattern import Pattern


class PatternRunner:
    """MBQC pattern runner.

    Executes the measurement pattern.
    """

    def __init__(self, pattern: Pattern, backend: str = "ibmq", instance: str | None = None, resource: str | None = None, save_statevector: bool = False, optimizer_level: int = 1, shots: int = 1024) -> None:
        """Instantiate a pattern runner.

        Parameters
        ----------
        pattern: :class:`graphix.pattern.Pattern` object
            MBQC pattern to be executed.
        backend: str
            execution backend (optional, default is 'ibmq')
        instance: str | None
            instance name (optional, backend specific)
        resource: str | None
            resource name (optional, backend specific)
        instance: str | None
            instance name (optional, backend specific)
        save_statevector: bool
            whether to save the statevector before the measurements of output qubits (optional, default is 'False', backend specific)
        optimizer_level: int
            optimizer level (optional, default is '1', backend specific)
        shots: int
            number of shots (optional, default is '1024', backend specific)
        """
        self.pattern = pattern
        self.backend_name = backend

        if self.backend_name == "ibmq":
            try:
                from graphix_ibmq.runner import IBMQBackend
            except ImportError as e:
                raise ImportError(
                    "Failed to import graphix_ibmq. Please install graphix_ibmq by `pip install graphix-ibmq`."
                ) from e
            self.backend = IBMQBackend(pattern)
            kwargs_get_backend = {}
            if instance is not None:
                kwargs_get_backend["instance"] = instance
            if resource is not None:
                kwargs_get_backend["resource"] = resource
            self.backend.get_backend(**kwargs_get_backend)
            self.backend.to_qiskit(save_statevector)
            self.backend.transpile(optimizer_level)
            self.shots = shots
        else:
            raise ValueError("unknown backend")

    def simulate(self, **kwargs) -> Any:
        """Perform the simulation.

        Parameters
        ----------
        kwargs: dict
            keyword args for specified backend.

        Returns
        -------
        result: Any
            the simulation result,
            in the representation depending on the backend used.
        """
        if self.backend_name == "ibmq":
            shots = kwargs.get("shots", self.shots)
            noise_model = kwargs.get("noise_model", None)
            format_result = kwargs.get("format_result", True)

            result = self.backend.simulate(shots=shots, noise_model=noise_model, format_result=format_result)

        return result

    def run(self, **kwargs) -> Any:
        """Perform the execution.

        Parameters
        ----------
        kwargs: dict
            keyword args for specified backend.

        Returns
        -------
        result: Any
            the measurement result,
            in the representation depending on the backend used.
        """
        if self.backend_name == "ibmq":
            shots = kwargs.get("shots", self.shots)
            format_result = kwargs.get("format_result", True)
            optimization_level = kwargs.get("optimizer_level", 1)

            result = self.backend.run(shots=shots, format_result=format_result, optimization_level=optimization_level)

        return result

    def retrieve_result(self, **kwargs) -> Any:
        """Retrieve the execution result.

        Parameters
        ----------
        kwargs: dict
            keyword args for specified backend.

        Returns
        -------
        result: Any
            the measurement result,
            in the representation depending on the backend used.
        """
        if self.backend_name == "ibmq":
            job_id = kwargs.get("job_id", None)
            result = self.backend.retrieve_result(job_id)

        return result
