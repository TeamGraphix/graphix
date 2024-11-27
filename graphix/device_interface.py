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

    def __init__(self, pattern: Pattern, backend: str = "ibmq", **kwargs) -> None:
        """Instantiate a pattern runner.

        Parameters
        ----------
        pattern: :class:`graphix.pattern.Pattern` object
            MBQC pattern to be executed.
        backend: str
            execution backend (optional, default is 'ibmq')
        kwargs: dict
            keyword args for specified backend.
        """
        self.pattern = pattern
        self.backend_name = backend

        if self.backend_name == "ibmq":
            try:
                from graphix_ibmq.runner import IBMQBackend
            except Exception as e:
                raise ImportError(
                    "Failed to import graphix_ibmq. Please install graphix_ibmq by `pip install graphix-ibmq`."
                ) from e
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
            except Exception:  # noqa: BLE001
                save_statevector = kwargs.get("save_statevector", False)
                optimization_level = kwargs.get("optimizer_level", 1)
                self.backend.to_qiskit(save_statevector)
                self.shots = kwargs.get("shots", 1024)
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
