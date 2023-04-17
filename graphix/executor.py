"""MBQC executor

Executes MBQC using actual quantum devices.

"""

from .exe.ibmq import IBMQBackend

class PatternExecutor:
    """MBQC exeulator

    Executes the measurement pattern.
    """

    def __init__(self, pattern, backend, **kwargs):
        """
        Parameteres
        -----------
        pattern: :class:`graphix.pattern.Pattern` object
            MBQC pattern to be executed.
        backend_name: str, 'ibmq'
            execution backend (optional), default is 'ibmq'.
        kwargs: keyword args for specified backend.
        """
        self.pattern = pattern
        self.backend_name = backend

        if self.backend_name == 'ibmq':
            self.backend = IBMQBackend(pattern, **kwargs)
        else:
            raise ValueError("unknown backend")

    def run(self):
        """Perform the execution.

        Returns
        -------
        result :
            the measurement result,
            in the representation depending on the backend used.
        """
        if self.backend_name == 'ibmq':
            self.job = self.backend.backend.run(self.backend.circ, shots = self.backend.shots, dynamic=True)
            print(f"Your job's id: {self.job.job_id()}")
            result = self.job.result()

        return result
    
    def retrieve_result(self, job_id):
        """Retrieve the execution result.

        Returns
        -------
        result :
            the measurement result,
            in the representation depending on the backend used.
        """
        if self.backend_name == 'ibmq':
            self.job = self.backend.backend.retrieve_job(job_id)
            result = self.job.result()

        return result