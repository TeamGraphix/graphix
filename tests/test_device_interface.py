from graphix.device_interface import CompileOptions, DeviceBackend, JobHandler
from graphix.pattern import Pattern


class DummyOptions(CompileOptions):
    def __repr__(self) -> str:
        return "DummyOptions()"


class DummyJob(JobHandler):
    def get_id(self):
        return "dummy"

    def is_done(self):
        return True

    def cancel(self):
        pass


class DummyBackend(DeviceBackend):
    def compile(self, options: CompileOptions = None):
        pass

    def submit_job(self, shots: int) -> JobHandler:
        return DummyJob()

    def retrieve_result(self, job_handle: JobHandler):
        return "ok"


def test_dummy_backend_can_be_instantiated():
    backend = DummyBackend()
    pattern = Pattern()
    backend.set_pattern(pattern)
    backend.compile(DummyOptions())
    job = backend.submit_job(shots=100)
    assert isinstance(job, JobHandler)
    result = backend.retrieve_result(job)
    assert result == "ok"
