from __future__ import annotations

from graphix.device_interface import CompileOptions, DeviceBackend, Job
from graphix.pattern import Pattern


class DummyOptions(CompileOptions):
    def __init__(self) -> str:
        """Define a dummy abstract method to satisfy ABC requirements."""


class DummyJob(Job):
    def get_id(self):
        return "dummy"

    def is_done(self):
        return True

    def cancel(self):
        pass

    def retrieve_result(self):
        return {"result": "dummy_result"}


class DummyBackend(DeviceBackend):
    def compile(self, options: CompileOptions = None):
        pass

    def submit_job(self, shots: int) -> Job:
        return DummyJob()


def test_dummy_backend_can_be_instantiated():
    backend = DummyBackend()
    pattern = Pattern()
    backend.set_pattern(pattern)
    backend.compile(DummyOptions())
    job = backend.submit_job(shots=100)
    assert isinstance(job, Job)
    result = job.retrieve_result()
    assert result == {"result": "dummy_result"}
