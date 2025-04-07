import pytest
from graphix.device_interface import MBQCBackend, JobHandle, CompileOptions
from graphix.pattern import Pattern


class DummyOptions(CompileOptions):
    pass


class DummyJob(JobHandle):
    def get_id(self): return "dummy"
    def is_done(self): return True
    def cancel(self): pass


class DummyBackend(MBQCBackend):
    def compile(self, options: CompileOptions = None): pass
    def submit_job(self, shots: int) -> JobHandle: return DummyJob()
    def retrieve_result(self, job_handle: JobHandle): return "ok"


def test_dummy_backend_can_be_instantiated():
    backend = DummyBackend()
    pattern = Pattern()
    backend.set_pattern(pattern)
    backend.compile(DummyOptions())
    job = backend.submit_job(shots=100)
    assert isinstance(job, JobHandle)
    result = backend.retrieve_result(job)
    assert result == "ok"
