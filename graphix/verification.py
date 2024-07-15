import abc
import graphix.states
import graphix.sim.base_backend


class Run(abc.ABC) :
    def __init__(self, pattern) -> None:
        self.pattern = pattern
        self.prepared_states = None


    @abc.abstractmethod
    def delegate(self, backend:graphix.sim.base_backend.Backend):
        ...

class ComputationRun(Run) :
    def __init__(self, pattern) -> None:
        super().__init__(pattern)
        self.prepared_states = 0

    



class TestRun(Run) :
    def __init__(self) -> None:
        super().__init__()
