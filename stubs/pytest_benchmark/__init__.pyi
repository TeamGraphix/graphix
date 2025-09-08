from typing import Callable, ParamSpec, Protocol, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

class BenchmarkFixture(Protocol):
    def __call__(self, func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R: ...
