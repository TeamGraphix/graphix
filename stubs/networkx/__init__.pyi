from collections.abc import Collection
from typing import Any, TypeVar

import numpy.typing as npt
from networkx.classes.graph import Graph

_G = TypeVar("_G", bound=Graph)

# parameter `nodelist` is not included in networkx-types
# https://github.com/python/typeshed/blob/main/stubs/networkx/networkx/convert_matrix.pyi
def from_numpy_array(adj_mat: npt.NDArray[Any], create_using: type[_G], *, nodelist: Collection[int]) -> _G: ...
