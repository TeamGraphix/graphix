"""Types."""

from __future__ import annotations

from typing import TypeVar

import annotated_types
import typing_extensions

if typing_extensions.TYPE_CHECKING:
    from collections.abc import Iterable

PositiveOrNullInt = typing_extensions.Annotated[int, annotated_types.Ge(0)]  # includes 0

T = TypeVar("T")


def check_list_elements(l: Iterable[T], ty: type[T]) -> None:
    """Check that every element of the list has the given type."""
    for index, item in enumerate(l):
        if not isinstance(item, ty):
            raise TypeError(f"data[{index}] has type {type(item)} whereas {ty} is expected")
