"""Types."""

from __future__ import annotations

import annotated_types
import typing_extensions

PositiveOrNullInt = typing_extensions.Annotated[int, annotated_types.Ge(0)]  # includes 0


def check_list_elements(l, ty):
    """Check that every element of the list has the given type."""
    for index, item in enumerate(l):
        if not isinstance(item, ty):
            raise TypeError(f"data[{index}] has type {type(item)} whereas {ty} is expected")
