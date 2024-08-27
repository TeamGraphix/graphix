from __future__ import annotations


def check_list_elements(l, ty):
    for index, item in enumerate(l):
        if not isinstance(item, ty):
            raise TypeError(f"data[{index}] has type {type(item)} whereas {ty} is expected")
