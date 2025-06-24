"""Mixins for eval-friendly `repr` for dataclasses and Enum members."""

from __future__ import annotations

import dataclasses
from dataclasses import MISSING
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # these live only in the stub package, not at runtime
    from _typeshed import DataclassInstance


class DataclassReprMixin:
    """
    Mixin for a concise, eval-friendly `repr` of dataclasses.

    Compared to the default dataclass `repr`:
      - Class variables are omitted (dataclasses.fields only returns actual fields).
      - Fields whose values equal their defaults are omitted.
      - Field names are only shown when preceding fields have been omitted, ensuring positional listings when possible.

    Use with `@dataclass(repr=False)` on the target class.
    """

    def __repr__(self: DataclassInstance) -> str:
        """Return a representation string for a dataclass."""
        cls_name = type(self).__name__
        arguments = []
        saw_omitted = False
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.default is not MISSING or field.default_factory is not MISSING:
                default = field.default_factory() if field.default_factory is not MISSING else field.default
                if value == default:
                    saw_omitted = True
                    continue
            custom_repr = field.metadata.get("repr")
            value_str = custom_repr(value) if custom_repr else repr(value)
            if saw_omitted:
                arguments.append(f"{field.name}={value_str}")
            else:
                arguments.append(value_str)
        arguments_str = ", ".join(arguments)
        return f"{cls_name}({arguments_str})"


class EnumReprMixin:
    """
    Mixin to provide a concise, eval-friendly repr for Enum members.

    Compared to the default `<ClassName.MEMBER_NAME: value>`, this mixin's `__repr__`
    returns `ClassName.MEMBER_NAME`, which can be evaluated in Python (assuming the
    enum class is in scope) to retrieve the same member.
    """

    def __repr__(self) -> str:
        """
        Return a representation string of an Enum member.

        Returns
        -------
        str
            A string in the form `ClassName.MEMBER_NAME`.
        """
        # Equivalently (as of Python 3.12), `str(value)` also produces
        # "ClassName.MEMBER_NAME", but we build it explicitly here for
        # clarity.
        if not isinstance(self, Enum):
            msg = "EnumMixin can only be used with Enum classes."
            raise TypeError(msg)
        return f"{self.__class__.__name__}.{self.name}"
