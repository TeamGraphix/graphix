"""Utilities."""

from __future__ import annotations

import inspect
import sys
import typing
from typing import TYPE_CHECKING, Any, ClassVar, Literal, SupportsInt, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt

# override introduced in Python 3.12

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_T = TypeVar("_T")


def check_list_elements(l: Iterable[_T], ty: type[_T]) -> None:
    """Check that every element of the list has the given type."""
    for index, item in enumerate(l):
        if not isinstance(item, ty):
            raise TypeError(f"data[{index}] has type {type(item)} whereas {ty} is expected")


def check_kind(cls: type, scope: dict[str, Any]) -> None:
    """Check that the class has a kind attribute."""
    if not hasattr(cls, "kind"):
        msg = f"{cls.__name__} must have a tag attribute named kind."
        raise TypeError(msg)

    module = sys.modules.get(cls.__module__)
    # Patch in a self-reference: while __init_subclass__ runs, `cls` has not
    # yet been bound to its name in the module namespace, so any field whose
    # annotation refers back to the class being defined (e.g. CONDINSTR)
    # would otherwise fail to resolve.
    globalns = {**getattr(module, "__dict__", {}), cls.__name__: cls}

    # Type annotation to work around a regression in mypy 1.17, see https://github.com/python/mypy/issues/19458
    ann: Any | None = inspect.get_annotations(cls, eval_str=True, locals=scope, globals=globalns).get("kind")
    if ann is None:
        msg = "kind must be annotated."
        raise TypeError(msg)
    if typing.get_origin(ann) is not ClassVar:
        msg = "Tag attribute must be a class variable."
        raise TypeError(msg)
    (ann,) = typing.get_args(ann)
    if typing.get_origin(ann) is not Literal:
        msg = "Tag attribute must be a literal."
        raise TypeError(msg)


def is_integer(value: SupportsInt) -> bool:
    """Return `True` if `value` is an integer, `False` otherwise."""
    return value == int(value)


G = TypeVar("G", bound=np.generic)


@typing.overload
def lock(data: npt.NDArray[Any]) -> npt.NDArray[np.complex128]: ...


@typing.overload
def lock(data: npt.NDArray[Any], dtype: type[G]) -> npt.NDArray[G]: ...


def lock(data: npt.NDArray[Any], dtype: type = np.complex128) -> npt.NDArray[Any]:
    """Create a true immutable view.

    data must not have aliasing references, otherwise users can still turn on writeable flag of m.
    """
    m: npt.NDArray[Any] = data.astype(dtype)
    m.flags.writeable = False
    v = m.view()
    assert not v.flags.writeable
    return v


def iter_empty(it: Iterator[_T]) -> bool:
    """Check if an iterable is empty.

    Notes
    -----
    This function consumes the iterator.
    """
    return all(False for _ in it)


def check_probability(value: float) -> None:
    """Raise :class:`ValueError` if ``value`` is not a probability (between 0 and 1)."""
    if not 0 <= value <= 1:
        raise ValueError(f"Expected {value!r} to be a probability, between 0 and 1")


Qubits: TypeAlias = int | tuple["Qubits", ...]


def flat_qubits(qubits: Qubits) -> tuple[int, ...]:
    """Return a flat tuple of ints from nested qubits."""
    if isinstance(qubits, int):
        return (qubits,)
    return tuple(qubit for item in qubits for qubit in flat_qubits(item))


QubitTemplate = TypeVar("QubitTemplate", bound=tuple[int, ...])


def extract_qubits(qubits: Qubits, template: QubitTemplate) -> QubitTemplate:
    """Return a flat tuple of ints from nested qubits and check its length.

    Notes
    -----
    The expected length is specified by a tuple, since there is no
    direct way to express a tuple length dependent on an integer
    argument.  The type variable ``QubitTemplate`` occurs both as a
    parameter and as a result, ensuring that the result has the same
    type as the given ``template``. The ``template`` parameter can be
    of any tuple type containing ints. In particular, fixed-length
    tuple types allow the result to be deconstructed with the
    corresponding number of components.

    The following line is well-typed:

    .. code-block:: python

        _a, _b = extract_qubits((0, 1), (0, 0))

    but the following line is not:

    .. code-block:: python

        _a, _b, _c = extract_qubits((0, 1), (0, 0))

    The function is only intended to be called with tuple literals for
    the ``template`` argument.  It is unsafe to use it with arbitrary
    values. For example, the following assignments are well-typed but
    result in ``_x`` being assigned ``(1,)`` while having the type
    ``tuple[Literal[0]]``.

    .. code-block:: python

        s: tuple[Literal[0]] = (0,)
        _x: tuple[Literal[0]] = extract_qubits((1,), s)

    Parameters
    ----------
    qubits: Qubits
        Qubits, given as arbitrary nested tuples of ints
    template: QubitTemplate
        Int-tuple of the expected length.
    """
    flat = flat_qubits(qubits)
    if len(flat) != len(template):
        msg = f"{len(template)} qubits expected but {len(flat)} provided."
        raise ValueError(msg)
    return cast("QubitTemplate", flat)
