from typing import Any, TypeVar

_T = TypeVar("_T")
_U = TypeVar("_U")


def get_or(maybe: _T | None, value: _U) -> _T | _U:
    return value if maybe is None else maybe
