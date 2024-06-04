import functools as ft
from typing import Callable, Concatenate, ParamSpec, TypeVar, overload

_P1 = ParamSpec("_P1")
_P2 = ParamSpec("_P2")
_P3 = ParamSpec("_P3")
_P4 = ParamSpec("_P4")
_T = TypeVar("_T")
_R_co = TypeVar("_R_co", covariant=True)

_Fn2 = Callable[Concatenate[_P1, _P2], _R_co]
_Fn3 = Callable[Concatenate[_P1, _P2, _P3], _R_co]
_Fn4 = Callable[Concatenate[_P1, _P2, _P3, _P4], _R_co]


@overload
def partial(fn: _Fn2, arg0: _P1) -> Callable[_P2, _R_co]:
    ...


@overload
def partial(fn: _Fn3, arg0: _P1, arg1: _P2) -> Callable[_P3, _R_co]:
    ...


@overload
def partial(fn: _Fn4, arg0: _P1, arg1: _P2, arg2: _P3) -> Callable[_P4, _R_co]:
    ...


def partial(fn: Callable[_P1, _R_co], *args, **kwargs) -> Callable[..., _R_co]:
    return ft.partial(fn, *args, **kwargs)
