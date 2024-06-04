from typing import Generator, Iterable, TypeVar

_Elem = TypeVar("_Elem")


def signal_last_enumerate(it: Iterable[_Elem]) -> Generator[tuple[bool, int, _Elem], None, None]:
    iterable = iter(it)
    count = 0
    ret_var = next(iterable)
    for val in iterable:
        yield False, count, ret_var
        count += 1
        ret_var = val
    yield True, count, ret_var


def signal_last(it: Iterable[_Elem]) -> Generator[tuple[bool, _Elem], None, None]:
    iterable = iter(it)
    ret_var = next(iterable)
    for val in iterable:
        yield False, ret_var
        ret_var = val
    yield True, ret_var


def signal_last_range(stop: int) -> Generator[tuple[bool, int], None, None]:
    return signal_last(range(stop))
