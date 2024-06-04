from typing import TypeVar

import jax.numpy as jnp
import numpy as np

from pncbf.utils.jax_types import Shape, as_shape
from pncbf.utils.none import get_or

_Arr = TypeVar("_Arr", np.ndarray, jnp.ndarray, bool)


def assert_shape(arr: _Arr, shape: int | Shape, label: str | None = None) -> _Arr:
    shape = as_shape(shape)
    label = get_or(label, "array")
    if arr.shape != shape:
        raise AssertionError(f"Expected {label} of shape {shape}, but got shape {arr.shape} of type {type(arr)}!")
    return arr


def assert_scalar(arr: _Arr, label: str | None = None) -> _Arr:
    label = get_or(label, "scalar")
    is_scalar = isinstance(arr, float) or arr.shape == tuple()
    if not is_scalar:
        raise AssertionError(f"Expected {label} but got shape {arr.shape} of type {type(arr)}!")
    return arr
