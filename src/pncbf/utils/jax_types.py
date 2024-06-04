from typing import NamedTuple, Union

import numpy as np
from jaxtyping import Array, Bool, Float, Int, Shaped

Arr = Union[np.ndarray, Array]

AnyShaped = Shaped[Arr, "*"]
AnyFloat = Float[Arr, "*"]
Shape = tuple[int, ...]

Vec2 = Float[Arr, "2"]
Vec3 = Float[Arr, "3"]

BVec2 = Float[Arr, "b 2"]
BVec3 = Float[Arr, "b 3"]

BBVec3 = Float[Arr, "b1 b2 3"]

FloatScalar = float | Float[Arr, ""]
IntScalar = int | Int[Arr, ""]
BoolScalar = bool | Bool[Arr, ""]

BFloat = Float[Arr, "b"]
BInt = Int[Arr, "b"]
BBool = Bool[Arr, "b"]

BBFloat = Float[Arr, "b1 b2"]
BBBool = Bool[Arr, "b1 b2"]
BBInt = Int[Arr, "b1 b2"]

TFloat = Float[Arr, "T"]
Tp1Float = Float[Arr, "Tp1"]

TBool = Bool[Arr, "T"]

BTFloat = Float[Arr, "b T"]
BTInt = Int[Arr, "b T"]
BTBool = Bool[Arr, "b T"]
BHBool = Bool[Arr, "b h"]
BTHBool = Bool[Arr, "b T h"]

BTNFloat = Float[Arr, "b T N"]

RotMat3D = Float[Arr, "3 3"]
RotMat2D = Float[Arr, "2 2"]

_CurrentFloat = np.float32

MetricsDict = dict[str, Union[FloatScalar, "MetricsDict"]]


def set_current_float(dtype):
    global _CurrentFloat
    _CurrentFloat = dtype


def current_float():
    return _CurrentFloat


# Special value to mark that the shape can be an arbitrary batch size.
ANY_DIM = -69


class PytreeShape(NamedTuple):
    """Special marker to denote that it's not an array but a pytree."""

    pass


def as_shape(shape: int | Shape | PytreeShape) -> Shape | PytreeShape:
    if isinstance(shape, PytreeShape):
        return shape
    if isinstance(shape, int):
        shape = (shape,)
    if not isinstance(shape, tuple):
        raise ValueError(f"Expected shape {shape} to be a tuple!")
    return shape
