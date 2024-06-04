import dataclasses as dc

import jax.random as jr
import numpy as np

from pncbf.utils.jax_types import Array, Float, Shape, as_shape

PRNGKey = Float[Array, "2"]
PRNGKeyDtype = np.uint32


def jr_split_shape(key: PRNGKey, shape: int | Shape) -> PRNGKey:
    shape = as_shape(shape)
    total_keys = np.prod(shape)
    keys = jr.split(key, total_keys)
    return keys.reshape(shape + (2,))


def _get_field(fname, ftype):
    field = dc.field()
    field.name = fname
    field.type = ftype
    # noinspection PyProtectedMember
    field._field_type = dc._FIELD
    field.kw_only = False
    return field


class KeyGen:
    key: PRNGKey
    __dataclass_fields__ = {"key": _get_field("key", PRNGKey)}

    def __init__(self, key: PRNGKey | int):
        if isinstance(key, int):
            key = jr.PRNGKey(key)
        self.key = key

    def __call__(self) -> PRNGKey:
        key = self.key
        self.key, key = jr.split(key, 2)
        return key

    def __invert__(self) -> PRNGKey:
        return self.__call__()
