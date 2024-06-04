from typing import Sequence, Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike


def is_all_np(arrs: Sequence[np.ndarray | jnp.ndarray | ArrayLike]) -> bool:
    return not any(isinstance(arr, jnp.ndarray) for arr in arrs)


def concatenate(arrays: Union[np.ndarray, jnp.ndarray, Sequence[ArrayLike]], axis=None):
    if is_all_np(arrays):
        return np.concatenate(arrays, axis=axis)

    return jnp.concatenate(arrays, axis=axis)
