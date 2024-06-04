from typing import Callable, NamedTuple
import einops as ei

import jax
import jax.numpy as jnp

from pncbf.dyn.dyn_types import HFloat, State


def hocbf(h_fn: Callable, f_fn, alpha0: float | HFloat, state: State) -> HFloat:
    h_h = h_fn(state)
    hx_hx = jax.jacfwd(h_fn)(state)
    nh, nx = hx_hx.shape
    assert h_h.shape == (nh,)
    f = f_fn(state)
    assert f.shape == (nx,)

    h_hdot = ei.einsum(hx_hx, f, "h nx, nx -> h")
    return h_hdot + alpha0 * h_h


def stack_cbf_fns(B_fns: list[Callable]):
    def h_B_fn(state: State):
        return jnp.stack([B_fn(state) for B_fn in B_fns], axis=0)

    return h_B_fn
