import jax.numpy as jnp

from osdyn.dyn.dyn_types import THFloat, TState
from pncbf.utils.shape_utils import assert_shape


def trunc_at_max_unsafe(T_xtraj: TState, T_hs: THFloat):
    T, nx = T_xtraj.shape
    T_h = assert_shape(jnp.max(T_hs, axis=1), T)

    argmax_h = jnp.argmax(T_h)

    idx_trunc_from = argmax_h
    should_set_to_trunc = jnp.arange(T) >= idx_trunc_from
    trunc_state = T_xtraj[argmax_h]
    T_xtraj_trunc = jnp.where(should_set_to_trunc[:, None], trunc_state, T_xtraj)

    # Only truncate if we hit an unsafe state. If all states are safe, then don't truncate.
    is_safe = jnp.max(T_h) < 0
    T_xtraj_trunc = jnp.where(is_safe, T_xtraj, T_xtraj_trunc)

    return T_xtraj_trunc
