import functools as ft

import einops as ei
import jax.numpy as jnp
import numpy as np
from jaxproxqp.jaxproxqp import JaxProxQP, QPSolution

from pncbf.dyn.dyn_types import Control, HFloat, HState, State
from pncbf.solvers.qp import get_relaxed_constr_Gh, jaxopt_osqp
from pncbf.utils.jax_types import FloatScalar
from pncbf.utils.jax_utils import jax_vmap


def min_norm_cbf_qp_mats(
    alpha: FloatScalar,
    u_lb: Control,
    u_ub: Control,
    h_V: HFloat,
    hx_Vx: HState,
    f: State,
    G,
    u_nom: Control,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 5.0,
) -> JaxProxQP.QPModel:
    nx, nu = G.shape
    dtype = h_V.dtype
    assert u_lb.shape == u_ub.shape == (nu,)

    if h_V.ndim == 0:
        h_V = h_V[None]

        assert hx_Vx.shape == (nx,)
        hx_Vx = hx_Vx[None]

    H = np.eye(nu + 1, dtype=dtype)
    # 0.5 * penalty * ( r + relax_eps2 )^2
    # = 0.5 * penalty * ( r^2 + 2 r relax_eps2 + relax_eps2^2 )
    H[-1, -1] = penalty
    # We now have a nominal control.
    g = jnp.concatenate([-u_nom, jnp.array([penalty * relax_eps2], dtype=dtype)], axis=0)
    assert g.shape == (nu + 1,)

    # Get G and h for each CBF constraint.
    if isinstance(alpha, float):
        alpha = jnp.array(alpha)

    if alpha.ndim == 0:
        h_G, h_h = jax_vmap(ft.partial(get_relaxed_constr_Gh, f, G, alpha))(h_V, hx_Vx)
    else:
        nh = len(h_V)
        assert alpha.shape == (nh,)
        h_alpha = alpha
        h_G, h_h = jax_vmap(ft.partial(get_relaxed_constr_Gh, f, G))(h_alpha, h_V, hx_Vx)

    # -eps1 <= r <= infty
    r_lb = jnp.array(-relax_eps1, dtype=dtype)
    r_ub = jnp.array(1e9, dtype=dtype)

    l_box = jnp.concatenate([u_lb, r_lb[None]], axis=0)
    u_box = jnp.concatenate([u_ub, r_ub[None]], axis=0)

    # C = jnp.stack([*h_G], axis=0)
    # u = jnp.array([*h_h])
    C = h_G
    u = h_h

    return JaxProxQP.QPModel.create(H, g, C, u, l_box, u_box)


def min_norm_cbf(
    alpha: FloatScalar,
    u_lb: Control,
    u_ub: Control,
    h_V: HFloat,
    hx_Vx: HState,
    f: State,
    G,
    u_nom: Control,
    penalty: float = 10.0,
    relax_eps1: float = 5e-1,
    relax_eps2: float = 20.0,
    settings: JaxProxQP.Settings = None,
) -> tuple[Control, FloatScalar, QPSolution]:
    nx, nu = G.shape

    qp = min_norm_cbf_qp_mats(alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom, penalty, relax_eps1, relax_eps2)
    if settings is None:
        settings = JaxProxQP.Settings.default()
    solver = JaxProxQP(qp, settings)
    sol = solver.solve()

    assert sol.x.shape == (nu + 1,)
    u_opt, r = sol.x[:nu], sol.x[-1]
    return u_opt, r, sol


def lb_bangbang(u_lb: Control, u_ub: Control, G, Vx):
    # Minimize V.
    LG_V = ei.einsum(Vx, G, "nx, nx nu -> nu")
    # To minimize, if LG_V is negative, we want u to be as large as possible.
    u_opt = jnp.where(LG_V < 0, u_ub, u_lb)
    return u_opt
