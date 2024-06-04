import functools as ft

import einops as ei
import jax.numpy as jnp
import jaxopt
import numpy as np
from jaxopt._src import base
from jaxopt._src.osqp import BoxOSQPState, OSQP_to_BoxOSQP

from pncbf.dyn.dyn_types import Control, HFloat, HState, State
from pncbf.solvers.qp import get_relaxed_constr_Gh, jaxopt_osqp
from pncbf.utils.jax_types import FloatScalar
from pncbf.utils.jax_utils import jax_vmap


class MyOSQP(jaxopt.OSQP):
    def run(self, init_params=None, params_obj=None, params_eq=None, params_ineq=None):
        assert params_obj is not None
        if init_params is None:
            init_params = self.init_params(None, params_obj, params_eq, params_ineq)
        init_params, hyper_params, eq_ineq_size = OSQP_to_BoxOSQP.transform(
            self.matvec_A_box, init_params, params_obj, params_eq, params_ineq
        )
        box_osqp_state: BoxOSQPState
        sol, box_osqp_state = self._box_osqp.run(init_params, **hyper_params)
        sol = OSQP_to_BoxOSQP.inverse_transform(self.matvec_A_box, eq_ineq_size, sol)
        return base.OptStep(params=sol, state=box_osqp_state)


def min_norm_cbf_qp_mats2(
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
):
    nx, nu = G.shape

    assert u_lb.shape == u_ub.shape == (nu,)

    if h_V.ndim == 0:
        h_V = h_V[None]

        assert hx_Vx.shape == (nx,)
        hx_Vx = hx_Vx[None]

    Q = np.eye(nu + 1)
    # 0.5 * penalty * ( r + relax_eps )^2
    Q[-1, -1] = penalty
    # We now have a nominal control.
    c = jnp.concatenate([-u_nom, jnp.array([penalty * relax_eps2])], axis=0)
    assert c.shape == (nu + 1,)

    # Get G and h for each CBF constraint.
    h_G, h_h = jax_vmap(ft.partial(get_relaxed_constr_Gh, f, G, alpha))(h_V, hx_Vx)

    # r >= -eps1     <=>      -r <= eps1
    G2 = np.zeros(nu + 1)
    G2[-1] = -1
    h2 = relax_eps1

    G = jnp.stack([*h_G, G2], axis=0)
    h = jnp.array([*h_h, h2])

    return Q, c, G, h, u_lb, u_ub


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
    relax_eps2: float = 5.0,
    maxiter: int = 200,
    tol: float = 5e-4,
):
    nx, nu = G.shape
    solver = MyOSQP(eq_qp_solve="lu", maxiter=maxiter, tol=tol)

    Q, c, G, h, u_lb, u_ub = min_norm_cbf_qp_mats2(
        alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom, penalty, relax_eps1, relax_eps2
    )
    params, state, qp_mats = jaxopt_osqp(solver, Q, c, G, h, u_lb, u_ub)
    u_opt, r = params.primal[:nu], params.primal[-1]
    assert u_opt.shape == (nu,)
    return u_opt, r, (state, qp_mats)
