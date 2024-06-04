import enum
from typing import Any, NamedTuple

import einops as ei
import ipdb
import jax.numpy as jnp
import jaxopt
from jaxopt._src.base import KKTSolution

from pncbf.dyn.dyn_types import State
from pncbf.utils.jax_types import FloatScalar
from pncbf.utils.shape_utils import assert_scalar, assert_shape


class OSQPStatus(enum.IntEnum):
    UNSOLVED = 0
    SOLVED = 1
    DUAL_INFEASIBLE = 2
    PRIMAL_INFEASIBLE = 3


class BoxOSQPState(NamedTuple):
    iter_num: int
    error: float
    status: OSQPStatus
    primal_residuals: float
    dual_residuals: float
    rho_bar: float


def get_relaxed_constr_Gh(f: State, G, alpha: FloatScalar, V: FloatScalar, Vx: State):
    """"""
    nx, nu = G.shape
    assert_scalar(V)
    assert_shape(Vx, nx)

    Lf_V = jnp.dot(Vx, f)
    LG_V = ei.einsum(Vx, G, "nx, nx nu -> nu")

    # Relaxed constraint: LfV + LGV u + alpha V - r <= 0
    G = jnp.concatenate([LG_V, -jnp.ones(1)], axis=0)
    h = -(Lf_V + alpha * V)

    return G, h


def jaxopt_osqp(solver: jaxopt.OSQP, Q, c, G, h, lb, ub) -> tuple[KKTSolution, BoxOSQPState, Any]:
    nx = len(Q)
    G_ub, G_lb = jnp.eye(nx), -jnp.eye(nx)
    G_ub, G_lb = G_ub[: len(lb)], G_lb[: len(lb)]
    G_new = jnp.concatenate([G, G_ub, G_lb], axis=0)

    h_new = jnp.concatenate([h, ub, -lb], axis=0)

    params, state = solver.run(params_obj=(Q, c), params_ineq=(G_new, h_new))
    state = BoxOSQPState(
        state.iter_num, state.error, state.status, state.primal_residuals, state.dual_residuals, state.rho_bar
    )
    return params, state, (Q, c, G_new, h_new)
