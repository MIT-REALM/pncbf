import functools as ft
from typing import NamedTuple

import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from pncbf.dyn.dyn_types import HFloat, THFloat
from pncbf.utils.jax_types import FloatScalar, TFloat
from pncbf.utils.jax_utils import jax_vmap


def cum_max_h(Th_h: TFloat) -> THFloat:
    TVh_h = jax_vmap(cum_max, in_axes=1, out_axes=1)(Th_h)
    assert TVh_h.shape == Th_h.shape
    return TVh_h


def cum_max(T_h: TFloat) -> TFloat:
    def body(carry, h):
        carry_new = jnp.maximum(carry, h)
        return carry_new, carry_new

    _, T_out = lax.scan(body, -jnp.inf, T_h[::-1], length=len(T_h))
    return T_out[::-1]


def compute_disc_avoid_h(lambd: FloatScalar, dt: FloatScalar, Th_h: THFloat) -> THFloat:
    return jax_vmap(ft.partial(compute_disc_avoid, lambd, dt), in_axes=1)(Th_h)


def compute_disc_avoid(lambd: FloatScalar, dt: FloatScalar, T_h: TFloat) -> TFloat:
    """Compute the discrete (discounted) avoid cost.

    PDE:
        max_s   (λ int_0^s  exp(-λ τ) h(τ) dτ) + exp(-λ s) h(s)

    We discretize the integral using the left endpoint of h and the analytical solution of the remaining part.
    If we take
                    γ = exp(-λ Δt)
    then,
        λ int_0^(Δt) exp(-λ τ) h(τ) dτ  ≈  [1 - γ] h(0)

        λ int_0^{k Δt) exp(-λ τ) h(τ) dτ  ≈  [1 - γ] sum_{i=0}^{k-1} γ^i h(i)

    Hence, the total discretized expression is

        max_k  ( [1 - γ] sum_{i=0}^{k-1} γ^i h(i) ) + γ^k h(k)

    This discretization has the property that, for constant h, the discretized expression is equal to the analytical.
    i.e.,
        max_k ... = h
    """
    T = len(T_h)

    # 1: Compute the discretized integral.
    gamma = jnp.exp(-lambd * dt)
    T_disc = gamma ** jnp.arange(T)
    T_h_disc = T_disc * T_h
    T_h_disc = T_h_disc.at[0].set(0)
    T_int = jnp.cumsum(T_h_disc)

    T_terms = (1 - gamma) * T_int + T_disc * T_h
    return jnp.max(T_terms)


class DiscAvoidTerms(NamedTuple):
    """Terms needed to compute V(x_0) = max{ h_max_lhs, h_disc_int_rhs + discount * V(x_t) }."""

    discount: FloatScalar
    Th_disc_int_lhs: THFloat
    h_max_lhs: HFloat
    h_disc_int_rhs: HFloat


class AllDiscAvoidTerms(NamedTuple):
    """Terms needed to compute V(x_t) = max{ h_max_lhs, h_disc_int_rhs + discount * V(x_T) } for all t."""

    Th_max_lhs: THFloat
    Th_disc_int_rhs: THFloat
    T_discount_rhs: TFloat


def compute_all_disc_avoid_terms(lambd: FloatScalar, dt: FloatScalar, Th_h: THFloat) -> AllDiscAvoidTerms:
    """Compute all the terms we need (except for the value function part) to compute the
    discrete (discounted) avoid cost.

    From the DPP, we have
        V(x_0) = max{ max_{s∈[0, t)}  [ λ int_0^Δ exp(-λ τ) h(τ) dτ + exp(-λ s) h(x_s) ],
                                                    λ int_0^t exp(-λ τ) h(τ) dτ + exp(-λ t) V(x_t) }

    We want to compute the left branch of the max and the integral in the right branch, so that we can evaluate
    V(x_0) easily for any V(x_t) when V gets updated in the update.
    """
    T, nh = Th_h.shape
    gamma = jnp.exp(-lambd * dt)
    one_minus_gamma = -jnp.expm1(-lambd * dt)

    # 1: Compute the discretized integral. Do a linear approximation, then compute the integral exactly.
    #       λ int_0^Δ exp(-λ τ) h(τ) dτ  ≈  λ int_0^Δ exp(-λ τ) [ (1 - τ/Δ) h(0) +  τ/Δ h(Δ) ] dτ
    #                                    =  (1 - (1-γ)/(λΔ) ) h0 + ((1-γ)/(λΔ) - γ) h1
    Tm1h_hl = Th_h[:-1, :]
    Tm1h_hr = Th_h[1:, :]
    #       If lam=0, then the integral is just 0. However, we are dividing by lambd, so we need to handle the nan.
    safe_lambd_dt = jnp.where(lambd == 0, 1.0, lambd * dt)
    coeff = one_minus_gamma / safe_lambd_dt
    Tm1h_int_terms = (1 - coeff) * Tm1h_hl + (coeff - gamma) * Tm1h_hr
    Th_disc_int_terms = jnp.concatenate([jnp.zeros((1, nh)), Tm1h_int_terms], axis=0)
    assert Th_disc_int_terms.shape == (T, nh)
    Th_disc_int_terms = jnp.where(lambd == 0, 0.0, Th_disc_int_terms)

    # Tm1h_disc_int = jnp.cumsum(T_disc[:-1, None] * Tm1h_int_terms, axis=0)
    # assert Tm1h_disc_int.shape == Tm1h_int_terms.shape == (T - 1, nh)
    # Th_disc_int = jnp.concatenate([jnp.zeros((1, nh)), Tm1h_disc_int], axis=0)

    # 2: Compute the left branch of the max for all states. We do this via a scan, along the time.
    def body(carry, inp):
        Th_int, T_gam_coeff, Th_V, t = carry
        h_disc_int, h_h = inp

        # 1: Compute mask so that V(x_t) doesn't use earlier information.
        T_should_max = jnp.arange(T) <= t

        # 2: Compute the new lhs.
        # 2.1:  prev int + gamma^k * new_int
        #       T_gam_coeff:  [ 0 0 0 ...]  ->  [ 1 0 0 ... ] -> [ γ 1 0 ... ]
        #                  0:   0 * int_0=0    +    1 * int_1     +    γ * int_2
        #                  1:   0 * int_0=0    +    0 * int_1     +    1 * int_2
        Th_int_term_new = Th_int + T_gam_coeff[:, None] * h_disc_int
        assert Th_int_term_new.shape == (T, nh)

        # 2.2:  Update T_gam_coeff.
        T_gam_coeff_new = gamma * T_gam_coeff
        T_gam_coeff_new = T_gam_coeff_new.at[t].set(1.0)

        # 2.3:  Compute the full lhs term.
        Th_lhs_term = Th_int_term_new + T_gam_coeff_new[:, None] * h_h

        # jd.print("int_new: {}, gam_new: {}, h: {}", Th_int_term_new.flatten(), T_gam_coeff_new, h_h, ordered=True)

        # 3: Take the max of Th_lhs_term and the existing Th_V.
        Th_V_new = jnp.maximum(Th_V, Th_lhs_term)

        # 4: If we shouldn't be updating Th_V, then make it BIG_NEG_NUM.
        Th_V_new = jnp.where(T_should_max[:, None], Th_V_new, BIG_NEG_NUM)

        carry_new = (Th_int_term_new, T_gam_coeff_new, Th_V_new, t + 1)
        return carry_new, None

    # Tm1h_disc_int = jnp.cumsum(T_disc[:-1, None] * Tm1h_int_terms, axis=0)
    # jd.print("Tm1h_disc_int: {}", Tm1h_disc_int.flatten(), ordered=True)

    BIG_NEG_NUM = -1e8
    Th_int_init = jnp.zeros((T, nh))
    T_coeff_init = np.zeros(T)
    Th_V_init = jnp.full((T, nh), BIG_NEG_NUM)
    t_init = 0
    inits = (Th_int_init, T_coeff_init, Th_V_init, t_init)
    carry, _ = lax.scan(body, inits, (Th_disc_int_terms, Th_h), length=T)
    Th_disc_int, _, Th_V, _ = carry

    # Coeff that V(x_T) should be multiplied by.
    # [ γ^{T-1}, γ^{T-2}, ..., γ^0 ]
    T_gammas = gamma ** jnp.arange(T - 1, -1, -1)

    return AllDiscAvoidTerms(Th_V, Th_disc_int, T_gammas)
