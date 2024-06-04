import math

import jax.nn as jnn
import jax.numpy as jnp

from pncbf.utils.jax_types import AnyFloat, FloatScalar


def scale_dict_scalar(float_dict: dict[str, FloatScalar], coeff: float) -> dict[str, FloatScalar]:
    return {k: coeff * v for k, v in float_dict.items()}


def scale_and_sum_dict(costs: dict[str, FloatScalar], scale: dict[str, FloatScalar] | None = None) -> FloatScalar:
    """Sum together a cost dict, with an optional scale used to DIVIDE costs."""
    total_cost = []
    for k, v in costs.items():
        if scale is None:
            total_cost.append(v)
        else:
            assert k in scale, f"Didn't find cost key {k} in scale dict! Keys: {scale.keys()}"
            total_cost.append(v / scale[k])
    return jnp.array(total_cost).sum()


def clipped_log1p(h: FloatScalar, min_val: float | FloatScalar) -> FloatScalar:
    """
    Transforms h, a function that has bounded negative part to unbounded negative (but with min val) using log.
    """
    assert min_val < 0, "min_val should be negative to preserve semantics of h."
    log1p_min = -1.0 + 1e-4
    return jnp.clip(jnp.log1p(jnp.clip(h, a_min=log1p_min)), a_min=min_val)


def poly2_clip_max(x: AnyFloat, max_val: float = 1.0) -> AnyFloat:
    """Polynomial symmetric smoothed clipping function that preserves sign and clips positive values only.
    for max_val = 1: x <= 1
    f(0)   = 0                          f(1)  = 1.0
    f'(0)  = 2                          f'(1) = 0.0
    f''(0) = f'''(0) = 0
    """
    x = jnp.minimum(x, max_val)
    y = x / max_val
    clip_branch = 2 * y - y**2
    return max_val * jnp.where(x >= 0, clip_branch, 2 * y)


def poly2_clip_max_flat(x, max_val: float = 1.0) -> AnyFloat:
    return poly2_clip_max(x * 1 / 2, max_val)


def poly4_clip_max(x: AnyFloat, max_val: float = 1.0) -> AnyFloat:
    """Polynomial symmetric smoothed clipping function that preserves sign and clips positive values only.
    for max_val = 1: x <= 1
    f(0)   = 0                          f(1)  = 1.0
    f'(0)  = 4/3                        f'(1) = 0.0
    """
    x = jnp.minimum(x, max_val)
    y = x / max_val
    clip_branch = (4 / 3 * y) - (y**4 / 3)
    return max_val * jnp.where(x >= 0, clip_branch, 4 / 3 * y)


def poly4_clip_max_flat(x, max_val: float = 1.0) -> AnyFloat:
    return poly4_clip_max(x * 3 / 4, max_val)


def poly4_softclip(x, m: float = 0.1):
    a1 = (12 - 7 * m) / 6
    a3 = (5 * m - 4) / 2
    a4 = (3 - 4 * m) / 3

    safe_x = jnp.clip(x, a_min=1)

    branch_lin = a1 * x
    branch_poly = a1 * x + a3 * (x**3) + a4 * (x**4)
    branch_log = m * jnp.log(safe_x) + 1.0

    return jnp.where(x <= 0, branch_lin, jnp.where(x <= 1, branch_poly, branch_log))


def poly4_softclip_flat(x, m: float = 0.1):
    a1 = (12 - 7 * m) / 6
    return poly4_softclip(x / a1, m)


def poly6_clip_max(x: AnyFloat, max_val: float = 1.0) -> AnyFloat:
    """Polynomial symmetric smoothed clipping function that preserves sign and clips positive values only.
    for max_val = 1: x <= 1
    f(0)   = 0                          f(1)  = 1.0
    f'(0)  = 3/2                        f'(1) = 0.0
    f''(0) = f'''(0) = f''''(0) = 0
    """
    a0 = 3
    a4 = -3
    a5 = 2
    x = jnp.minimum(x, max_val)
    y = x / max_val
    clip_branch = ((a5 * y + a4) * (y**4) + a0) * x / 2
    return max_val * jnp.where(x >= 0, clip_branch, 1.5 * y)


def poly6_clip_max_flat(x, max_val: float = 1.0) -> AnyFloat:
    return poly6_clip_max(x * 2 / 3, max_val)


def poly4_clip_min(x: AnyFloat, min_val: float) -> AnyFloat:
    return -poly4_clip_max(-x, -min_val)


def softclip_lo(x: AnyFloat, minval: float) -> AnyFloat:
    """Soft clip of negative values. Is strictly monotonic, zero at zero, gradient approaches 1 at infinity."""
    if minval >= 0:
        raise ValueError("minval should be negative!")
    b = -math.log(math.exp(-minval) - 1)
    return jnn.softplus(x - b) + minval


def softclip_grad_lo(x: AnyFloat, minval: float, alpha: float) -> AnyFloat:
    """Clip the gradients of negative values. Strictly monotonic, zero at zero, gradient approaches 1 at infinity."""
    return alpha * x + (1 - alpha) * softclip_lo(x, minval)


def add_constr_margin(h: AnyFloat, margin: FloatScalar) -> AnyFloat:
    """Adds a margin between the unsafe and safe sets in terms of the value of h."""
    return jnp.where(h < 0, h - margin / 2, h + margin / 2)
