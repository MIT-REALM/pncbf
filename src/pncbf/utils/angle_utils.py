import numpy as np
from jax import numpy as jnp

from pncbf.utils.jax_types import AnyFloat, FloatScalar, FloatScalar, RotMat2D, RotMat3D, Vec2, Vec3


def sincos(theta):
    return jnp.sin(theta), jnp.cos(theta)


def wrap_to_pi(x: AnyFloat) -> AnyFloat:
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def rotz(psi: FloatScalar) -> RotMat3D:
    c, s = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def roty(theta: FloatScalar) -> RotMat3D:
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotx(phi: FloatScalar) -> RotMat3D:
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def get_unitvec(theta: FloatScalar) -> Vec2:
    if isinstance(theta, jnp.ndarray):
        return jnp.array([jnp.cos(theta), jnp.sin(theta)])
    else:
        return np.array([np.cos(theta), np.sin(theta)])


def unitvec_to_rot(vec: Vec2) -> RotMat2D:
    c, s = vec
    return jnp.array([[c, -s], [s, c]])


def rot2d(theta: FloatScalar) -> RotMat2D:
    if isinstance(theta, jnp.ndarray):
        c, s = jnp.cos(theta), jnp.sin(theta)
        return jnp.array([[c, -s], [s, c]])
    else:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])


def rpy_ned_to_enu(ned_rpy: Vec3) -> Vec3:
    """Convert from RPY in NED=(front-right-down) frame to ENU=(right-front-up) frame.
    In NED,
        1: positive yaw is body-frame CW. Zero degrees is NORTH.
        2: positive pitch is body-frame pitch UP
        3: positive roll is body-frame RIGHT

    In ENU,
        1: positive yaw is body-frame CCW. Zero degrees is EAST.
        2: positive pitch is body-frame pitch DOWN
        3: positive roll is body-frame RIGHT
    """
    assert ned_rpy.shape == (3,)
    ned_r, ned_p, ned_y = ned_rpy

    enu_y = np.pi / 2 - ned_y
    enu_p = -ned_p
    enu_r = ned_r

    return jnp.array([enu_r, enu_p, enu_y])
