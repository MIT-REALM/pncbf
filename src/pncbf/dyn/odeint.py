import numpy as np
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

from pncbf.dyn.dyn_types import State
from pncbf.utils.jax_types import FloatScalar
from pncbf.utils.shape_utils import assert_shape


def rk4(dt: FloatScalar, xdot, x1: State) -> State:
    k1 = xdot(x1)

    x2 = x1 + k1 * dt * 0.5
    k2 = xdot(x2)

    x3 = x1 + k2 * dt * 0.5
    k3 = xdot(x3)

    x4 = x1 + k3 * dt
    k4 = xdot(x4)

    return x1 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def tsit5(dt: FloatScalar, n_knots: int, xdot, x1: State) -> State:
    def wrapper(t, x, args):
        return xdot(x)

    term = ODETerm(wrapper)
    solver = Tsit5()
    saveat = SaveAt(ts=np.linspace(0, dt, num=n_knots + 1))
    solution = diffeqsolve(term, solver, t0=0, t1=dt, dt0=dt / n_knots, y0=x1, saveat=saveat)
    sol = solution.ys
    sol_shape = (n_knots + 1,) + x1.shape
    return assert_shape(sol, sol_shape)


def tsit5_dense(dt: FloatScalar, dt0_knots: int, xdot, x1: State):
    def wrapper(t, x, args):
        return xdot(x)

    term = ODETerm(wrapper)
    solver = Tsit5()
    saveat = SaveAt(dense=True)
    solution = diffeqsolve(term, solver, t0=0, t1=dt, dt0=dt / dt0_knots, y0=x1, saveat=saveat)
    interp = solution.interpolation
    assert interp is not None

    return interp


def tsit5_dense_pid(tf: FloatScalar, dt0: float, xdot, x1: State, max_steps: int = 1024):
    def wrapper(t, x, args):
        return xdot(x)

    term = ODETerm(wrapper)
    solver = Tsit5()
    saveat = SaveAt(dense=True)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    solution = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=tf,
        dt0=dt0,
        y0=x1,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    interp = solution.interpolation
    assert interp is not None

    return interp
