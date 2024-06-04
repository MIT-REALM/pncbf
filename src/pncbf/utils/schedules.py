import math

import jax.numpy as jnp
import numpy as np
import optax
from attrs import asdict, define

from pncbf.utils.jax_types import FloatScalar, IntScalar


@define
class Schedule:
    def as_dict(self):
        return {"type": type(self).__name__, **asdict(self)}

    @property
    def total_steps(self) -> int:
        return 0

    def make(self) -> optax.Schedule:
        ...


def as_schedule(val: Schedule | float | int) -> Schedule:
    if isinstance(val, Schedule):
        return val

    return Constant(val)


@define
class Constant(Schedule):
    value: float

    def make(self) -> optax.Schedule:
        return optax.constant_schedule(self.value)


@define
class LinWarmup(Schedule):
    schedule: Schedule
    warmup_div: float
    warmup_steps: int

    def make(self) -> optax.Schedule:
        base_schedule = self.schedule.make()

        def schedule(count: IntScalar) -> FloatScalar:
            frac = jnp.clip(count / self.warmup_steps, a_max=1.0)
            base_val = base_schedule(count)
            coeff = 1 / self.warmup_div
            multiplier = (1 - coeff) * frac + coeff
            return multiplier * base_val

        return schedule


@define
class ExpDecay(Schedule):
    init_value: float
    trans_steps: int
    decay_rate: float
    trans_begin: int
    staircase: bool = False
    end_value: float = 1e-6

    def make(self) -> optax.Schedule:
        return optax.exponential_decay(
            self.init_value, self.trans_steps, self.decay_rate, self.trans_begin, self.staircase, self.end_value
        )


@define
class LinDecay(Schedule):
    init: float
    decay_ratio: float
    warmup_steps: int
    trans_steps: int

    def make(self):
        return linear_with_warmup(self.init, self.init / self.decay_ratio, self.warmup_steps, self.trans_steps)


def linear_with_warmup(init_value: float, end_value: float, warmup_steps: int, transition_steps: int) -> optax.Schedule:
    warmup_sched = optax.constant_schedule(init_value)
    linear_sched = optax.linear_schedule(init_value, end_value, transition_steps)

    return optax.join_schedules([warmup_sched, linear_sched], [warmup_steps])


@define
class Lin(Schedule):
    init: float
    end: float
    steps: int
    warmup: int = 0

    @property
    def total_steps(self):
        return self.steps + self.warmup

    def make(self):
        return optax.linear_schedule(self.init, self.end, self.steps, self.warmup)

    def __str__(self):
        return "Lin({} -> {}, s{}, w{})".format(self.init, self.end, self.steps, self.warmup)


@define
class MultScheds(Schedule):
    schedules: list[Schedule]

    def make(self):
        schedules = [sched.make() for sched in self.schedules]

        def multscheds_schedule(count: IntScalar) -> FloatScalar:
            return jnp.array([sched(count) for sched in schedules]).prod()

        return multscheds_schedule


@define
class SchedAtK(Schedule):
    base_sched: Schedule
    k: int

    @property
    def total_steps(self) -> int:
        return self.base_sched.total_steps

    def make(self) -> optax.Schedule:
        return sched_at_k(self.base_sched, self.k)


@define
class SchedCtsHorizon(Schedule):
    """Given a schedule for the effective horizon, return a schedule for the continuous time discount factor.
    We express the effective horizon as

        T_eff = 1 / (1 - gamma)

    where gamma is the discrete time discount factor.
    """

    horizon_sched: Schedule
    dt: float

    @property
    def total_steps(self) -> int:
        return self.horizon_sched.total_steps

    def make(self) -> optax.Schedule:
        return sched_cts_horizon(self.horizon_sched, self.dt)

    def __str__(self) -> str:
        return "SchedCts({}, dt={})".format(self.horizon_sched, self.dt)


@define
class SchedEffHorizon(Schedule):
    """Given a schedule for the horizon, return a schedule for the discrete time discount factor.
    We express the effective horizon as


        T_eff = 1 / (1 - gamma)

    where gamma is the discrete time discount factor.
    """

    base_sched: Schedule

    def make(self) -> optax.Schedule:
        return sched_eff_horizon(self.base_sched)


@define
class ExpAnneal(Schedule):
    # Initial discount at timestep k.
    init: float
    mid: float
    mid_at: int
    final: float

    def make(self):
        lambd = math.log((self.final - self.init) / (self.final - self.mid)) / self.mid_at
        return exp_anneal(self.init, self.final, lambd)


def exp_anneal(init: float, final: float, lambd: float):
    a = final - init
    c = init

    def schedule(count: IntScalar) -> FloatScalar:
        count = jnp.clip(count, a_min=0)
        return a * (1 - jnp.exp(-lambd * count)) + c

    return schedule


def sched_at_k(base_sched: Schedule, k: int):
    base_sched = base_sched.make()

    if not (k > 0):
        raise ValueError(f"k {k} should be >0!")

    def schedule(count: IntScalar) -> FloatScalar:
        base_val = jnp.clip(base_sched(count), a_min=0.0)
        return base_val ** (1 / k)

    return schedule


def horizon_to_lam(horizon: float, dt: float):
    return -jnp.log(1 - 1 / horizon) / dt


def lam_to_horizon(lam: float, dt: float):
    return 1 / (1 - jnp.exp(-lam * dt))


def sched_cts_horizon(horizon_sched: Schedule, dt: float):
    horizon_sched = horizon_sched.make()

    def schedule(count: IntScalar) -> FloatScalar:
        horizon = jnp.clip(horizon_sched(count), a_min=1.5)
        return -jnp.log(1 - 1 / horizon) / dt

    return schedule


def sched_eff_horizon(horizon_sched: Schedule):
    horizon_sched = horizon_sched.make()

    def schedule(count: IntScalar) -> FloatScalar:
        horizon = jnp.clip(horizon_sched(count), a_min=1.5)
        return 1 - 1 / horizon

    return schedule


@define
class PiecewiseLinear(Schedule):
    xs: np.ndarray | list[float]
    ys: np.ndarray | list[float]

    def make(self):
        xs, ys = np.array(self.xs), np.array(self.ys)
        assert xs.shape == ys.shape and xs.ndim == 1
        return piecewise_linear(xs, ys)


def piecewise_linear(xs: np.ndarray, ys: np.ndarray):
    xmin, xmax = xs.min(), xs.max()
    norm_xs = (xs - xmin) / (xmax - xmin)

    def schedule(count: IntScalar) -> FloatScalar:
        # [0, 1]
        frac = jnp.clip((count - xmin) / (xmax - xmin), 0, 1)
        return jnp.interp(frac, norm_xs, ys)

    return schedule


@define
class JoinSched(Schedule):
    sched1: Schedule
    sched2: Schedule
    sched2_start: int

    @property
    def total_steps(self) -> int:
        return self.sched2_start + self.sched2.total_steps

    def make(self):
        return join_sched(self.sched1, self.sched2, self.sched2_start)

    def __str__(self):
        return "Join({}, {}) @ {}".format(self.sched1, self.sched2, self.sched2_start)


def join_sched(sched1: Schedule, sched2: Schedule, sched2_start: int):
    def schedule(count: IntScalar) -> FloatScalar:
        sched1_count = jnp.minimum(count, sched2_start)
        sched2_count = jnp.maximum(count - sched2_start, 0)
        use_sched2 = count >= sched2_start
        return jnp.where(use_sched2, sched2(sched2_count), sched1(sched1_count))

    sched1, sched2 = sched1.make(), sched2.make()
    return schedule
