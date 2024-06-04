import jax.debug as jd
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from diffrax import (
    Bosh3,
    ConstantStepSize,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from jax.experimental import io_callback
from rich.progress import Progress, TimeElapsedColumn

from pncbf.dyn.dyn_types import State
from pncbf.dyn.task import Task
from pncbf.utils.jax_utils import merge01
from pncbf.utils.none import get_or


class SimCtsPbar:
    def __init__(
        self,
        task: Task,
        policy,
        T: int,
        result_dt: float,
        dt0: float = None,
        use_obs: bool = False,
        max_steps: int = 256,
        use_pid: bool = True,
        solver: str = "tsit5",
        n_updates: int = 5,
    ):
        self.task = task
        self.policy = policy
        self.result_dt = result_dt
        self.T = T
        self.use_obs = use_obs
        self.max_steps = max_steps
        self.use_pid = use_pid
        self.dt0 = get_or(dt0, result_dt)
        self.solver = solver
        self.n_updates = n_updates

        assert T % n_updates == 0, "T must be divisible by n_updates"

        self.pbar: Progress | None = None
        self.pbar_task = None

    @property
    def use_pbar(self):
        return self.n_updates > 1

    def start_pbar(self):
        self.pbar = Progress(*Progress.get_default_columns(), TimeElapsedColumn())
        self.pbar.start()
        self.pbar_task = self.pbar.add_task("Integrating...", total=self.n_updates)

    def update_pbar(self, ii: int):
        self.pbar.update(self.pbar_task, advance=1)
        # print("ii: {}".format(ii))

    def finish_pbar(self):
        self.pbar.stop()
        # print("finished!")

    def get_control(self, state):
        if self.use_obs:
            Vobs, polobs = self.task.get_obs(state)
            return self.policy(polobs)

        return self.policy(state)

    def get_solver_args(self):
        def body(t, state, args):
            control = self.get_control(state)
            return self.task.xdot(state, control)

        term = ODETerm(body)

        if self.solver == "bosh3":
            solver = Bosh3()
        else:
            solver = Tsit5()

        adjoint = RecursiveCheckpointAdjoint()
        if self.use_pid:
            stepsize_controller = PIDController(pcoeff=0.1, icoeff=0.4, rtol=1e-5, atol=1e-5)
        else:
            stepsize_controller = ConstantStepSize()

        return dict(
            terms=term,
            solver=solver,
            adjoint=adjoint,
            stepsize_controller=stepsize_controller,
            max_steps=int(np.ceil(self.max_steps / self.n_updates)) + 2,
            dt0=self.dt0,
        )

    def rollout_plot(self, x0: State):
        solver_args = self.get_solver_args()

        # Fire off the beginning.
        if self.use_pbar:
            io_callback(self.start_pbar, None, ordered=True)

        tf = self.T * self.result_dt
        dt_per_update = self.T // self.n_updates
        update_dt = self.result_dt * dt_per_update

        def body(carry, ii):
            t0_step = ii * update_dt
            tf_step = (ii + 1) * update_dt
            ts = t0_step + np.arange(dt_per_update + 1) * self.result_dt
            # In case there's any rounding issues.
            ts = jnp.clip(ts, t0_step, tf_step)
            saveat = SaveAt(ts=ts)
            solution = diffeqsolve(
                t0=t0_step,
                t1=tf_step,
                y0=carry,
                saveat=saveat,
                **solver_args,
            )
            xs_out = solution.ys[1:]
            ts_out = solution.ts[1:]
            x_last = solution.ys[-1]

            if self.use_pbar:
                io_callback(self.update_pbar, None, ii + 1, ordered=True)
            return x_last, (ts_out, xs_out)

        x_final, (bT_ts, bT_xs) = lax.scan(body, x0, jnp.arange(self.n_updates), length=self.n_updates)
        T_ts, T_xs = merge01(bT_ts), merge01(bT_xs)

        t0 = jnp.array(0.0)
        T_xs = jnp.concatenate([x0[None], T_xs], axis=0)
        T_ts = jnp.concatenate([t0[None], T_ts], axis=0)

        assert T_xs.shape[0] == T_ts.shape[0] == 1 + self.T

        if self.use_pbar:
            io_callback(self.finish_pbar, None, ordered=True)

        return T_xs, T_ts
