import functools as ft
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jaxproxqp.jaxproxqp import JaxProxQP
from jaxtyping import Float
from loguru import logger

import pncbf.pncbf.min_norm_cbf as cbf_old
import pncbf.utils.smart_np as snp
from pncbf.dyn.dyn_types import BState, Control, Disturb, HFloat, PolObs, State, TState, VObs
from pncbf.dyn.odeint import rk4, tsit5
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.dyn.task import Task
from pncbf.plotting.circ_artist import CircArtist
from pncbf.qp.min_norm_cbf import min_norm_cbf
from pncbf.utils.costconstr_utils import poly4_clip_max_flat, poly4_softclip_flat
from pncbf.utils.hocbf import hocbf, stack_cbf_fns
from pncbf.utils.jax_types import Arr, BBFloat, BoolScalar, TFloat, Vec2
from pncbf.utils.jax_utils import jax_vmap, rep_vmap
from pncbf.utils.none import get_or
from pncbf.utils.paths import get_paper_data_dir
from pncbf.utils.rng import PRNGKey
from pncbf.utils.sampling_utils import get_mesh_np
from pncbf.utils.shape_utils import assert_shape


class QuadCircle(Task):
    NX_QUAD = 4
    NU_QUAD = 2
    NX_OBS = 4
    NX = 2 * NX_QUAD + NX_OBS
    NU = 2 * NU_QUAD

    PX1, PY1, VX1, VY1, PX2, PY2, VX2, VY2, PXO, PYO, VXO, VYO = range(NX)
    VXD1, VYD1, VXD2, VYD2 = range(NU)

    QUAD1 = np.array([PX1, PY1, VX1, VY1])
    POS_QUAD1 = np.array([PX1, PY1])
    VEL_QUAD1 = np.array([VX1, VY1])

    QUAD2 = np.array([PX2, PY2, VX2, VY2])
    POS_QUAD2 = np.array([PX2, PY2])
    VEL_QUAD2 = np.array([VX2, VY2])

    OBS = np.array([PXO, PYO, VXO, VYO])
    POS_OBS = np.array([PXO, PYO])
    VEL_OBS = np.array([VXO, VYO])

    DT = 0.1

    def __init__(self):
        # jjj
        self.quad_max_dist = 1.2

        # The quad body is 0.16m. Adding in two props each of 5 inches=0.127 = 0.254m => ~ 0.42 m diameter.
        # Round up to 0.22 m radius.
        self.quad_r = 0.22
        # Keep one quad's radius away.
        self.quadquad_dmin = 0.22

        self.obs_r = 0.6
        # Keep one quad's radius away.
        self.quadobs_dmin = 0.22

        # For the velocity pid loop.
        # self.k_v = 2.0
        self.k_v = 0.5
        # self.k_v = 0.4

        # For the nominal policy.
        self.nom_r = 2.0
        self.c_r = 0.2
        # v = c_theta * nom_r. If we want to bound v, then we need to make c_theta slower.
        self.c_theta = 0.35

        self.pos_max = 3.95

        self.hand_cbf_alpha = 5.0

        self._dt = QuadCircle.DT

        self.buffer_assert = {"quadquad_mindist": self.quad_r, "quadquad_maxdist": self.quad_r, "quadobs": self.quad_r}

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def n_Vobs(self) -> int:
        # return 12
        # return 18
        return 27

    @property
    def quad_x_labels(self) -> list[str]:
        return [r"$p_x$", r"$p_y$", r"$v_x$", r"$v_y$"]

    @property
    def x_labels(self) -> list[str]:
        quad1_labels = [r"$1p_x$", r"$1p_y$", r"$1v_x$", r"$1v_y$"]
        quad2_labels = [r"$2p_x$", r"$2p_y$", r"$2v_x$", r"$2v_y$"]
        obs_labels = [r"$op_x$", r"$op_y$", r"$ov_x$", r"$ov_y$"]
        return quad1_labels + quad2_labels + obs_labels

    @property
    def u_labels(self) -> list[str]:
        quad1_labels = [r"$1vd_x$", r"$1vd_y$"]
        quad2_labels = [r"$2vd_x$", r"$2vd_y$"]
        return quad1_labels + quad2_labels

    @property
    def h_labels(self) -> list[str]:
        return [
            "quadquad_col",
            "quadquad_maxdist",
            "quad1obs",
            "quad2obs",
            "q1_pos",
            "q2_pos",
        ]

    @property
    def h_max(self) -> float:
        return 1.0

    @property
    def h_min(self) -> float:
        return -1.0

    @property
    def u_min(self) -> Control:
        return np.full(self.nu, -1.0)

    @property
    def u_max(self) -> Control:
        return np.full(self.nu, +1.0)

    def h_components(self, state: State) -> HFloat:
        state = self.chk_x(state)
        p_q1, p_q2, p_o = state[self.POS_QUAD1], state[self.POS_QUAD2], state[self.POS_OBS]

        quadquad_dist = jnp.linalg.norm(p_q1 - p_q2)
        quad1obs_dist = jnp.linalg.norm(p_q1 - p_o)
        quad2obs_dist = jnp.linalg.norm(p_q2 - p_o)

        h_quadquadcol = -(quadquad_dist - 2 * self.quad_r - self.quadquad_dmin)
        h_quadquad_maxdist = quadquad_dist - self.quad_max_dist

        h_quad1obscol = -(quad1obs_dist - self.quad_r - self.quadobs_dmin - self.obs_r)
        h_quad2obscol = -(quad2obs_dist - self.quad_r - self.quadobs_dmin - self.obs_r)

        h_quad1_xu = p_q1[0] - self.pos_max
        h_quad1_xl = -(p_q1[0] + self.pos_max)
        h_quad1_yu = p_q1[1] - self.pos_max
        h_quad1_yl = -(p_q1[1] + self.pos_max)
        h_quad2_xu = p_q2[0] - self.pos_max
        h_quad2_xl = -(p_q2[0] + self.pos_max)
        h_quad2_yu = p_q2[1] - self.pos_max
        h_quad2_yl = -(p_q2[1] + self.pos_max)

        h_quad1_pos = jnp.stack([h_quad1_xl, h_quad1_xu, h_quad1_yl, h_quad1_yu]).max()
        h_quad2_pos = jnp.stack([h_quad2_xl, h_quad2_xu, h_quad2_yl, h_quad2_yu]).max()

        hs = jnp.array(
            [
                h_quadquadcol,
                h_quadquad_maxdist,
                h_quad1obscol,
                h_quad2obscol,
                h_quad1_pos,
                h_quad2_pos,
            ]
        )
        # h <= 1
        hs = poly4_clip_max_flat(hs)
        # softclip the minimum.
        hs = -poly4_softclip_flat(-hs, m=0.8)
        return hs

    def split_state(self, state: State):
        self.chk_x(state)
        return state[0:4], state[4:8], state[8:12]

    def posrel_embed(self, pos: Vec2, halfrange: float):
        """Represent relative position vector as an angle (sincos) and a distance (with gauss embed)"""
        assert pos.shape == (2,)
        norm_sq = jnp.sum(pos**2)
        norm = jnp.sqrt(norm_sq + 1e-6)
        sincos = pos / norm
        inv_norm = jnp.exp(-norm_sq / (2 * halfrange**2))
        embed = jnp.array([sincos[0], sincos[1], inv_norm])
        return embed

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        q1, q2, obs = self.split_state(state)
        Q_quads = jnp.stack([q1, q2], axis=0)

        # Rel pos between quads
        quadquad_pos_rel = q2[:2] - q1[:2]
        # Rel vel between quads
        quadquad_rel_vel = q2[2:] - q1[2:]

        # Rel pos between quads and obs
        Q_quadobs_pos_rel = obs[:2] - Q_quads[:, :2]
        # Rel vel between quads and obs
        Q_quadobs_vel_rel = obs[2:] - Q_quads[:, 2:]
        # quadobs_pos_rel = Q_quadobs_pos_rel.flatten()
        quadobs_vel_rel = Q_quadobs_vel_rel.flatten()

        # We need absolute positions of the quads because the nominal will go to the ring.
        Q_r = jnp.sqrt(jnp.sum(Q_quads[:, :2] ** 2, axis=-1) + 1e-6)
        Q_cos = Q_quads[:, 0] / Q_r
        Q_sin = Q_quads[:, 1] / Q_r
        assert Q_r.shape == Q_cos.shape == Q_sin.shape == (2,)

        # Absolute velocities as well.
        vel = jnp.concatenate([q1[2:], q2[2:], obs[2:]], axis=0)

        # For the rel position obs, transform into direction (sincos) + distance.
        quadquad_pos_emb = self.posrel_embed(quadquad_pos_rel, halfrange=1.5 * self.quad_max_dist)
        Q_quadobs_pos_emb = jax_vmap(ft.partial(self.posrel_embed, halfrange=5.0))(Q_quadobs_pos_rel)
        quadobs_pos_emb = Q_quadobs_pos_emb.flatten()

        # [3, 6, 2, 2, 4, 6, 2, 2] = 27
        obs = jnp.concatenate(
            [quadquad_pos_emb, quadobs_pos_emb, Q_r, quadquad_rel_vel, quadobs_vel_rel, vel, Q_cos, Q_sin]
        )
        assert obs.shape == (self.n_Vobs,)

        return obs, obs

    def f(self, state: State) -> State:
        self.chk_x(state)
        q1, q2, obs = self.split_state(state)

        zeros = jnp.zeros(2)
        dq1 = jnp.concatenate([q1[2:], -self.k_v * q1[2:]])
        dq2 = jnp.concatenate([q2[2:], -self.k_v * q2[2:]])
        dobs = jnp.concatenate([obs[2:], zeros])

        return jnp.concatenate([dq1, dq2, dobs])

    def G(self, state: State) -> State:
        self.chk_x(state)
        kv = self.k_v
        G = np.zeros((self.NX, self.NU))
        G[self.VX1, self.VXD1] = kv
        G[self.VY1, self.VYD1] = kv
        G[self.VX2, self.VXD2] = kv
        G[self.VY2, self.VYD2] = kv
        return G

    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        xdot_with_u = ft.partial(self.xdot, control=control)
        return rk4(self.dt, xdot_with_u, state)

    def step_plot(
        self, state: State, control: Control, disturb: Disturb = None, dt: float = None
    ) -> tuple[TState, TFloat]:
        xdot_with_u = ft.partial(self.xdot, control=control)
        dt = get_or(dt, self.dt)
        return tsit5(dt, 4, xdot_with_u, state), np.linspace(0, dt, num=5)

    def has_eq_state(self) -> bool:
        return False

    def assert_is_safe(self, state: State) -> BoolScalar:
        state = self.chk_x(state)
        p_q1, p_q2, p_o = state[self.POS_QUAD1], state[self.POS_QUAD2], state[self.POS_OBS]
        quadquad_dist = jnp.linalg.norm(p_q1 - p_q2)
        quad1obs_dist = jnp.linalg.norm(p_q1 - p_o)
        quad2obs_dist = jnp.linalg.norm(p_q2 - p_o)

        quadquad_mindist_buffer_assert = self.buffer_assert["quadquad_mindist"]
        quadquad_maxdist_buffer_assert = self.buffer_assert["quadquad_maxdist"]
        quadobs_buffer_assert = self.buffer_assert["quadobs"]

        safe_quadquad_nocol = quadquad_dist > (2 * self.quad_r + self.quadquad_dmin) + quadquad_mindist_buffer_assert
        safe_quadquad_maxdist = quadquad_dist + quadquad_maxdist_buffer_assert < self.quad_max_dist
        safe_quad1obscol = quad1obs_dist > (self.quad_r + self.quadobs_dmin + self.obs_r) + quadobs_buffer_assert
        safe_quad2obscol = quad2obs_dist > (self.quad_r + self.quadobs_dmin + self.obs_r) + quadobs_buffer_assert

        h_safe = jnp.array([safe_quadquad_nocol, safe_quadquad_maxdist, safe_quad1obscol, safe_quad2obscol])
        return jnp.all(h_safe)

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        quad_bounds = np.array([(-6.0, 6.0), (-6.0, 6.0), (-1.2, 1.2), (-1.2, 1.2)])
        obs_bounds = np.array([(-6.0, 6.0), (-6.0, 6.0), (-1.2, 1.2), (-1.2, 1.2)])
        return np.concatenate([quad_bounds, quad_bounds, obs_bounds], axis=0).T

    def sample_train_x0(self, key: PRNGKey, n_sample: int) -> BState:
        key_choose, key_uniform, key_circ = jr.split(key, 3)
        x_lb, x_ub = self.train_bounds()
        key_uniform0, key_uniform1, key_uniform2 = jr.split(key_uniform, 3)
        b_x0_uniform = jr.uniform(key_uniform0, (n_sample, self.nx), minval=x_lb, maxval=x_ub)

        # # Sample quad2 to be somewhat close to quad1.
        # b_q1_th = jr.uniform(key_uniform1, (n_sample,), minval=0.0, maxval=2 * np.pi)
        # b_q1_r = jr.uniform(key_uniform2, (n_sample,), minval=0.0, maxval=4 * self.quad_max_dist + 2 * self.quad_r)
        # b_q1_pos = b_q1_r[:, None] * jnp.stack([jnp.cos(b_q1_th), jnp.sin(b_q1_th)], axis=1)
        # b_q1_pos = b_x0_uniform[:, self.POS_QUAD1] + b_q1_pos
        # b_x0_uniform = b_x0_uniform.at[:, self.POS_QUAD2].set(b_q1_pos)

        # For the other one, we sample the obstacle on the circle,
        # and sample the two quads nearby.
        key_obs_theta, key_obs_r, key_q0_th, key_q0_r, key_q1_th, key_q1_r = jr.split(key_circ, 6)
        b_obs_th = jr.uniform(key_obs_theta, (n_sample,), minval=0.0, maxval=2 * np.pi)
        b_obs_r = jr.uniform(key_obs_r, (n_sample,), minval=0.95 * self.nom_r, maxval=1.05 * self.nom_r)
        # (n_sample, 2)
        b_obs_pos = b_obs_r[:, None] * jnp.stack([jnp.cos(b_obs_th), jnp.sin(b_obs_th)], axis=1)

        b_q0_th = b_obs_th - jr.uniform(key_q0_th, (n_sample,), minval=0.0, maxval=0.6 * np.pi)
        b_q1_th = b_obs_th - jr.uniform(key_q1_th, (n_sample,), minval=0.0, maxval=0.6 * np.pi)

        b_q0_r = jr.uniform(key_q0_r, (n_sample,), minval=0.95 * self.nom_r, maxval=1.05 * self.nom_r)
        b_q1_r = jr.uniform(key_q1_r, (n_sample,), minval=0.95 * self.nom_r, maxval=1.05 * self.nom_r)

        b_q0_pos = b_q0_r[:, None] * jnp.stack([jnp.cos(b_q0_th), jnp.sin(b_q0_th)], axis=1)
        b_q1_pos = b_q1_r[:, None] * jnp.stack([jnp.cos(b_q1_th), jnp.sin(b_q1_th)], axis=1)

        b_q0_vel = b_x0_uniform[:, self.VEL_QUAD1]
        b_q1_vel = b_x0_uniform[:, self.VEL_QUAD2]
        b_obs_vel = b_x0_uniform[:, self.VEL_OBS]

        b_x0_circle = jnp.concatenate([b_q0_pos, b_q0_vel, b_q1_pos, b_q1_vel, b_obs_pos, b_obs_vel], axis=1)

        p_uniform = 0.95
        b_use_uniform = jr.bernoulli(key_choose, p_uniform, shape=(n_sample,))
        b_x0 = jnp.where(b_use_uniform[:, None], b_x0_uniform, b_x0_circle)

        return assert_shape(b_x0, (n_sample, self.nx))

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        quad_bounds = np.array([(-6.0, 6.0), (-6.0, 6.0), (-1.2, 1.2), (-1.2, 1.2)])
        obs_bounds = np.array([(-6.0, 6.0), (-6.0, 6.0), (-1.2, 1.2), (-1.2, 1.2)])
        return np.concatenate([quad_bounds, quad_bounds, obs_bounds], axis=0).T

    def ci_bounds(self, setup_name: str) -> Float[Arr, "2 nx"]:
        bounds = self.contour_bounds()
        if setup_name == "quad1_pos":
            bounds[:, self.PX1] = np.array([0.0, 5.0])
            bounds[:, self.PY1] = np.array([-4.0, 1.0])
        if setup_name == "quad2_pos":
            bounds[:, self.PX2] = np.array([0.0, 5.0])
            bounds[:, self.PY2] = np.array([-2.1, 2.1])
        if setup_name == "obs_pos":
            bounds[:, self.PXO] = np.array([-6.0, 6.0])
            bounds[:, self.PYO] = np.array([-6.0, 6.0])

        return bounds

    def get_ci_x0(self, setup: int = 0, n_pts: int = 80):
        with jax.ensure_compile_time_eval():
            phase2d_setup = self.phase2d_setups()[setup]
            name = phase2d_setup.plot_name
            idxs = phase2d_setup.get2d_fn.idxs
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(self.ci_bounds(name), idxs, n_pts, n_pts, self.nominal_val_state())
        return bb_x0, bb_Xs, bb_Ys

    def get_metric_x0(self) -> BState:
        with jax.ensure_compile_time_eval():
            # We're safe most of the time, so just sample around the nominal.
            x_nom = self.nominal_val_state()
            # Perturb positions and velocities slightly
            q0_bounds = np.array([(-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2)])
            q1_bounds = np.array([(-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2)])
            obs_bounds = np.array([(-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2), (-0.2, 0.2)])
            perturb_bounds = np.concatenate([q0_bounds, q1_bounds, obs_bounds], axis=0).T
            # (2, nx)
            bounds = x_nom + perturb_bounds

            key = jr.PRNGKey(8182417)
            n_sample = 256
            b_x0 = jr.uniform(key, (n_sample, self.nx), minval=bounds[0], maxval=bounds[1])
            # Only keep the ones that are inside.
            b_in_ci = jax_vmap(self.in_ci_approx)(b_x0)
            b_x0 = b_x0[b_in_ci]

        return b_x0

    def get_loss_x0(self, n_sample: int = 128) -> BState:
        b_x0 = self.sample_train_x0(jr.PRNGKey(314159), n_sample)
        return b_x0

    @property
    def nom_v(self) -> float:
        return self.c_theta * self.nom_r

    def nominal_val_state(self):
        sep_frac = 0.8
        dtheta = -np.arccos(0.5 * (2 - (sep_frac * self.quad_max_dist / self.nom_r) ** 2))
        nom_r, nom_v = self.nom_r, self.nom_v

        # Sanity check.
        quad_dist = self.nom_r * np.sqrt((np.cos(dtheta) - 1) ** 2 + np.sin(dtheta) ** 2)
        assert np.allclose(quad_dist, sep_frac * self.quad_max_dist)

        # Compute obstacle position.
        ttc = 2.0
        frac_to_q2 = 0.1  # 0 => q1,  1 => q2
        # frac_to_q2 = 0.5  # 0 => q1,  1 => q2

        obs_theta = 0.9 * np.pi / 4
        # obs_v = 1.1 * np.array([np.cos(obs_theta), np.sin(obs_theta)])
        obs_v = 1.2 * np.array([np.cos(obs_theta), np.sin(obs_theta)])
        # obs_v = 1.5 * np.array([np.cos(obs_theta), np.sin(obs_theta)])

        q1_theta = ttc * self.c_theta
        q1_pos = nom_r * np.array([np.cos(q1_theta), np.sin(q1_theta)])
        q2_pos = nom_r * np.array([np.cos(q1_theta + dtheta), np.sin(q1_theta + dtheta)])
        target_pos = (1 - frac_to_q2) * q1_pos + frac_to_q2 * q2_pos
        obs_pos = target_pos - ttc * obs_v

        q0 = np.array([nom_r * 1, 0, 0, nom_v])
        q1 = np.array([nom_r * np.cos(dtheta), nom_r * np.sin(dtheta), -nom_v * np.sin(dtheta), nom_v * np.cos(dtheta)])
        obs = np.concatenate([obs_pos, obs_v])

        return np.concatenate([q0, q1, obs])

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [
            Task.Phase2DSetup("quad1_pos", ft.partial(self.plot_pos, who="quad1"), Task.mk_get2d([self.PX1, self.PY1])),
            Task.Phase2DSetup("quad2_pos", ft.partial(self.plot_pos, who="quad2"), Task.mk_get2d([self.PX2, self.PY2])),
            Task.Phase2DSetup("obs_pos", ft.partial(self.plot_pos, who="obs"), Task.mk_get2d([self.PXO, self.PYO])),
        ]

    def plot_quadquad_col(self, ax: plt.Axes, quad_state: State, color, add_q_r: bool = True):
        """Circle around quad. Take the quad radius into account."""
        assert quad_state.shape == (4,)

        self_q_r = self.quad_r if add_q_r else 0
        r = self.quad_r + self.quadquad_dmin + self_q_r
        pos = quad_state[:2]
        circle = plt.Circle(pos, r, color=color, fill=False, alpha=0.4, zorder=3)
        ax.add_artist(circle)

    def plot_quadquad_maxdist(self, ax: plt.Axes, quad_state: State, color):
        """Circle around quad. Take the quad radius into account."""
        assert quad_state.shape == (4,)

        r = self.quad_max_dist
        pos = quad_state[:2]
        circle = plt.Circle(pos, r, color=color, fill=False, alpha=0.4, zorder=3)
        ax.add_artist(circle)

    def plot_obs_col(self, ax: plt.Axes, obs_state: State, add_q_r: bool = True):
        """Circle around obstacle. Take the quad radius and obstacle radius into account."""
        assert obs_state.shape == (4,)

        self_q_r = self.quad_r if add_q_r else 0
        r = self_q_r + self.quadobs_dmin + self.obs_r
        pos = obs_state[:2]
        circle = plt.Circle(pos, r, color="C6", fill=False, alpha=0.8, zorder=3)
        ax.add_artist(circle)

    def plot_obs_col_at_quad(self, ax: plt.Axes, quad_state: State):
        """Circle around quad. Take the quad radius and obstacle radius into account."""
        assert quad_state.shape == (4,)

        r = self.quad_r + self.quadobs_dmin + self.obs_r
        pos = quad_state[:2]
        circle = plt.Circle(pos, r, color="C6", fill=False, alpha=0.3, zorder=3)
        ax.add_artist(circle)

    def get_artists(
        self, show_nom: bool = True, show_control: bool = False, rot: bool = False
    ) -> tuple[CircArtist, CircArtist, CircArtist]:
        color_quad1 = "C1"
        color_quad2 = "C2"
        color_obs = "C6"

        # The full quad must be outside this circle.
        r_quadquad_col = self.quad_r + self.quadquad_dmin
        # # The center of the quad must be inside this distance.
        # r_quadquad_maxdist = self.quad_max_dist
        # The full quad must be inside this distance.
        r_quadquad_maxdist = self.quad_max_dist + self.quad_r
        # The full quad must be outside this circle.
        r_obsquad_col = self.quadobs_dmin + self.obs_r
        quad_lines = [r_quadquad_col, r_quadquad_maxdist]
        obs_lines = [r_obsquad_col]
        q1_artist = CircArtist(
            self.quad_r, color_quad1, 4, quad_lines, show_nom=show_nom, show_control=show_control, rot=rot
        )
        q2_artist = CircArtist(
            self.quad_r, color_quad2, 4, quad_lines, show_nom=show_nom, show_control=show_control, rot=rot
        )
        obs_artist = CircArtist(self.obs_r, color_obs, 4, obs_lines, line_alpha=0.95, rot=rot)
        return q1_artist, q2_artist, obs_artist

    def viz_obs_path(self, ax: plt.Axes, state: State):
        q1, q2, obs = self.split_state(state)
        p_obs, v_obs = obs[:2], obs[2:]
        # Draw a line from t=-1 to t=8.
        p_obs_neg = p_obs - v_obs
        p_obs_pos = p_obs + 8 * v_obs
        p_obs_line = np.stack([p_obs_neg, p_obs_pos], axis=0)

        ax.plot(p_obs_line[:, 0], p_obs_line[:, 1], color="C6", ls="--", zorder=2.2)

    def setup_ax_pos(self, ax: plt.Axes, rot: bool = False):
        PLOT_XMIN, PLOT_XMAX = -5.5, 5.5
        PLOT_YMIN, PLOT_YMAX = -5.5, 5.5
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        if rot:
            ax.set(xlabel=r"$p_y$", ylabel=r"$p_x$")
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
        else:
            ax.set(xlabel=r"$p_x$", ylabel=r"$p_y$")
        ax.set_aspect("equal")

        # Plot the nominal circle.
        circle = plt.Circle((0, 0), self.nom_r, color="0.6", ls="--", fill=False, zorder=2.5)
        ax.add_artist(circle)

    def plot_pos(self, ax: plt.Axes, state: State = None, who: Literal["quad1", "quad2", "obs", "none"] = None):
        """(Theta, Omega) plot."""
        self.setup_ax_pos(ax)
        state = get_or(state, self.nominal_val_state())
        assert who is not None

        self.chk_x(state)
        q1, q2, obs = self.split_state(state)

        color_quad1 = "C1"
        color_quad2 = "C2"
        color_obs = "C6"
        draw_q1, draw_q2, draw_obs = True, True, True
        if who == "quad1":
            self.plot_quadquad_col(ax, q2, color_quad2)
            self.plot_quadquad_maxdist(ax, q2, color_quad2)
            self.plot_obs_col(ax, obs)
            draw_q1 = False
        elif who == "quad2":
            self.plot_quadquad_col(ax, q1, color_quad1)
            self.plot_quadquad_maxdist(ax, q1, color_quad1)
            self.plot_obs_col(ax, obs)
            draw_q2 = False
        elif who == "obs":
            self.plot_quadquad_col(ax, q1, color_quad1)
            self.plot_quadquad_col(ax, q2, color_quad2)
            self.plot_quadquad_maxdist(ax, q1, color_quad1)
            self.plot_quadquad_maxdist(ax, q2, color_quad2)

            self.plot_obs_col_at_quad(ax, q1)
            self.plot_obs_col_at_quad(ax, q2)
            draw_obs = False
        elif who == "none":
            self.plot_quadquad_col(ax, q1, color_quad1, add_q_r=False)
            self.plot_quadquad_maxdist(ax, q1, color_quad1)

            self.plot_quadquad_col(ax, q2, color_quad2, add_q_r=False)
            self.plot_quadquad_maxdist(ax, q2, color_quad2)

            self.plot_obs_col(ax, obs, add_q_r=False)

        if draw_q1:
            circ_q1 = plt.Circle(q1[:2], self.quad_r, color=color_quad1, alpha=0.7, zorder=4)
            ax.add_artist(circ_q1)
        if draw_q2:
            circ_q2 = plt.Circle(q2[:2], self.quad_r, color=color_quad2, alpha=0.7, zorder=4)
            ax.add_artist(circ_q2)
        if draw_obs:
            circ_obs = plt.Circle(obs[:2], self.obs_r, color=color_obs, alpha=0.7, zorder=4)
            ax.add_artist(circ_obs)

    def nom_pol_vf(self, state: State) -> Control:
        """Vector field based control. Limit cycle of circle."""
        self.chk_x(state)
        q1, q2, obs = self.split_state(state)

        def vf(quad_state):
            r_sq = quad_state[0] ** 2 + quad_state[1] ** 2
            A_theta = np.array([[0.0, -1.0], [1.0, 0.0]])
            A_radius = -(r_sq - self.nom_r**2) * np.eye(2)
            A = self.c_theta * A_theta + self.c_r * A_radius
            return A @ quad_state[:2]

        u1_des = vf(q1)
        u2_des = vf(q2)
        u1 = u1_des
        u2 = u2_des

        u = self.chk_u(snp.concatenate([u1, u2], axis=0))
        u = u.clip(-1, 1)
        return u

    def nom_pol_h_h(self, state: State):
        def h_quadquad_col(state: State):
            p_q1, p_q2 = state[self.POS_QUAD1], state[self.POS_QUAD2]
            quadquad_distsq = np.sum((p_q1 - p_q2) ** 2)
            col_dist = 2 * self.quad_r + self.quadquad_dmin
            # col_dist^2 <= dist^2    =>    col_dist^2 - dist^2 <= 0
            return col_dist**2 - quadquad_distsq

        def h_quadquad_maxdist(state: State):
            p_q1, p_q2 = state[self.POS_QUAD1], state[self.POS_QUAD2]
            quadquad_distsq = np.sum((p_q1 - p_q2) ** 2)
            col_dist = self.quad_max_dist
            # dist^2 <= col_dist^2
            return quadquad_distsq - col_dist**2

        def quad1_obs(state: State):
            p_q1, p_obs = state[self.POS_QUAD1], state[self.POS_OBS]
            quad_obs_distsq = np.sum((p_q1 - p_obs) ** 2)
            col_dist = self.quad_r + self.quadobs_dmin + self.obs_r
            # col_dist^2 <= dist^2
            return col_dist**2 - quad_obs_distsq

        def quad2_obs(state: State):
            p_q2, p_obs = state[self.POS_QUAD2], state[self.POS_OBS]
            quad_obs_distsq = np.sum((p_q2 - p_obs) ** 2)
            col_dist = self.quad_r + self.quadobs_dmin + self.obs_r
            # col_dist^2 <= dist^2
            return col_dist**2 - quad_obs_distsq

        h_fns = [h_quadquad_col, h_quadquad_maxdist, quad1_obs, quad2_obs]
        h_h_fn = stack_cbf_fns(h_fns)
        return h_h_fn(state)

    def nom_pol_h_B(self, state: State):
        self.chk_x(state)
        alpha0s = np.array([2.0, 2.0, 2.0, 2.0])
        return hocbf(self.nom_pol_h_h, self.f, alpha0s, state)

    def nom_pol_handcbf(self, state: State) -> Control:
        h_B = self.nom_pol_h_B(state)
        hx_Bx = jax.jacfwd(self.nom_pol_h_B)(state)

        # Compute QP sol.
        u_nom = self.nom_pol_vf(state)
        u_lb, u_ub = self.u_min, self.u_max
        f, G = self.f(state), self.G(state)

        # u_qp, r, (qp_state, qp_mats) = cbf_old.min_norm_cbf(self.hand_cbf_alpha, u_lb, u_ub, h_B, hx_Bx, f, G, u_nom)
        settings = JaxProxQP.Settings.default()
        settings.max_iter = 15
        settings.max_iter_in = 5
        settings.max_iterative_refine = 5
        u_qp, r, sol = min_norm_cbf(self.hand_cbf_alpha, u_lb, u_ub, h_B, hx_Bx, f, G, u_nom, settings=settings)

        u_qp = self.chk_u(u_qp.clip(-1, 1))
        return u_qp

    @ft.partial(jax.jit, static_argnums=0)
    def get_bb_V_noms(self) -> dict[str, BBFloat]:
        tf = 8.0
        result_dt = 0.1
        sim = SimCtsReal(self, self.nom_pol_vf, tf, result_dt)

        out = {}
        for ii, setup in enumerate(self.phase2d_setups()):
            bb_x0, bb_Xs, bb_Ys = self.get_contour_x0(ii)
            bbT_x, _, _ = rep_vmap(sim.rollout_plot, rep=2)(bb_x0)
            bbT_h = rep_vmap(self.h, rep=3)(bbT_x)
            out[setup.plot_name] = bbT_h.max(axis=2)

        return out

    def get_state(self, q1: State | None = None, q2: State | None = None, obs: State | None = None) -> State:
        q1_nom, q2_nom, obs_nom = self.split_state(self.nominal_val_state())
        q1 = get_or(q1, q1_nom)
        q2 = get_or(q2, q2_nom)
        obs = get_or(obs, obs_nom)
        return np.concatenate([q1, q2, obs], axis=0)

    def _sample_stats_x0(self, key: PRNGKey, n_samples: int) -> BState:
        """Samples random states for evaluating safety stats. To make sampling more efficient,
        instead of sampling both the quad positions randomly, constrain the second quad to be within
        the correct radius of the first."""
        key_obs, key_quad1, key_quad2_radius, key_quad2_angle, key_quad2_vel = jr.split(key, 5)
        train_bounds = self.train_bounds()
        b_obs_state = jr.uniform(
            key_obs, (n_samples, 4), minval=train_bounds[0, self.OBS], maxval=train_bounds[1, self.OBS]
        )
        b_q1_state = jr.uniform(
            key_quad1, (n_samples, 4), minval=train_bounds[0, self.QUAD1], maxval=train_bounds[1, self.QUAD1]
        )
        r_min, r_max = self.quadquad_dmin + 2 * self.quad_r, self.quad_max_dist
        b_q2_radius = jr.uniform(key_quad2_radius, (n_samples,), minval=r_min, maxval=r_max)
        b_q2_angle = jr.uniform(key_quad2_angle, (n_samples,), minval=0, maxval=2 * np.pi)
        b_q2_pos = b_q1_state[:, :2] + b_q2_radius[:, None] * np.stack([np.cos(b_q2_angle), np.sin(b_q2_angle)], axis=1)
        b_q2_vel = jr.uniform(key_quad2_vel, (n_samples, 2), minval=train_bounds[0, self.VEL_QUAD2], maxval=1)

        b_state = jnp.concatenate([b_q1_state, b_q2_pos, b_q2_vel, b_obs_state], axis=1)
        assert b_state.shape == (n_samples, self.nx)
        return b_state

    def sample_stats_x0(self):
        n_desired = 32768

        # Cache it so its faster.
        path_cache_npy = get_paper_data_dir() / "quadint/stats_x0_cache.npy"

        if path_cache_npy.exists():
            b_x0 = np.load(path_cache_npy)
            logger.info("Loaded cache from {}!".format(path_cache_npy))
            assert b_x0.shape == (n_desired, self.nx)
        else:
            eps = 5e-3
            key = jr.PRNGKey(58124123)
            b_x0_list = []
            n_x0 = 0

            ii = 0
            while n_x0 < n_desired:
                key_tmp = jr.fold_in(key, ii)
                b_x0 = np.array(self._sample_stats_x0(key_tmp, 2048))
                b_h = np.array(jax_vmap(self.h)(b_x0))
                b_x0 = b_x0[b_h < -eps]
                n_x0 += len(b_x0)
                b_x0_list.append(b_x0)
                ii += 1
            b_x0 = np.concatenate(b_x0_list, axis=0)
            b_x0 = b_x0[:n_desired]

            # Save cache for next time.
            np.save(path_cache_npy, b_x0)
            logger.info("Saved cache to {}!".format(path_cache_npy))

        return b_x0

    def get_cbf_alphas(self):
        # quadquad_col, quadquad_maxdist, quad1obs, quad2obs
        alpha_safe = np.full(self.nh, 1.0)
        alpha_safe[1] = 2.0
        alpha_unsafe = np.full(self.nh, 40.0)
        # alpha_safe = np.array([1.0, 2.0, 1.0, 1.0])
        # alpha_unsafe = np.array([50.0, 2.0, 50.0, 50.0])
        return alpha_safe, alpha_unsafe
