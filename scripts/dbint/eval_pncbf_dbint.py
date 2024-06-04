import functools as ft
import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.int_avoid.doubleintwall_cfg
from pncbf.dyn.doubleint_wall import DoubleIntWall
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.plotting.contour_utils import centered_norm
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.pncbf.pncbf import PNCBF
from pncbf.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    seed = 0

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    task = DoubleIntWall()

    # nom_pol = task.nom_pol_osc
    nom_pol = task.nom_pol_rng3

    CFG = run_config.int_avoid.doubleintwall_cfg.get(seed)
    alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    # Plot how V varies along a trajectory.
    # x0 = np.array([-0.6, 1.7])
    # x0 = np.array([0.5, -1.7])
    x0 = np.array([0.8, 0.3])
    T = 80
    tf = T * task.dt

    # Original nominal policy.
    logger.info("Sim nom...")
    sim = SimCtsReal(task, nom_pol, tf, task.dt, use_pid=True)
    T_x_nom, T_t_nom, _ = jax2np(jax_jit(sim.rollout_plot)(x0))

    # alphas = np.array([0.1, 1.0, 5.0, 10.0])
    alphas = np.array([0.001, 0.01, 0.1, 5.0])

    def int_pol_for_alpha(alpha_safe):
        alpha_unsafe = 10.0
        pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, V_shift=1e-2)
        sim = SimCtsReal(task, pol, tf, 0.5 * task.dt, use_obs=False, use_pid=False, max_steps=512)
        T_x, T_t, _ = sim.rollout_plot(x0)
        return T_x, T_t

    logger.info("Sim pol for different alphas...")
    bT_x, bT_t = [], []
    for alpha in alphas:
        T_x, T_t = jax2np(jax_jit(int_pol_for_alpha)(alpha))
        bT_x.append(T_x)
        bT_t.append(T_t)
    bT_x = np.stack(bT_x, axis=0)
    bT_t = np.stack(bT_t, axis=0)

    logger.info("bbh_Vh...")
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=192)
    bbh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bb_x))
    bb_Vh = bbh_Vh.max(-1)

    #####################################################
    logger.info("Plotting...")
    h_labels = task.h_labels

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x_nom[:, 0], T_x_nom[:, 1], color="C3", ls="--", label="Nominal")
    for ii, alpha in enumerate(alphas):
        ax.plot(bT_x[ii, :, 0], bT_x[ii, :, 1], color=f"C{ii}", lw=0.5, ls="--", label=f"QP ({alpha})", zorder=100)
    ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.6, linewidths=1.0)
    task.plot_phase(ax)
    ax.legend()
    fig.savefig(plot_dir / "eval_phase.pdf")
    plt.close(fig)

    norm = centered_norm(bbh_Vh.min(), bbh_Vh.max())

    levels = 31
    figsize = (8, 4)
    fig, axes = plt.subplots(1, task.nh, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_Vh[:, :, ii], norm=norm, levels=levels, cmap="RdBu_r", alpha=0.9)
        cs1 = ax.contour(
            bb_Xs, bb_Ys, bbh_Vh[:, :, ii], levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0
        )
        cbar = fig.colorbar(cs0, ax=ax)
        cbar.add_lines(cs1)
        task.plot_phase(ax)
        ax.set_title(h_labels[ii])
    fig.savefig(plot_dir / "eval_Vh.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
