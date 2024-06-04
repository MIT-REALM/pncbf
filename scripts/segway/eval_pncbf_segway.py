import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.int_avoid.segway_cfg
from pncbf.dyn.segway import Segway
from pncbf.plotting.contour_utils import centered_norm
from pncbf.plotting.legend_helpers import lline
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.pncbf.pncbf import PNCBF
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt
from pncbf.utils.jax_utils import jax2np, jax_jit, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    task = Segway()
    CFG = run_config.int_avoid.segway_cfg.get(seed)
    CFG.alg_cfg.eval_cfg.eval_rollout_T = 200
    nom_pol = task.nom_pol_lqr

    alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    h_labels = task.h_labels
    #####################################################
    # 1: Detailed plot of Vh.
    for ii, setup in enumerate(task.phase2d_setups()):
        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup=ii, n_pts=256)
        bbh_Vh_pred = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bb_x))

        vmin, vmax = bbh_Vh_pred.min(), bbh_Vh_pred.max()
        norm = centered_norm(vmin, vmax)
        logger.info("vmin: {}, vmax: {}, halfrange: {}".format(vmin, vmax, norm.halfrange))
        levels = np.linspace(-norm.halfrange, norm.halfrange, num=31)
        style = dict(norm=norm, levels=levels, cmap="RdBu_r", alpha=0.9)
        fig, axes = plt.subplots(2, 2, layout="constrained")
        axes = axes.flatten()
        for ii, ax in enumerate(axes):
            cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_Vh_pred[:, :, ii], **style)
            zero_style = dict(levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0)
            cs1 = ax.contour(bb_Xs, bb_Ys, bbh_Vh_pred[:, :, ii], **zero_style)
            setup.plot(ax)
            ax.set_title(h_labels[ii])
        cbar = fig.colorbar(cs0, ax=axes.tolist())
        cbar.add_lines(cs1)
        fig.savefig(plot_dir / f"Vh_detailed_{setup.plot_name}{cid}.pdf")
        plt.close(fig)

    #####################################################
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0()
    logger.info("V_pred")
    bb_V_pred = jax2np(jax_jit(rep_vmap(alg.get_V, rep=2))(bb_x))
    logger.info("V_nom")
    bb_V_nom = jax2np(jax_jit(alg.get_bb_V_nom)())
    logger.info("V_sloped")
    bb_V_qp = jax2np(jax_jit(alg.get_bb_V_sloped)())

    #####################################################
    # Compare the CI between the two.
    logger.info("Plotting...")
    # Plot phase.
    fig, ax = plt.subplots(layout="constrained")
    # levels = [task.h_safe_ub, 0]
    # linestyles = ["--", "-"]
    levels = [0]
    linestyles = ["-"]
    ax.contour(bb_Xs, bb_Ys, bb_V_pred, levels=levels, linestyles=linestyles, colors=["C0"], zorder=3.5)
    ax.contour(bb_Xs, bb_Ys, bb_V_nom, levels=levels, linestyles=linestyles, colors=["C5"], zorder=3.5)
    ax.contour(bb_Xs, bb_Ys, bb_V_qp, levels=levels, linestyles=linestyles, colors=["C6"], zorder=3.5)
    task.plot_pend(ax)

    lines = [lline("C0"), lline("C5"), lline("C6")]
    ax.legend(lines, ["CBF Pred CI", "Nominal CI", "QP CI"], loc="upper right")
    fig.savefig(plot_dir / f"compare_ci{cid}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
