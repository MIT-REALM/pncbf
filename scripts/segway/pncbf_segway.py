import functools as ft
import pathlib

import ipdb
import numpy as np
import typer
from loguru import logger

import run_config.int_avoid.segway_cfg
import wandb
from pncbf.dyn.segway import Segway
from pncbf.pncbf.pncbf import PNCBF
from pncbf.pncbf.stateful_dset_buffer import StatefulDsetBuffer
from pncbf.plotting.plot_task_summary import plot_task_summary
from pncbf.plotting.plotter import MPPlotter
from pncbf.training.ckpt_manager import get_ckpt_manager, save_create_args
from pncbf.training.run_dir import init_wandb_and_get_run_dir
from pncbf.utils.ckpt_utils import load_ckpt_with_step
from pncbf.utils.jax_utils import jax2np, jax_default_x32
from pncbf.utils.logging import set_logger_format


def main(
    name: str = typer.Option(..., help="Name of the run."),
    group: str = typer.Option(None),
    ckpt: pathlib.Path = None,
    seed: int = 7957821,
):
    jax_default_x32()
    set_logger_format()

    task = Segway()

    CFG = run_config.int_avoid.segway_cfg.get(seed)
    nom_T = 500

    nom_pol = task.nom_pol_lqr

    alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    CFG.extras = {"nom_pol": "lqr"}

    if ckpt is not None:
        alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
        alg, ckpt_path = load_ckpt_with_step(alg, ckpt)
        logger.info("Loaded ckpt from {}!".format(ckpt_path))
        alpha_safe, alpha_unsafe = 2.0, 100.0
        nom_pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, nom_pol=alg.nom_pol)
        CFG.extras = {"nom_pol": "ckpt_cbf"}
        CFG = run_config.int_avoid.segway_cfg.get_pi(seed)

    # loss_weights = {"Loss/Vh_mse": 1.0, "Loss/Now": 0.0, "Loss/Future": 1.0, "Loss/PDE": 0.0}
    loss_weights = {"Loss/Vh_mse": 1.0}
    CFG.extras["loss_weights"] = loss_weights

    run_dir = init_wandb_and_get_run_dir(CFG, "pncbf_segway", "pncbf_segway", name, group=group)
    plot_dir, ckpt_dir = run_dir / "plots", run_dir / "ckpts"
    plotter = MPPlotter(task, plot_dir)

    LCFG = CFG.loop_cfg

    ckpt_manager = get_ckpt_manager(ckpt_dir)
    save_create_args(ckpt_dir, [seed, task, CFG.alg_cfg])
    plot_task_summary(task, plotter, nom_pol=nom_pol)

    _, bb_Xs, bb_Ys = task.get_contour_x0()
    del _

    bb_V_nom = jax2np(alg.get_bb_V_nom(nom_T))
    V_nom_line = [(bb_V_nom, "C5")]
    V_nom_line2 = [(bb_Xs, bb_Ys, bb_V_nom, "C5")]

    dset_buf = StatefulDsetBuffer(seed=58123, dset_len_max=16)
    for idx in range(LCFG.n_iters + 1):
        should_log = idx % LCFG.log_every == 0
        should_eval = idx % LCFG.eval_every == 0
        should_ckpt = idx % LCFG.ckpt_every == 0

        if (idx % 1_000 == 0) or not dset_buf.is_full:
            alg, updated = alg.update_lam()
            alg, dset = alg.sample_dset()
            dset_buf.add_dset(dset, alg.get_vterms)

        # Randomly sample x0. Half is random, half is t=0.
        n_rng = alg.train_cfg.batch_size // 2
        n_zero = alg.train_cfg.batch_size - n_rng
        batch = dset_buf.sample_batch(n_rng, n_zero)
        alg, loss_info = alg.update(batch, loss_weights)

        if should_log:
            log_dict = {f"train/{k}": np.mean(v) for k, v in loss_info.items()}
            log_dict["train/collect_idx"] = alg.collect_idx
            logger.info(f"[{idx:8}]   ")
            wandb.log(log_dict, step=idx)
        del loss_info

        if should_eval:
            eval_data: PNCBF.EvalData = jax2np(alg.eval())
            suffix = "{}.jpg".format(idx)

            bb_Xs, bb_Ys = eval_data.bb_Xs, eval_data.bb_Ys

            plotter.mp_run.remove_finished()
            plotter.batch_phase2d(eval_data.bT_x_plot, f"phase/phase_{suffix}", extra_lines=V_nom_line2)
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_V.max(-1), f"V/V_{suffix}", extra_lines=V_nom_line)
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_Vdot.max(-1), f"Vdot_max/Vdot_max_{suffix}")
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_Vdot_disc.max(-1), f"Vdot_disc_max/Vdot_disc_max_{suffix}")
            # plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_hmV.max(-1), f"hmV_max/hmV_max_{suffix}", extra_lines=V_nom_line)
            # plotter.V_div(
            #     bb_Xs, bb_Ys, eval_data.bbh_compl.max(-1), f"compl_max/compl_max_{suffix}", extra_lines=V_nom_line
            # )
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bb_u[:, :, 0], f"u/u_{suffix}")

            log_dict = {f"eval/{k}": v.sum() for k, v in eval_data.info.items()}
            wandb.log(log_dict, step=idx)

        if should_ckpt:
            ckpt_manager.save(idx, alg)
            logger.info("[{:5}] - Saving ckpt...".format(idx))

    # Save last.
    logger.info("[{:5}] - Saving ckpt...".format(idx))
    ckpt_manager.save(idx, alg)
    ckpt_manager.wait_until_finished()
    plotter.mp_run.remove_finished()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
