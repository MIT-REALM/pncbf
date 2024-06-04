import ipdb
import numpy as np
import typer
from flax.training import orbax_utils
from loguru import logger

import run_config.int_avoid.f16gcas_cfg
import wandb
from pncbf.dyn.f16_gcas import F16GCAS
from pncbf.pncbf.pncbf import PNCBF
from pncbf.plotting.plot_task_summary import plot_task_summary
from pncbf.plotting.plotter import MPPlotter, Plotter
from pncbf.training.ckpt_manager import get_ckpt_manager, save_create_args
from pncbf.training.run_dir import init_wandb_and_get_run_dir
from pncbf.utils.jax_utils import jax2np, jax_jit, tree_cat, tree_copy
from pncbf.utils.logging import set_logger_format


def main(name: str = typer.Option(..., help="Name of the run."), group: str = typer.Option(None), seed: int = 7957821):
    set_logger_format()

    task = F16GCAS()

    CFG = run_config.int_avoid.f16gcas_cfg.get(seed)

    nom_pol = task.nom_pol_pid

    alg: PNCBF = PNCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    CFG.extras = {"nom_pol": "pid"}

    # loss_weights = {"Loss/Vh_mse": 1.0, "Loss/Now": 1.0, "Loss/Future": 1.0, "Loss/PDE": 0.0}
    loss_weights = {"Loss/Vh_mse": 1.0, "Loss/Now": 1.0, "Loss/Future": 1.0, "Loss/Equil": 1.0}
    CFG.extras["loss_weights"] = loss_weights

    run_dir = init_wandb_and_get_run_dir(CFG, "pncbf_f16gcas", "pncbf_f16gcas", name, group=group)
    plot_dir, ckpt_dir = run_dir / "plots", run_dir / "ckpts"
    plotter = MPPlotter(task, plot_dir)

    LCFG = CFG.loop_cfg

    ckpt_manager = get_ckpt_manager(ckpt_dir)
    save_create_args(ckpt_dir, [seed, task, CFG.alg_cfg])
    plot_task_summary(task, plotter, nom_pol=nom_pol)

    _, bb_Xs, bb_Ys = task.get_contour_x0()
    del _

    bb_V_noms = jax2np(task.get_bb_V_noms())
    Vnom_ln = [(bb_V_noms, "C5")]

    dset_list = []
    dset_len_max = 8

    rng = np.random.default_rng(seed=58123)
    dset: PNCBF.CollectData | None = None
    for idx in range(LCFG.n_iters + 1):
        should_log = idx % LCFG.log_every == 0
        should_eval = idx % LCFG.eval_every == 0
        should_ckpt = idx % LCFG.ckpt_every == 0

        # if idx == 150_000:
        #     loss_weights = {"Loss/Now": 1.0, "Loss/Future": 1.0, "Loss/PDE": 1.0}

        if (idx % 500 == 0) or len(dset_list) < dset_len_max:
            alg, updated = alg.update_lam()
            alg, dset = alg.sample_dset()
            dset_list.append(jax2np(dset))

            if len(dset_list) > dset_len_max:
                del dset_list[0]

            # If we have updated lambda, then we need to recompute vterms using the new lambda.
            for ii, dset_item in enumerate(dset_list[:-1]):
                dset_list[ii] = dset_list[ii]._replace(b_vterms=jax2np(alg.get_vterms(dset_item.bT_x)))

            dset = tree_cat(dset_list, axis=0)
            b = dset.bT_x.shape[0]
            b_times_Tm1 = b * (dset.bT_x.shape[1] - 1)

        # Randomly sample x0. Half is random, half is t=0.
        n_rng = alg.train_cfg.batch_size // 2
        n_zero = alg.train_cfg.batch_size - n_rng

        b_idx_rng = rng.integers(0, b_times_Tm1, size=(n_rng,))
        b_idx_b_rng = b_idx_rng // (dset.bT_x.shape[1] - 1)
        b_idx_t_rng = 1 + (b_idx_rng % (dset.bT_x.shape[1] - 1))

        b_idx_b_zero = rng.integers(0, b, size=(n_zero,))
        b_idx_t_zero = np.zeros_like(b_idx_b_zero)

        b_idx_b = np.concatenate([b_idx_b_rng, b_idx_b_zero], axis=0)
        b_idx_t = np.concatenate([b_idx_t_rng, b_idx_t_zero], axis=0)

        b_x0 = dset.bT_x[b_idx_b, b_idx_t]
        b_xT = dset.bT_x[b_idx_b, -1]
        bh_lhs = dset.b_vterms.Th_max_lhs[b_idx_b, b_idx_t, :]
        bh_int_rhs = dset.b_vterms.Th_disc_int_rhs[b_idx_b, b_idx_t, :]
        b_discount_rhs = dset.b_vterms.T_discount_rhs[b_idx_b, b_idx_t]
        batch = alg.Batch(b_x0, b_xT, bh_lhs, bh_int_rhs, b_discount_rhs)

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
            bb_dV_max = eval_data.bbh_Vdot.max(-1)
            bb_dV_disc_max = eval_data.bbh_Vdot_disc.max(-1)

            logger.info("Plotting...")
            plotter.mp_run.remove_finished()
            logger.info("Removed finished!")
            plotter.batch_phase2d(eval_data.bT_x_plot, f"phase/phase_{suffix}", extra_lines=Vnom_ln)
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_V.max(-1), f"V/V_{suffix}", extra_lines=Vnom_ln)
            plotter.V_div(bb_Xs, bb_Ys, bb_dV_max, f"dV_max/dV_max_{suffix}", extra_lines=Vnom_ln)
            plotter.V_div(bb_Xs, bb_Ys, bb_dV_disc_max, f"dV_disc_max/dV_disc_max_{suffix}", extra_lines=Vnom_ln)
            # plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_hmV.max(-1), f"hmV_max/hmV_max_{suffix}", extra_lines=V_nom_line)
            # plotter.V_div(
            #     bb_Xs, bb_Ys, eval_data.bbh_compl.max(-1), f"compl_max/compl_max_{suffix}", extra_lines=V_nom_line
            # )
            # plotter.V_div(bb_Xs, bb_Ys, eval_data.bb_u[:, :, 0], f"u/u_{suffix}")
            logger.info("End of plot")

            log_dict = {f"eval/{k}": v.sum() for k, v in eval_data.info.items()}
            wandb.log(log_dict, step=idx)

        if should_ckpt:
            alg_copy = tree_copy(alg)
            save_args = orbax_utils.save_args_from_target(alg_copy)
            ckpt_manager.save(idx, alg_copy, save_kwargs={"save_args": save_args})
            logger.info("[{:5}] - Saving ckpt...".format(idx))

    # Save last.
    logger.info("[{:5}] - Saving ckpt...".format(idx))
    save_args = orbax_utils.save_args_from_target(alg)
    ckpt_manager.save(idx, alg, save_kwargs={"save_args": save_args})


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
