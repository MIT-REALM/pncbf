import numpy as np

from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.dyn.task import Task
# from pncbf.nclf.sim_nclf import SimNCLF
from pncbf.plotting.plotter import Plotter
from pncbf.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from pncbf.utils.mathtext import from_mathtext


def plot_task_summary(task: Task, plotter: Plotter, nom_pol=None):
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0()
    bb_hs = jax2np(rep_vmap(task.h_components, rep=2)(bb_x))
    bb_ls = jax2np(rep_vmap(task.l_components, rep=2)(bb_x))

    # Plot each h.
    for ii, h_label in enumerate(task.h_labels):
        plotter.V_div(bb_Xs, bb_Ys, bb_hs[:, :, ii], from_mathtext(f"h_{ii:02}_{h_label}.pdf"), title=h_label)

    # Plot total h.
    if task.nh > 0:
        bb_h = np.max(bb_hs, axis=-1)
        plotter.V_div(bb_Xs, bb_Ys, bb_h, f"max_h_fn.pdf")

    # # Plot each l.
    # for ii, l_label in enumerate(task.l_labels):
    #     plotter.V_div(bb_Xs, bb_Ys, bb_ls[:, :, ii], from_mathtext(f"l_{ii:02}_{l_label}.pdf"), title=l_label)
    # # Plot total l.
    # bb_l = np.sum(bb_ls, axis=-1)
    # plotter.V_div(bb_Xs, bb_Ys, bb_l, f"sum_l_fn.pdf")

    if nom_pol is None:
        return

    # Plot the nominal policy and its CI
    T = 80
    tf = T * task.dt

    sim = SimCtsReal(task, nom_pol, tf, task.dt, use_obs=False)
    bbT_x, _, _ = jax2np(jax_jit(rep_vmap(sim.rollout_plot, rep=2))(bb_x))
    bbT_h = jax2np(rep_vmap(task.h, rep=3)(bbT_x))
    bb_V_nom = bbT_h.max(-1)

    b_x0_plot = task.get_plot_x0()
    bT_x_plot, _, _ = jax2np(jax_jit(jax_vmap(sim.rollout_plot))(b_x0_plot))
    plotter.batch_phase2d(bT_x_plot, "phase_nom_pol.pdf", extra_lines=[(bb_Xs, bb_Ys, bb_V_nom, "C5")])

    # Also see what the nominal policy is like.
    if task.nu != 1:
        return

    bb_u = jax2np(jax_jit(rep_vmap(nom_pol, rep=2))(bb_x))
    plotter.V_div(bb_Xs, bb_Ys, bb_u[:, :, 0], "nom_pol.pdf")
