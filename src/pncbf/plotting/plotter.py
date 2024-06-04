import inspect
import os
import pathlib
from typing import Any, Callable, TypeVar

import einops as ei
import matplotlib.colors as mcolors
import numpy as np
from attrs import define
from jaxtyping import Float
from loguru import logger
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import CenteredNorm, Normalize, TwoSlopeNorm

import pncbf.utils.typed_mpl as plt
from pncbf.dyn.dyn_types import BBState, BTState, ZBBFloat, ZFloat
from pncbf.dyn.task import Task
from pncbf.plotting.ax_sizing import scale_ax_lims
from pncbf.plotting.fast_savefig import fast_savefig
from pncbf.plotting.mp_run import MPRun
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.plotting.register_sns_cmaps import register_cmaps

from pncbf.utils.dict_utils import dict_index

from pncbf.utils.jax_types import Arr, BBBool, BBFloat, BBool, BTBool, BTFloat, FloatScalar
from pncbf.utils.none import get_or
from pncbf.utils.shape_utils import assert_shape

PathOrStr = pathlib.Path | str

na = np.ndarray


def pretty_str(x):
    if isinstance(x, (int, np.int32, np.int64)):
        return "{}".format(x)

    if isinstance(x, (float, np.float32, np.float64)):
        if x == 0.0:
            return "0."

        if 0.0099 < abs(x) < 100:
            return "{:.2f}".format(x)

        return "{:.2e}".format(x)

    raise NotImplementedError(f"Unknown type {type(x).__name__}!")


def format_bounds_str(minval: FloatScalar, maxval: FloatScalar, variable_name: str) -> str:
    return "{}∈[{}, {}]".format(variable_name, pretty_str(minval), pretty_str(maxval))


def two_slope_norm(data: np.ndarray, vmin: float | None = None, vmax: float | None = None):
    eps = 1e-3
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    vmin = np.clip(vmin, a_min=None, a_max=-eps)
    vmax = np.clip(vmax, a_min=eps, a_max=None)

    return TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)


def centered_norm(vmin: float, vmax: float):
    return CenteredNorm(vcenter=0.0, halfrange=max(-vmin, vmax))


def _add_prefix(path: PathOrStr, prefix: str, sep: str) -> PathOrStr:
    is_str = isinstance(path, str)
    if is_str:
        path = pathlib.Path(path)

    if sep == "/":
        path = path.parent / prefix / path.name
    else:
        path = path.with_stem("{}{}{}".format(prefix, sep, path.stem))

    if is_str:
        path = str(path)
    return path


@define
class PlotterCfg:
    dpi: int = 350

    div_cmap: str = "RdBu_r"
    seq_cmap: str = "rocket"
    seq_cmap_r: str = "rocket_r"


class _Plotter:
    def __init__(self, task: Task, rel_path: pathlib.Path | None = None, cfg: PlotterCfg = PlotterCfg()):
        self._task = task
        self._rel_path = rel_path
        self._cfg = cfg
        self._ext_map = [("pdf", ""), ("png", "web")]
        register_cmaps()

    @staticmethod
    def use_grid(ax: plt.Axes):
        ax.grid(visible=True, color="#b2b2b2", linestyle="--", linewidth=0.5)

    def plot_task(self, ax: plt.Axes, setup_idx: int = 0) -> None:
        self._task.plot(ax, setup_idx=setup_idx)

    def savefig(self, fig: plt.Figure, path: PathOrStr, closefig: bool = True) -> None:
        path = pathlib.Path(path)

        if path.suffix == "":
            # Save using the ext_map.
            for ext, folder in self._ext_map:
                fig_path = path.parent / folder / "{}.{}".format(path.stem, ext)
                self.savefig(fig, fig_path, closefig=False)
        else:
            if self._rel_path is not None:
                # assert isinstance(path, str), "[savefig] path argument must be a str but got {}".format(path)
                assert isinstance(path, str) or (isinstance(path, pathlib.Path) and not path.is_absolute())
                fig_path = self._rel_path / path
            else:
                fig_path = path
            fig_path.parent.mkdir(exist_ok=True, parents=True)
            if path.suffix in [".png", ".jpg"]:
                fig.set_dpi(self._cfg.dpi)
                fast_savefig(fig, fig_path)
            else:
                fig.savefig(fig_path)

        if closefig:
            plt.close(fig)

    def batch_plot(self, b_values: BTFloat, path: PathOrStr):
        """Just ax.plot(values)."""
        b, T = b_values.shape
        b_ts = ei.repeat(np.arange(T), "T -> b T", b=b)
        # (b T 2)
        segs = np.stack([b_ts, b_values], axis=2)

        fig, ax = plt.subplots(dpi=self._cfg.dpi)
        line_col = LineCollection(segs, lw=1.0, alpha=0.9)
        ax.add_collection(line_col)
        ax.autoscale_view()

        self.savefig(fig, path)

    def grid_batch_plot(
        self,
        zbT_vals: Float[na, "ngrid batch T"] | dict[str, Float[na, "ngrid batch T"]],
        vals: Float[na, "ngrid"],
        fig_path: PathOrStr,
        valname: str = "z",
    ):
        """Just ax.plot(values)."""
        if isinstance(zbT_vals, np.ndarray):
            zbT_vals = {"": zbT_vals}

        ngrid, b, T = [*zbT_vals.values()][0].shape
        b_ts = ei.repeat(np.arange(T), "T -> b T", b=b)

        if ngrid == 1:
            return self.batch_plot(dict_index(zbT_vals, 0), fig_path)

        fig, axes = plt.subplots(ngrid, figsize=(8, 2 * ngrid), dpi=self._cfg.dpi)
        for ii, ax in enumerate(axes):
            for jj, (k, v) in enumerate(zbT_vals.items()):
                bT_vals = v[ii]
                segs = np.stack([b_ts, bT_vals], axis=-1)

                line_col = LineCollection(segs, lw=1.0, alpha=0.9, label=k, color=f"C{jj}")
                ax.add_collection(line_col)
                ax.autoscale_view()

            if len(zbT_vals) == 1:
                vmin, vmax = v.min(), v.max()
                title_str = "{}={:.2f} {}".format(valname, vals[ii], format_bounds_str(vmin, vmax, "x"))
            else:
                title_str = "{}={:.2f}".format(valname, vals[ii])
            ax.set(title=title_str)

            if len(zbT_vals) > 1:
                ax.legend()

        self.savefig(fig, fig_path)

    def batch_phase2d(
        self,
        states: BTState,
        path: PathOrStr,
        is_valid: BTBool | BBool = None,
        setup_idx: int = None,
        stop_at_invalid: bool = True,
        extra_lines: list[tuple[BBFloat, BBFloat, BBFloat, str] | tuple[dict[str, BBFloat], str]] = [],
    ):
        assert states.ndim == 3, "Must have shape (batch, T, 2), got {}".format(states.shape)

        n_2d_plots = len(self._task.phase2d_setups())
        phase2d_setups = self._task.phase2d_setups()

        if setup_idx is not None:
            phase2d_setups = [phase2d_setups[setup_idx]]
            n_2d_plots = 1

        b, T, nx = states.shape
        is_valid = get_or(is_valid, np.ones((b, T), dtype=bool))
        if is_valid.ndim == 1 and len(is_valid) == b:
            is_valid = ei.repeat(is_valid, "b -> b T", T=T)

        bT_isvalid = is_valid
        del is_valid

        # (b, )
        has_frozen = np.max(~bT_isvalid, axis=1)
        valid_lens = np.where(has_frozen, bT_isvalid.argmin(axis=1), T)
        n_valid, n_invalid = np.sum(valid_lens == T), np.sum(valid_lens < T)

        for stpidx, (plot2d_name, plot_avoid_fn, get2d, _) in enumerate(phase2d_setups):
            stpidx = setup_idx if n_2d_plots == 1 else stpidx

            pos = get2d(states)
            fig, ax = plt.subplots(dpi=self._cfg.dpi)

            valid_segs, invalid_segs, invalid_ends = [], [], []
            for ii, valid_len in enumerate(valid_lens):
                if valid_len == T:
                    valid_segs.append(pos[ii, :valid_len, :])
                else:
                    invalid_start = max(0, valid_len - 1)
                    invalid_segs.append(pos[ii, : invalid_start + 2])
                    invalid_ends.append(invalid_segs[-1][-1])

            valid_color = "#348ABD"
            invalid_color = "#E24A33"

            valid_col = LineCollection(valid_segs, lw=0.3, zorder=5, colors=valid_color)
            invalid_col = LineCollection(invalid_segs, lw=0.3, zorder=5, colors=invalid_color)
            ax.add_collection(valid_col)
            ax.add_collection(invalid_col)

            # Ends.
            end_style = dict(s=1**2, zorder=7, marker="o")
            if n_valid > 0:
                valid_ends = np.stack(
                    [pos[ii, -1] for ii, valid_len in enumerate(valid_lens) if valid_len == T], axis=0
                )
                ax.scatter(valid_ends[:, 0], valid_ends[:, 1], color=PlotStyle.valid_color, **end_style)
            if n_invalid > 0:
                invalid_ends = np.stack(invalid_ends, axis=0)
                ax.scatter(invalid_ends[:, 0], invalid_ends[:, 1], color=PlotStyle.invalid_color, **end_style)

            plot_avoid_fn(ax)

            for extra_line in extra_lines:
                if len(extra_line) == 4:
                    bb_X, bb_Y, bb_V_other, color = extra_line
                    ax.contour(
                        bb_X, bb_Y, bb_V_other, levels=[0.0], colors=[color], alpha=0.98, linewidths=1.0, zorder=3.2
                    )
                elif len(extra_line) == 2 and not isinstance(extra_lines[0], dict):
                    if stpidx == 0:
                        bb_V_other, color = extra_line
                        _, bb_X, bb_Y = self._task.get_contour_x0(setup=0)
                        ax.contour(
                            bb_X, bb_Y, bb_V_other, levels=[0.0], colors=[color], alpha=0.98, linewidths=1.0, zorder=3.2
                        )
                else:
                    assert len(extra_line) == 2
                    extra_line_dict, color = extra_line
                    bb_V_other = extra_line_dict[plot2d_name]
                    _, bb_X, bb_Y = self._task.get_contour_x0(setup=stpidx)
                    ax.contour(
                        bb_X, bb_Y, bb_V_other, levels=[0.0], colors=[color], alpha=0.98, linewidths=1.0, zorder=3.2
                    )

            # Starts.
            ax.scatter(pos[:, 0, 0], pos[:, 0, 1], color="black", s=1**2, zorder=6, marker="s")
            ax.set(title="Phase Plot")
            ax.autoscale_view()

            if n_2d_plots > 1:
                tmp_fig_path = _add_prefix(path, plot2d_name, sep="/")
            else:
                tmp_fig_path = path
            self.savefig(fig, tmp_fig_path)

    def density_hist(self, b_x: BTState, path: PathOrStr, setup_idx: int = None):
        assert b_x.ndim == 2, "Must have shape (batch, 2), got {}".format(b_x.shape)

        n_2d_plots = len(self._task.phase2d_setups())
        phase2d_setups = self._task.phase2d_setups()

        if setup_idx is not None:
            phase2d_setups = [phase2d_setups[setup_idx]]
            n_2d_plots = 1

        for plot2d_name, plot_avoid_fn, get2d, _ in phase2d_setups:
            b_pos = get2d(b_x)
            fig, ax = plt.subplots(dpi=self._cfg.dpi)

            h, xedges, yedges = np.histogram2d(b_pos[:, 0], b_pos[:, 1], bins=(64, 64))
            n_colors = h.max() + 1

            # Highlight all values with 0 data in blue.
            cmap_orig = plt.get_cmap(self._cfg.seq_cmap)
            c_vals = np.arange(n_colors)
            c_vals_scaled = c_vals / (n_colors - 1)
            colors = cmap_orig(c_vals_scaled)
            # ggplot pink.
            colors[c_vals <= 5] = mcolors.to_rgba("#FFB5B8")
            # ggplot blue.
            colors[c_vals == 0] = mcolors.to_rgba("#348ABD")
            cmap = mcolors.ListedColormap(colors)

            pc = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap)
            fig.colorbar(pc, ax=ax, shrink=0.8)
            plot_avoid_fn(ax)
            scale_ax_lims(3.0, ax)
            ax.set(title="Density Plot")
            if n_2d_plots > 1:
                tmp_fig_path = _add_prefix(path, plot2d_name, sep="/")
            else:
                tmp_fig_path = path
            self.savefig(fig, tmp_fig_path)

    def get_grid_info(self, ngrid: int):
        if ngrid <= 4:
            nrow, ncol, figsize = 2, 2, np.array([5.33, 4])
        elif ngrid <= 9:
            # nrow, ncol, figsize = 3, 3, (8, 6)
            nrow, ncol, figsize = 3, 3, 1.5 * np.array([8, 6])
        else:
            raise NotImplementedError("")

        return nrow, ncol, tuple(figsize)

    def grid_batch_phase2d(
        self,
        states: Float[na, "ngrid batch T nx"],
        vals: Float[na, "ngrid"],
        fig_path: PathOrStr,
        valname: str = "z",
    ):
        ngrid, batch, T, nx = states.shape
        assert_shape(vals, ngrid)

        if ngrid == 1:
            return self.batch_phase2d(states[0], fig_path)

        nrow, ncol, figsize = self.get_grid_info(ngrid)

        n_2d_plots = len(self._task.phase2d_setups())
        for plot2d_name, plot_avoid_fn, get2d, _ in self._task.phase2d_setups():
            if states.shape[2] == 2:
                pos = states
            else:
                pos = get2d(states)

            fig, axes = plt.subplots(nrow, ncol, figsize=figsize, flat=True, dpi=self._cfg.dpi)
            assert len(axes) >= ngrid

            for grididx, ax in enumerate(axes[:ngrid]):
                this_states, this_pos = states[grididx], pos[grididx]

                col = LineCollection(this_pos, lw=0.2, zorder=5, alpha=0.95)
                ax.add_collection(col)
                # Ends.
                valid_green = "lime"
                ax.scatter(this_pos[:, -1, 0], this_pos[:, -1, 1], color=valid_green, s=1**2, zorder=7, marker="o")

                plot_avoid_fn(ax)

                # Starts.
                ax.scatter(this_pos[:, 0, 0], this_pos[:, 0, 1], color="black", s=1**2, zorder=6, marker="s")

                title_str = "{}={:.2f}".format(valname, vals[grididx])
                ax.set(title=title_str)
                ax.autoscale_view()

            for ax in axes[ngrid:]:
                ax.set_axis_off()

            if n_2d_plots > 1:
                tmp_fig_path = _add_prefix(fig_path, plot2d_name, sep="/")
            else:
                tmp_fig_path = fig_path

            self.savefig(fig, tmp_fig_path)

    def V_seq(
        self,
        Xs: BBFloat,
        Ys: BBFloat,
        Vs: BBFloat,
        path: PathOrStr,
        setup_idx: int = 0,
        levels: int = 11,
        vmax: float = None,
    ):
        Vmin, Vmax = Vs.min(), Vs.max()
        vmax = get_or(vmax, Vmax)
        if Vmax < 0.0:
            norm = Normalize(vmin=Vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=0.0, vmax=vmax)

        fig, ax = plt.subplots(dpi=self._cfg.dpi)
        cs0 = ax.contourf(Xs, Ys, Vs, levels=levels, norm=norm, cmap=self._cfg.seq_cmap_r, zorder=3.5, alpha=0.8)

        # Plot minimum.
        min_idx = np.unravel_index(Vs.argmin(), Vs.shape)
        ax.plot([Xs[min_idx]], [Ys[min_idx]], marker="*")

        if Vs.dtype == bool:
            vmin, vmax = 0, 1
        else:
            vmin, vmax = Vs.min(), Vs.max()
        ax.set_title("x∈[{}, {}]".format(pretty_str(vmin), pretty_str(vmax)))

        self.plot_task(ax, setup_idx)
        self.use_grid(ax)
        fig.colorbar(cs0, ax=ax, shrink=0.8)

        self.savefig(fig, path)

    def V_div(
        self,
        Xs: BBFloat,
        Ys: BBFloat,
        Vs: BBFloat,
        path: PathOrStr,
        setup_idx: int = 0,
        levels: int = 11,
        twoslope: bool = False,
        title: str | None = None,
        extra_lines: list[tuple[BBFloat | dict[str, BBFloat], str]] = {},
        hold: bool = False,
    ):
        Vmin, Vmax = Vs.min(), Vs.max()
        if twoslope:
            norm = two_slope_norm(Vs, Vmin, Vmax)
        else:
            norm = CenteredNorm(vcenter=0.0)

        fig, ax = plt.subplots(dpi=self._cfg.dpi)
        cs0 = ax.contourf(Xs, Ys, Vs, levels=levels, norm=norm, cmap=self._cfg.div_cmap, zorder=3.5, alpha=0.8)
        if Vmin < 0 < Vmax:
            ax.contour(Xs, Ys, Vs, levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0)

        for bb_V_other, color in extra_lines:
            if isinstance(bb_V_other, dict):
                phase2d_name = self._task.phase2d_setups()[setup_idx].plot_name
                bb_V_other = bb_V_other[phase2d_name]
            ax.contour(Xs, Ys, bb_V_other, levels=[0.0], colors=[color], alpha=0.98, linewidths=1.0, zorder=3.2)

        # Plot minimum.
        min_idx = np.unravel_index(Vs.argmin(), Vs.shape)
        ax.plot([Xs[min_idx]], [Ys[min_idx]], marker="*")

        title_str = format_bounds_str(Vmin, Vmax, "x")
        if title is not None:
            title_str = f"{title} - {title_str}"
        ax.set_title(title_str)

        self.plot_task(ax, setup_idx)
        self.use_grid(ax)
        fig.colorbar(cs0, ax=ax, shrink=0.8)

        if not hold:
            self.savefig(fig, path)
        return fig

    def vectorfield(self, bb_Xs: BBFloat, bb_Ys: BBFloat, bb_Vx: BBState, path: PathOrStr, setup_idx: int = 0):
        bb_V_x, bb_V_y = bb_Vx[:, :, 0], bb_Vx[:, :, 1]
        bb_norm = np.linalg.norm(bb_Vx, axis=-1)

        fig, ax = plt.subplots(dpi=self._cfg.dpi)
        streamplotset = ax.streamplot(
            bb_Xs, bb_Ys, bb_V_x, bb_V_y, color=bb_norm, cmap=self._cfg.seq_cmap_r, zorder=3.5
        )

        self.plot_task(ax, setup_idx)
        fig.colorbar(streamplotset.lines, ax=ax, shrink=0.8)

        self.savefig(fig, path)

    def grid_V_seq(
        self,
        Xs: BBFloat,
        Ys: BBFloat,
        grid_Vs: ZBBFloat,
        vals: ZFloat,
        fig_path: PathOrStr,
        valname: str = "z",
        nlevels: int = 9,
        ndifflevels: int = 5,
        mask: BBBool = None,
    ):
        ngrid, ny, nx = grid_Vs.shape
        assert_shape(vals, ngrid)
        assert ngrid == 8

        diff_V = grid_Vs[-1] - grid_Vs[0]

        if grid_Vs.max() < 0:
            logger.error("grid_V_seq, but grid_Vs.max() is negative {}!".format(grid_Vs.max()))
        norm = Normalize(vmin=grid_Vs.min(), vmax=grid_Vs.max())

        diffnorm = two_slope_norm(diff_V)

        levels = np.linspace(norm.vmin, norm.vmax, num=nlevels)
        difflevels = np.linspace(diffnorm.vmin, diffnorm.vmax, num=ndifflevels)

        nrow, ncol, figsize = self.get_grid_info(ngrid)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, flat=True, dpi=self._cfg.dpi)

        # First, plot the Vs.
        for grididx, ax in enumerate(axes[:ngrid]):
            Vs = grid_Vs[grididx]
            if mask is not None:
                Vs = np.ma.array(Vs, mask=mask)
            cs0 = ax.contourf(Xs, Ys, Vs, levels=levels, norm=norm, cmap=self._cfg.seq_cmap_r)
            self.plot_task(ax)
            self.use_grid(ax)

            title_str = "{}={:.2f}".format(valname, vals[grididx])
            ax.set(title=title_str)

        # Finally, plot the diff.
        diffax = axes[-1]
        cs2 = diffax.contourf(Xs, Ys, diff_V, levels=difflevels, norm=diffnorm, cmap=self._cfg.div_cmap)
        cs3 = diffax.contour(Xs, Ys, diff_V, levels=[0.0], colors=["cyan"])
        self.plot_task(diffax)

        # Colorbar for both.
        diffcbar = fig.colorbar(cs2, ax=axes, shrink=0.8)
        diffcbar.ax.set_ylabel("diff", rotation=270)
        diffcbar.add_lines(cs3)

        cbar = fig.colorbar(cs0, ax=axes, shrink=0.8)

        self.savefig(fig, fig_path)

    def grid_V_div(
        self,
        Xs: BBFloat,
        Ys: BBFloat,
        grid_Vs: ZBBFloat,
        vals: ZFloat,
        fig_path: PathOrStr,
        nlevels: int = 9,
        ndifflevels: int = 5,
        valname: str = "z",
        plotzero: bool = True,
        mask: BBBool | None = None,
    ):
        ngrid, ny, nx = grid_Vs.shape
        assert_shape(vals, ngrid)
        assert ngrid == 8

        if mask is not None:
            if mask.shape == (ny, nx):
                mask = ei.repeat(mask, "ny nx -> ngrid ny nx", ngrid=ngrid)
            grid_Vs = np.ma.array(grid_Vs, mask=mask)

        diff_V = grid_Vs[-1] - grid_Vs[0]

        norm = two_slope_norm(grid_Vs)
        diffnorm = two_slope_norm(diff_V)

        levels = np.linspace(norm.vmin, norm.vmax, num=nlevels)
        difflevels = np.linspace(diffnorm.vmin, diffnorm.vmax, num=ndifflevels)

        nrow, ncol, figsize = self.get_grid_info(ngrid)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, flat=True, dpi=self._cfg.dpi)

        # First, plot the Vs.
        for grididx, ax in enumerate(axes[:ngrid]):
            Vs = grid_Vs[grididx]
            cs0 = ax.contourf(Xs, Ys, Vs, levels=levels, norm=norm, cmap=self._cfg.div_cmap)
            if plotzero:
                cs1 = ax.contour(Xs, Ys, grid_Vs[grididx], levels=[0.0], colors=["cyan"])
            self.plot_task(ax)
            self.use_grid(ax)

            title_str = "{}={:.2f}".format(valname, vals[grididx])
            ax.set(title=title_str)

        # Finally, plot the diff.
        diffax = axes[-1]
        cs2 = diffax.contourf(Xs, Ys, diff_V, levels=difflevels, norm=diffnorm, cmap=self._cfg.div_cmap, zorder=3.5)
        cs3 = diffax.contour(Xs, Ys, diff_V, levels=[0.0], colors=["cyan"], zorder=4)
        self.plot_task(diffax)
        self.use_grid(diffax)

        # Colorbar for both.
        diffcbar = fig.colorbar(cs2, ax=axes, shrink=0.8)
        diffcbar.ax.set_ylabel("diff", rotation=270)
        diffcbar.add_lines(cs3)

        cbar = fig.colorbar(cs0, ax=axes, shrink=0.8)
        if plotzero:
            cbar.add_lines(cs1)

        self.savefig(fig, fig_path)

    def grid_hs(self, Xs: BBFloat, Ys: BBFloat, hs: Float[Arr, "b1 b2 nh"], fig_path: PathOrStr, levels: int = 8):
        ny, nx, nh = hs.shape
        assert nh == self._task.nh, f"Expected nh = nh! {nh} {self._task.nh}"

        # Plot argmax for the last one.
        nrow, ncol, figsize = self.get_grid_info(nh + 1)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, flat=True, dpi=self._cfg.dpi)

        # First, plot each h.
        vmin, vmax = hs.min(), hs.max()
        norm = centered_norm(vmin, vmax)

        levels = np.linspace(vmin, vmax, num=levels)

        labels = self._task.h_labels
        for ii, ax in enumerate(axes[:nh]):
            h = hs[:, :, ii]
            cs0 = ax.contourf(Xs, Ys, h, levels=levels, norm=norm, cmap=self._cfg.div_cmap)
            cs1 = ax.contour(Xs, Ys, h, levels=[0.0], colors=[PlotStyle.ZeroColor])
            self.plot_task(ax)
            self.use_grid(ax)

            title_str = format_bounds_str(h.min(), h.max(), labels[ii])
            ax.set(title=title_str)

        # Finally, plot argmax.
        cmap = mcolors.ListedColormap(
            ["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"][:nh]
        )
        h_argmax = hs.argmax(axis=-1)
        diffax = axes[-1]
        argmax_norm = mcolors.Normalize(vmin=0, vmax=nh - 1)
        extent = (Xs.min(), Xs.max(), Ys.min(), Ys.max())
        im = diffax.imshow(
            h_argmax, origin="lower", interpolation="nearest", extent=extent, cmap=cmap, norm=argmax_norm, aspect="auto"
        )
        self.plot_task(diffax)
        self.use_grid(diffax)

        # Colorbar for argmax.
        cbar = fig.colorbar(im, ax=axes, shrink=0.9)
        for ii, label in enumerate(labels):
            pos_start = (ii + 0.5) * len(labels) / (len(labels) + 1.0)
            cbar.ax.text(0.5, pos_start, label, ha="center", va="center")

        # Colorbar for hs.
        cbar = fig.colorbar(cs0, ax=axes, shrink=0.9)

        self.savefig(fig, fig_path)

    def pbb_bound(
        self,
        Xs: BBFloat,
        Ys: BBFloat,
        psafe: BBFloat,
        bound: FloatScalar,
        delta_total: float,
        fig_path: PathOrStr,
        levels: int = 8,
        setup_idx: int = 0,
    ):
        pmin, pmax = psafe.min(), psafe.max()
        norm = CenteredNorm(vcenter=0.5)
        fig, ax = plt.subplots(dpi=self._cfg.dpi)
        cs0 = ax.contourf(Xs, Ys, psafe, levels=levels, norm=norm, cmap=self._cfg.div_cmap, zorder=3.5, alpha=0.8)

        cbar = fig.colorbar(cs0, ax=ax, shrink=0.8)
        ax.set_title("bound: {:8.2e}  (1-p>={:8.2e})".format(bound, delta_total))

        if pmin < bound < pmax:
            cs1 = ax.contour(
                Xs,
                Ys,
                psafe,
                levels=[bound],
                linestyles="--",
                colors=[PlotStyle.BoundColor],
                alpha=0.98,
                linewidths=0.75,
                zorder=5.0,
            )
            plt.setp(cs1.collections, path_effects=[patheffects.withStroke(linewidth=1.75, foreground="k")])
            cbar.add_lines(cs1)

        self.plot_task(ax, setup_idx)
        self.use_grid(ax)
        self.savefig(fig, fig_path)


_RetType = TypeVar("_RetType")
_FnWrap = Callable[[Callable], Callable[[Any], _RetType]]


def _id(x):
    return x


class Plotter(_Plotter):
    def __init__(
        self,
        task: Task,
        rel_path: pathlib.Path | None = None,
        cfg: PlotterCfg = PlotterCfg(),
        fnwrap: _FnWrap = None,
        log: bool = False,
    ):
        super().__init__(task, rel_path, cfg)
        if fnwrap is None:
            fnwrap = _id
        self._fnwrap = fnwrap
        self._log = log

    def run(self, fn, *args, **kwargs) -> _RetType:
        if self._log:
            args_list = list(args)
            for arg in reversed(args_list):
                if isinstance(arg, (pathlib.Path, str)):
                    fig_path = arg
                    logger.info(f"Plotting {fig_path}...")
        return self._fnwrap(fn)(*args, **kwargs)

    def _filt(self, d: dict[str, Any]) -> list[Any]:
        l = list(d.values())
        l = [item for item in l if item is not self]
        return l

    def __getattribute__(self, item: str):
        # Not a function of super, return normally.
        plot_fns = [
            _Plotter.batch_plot,
            _Plotter.grid_batch_plot,
            _Plotter.batch_phase2d,
            _Plotter.grid_batch_phase2d,
            _Plotter.V_seq,
            _Plotter.V_div,
            _Plotter.grid_V_seq,
            _Plotter.grid_V_div,
            _Plotter.grid_hs,
            _Plotter.pbb_bound,
        ]
        underlying = object.__getattribute__(self, item)
        try:
            is_plot_fn = inspect.ismethod(underlying) and underlying in plot_fns
            if not is_plot_fn:
                return underlying
        except:
            return underlying

        def wrap_fn(*args, **kwargs):
            orig_method = object.__getattribute__(self, item)
            return self.run(orig_method, *args, **kwargs)

        return wrap_fn


class MPPlotter(Plotter):
    def __init__(self, task: Task, log_dir: pathlib.Path, cfg: PlotterCfg = PlotterCfg()):
        self.mp_run = MPRun()
        if "DEBUG_PLOT" in os.environ and os.environ["DEBUG_PLOT"] == "1":
            logger.debug("Using sync Plotter for debugging!")
            super().__init__(task, log_dir, cfg)
        else:
            logger.debug("")
            super().__init__(task, log_dir, cfg, fnwrap=self.mp_run.wrap)
