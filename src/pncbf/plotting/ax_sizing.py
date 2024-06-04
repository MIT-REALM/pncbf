import einops as ei
import numpy as np
from jaxtyping import Float
from matplotlib import pyplot as plt

from pncbf.utils.jax_types import Arr


def expand_ax_to_fit(pos: Float[Arr, "... 2"], ax: plt.Axes, pad: float = 0.05):
    flat_pos = ei.rearrange(pos, "... two -> (...) two")
    x_min, y_min = np.min(flat_pos, axis=0)
    x_max, y_max = np.max(flat_pos, axis=0)
    dx, dy = x_max - x_min, y_max - y_min

    x_min2, x_max2 = x_min - pad * dx, x_max + pad * dx
    y_min2, y_max2 = y_min - pad * dy, y_max + pad * dy

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xlim = min(x_min2, xlim[0]), max(x_max2, xlim[1])
    ylim = min(y_min2, ylim[0]), max(y_max2, ylim[1])

    ax.set(xlim=xlim, ylim=ylim)


def scale_ax_lims(scale: float, ax: plt.Axes):
    xlim, ylim = np.array(ax.get_xlim()), np.array(ax.get_ylim())
    ax.set(xlim=scale * xlim, ylim=scale * ylim)
