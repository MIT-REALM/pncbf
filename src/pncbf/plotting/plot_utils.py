import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from pncbf.utils.jax_types import Arr


def plot_boundaries(axes: list[plt.Axes], bounds: Float[Arr, "2 nx"], color="C3", alpha=0.25):
    for ii, ax in enumerate(axes):
        ymin, ymax = ax.get_ylim()
        lb, ub = bounds[:, ii]

        if ymin < lb:
            ax.axhspan(ymin, lb, color=color, alpha=alpha)
        if ub < ymax:
            ax.axhspan(ub, ymax, color=color, alpha=alpha)


def plot_boundaries_with_clip(
    axes: list[plt.Axes], bounds: Float[Arr, "2 nx"], color="C3", alpha=0.25, expand_frac=0.1
):
    for ii, ax in enumerate(axes):
        ymin, ymax = ax.get_ylim()
        lb, ub = bounds[:, ii]
        width = ub - lb

        # Clip the axis limits to the bounds.
        ymin = max(ymin, lb - expand_frac * width)
        ymax = min(ymax, ub + expand_frac * width)

        if ymin < lb:
            ax.axhspan(ymin, lb, color=color, alpha=alpha)
        if ub < ymax:
            ax.axhspan(ub, ymax, color=color, alpha=alpha)

        ax.set_ylim(ymin, ymax)