import einops as ei
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.collections import LineCollection

from pncbf.utils.jax_types import Arr, Float
from pncbf.utils.shape_utils import assert_shape


def ez_line_collection(
    vals: Float[Arr, "b T"] | Float[Arr, "b T 2"], *, ax: plt.Axes | None = None, **kwargs
) -> LineCollection:
    if vals.ndim == 2:
        b, T = vals.shape
        b_ts = ei.repeat(np.arange(T), "T -> b T", b=b)
        segs = assert_shape(np.stack([b_ts, vals], axis=-1), (b, T, 2))
    else:
        assert vals.ndim == 3 and vals.shape[-1] == 2
        segs = vals

    kwargs = dict(kwargs)
    if "color" in kwargs:
        # logger.error("[ez_line_collection] color should instead be colors")
        colors = kwargs.pop("color")
        kwargs["colors"] = colors
    if "lw" in kwargs:
        # logger.error("[ez_line_collection] lw should instead be linewidths")
        linewidths = kwargs.pop("lw")
        kwargs["linewidths"] = linewidths

    col = LineCollection(segs, **kwargs)
    if ax is not None:
        ax.add_collection(col)
    return col
