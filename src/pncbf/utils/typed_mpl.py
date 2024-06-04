from typing import Literal, overload

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


@overload
def subplots(**kwargs) -> tuple[Figure, Axes]:
    ...


@overload
def subplots(nrows: int, figsize: tuple[float, float] | None = None, **kwargs) -> tuple[Figure, list[Axes]]:
    ...


@overload
def subplots(nrows: Literal[1], ncols: int, **kwargs) -> tuple[Figure, list[Axes]]:
    ...


@overload
def subplots(nrows: int, ncols: int, flat: Literal[True], **kwargs) -> tuple[Figure, list[Axes]]:
    ...


def subplots(nrows: int = 1, ncols: int = 1, *args, flat: bool = False, constrained_layout=True, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, *args, **kwargs, constrained_layout=constrained_layout)
    if flat:
        assert isinstance(axes, np.ndarray)
        axes = axes.ravel().tolist()
    return fig, axes


def show(*args, **kwargs):
    plt.show(*args, **kwargs)


def close(fig: Figure) -> None:
    plt.close(fig)


def get_cmap(name: str, lut: int | None = None) -> Colormap:
    return plt.get_cmap(name, lut)


def setp(obj: plt.Artist | list[plt.Artist], *args, **kwargs):
    return plt.setp(obj, *args, **kwargs)


def legend(*args, **kwargs):
    return plt.legend(*args, **kwargs)
