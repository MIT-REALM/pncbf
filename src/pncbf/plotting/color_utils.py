import colorsys

import numpy as np
from colour import Color
from matplotlib.colors import to_rgb


def reduce_sat(color, factor: float = 0.8) -> str:
    rgb = to_rgb(color)
    color = Color(rgb=rgb)
    sat = color.get_saturation()
    color.set_saturation(sat * factor)
    return color.get_hex()


def change_value(color, factor: float) -> str:
    rgb = to_rgb(color)
    hsv = np.array(colorsys.rgb_to_hsv(*rgb))
    hsv[2] *= factor
    hsv[2] = np.clip(hsv[2], 0, 1)
    rgb = colorsys.hsv_to_rgb(*hsv)
    return Color(rgb=rgb).get_hex()


def blend_colors(color1, color2, frac: float):
    rgb1 = to_rgb(color1)
    rgb2 = to_rgb(color2)
    rgb = np.array(rgb1) * (1 - frac) + np.array(rgb2) * frac
    return Color(rgb=rgb).get_hex()
