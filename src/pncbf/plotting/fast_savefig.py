import pathlib

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def mplfig_to_npimage(fig):
    """Converts a matplotlib figure to a RGB frame after updating the canvas"""
    #  only the Agg backend now supports the tostring_rgb function
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # update/draw the elements

    # get the width and the height to resize the matrix
    l, b, w, h = canvas.figure.bbox.bounds
    w, h = int(w), int(h)

    #  exports the canvas to a string buffer and then to a numpy nd.array
    buf = canvas.tostring_rgb()
    image = np.frombuffer(buf, dtype=np.uint8)
    return image.reshape((h, w, 3))


def fast_savefig(fig: plt.Figure, path: pathlib.Path):
    np_img = mplfig_to_npimage(fig)
    Image.fromarray(np_img).save(path)
