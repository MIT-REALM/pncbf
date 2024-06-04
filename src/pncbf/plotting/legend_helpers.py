import matplotlib.pyplot as plt


def lline(color: str, **kwargs) -> plt.Line2D:
    return plt.Line2D([0], [0], color=color, **kwargs)
