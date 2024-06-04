from matplotlib.colors import CenteredNorm


def centered_norm(vmin: float | list[float], vmax: float | list[float]):
    if isinstance(vmin, list):
        vmin = min(vmin)
    if isinstance(vmax, list):
        vmin = max(vmax)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(0, halfrange)
