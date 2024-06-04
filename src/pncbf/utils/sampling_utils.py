import einops as ei
import numpy as np
from jaxtyping import Float

from pncbf.dyn.dyn_types import BBState, State
from pncbf.utils.jax_types import Arr


def get_mesh_np(
    lims: Float[Arr, "2 nx"], idxs: tuple[int, int], n_xs: int, n_ys: int, nominal_state: State
) -> tuple[BBState, BBState, BBState]:
    x_idx, y_idx = idxs
    assert lims.shape[0] == 2
    xlims = lims[:, x_idx]
    ylims = lims[:, y_idx]
    assert xlims.shape == ylims.shape == (2,)

    xs = np.linspace(xlims[0], xlims[1], num=n_xs)
    ys = np.linspace(ylims[0], ylims[1], num=n_ys)

    Xs, Ys = np.meshgrid(xs, ys)

    (nx,) = nominal_state.shape

    bb_nominal = ei.repeat(nominal_state, "nx -> nys nxs nx", nys=n_ys, nxs=n_xs)
    bb_nominal[:, :, x_idx] = Xs
    bb_nominal[:, :, y_idx] = Ys
    assert bb_nominal.shape == (n_ys, n_xs, nx)

    return Xs, Ys, bb_nominal
