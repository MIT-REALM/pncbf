import matplotlib.pyplot as plt
import numpy as np
from attrs import define

from pncbf.dyn.dyn_types import State
from pncbf.utils.jax_types import Vec2


class CircArtist(plt.Artist):
    def __init__(
        self,
        radius: float,
        color: str,
        zorder: float,
        lines: list[float],
        line_alpha: float = 0.5,
        show_v: bool = True,
        show_nom: bool = False,
        show_control: bool = False,
        v_length_s: float = 2.0,
        rot: bool = False,
    ):
        super().__init__()
        self.radius = radius
        self.color = color
        self._lines = lines
        self.line_alpha = line_alpha

        self.show_v = show_v
        self.show_nom = show_nom
        self.show_control = show_control

        self.rot = rot

        # How long to draw the velocity vector, in seconds.
        self.v_length_s = v_length_s

        dummy_state = np.zeros(4)
        self.obs, self.lines, self.vecs = self._get_artists(dummy_state)
        self.set_zorder(zorder)

    @property
    def artists(self) -> list[plt.Artist]:
        return [*self.lines, self.obs, *self.vecs.values()]

    def set_figure(self, fig: plt.Figure):
        [artist.set_figure(fig) for artist in self.artists]

    def set_transform(self, t):
        [artist.set_transform(t) for artist in self.artists]

    def draw(self, renderer):
        [artist.draw(renderer) for artist in self.artists]

    def _get_artists(self, state: State):
        assert state.shape == (4,)
        pos = state[:2]
        obs_circ = plt.Circle(pos, self.radius, alpha=0.8, color=self.color)

        # Add a circle for each line.
        lines = []
        for radius in self._lines:
            lines.append(plt.Circle(pos, radius, color=self.color, alpha=self.line_alpha, fill=False))

        data = np.stack([pos, pos], axis=0)
        vecs = {}
        if self.show_v:
            vecs["vel"] = plt.Line2D(data[:, 0], data[:, 1], color=self.color, alpha=0.9, linestyle="-")
        if self.show_nom:
            vecs["nom"] = plt.Line2D(data[:, 0], data[:, 1], color="C3", alpha=0.9, linestyle="-")
        if self.show_control:
            vecs["control"] = plt.Line2D(data[:, 0], data[:, 1], color="C0", alpha=0.9, linestyle="--")

        return obs_circ, lines, vecs

    def rotate(self, vec: Vec2) -> Vec2:
        if not self.rot:
            return vec

        R_rot_W = np.array([[0, -1], [1, 0]])
        return R_rot_W @ vec

    def update_state(self, state: np.ndarray):
        assert state.shape == (4,)
        pos = self.rotate(state[:2])
        self.obs.set_center(pos)

        for line in self.lines:
            line.set_center(pos)

        # Update velocity vector.
        if self.show_v:
            dvel = state[2:4]
            dpos = self.rotate(dvel * self.v_length_s)
            data = np.stack([pos, pos + dpos], axis=0)
            self.vecs["vel"].set_data(data[:, 0], data[:, 1])

        self.stale = True

    def update_vecs(self, nom: Vec2 = None, control: Vec2 = None):
        # Should already be rotated.
        pos = self.obs.center
        if nom is not None:
            assert self.show_nom
            dpos = self.rotate(nom * self.v_length_s)
            data = np.stack([pos, pos + dpos], axis=0)
            self.vecs["nom"].set_data(data[:, 0], data[:, 1])

        if control is not None:
            assert self.show_control
            dpos = self.rotate(control * self.v_length_s)
            data = np.stack([pos, pos + dpos], axis=0)
            self.vecs["control"].set_data(data[:, 0], data[:, 1])
