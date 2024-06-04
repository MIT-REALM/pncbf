import jax.numpy as jnp
import numpy as np

from pncbf.dyn.dyn_types import State


class AngleEncoder:
    def __init__(self, nx: int, angle_dims: list[int]):
        self._nx = nx
        self._angle_dims = np.array(angle_dims)
        angle_mask = np.zeros(nx, dtype=bool)
        angle_mask[self._angle_dims] = True
        self._angle_mask = angle_mask
        self._normal_mask = ~angle_mask

        if len(angle_dims) == 0:
            raise ValueError("Don't use AngleEncoder if len(angle_dims) == 0! {}".format(angle_dims))

    def encode_state(self, state: State):
        assert state.shape == (self._nx,)

        normal_vals = state[self._normal_mask]
        assert normal_vals.shape == (self.nx - self.n_angles,)

        encoded_angles = self.encode_angles(state)
        enc_state = jnp.concatenate([normal_vals, encoded_angles], axis=0)
        assert enc_state.shape == (self.nx_encoded,)

        return enc_state

    def encode_angles(self, state: State):
        assert state.shape == (self._nx,)
        angle_vals = state[self._angle_mask]
        assert angle_vals.shape == (self.n_angles,)
        # (n_angles, 2)
        encoded_angles = jnp.stack([jnp.cos(angle_vals), jnp.sin(angle_vals)], axis=1)
        # (n_angles * 2, )
        encoded_angles = encoded_angles.flatten()
        assert encoded_angles.shape == (self.n_angles * 2,)
        return encoded_angles

    def encoded_state_labels(self, x_labels: list[str]) -> list[str]:
        assert len(x_labels) == self._nx
        labels = []
        angle_labels = []

        for ii, label in enumerate(x_labels):
            if ii in self._angle_dims:
                angle_labels.extend([rf"$\cos${label}", rf"$\sin${label}"])
            else:
                labels.append(label)

        labels.extend(angle_labels)
        return labels

    @property
    def n_angles(self) -> int:
        return len(self._angle_dims)

    @property
    def angle_dims(self) -> list[int]:
        return self._angle_dims.tolist()

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def nx_encoded(self) -> int:
        return self._nx + self.n_angles
