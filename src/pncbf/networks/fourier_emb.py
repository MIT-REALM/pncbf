from typing import Type

import flax.linen as nn
from jax import numpy as jnp
from jax import random as jr


def pos_embed_random(n_feats: int, x, scale: float = 1.0, seed: int = 58122347):
    is_scalar = x.ndim == 0
    if is_scalar:
        x = x[None]
    nx = x.shape[-1]

    key = jr.PRNGKey(seed)
    gauss_mat = scale * jnp.pi * jr.normal(key, (n_feats, nx))
    # (n_feats, )
    coords = jnp.sum(x * gauss_mat, axis=-1)

    # (n_feat * 2, )
    return jnp.concatenate([jnp.sin(coords), jnp.cos(coords)])


class PosEmbed(nn.Module):
    net_cls: Type[nn.Module]
    embed_dim: int = 32
    scale: float = 1.0

    @nn.compact
    def __call__(self, *args, **kwargs):
        out = self.net_cls()(*args, **kwargs)
        out = pos_embed_random(self.embed_dim // 2, out, self.scale)
        return out
