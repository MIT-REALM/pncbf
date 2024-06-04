import functools as ft

import flax.linen as nn
from flax.linen import initializers

from pncbf.networks.network_utils import ActFn, HidSizes, default_nn_init, scaled_init
from pncbf.utils.iter_utils import signal_last_enumerate
from pncbf.utils.jax_types import AnyFloat


class MLP(nn.Module):
    hid_sizes: HidSizes
    act: ActFn = nn.relu
    use_layernorm: bool = False
    act_final: bool = True
    scale_final: float | None = None

    @nn.compact
    def __call__(self, x: AnyFloat, apply_dropout: bool = False) -> AnyFloat:
        nn_init = default_nn_init
        for is_last_layer, ii, hid_size in signal_last_enumerate(self.hid_sizes):
            kernel_init, bias_init = nn_init(), initializers.zeros_init()
            if is_last_layer:
                if self.scale_final is not None:
                    kernel_init = scaled_init(kernel_init, self.scale_final)

            x = nn.Dense(hid_size, kernel_init=kernel_init, bias_init=bias_init)(x)

            no_activation = is_last_layer and not self.act_final
            if not no_activation:
                if self.use_layernorm:
                    x = nn.LayerNorm()(x)
                x = self.act(x)
        return x


def mlp_partial(
    hid_sizes: HidSizes,
    act: ActFn = nn.relu,
    use_layernorm: bool = False,
    act_final: bool = True,
    scale_final: float = None,
):
    def fn():
        return MLP(hid_sizes, act, use_layernorm, act_final, scale_final)

    return fn
