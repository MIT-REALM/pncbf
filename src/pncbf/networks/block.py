import flax.linen as nn
from flax.linen import initializers

from pncbf.networks.network_utils import ActFn, HidSizes, default_nn_init, scaled_init
from pncbf.utils.iter_utils import signal_last_enumerate
from pncbf.utils.jax_types import AnyFloat


class Block(nn.Module):
    hid_size: int
    act: ActFn = nn.relu

    @nn.compact
    def __call__(self, x: AnyFloat) -> AnyFloat:
        x_size = x.shape[-1]

        nn_init = default_nn_init
        out = nn.Dense(self.hid_size, kernel_init=nn_init())(x)
        out = nn.LayerNorm()(out)
        out = nn.Dense(self.hid_size, kernel_init=nn_init())(out)
        out = self.act(out)
        out = nn.Dense(x_size, kernel_init=nn_init())(out)

        return x + out


class TmpNet(nn.Module):
    blocks: int
    hid_size: int
    act: ActFn = nn.relu
    act_final: bool = True

    @nn.compact
    def __call__(self, x: AnyFloat) -> AnyFloat:
        nn_init = default_nn_init
        out = nn.Dense(self.hid_size, kernel_init=nn_init())(x)

        for ii in range(self.blocks):
            out = Block(self.hid_size, self.act)(out)

        if self.act_final:
            out = self.act(out)

        return out
