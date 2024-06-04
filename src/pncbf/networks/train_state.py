from typing import Any, Callable, Concatenate, Generic, ParamSpec, TypeVar

import flax.linen as nn
import optax
from flax import core, struct
from typing_extensions import Self

from pncbf.utils.jax_types import FloatScalar
from pncbf.utils.rng import PRNGKey

_Params = TypeVar("_Params")

_P = ParamSpec("_P")
_R = TypeVar("_R")
# i.e. a function that
_ApplyFn = Callable[Concatenate[_Params, _P], _R]


class ParamState(Generic[_P], struct.PyTreeNode):
    """Non-trainable version of TrainState."""

    apply_fn: _ApplyFn = struct.field(pytree_node=False)
    params: _Params


class TrainState(Generic[_R], struct.PyTreeNode):
    """Custom version of flax.training.TrainState but with better type information."""

    step: int
    apply_fn: _ApplyFn = struct.field(pytree_node=False)
    params: _Params
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState | optax.InjectHyperparamsState

    def apply(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.apply_fn(self.params, *args, **kwargs)

    def apply_gradients(self, grads: _Params, **kwargs) -> "TrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

    @property
    def lr(self) -> FloatScalar:
        hyperparams = self.opt_state.hyperparams
        lr_keys = ["lr", "learning_rate"]
        for key in lr_keys:
            if key in hyperparams:
                return hyperparams[key]
        raise KeyError(f"Couldn't find lr key in hyperparams! keys: {hyperparams.keys()}")

    @classmethod
    def create(cls, apply_fn: _ApplyFn, params: _Params, tx: optax.GradientTransformation, **kwargs) -> "TrainState":
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = None
        if tx is not None:
            opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    @classmethod
    def create_from_def(
        cls, key: PRNGKey, net_def: nn.Module, init_args: tuple, tx: optax.GradientTransformation, **kwargs
    ) -> "TrainState":
        params = net_def.init(key, *init_args)
        return cls.create(net_def.apply, params, tx, **kwargs)

    def strip(self) -> "TrainState":
        """Remove tx and opt_state."""
        return self.replace(tx=None, opt_state=None)
