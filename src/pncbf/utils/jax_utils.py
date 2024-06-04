import functools as ft
from typing import Any, Callable, Iterable, ParamSpec, Sequence, TypeVar

import einops as ei
import ipdb
import jax._src.dtypes
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax._src.lib import xla_client as xc
from jax._src.typing import ArrayLike
from loguru import logger

import pncbf.utils.jax_types
from pncbf.utils.jax_types import Arr, BFloat, BoolScalar, FloatScalar, Shape

_PyTree = TypeVar("_PyTree")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]


def jax_use_double():
    jax.config.update("jax_enable_x64", True)
    pncbf.utils.jax_types.set_current_float(np.float64)


def jax_default_x32():
    jax.config.update("jax_default_dtype_bits", "32")
    reset_default_types()


def reset_default_types():
    is_32 = jax.config.jax_default_dtype_bits == "32"
    jax._src.dtypes.int_ = np.int32 if is_32 else np.int64
    jax._src.dtypes.uint = np.uint32 if is_32 else np.uint64
    jax._src.dtypes.float_ = np.float32 if is_32 else np.float64
    jax._src.dtypes.complex_ = np.complex64 if is_32 else np.complex128
    jax._src.dtypes._default_types = {
        "b": jax._src.dtypes.bool_,
        "i": jax._src.dtypes.int_,
        "u": jax._src.dtypes.uint,
        "f": jax._src.dtypes.float_,
        "c": jax._src.dtypes.complex_,
    }
    dtypes = jax._src.dtypes

    jax.numpy.int_ = jnp.int32 if dtypes.int_ == np.int32 else jnp.int64
    jax.numpy.uint = jnp.uint32 if dtypes.uint == np.uint32 else jnp.uint64
    jax.numpy.float_ = jnp.float32 if dtypes.float_ == np.float32 else jnp.float64
    jax.numpy.complex_ = jnp.complex64 if dtypes.complex_ == np.complex64 else jnp.complex128


def jax_use_cpu() -> None:
    ctx = jax.default_device(get_cpu_device())
    ctx.__enter__()


def merge01(x):
    return ei.rearrange(x, "n1 n2 ... -> (n1 n2) ...")


def swaplast(x):
    return ei.rearrange(x, "... n1 n2 -> ... n2 n1")


def jax_vmap(fn: _Fn, in_axes: int | Sequence[Any] = 0, out_axes: Any = 0, rep: int = None) -> _Fn:
    if rep is not None:
        return rep_vmap(fn, rep=rep, in_axes=in_axes, out_axes=out_axes)

    return jax.vmap(fn, in_axes, out_axes)


def jax_jit(
    fn: _Fn,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] = (),
    device: xc.Device = None,
    *args,
    **kwargs,
) -> _Fn:
    return jax.jit(
        fn,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        device=device,
        *args,
        **kwargs,
    )


def jax_jit_np(
    fn: _Fn,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] = (),
    device: xc.Device = None,
    *args,
    **kwargs,
):
    jit_fn = jax_jit(fn, static_argnums, static_argnames, donate_argnums, device, *args, **kwargs)

    def wrapper(*args, **kwargs):
        return jax2np(jit_fn(*args, **kwargs))

    return wrapper


def rep_vmap(fn: _Fn, rep: int, in_axes: int | Sequence[Any] = 0, **kwargs) -> _Fn:
    for ii in range(rep):
        fn = jax.vmap(fn, in_axes=in_axes, **kwargs)
    return fn


def tree_index0(tree: _PyTree, idx: int) -> _PyTree:
    return jtu.tree_map(lambda x: x[idx], tree)


def tree_index(tree: _PyTree, idx: int | np.ndarray | Any, in_axes: int | Sequence[Any] | Any):
    def index_vmap_body(in_axes_spec: int | None, partial_pytree: _PyTree) -> _PyTree:
        if in_axes_spec is None:
            return partial_pytree

        return partial_pytree[idx]

    indexed_tree = jtu.tree_map(index_vmap_body, in_axes, tree, is_leaf=lambda x: x is None)
    return indexed_tree


def concat_at_end(arr1: Arr, arr2: Arr, axis: int) -> Arr:
    """
    :param arr1: (T, nx)
    :param arr2: (nx, )
    :param axis: Which axis for arr1 to concat under.
    :return: (T + 1, nx) with [arr1 arr2]
    """
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr1_shape = list(arr1.shape)
    del arr1_shape[axis]
    assert np.all(np.array(arr1_shape) == np.array(arr2.shape))

    return jnp.concatenate([arr1, jnp.expand_dims(arr2, axis=axis)], axis=axis)


def tree_concat_at_end(tree1: _PyTree, tree2: _PyTree, axis: int) -> _PyTree:
    def tree_concat_at_end_inner(arr1: Arr, arr2: Arr):
        return concat_at_end(arr1, arr2, axis=axis)

    return jtu.tree_map(tree_concat_at_end_inner, tree1, tree2)


def partial_dict_concat_at_end(d1: dict[str, Arr], d2: dict[str, Arr], axis: int) -> dict[str, Arr]:
    """Version of tree_concat_at_end that works when d2 is only a subset of d1."""
    out_dict = d1.copy()
    for k in d2.keys():
        if k not in d1:
            continue
        out_dict[k] = tree_concat_at_end(d1[k], d2[k], axis=axis)
    return out_dict


def tree_copy(tree: _PyTree) -> _PyTree:
    return jtu.tree_map(lambda x: x.copy(), tree)


def tree_add(tree1: _PyTree, tree2: _PyTree, alpha: float) -> _PyTree:
    def tree_add_inner(arr1: Arr, arr2: Arr):
        return arr1 + alpha * arr2

    return jtu.tree_map(tree_add_inner, tree1, tree2)


def tree_add_multi(tree1: _PyTree, others: list[_PyTree], coeffs: list[float]) -> _PyTree:
    if len(others) != len(coeffs):
        raise ValueError(f"len(others) {len(others)} should be same as len(coeffs) {coeffs}")

    def tree_add_multi_inner(arr1: Arr, *arrs: Arr):
        for coeff, arr in zip(coeffs, arrs):
            arr1 = arr1 + coeff * arr
        return arr1

    return jtu.tree_map(tree_add_multi_inner, tree1, *others)


def tree_map(f: Callable[..., Any], tree: _PyTree, *rest: _PyTree, is_leaf: Any = None) -> _PyTree:
    return jtu.tree_map(f, tree, *rest, is_leaf=is_leaf)


def tree_where(cond: BoolScalar | bool, true_val: _PyTree, false_val: _PyTree) -> _PyTree:
    if isinstance(cond, bool):
        cond = jnp.array(cond)

    def tree_where_inner(true_arr, false_arr):
        # Add dims appropriately.
        assert true_arr.shape == false_arr.shape, f"Shapes {true_arr.shape} {false_arr.shape} don't match"
        n_data_dim = len(true_arr.shape[1:])
        batched_cond = cond.reshape(cond.shape + (1,) * n_data_dim)
        return jnp.where(batched_cond, true_arr, false_arr)

    return jtu.tree_map(tree_where_inner, true_val, false_val)


def tree_stack(trees: list[_PyTree]) -> _PyTree:
    def tree_stack_inner(*arrs: Arr) -> Arr:
        arrs = list(arrs)
        if isinstance(arrs[0], np.ndarray):
            return np.stack(arrs, axis=0)
        return jnp.stack(arrs, axis=0)

    return jtu.tree_map(tree_stack_inner, *trees)


def tree_unstack(b_tree: _PyTree) -> list[_PyTree]:
    def get_idx(idx: int):
        def get_idx_inner(arr: Arr):
            return arr[idx]

        return get_idx_inner

    # First, see how many trees we end up with.
    tmp_leaves, treedef = jtu.tree_flatten(b_tree)
    n_output_trees = len(tmp_leaves[0])

    list_of_trees = [jtu.tree_map(get_idx(ii), b_tree) for ii in range(n_output_trees)]
    return list_of_trees


def tree_cat(trees: list[_PyTree], axis: int = 0) -> _PyTree:
    def tree_cat_inner(*arrs: Arr) -> Arr:
        arrs = list(arrs)
        if isinstance(arrs[0], np.ndarray):
            return np.concatenate(arrs, axis=axis)
        return jnp.concatenate(arrs, axis=axis)

    return jtu.tree_map(tree_cat_inner, *trees)


def tree_split(tree: _PyTree, indices_or_sections: int | ArrayLike, axis: int = 0) -> list[_PyTree]:
    def get_idx(idx: int):
        def tree_split_inner(arr: Arr):
            return jnp.split(arr, indices_or_sections, axis=axis)[idx]

        return tree_split_inner

    # First, see how many trees we end up with.
    tmp_leaves, treedef = jtu.tree_flatten(tree)
    n_output_trees = len(jnp.split(tmp_leaves[0], indices_or_sections, axis=axis))

    list_of_trees = [jtu.tree_map(get_idx(ii), tree) for ii in range(n_output_trees)]
    return list_of_trees


def unmergelast(arr: Arr, shape01: tuple[int, int]) -> Arr:
    assert arr.shape[-1] == np.prod(shape01)
    return arr.reshape(arr.shape[:-1] + shape01)


def unmerge01(arr: Arr, shape01: tuple[int, int]) -> Arr:
    assert arr.shape[0] == np.prod(shape01)
    return arr.reshape(shape01 + arr.shape[1:])


def tree_unmerge01(tree: _PyTree, shape01: tuple[int, int]) -> _PyTree:
    return jtu.tree_map(ft.partial(unmerge01, shape01=shape01), tree)


def tree_len(tree: _PyTree) -> int:
    leaves, treedef = jtu.tree_flatten(tree)
    return len(leaves[0])


def tree_combine_dims(tree: _PyTree, start: int = 0, count: int = 2) -> _PyTree:
    def tree_combine_dims_inner(arr: Arr) -> Arr:
        s = arr.shape
        target_shape = s[:start] + (-1,) + s[start + count :]
        return arr.reshape(target_shape)

    return jtu.tree_map(tree_combine_dims_inner, tree)


def tree_split_dims(tree: _PyTree, new_dims: Shape) -> _PyTree:
    prod_dims = np.prod(new_dims)

    def tree_split_dims_inner(arr: Arr) -> Arr:
        assert arr.shape[0] == prod_dims
        target_shape = new_dims + arr.shape[1:]
        return arr.reshape(target_shape)

    return jtu.tree_map(tree_split_dims_inner, tree)


def tree_any(fn: Callable[[Arr], bool], tree: _PyTree) -> bool:
    leaves, treedef = jtu.tree_flatten(tree)
    return jnp.any(jnp.stack([fn(leaf) for leaf in leaves]))


def normalize_vec(vec: BFloat) -> BFloat:
    assert vec.ndim == 1
    return vec / jnp.linalg.norm(vec)


def is_treedef_same(tree1: _PyTree, tree2: _PyTree) -> bool:
    _, treedef1 = jtu.tree_flatten(tree1)
    _, treedef2 = jtu.tree_flatten(tree2)
    return treedef1 == treedef2


def assert_treedef_same(tree1: _PyTree, tree2: _PyTree) -> None:
    _, treedef1 = jtu.tree_flatten(tree1)
    _, treedef2 = jtu.tree_flatten(tree2)
    if treedef1 != treedef2:
        raise ValueError(f"Treedef are different! Types: left={type(tree1).__name__}, right={type(tree2).__name__}")


def get_cpu_device():
    return jax.devices("cpu")[0]


def get_gpu_device():
    return jax.devices("gpu")[0]


def jax2cpu(pytree: _PyTree) -> _PyTree:
    return jax.device_put(pytree, get_cpu_device())


def jax2np(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(np.array, pytree)


def in_bounds(arr, lb, ub) -> BoolScalar:
    return (lb <= arr) & (arr <= ub)


def normalize_minmax(x, x_min, x_max):
    """Normalize x from [x_min, x_max] to [-1, +1]."""
    zero_one = (x - x_min) / (x_max - x_min)
    return 2.0 * zero_one - 1.0


def smoothmax(xs: BFloat, t: float) -> FloatScalar:
    return t * jnn.logsumexp(xs / t, axis=0)


def signif(x, p: int):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
