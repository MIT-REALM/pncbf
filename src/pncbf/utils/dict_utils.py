from typing import TypeVar

from pncbf.utils.jax_types import Arr
from pncbf.utils.none import get_or

_Dict = TypeVar("_Dict", bound=dict)


def modify_dict_keys(d: _Dict, *, pre: str | None = None, post: str | None = None) -> _Dict:
    pre, post = get_or(pre, ""), get_or(post, "")
    return {f"{pre}{k}{post}": v for k, v in d.items()}


def subdict_from_prefix(d: _Dict, pre: str, keep_prefix: bool = False) -> _Dict:
    """Extract a sub dictionary from an existing dictionary based on a prefix."""
    assert isinstance(pre, str)
    start_idx = 0 if keep_prefix else len(pre)
    return {k[start_idx:]: v for k, v in d.items() if k.startswith(pre)}


def dict_index(d: dict[str, Arr], idx: int):
    return {k: v[idx] for k, v in d.items()}
