import datetime
import pathlib
import pickle
from typing import Any

import ipdb
import jax
import orbax
import orbax.checkpoint

from pncbf.utils.path_utils import mkdir
import numpy as np


def get_ckpt_manager(ckpt_dir: pathlib.Path, max_to_keep: int = 100):
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=max_to_keep, keep_time_interval=datetime.timedelta(minutes=5), create=True
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointHandler()
    async_checkpointer = orbax.checkpoint.AsyncCheckpointer(checkpointer, timeout_secs=50)
    ckpt_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, async_checkpointer, options)
    return ckpt_manager


def get_ckpt_manager_sync(ckpt_dir: pathlib.Path, max_to_keep: int = 5):
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=max_to_keep, keep_time_interval=datetime.timedelta(minutes=5), create=True
    )
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, orbax_checkpointer, options)
    return ckpt_manager


def get_create_args_path(ckpt_dir: pathlib.Path):
    return ckpt_dir / "create_args.pkl"


def get_ckpt_dir_from_path(ckpt_path: pathlib.Path):
    # Either ckpt_path points to ckpts, or it points to a subdirectory in ckpts.
    if ckpt_path.name == "ckpts":
        return ckpt_path

    assert ckpt_path.parent.name == "ckpts"
    return ckpt_path.parent


def save_create_args(ckpt_dir: pathlib.Path, create_args: list[Any]):
    create_args_path = get_create_args_path(ckpt_dir)
    mkdir(create_args_path.parent)
    with open(create_args_path, "wb") as f:
        pickle.dump(create_args, f)


def load_create_args(ckpt_path: pathlib.Path):
    ckpt_dir = get_ckpt_dir_from_path(ckpt_path)
    create_args_path = get_create_args_path(ckpt_dir)
    with open(create_args_path, "rb") as f:
        return pickle.load(f)
