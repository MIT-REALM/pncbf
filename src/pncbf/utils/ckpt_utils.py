import pathlib
from typing import TypeVar

import orbax
import orbax.checkpoint
from flax import io, struct
from flax.training import checkpoints, orbax_utils
from flax.training.checkpoints import ORBAX_CKPT_FILENAME, natural_sort
from loguru import logger

from pncbf.training.ckpt_manager import get_ckpt_manager_sync

_Model = TypeVar("_Model", bound=struct.PyTreeNode)


def latest_checkpoint_step(ckpt_dir: pathlib.Path, prefix: str = "") -> int:
    """Retrieve the path of the latest checkpoint in a directory.

    Args:
      ckpt_dir: str: directory of checkpoints to restore from.
      prefix: str: name prefix of checkpoint files.

    Returns:
      The latest checkpoint path or None if no checkpoints were found.
    """
    checkpoint_files = list(ckpt_dir.iterdir())
    checkpoint_files = [str(f) for f in checkpoint_files if f.is_dir()]
    checkpoint_files = natural_sort(checkpoint_files)
    if not checkpoint_files:
        raise ValueError("Didn't find any checkpoint files!")

    latest = pathlib.Path(checkpoint_files[-1])
    # Extract step from it.
    return int(latest.name[len(prefix) :])


def get_run_path_from_ckpt(ckpt_path: pathlib.Path) -> pathlib.Path:
    # ckpt_path either points to the ckpts folder, or a ckpt within the ckpts folder.
    if ckpt_path.name == "ckpts":
        run_path = ckpt_path.parent
        assert (run_path / "plots").exists(), f"plots folder doesn't exist? {run_path}"
        return run_path
    elif (ckpt_path / "plots").exists() and (ckpt_path / "ckpts").exists():
        return ckpt_path

    return get_run_path_from_ckpt(ckpt_path.parent)


def get_id_from_ckpt(ckpt_path: pathlib.Path) -> str:
    assert io.isdir(ckpt_path)
    if io.exists(ckpt_path / "default" / ORBAX_CKPT_FILENAME):
        # This means the given dir is an orbax checkpoint.
        step = orbax.checkpoint.utils.step_from_checkpoint_name(ckpt_path.name)
    else:
        step = latest_checkpoint_step(ckpt_path)

    return "_{:07d}".format(step)


def load_ckpt_with_step(model: _Model, ckpt_path: pathlib.Path) -> tuple[_Model, pathlib.Path]:
    # Find which step we load. Do it here so we can log it for visibility instead of it being implicit....
    assert io.isdir(ckpt_path)
    if io.exists(ckpt_path / "default" / ORBAX_CKPT_FILENAME):
        # This means the given dir is an orbax checkpoint.
        step = orbax.checkpoint.utils.step_from_checkpoint_name(ckpt_path.name)
        ckpt_dir = ckpt_path.parent
    else:
        step = latest_checkpoint_step(ckpt_path)
        logger.info(f"Loading ckpt from step {step}!")

        ckpt_dir = ckpt_path

    ckpt_manager = get_ckpt_manager_sync(ckpt_dir)

    def restore_fn():
        return ckpt_manager.restore(step=step, items=model)
        # checkpointer: orbax.checkpoint.Checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # return checkpointer.restore(ckpt_dir, item=model)
        # return checkpoints.restore_checkpoint(
        #     ckpt_dir=ckpt_dir, target=model, step=step, orbax_checkpointer=checkpointer
        # )

    loaded = restore_fn()
    if model is loaded:
        raise ValueError("restore_checkpoint failed!")
    ckpt_path2 = ckpt_dir / "{}".format(step)
    assert ckpt_path2.exists()
    return loaded, ckpt_path2


def load_ckpt_without_step(model: _Model, ckpt_path: pathlib.Path) -> _Model:
    assert io.isdir(ckpt_path)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(ckpt_path, item=model)


def load_ckpt(model: _Model, ckpt_path: pathlib.Path) -> _Model:
    # Make sure ckpt_path is absolute.
    ckpt_path = ckpt_path.absolute()

    # Find which step we load. Do it here so we can log it for visibility instead of it being implicit....
    assert io.isdir(ckpt_path)
    if io.exists(ckpt_path / "default" / ORBAX_CKPT_FILENAME):
        # This means the given dir is an orbax checkpoint.
        step = orbax.checkpoint.utils.step_from_checkpoint_name(ckpt_path.name)
        ckpt_dir = ckpt_path.parent
    else:
        step = latest_checkpoint_step(ckpt_path)
        logger.info(f"Loading ckpt from step {step}!")

        ckpt_dir = ckpt_path

    ckpt_manager = get_ckpt_manager_sync(ckpt_dir)

    def restore_fn():
        return ckpt_manager.restore(step=step, items=model)
        # checkpointer: orbax.checkpoint.Checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # return checkpointer.restore(ckpt_dir, item=model)
        # return checkpoints.restore_checkpoint(
        #     ckpt_dir=ckpt_dir, target=model, step=step, orbax_checkpointer=checkpointer
        # )

    loaded = restore_fn()
    if model is loaded:
        raise ValueError("restore_checkpoint failed!")
    return loaded


def save_ckpt(
    model: _Model,
    ckpt_dir: pathlib.Path,
) -> None:
    checkpointer: orbax.checkpoint.Checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(model)
    checkpointer.save(ckpt_dir, model, save_args=save_args)
