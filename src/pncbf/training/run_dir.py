import os
import pathlib
import random
import socket
import string
from datetime import datetime

from loguru import logger

import wandb
from pncbf.utils.git_utils import log_git_info
from pncbf.utils.paths import get_runs_dir
from run_config.run_cfg import RunCfg


def get_date_stamp(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now()
    return now.strftime("%Y-%m-%d")


def get_time_stamp(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now()
    return now.strftime("%H-%M")


def get_datetime_stamp(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def get_prefix(now: datetime, cfg: RunCfg):
    time_str = get_time_stamp(now)
    dt_str = get_date_stamp(now)

    return f"{dt_str}_{time_str}_"


def get_unique_ident(now: datetime, run_dir: pathlib.Path, cfg: RunCfg, ident_len: int = 4) -> str:
    prefix = get_prefix(now, cfg)

    # We want the ident to be lexicographically larger than any existing runs.
    same_time_runs = list(run_dir.glob(f"{prefix}*"))
    prefix_len = len(prefix)
    same_time_idents = sorted([str(path.name)[prefix_len : prefix_len + ident_len] for path in same_time_runs])

    if len(same_time_idents) > 0 and "Z" * ident_len in same_time_idents:
        raise RuntimeError("cannot gen anything larger than ZZZZ....")

    while True:
        ident = "".join(random.choices(string.ascii_uppercase, k=ident_len))

        if len(same_time_idents) == 0 or ident > same_time_idents[-1]:
            return ident


def init_wandb_and_get_run_dir(
    cfg: RunCfg, project: str, job_type: str, name: str, ident_len: int = 4, group: str = None
):
    run_dir = get_runs_dir() / job_type
    ident: str = get_unique_ident(datetime.now(), run_dir, cfg, ident_len)

    wandb_name = f"{ident}_{name}"
    wandb_settings = wandb.Settings(_disable_stats=True)
    wandb.init(project=project, job_type=job_type, name=wandb_name, settings=wandb_settings, group=group)
    wandb.config.update(cfg.asdict())

    hostname = socket.gethostname()
    now = wandb.run.settings._start_datetime
    wandb_disabled = os.environ.get("WANDB_MODE", None) == "disabled"

    if isinstance(now, str) and not wandb_disabled:
        now = datetime.strptime(now, "%Y%m%d_%H%M%S")
    if not isinstance(now, datetime):
        assert wandb_disabled
        now = datetime.now()
    assert isinstance(now, datetime)

    prefix = get_prefix(now, cfg)
    run_name = f"{prefix}{ident}_{hostname}_{name}"
    log_dir = run_dir / run_name

    log_dir.parent.mkdir(exist_ok=True, parents=True)

    logger.info(f"run_dir: {log_dir}")
    log_dir.mkdir(exist_ok=True)

    # Create a symlink inside the log_dir to the wandb_dir, and vice versa.
    if not wandb_disabled:
        wandb_dir = pathlib.Path(wandb.run.dir).parent
        (log_dir / "wandb").symlink_to(wandb_dir)
        (wandb_dir / "local_dir").symlink_to(log_dir.absolute())
        logger.info(f"symlink: {wandb_dir}")

    log_file = log_dir / "log.txt"
    logger.add(log_file)
    log_git_info(log_dir)

    return log_dir
