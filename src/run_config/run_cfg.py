from typing import Any, Generic, TypeVar

from attrs import asdict, define
from typing_extensions import Self

AlgCfg = TypeVar("AlgCfg")
LoopCfg = TypeVar("LoopCfg")


@define
class RunCfg(Generic[AlgCfg, LoopCfg]):
    seed: int = 31415
    alg_cfg: AlgCfg = None
    loop_cfg: LoopCfg = None
    extras: dict = {}

    def asdict(self) -> dict[str, Any]:
        return asdict(self)
