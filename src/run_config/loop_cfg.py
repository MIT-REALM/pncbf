from attrs import define


@define
class LoopCfg:
    n_iters: int
    eval_every: int = 1_000
    log_every: int = 50
    plot_every: int = 1_000
    ckpt_every: int = 5_000
