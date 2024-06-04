from pncbf.dyn.doubleint_wall import DoubleIntWall
from pncbf.pncbf.pncbf import PNCBFCfg, PNCBFEvalCfg, PNCBFTrainCfg
from pncbf.utils.schedules import (
    Constant,
    JoinSched,
    Lin,
    SchedCtsHorizon,
    horizon_to_lam,
)
from run_config.loop_cfg import LoopCfg
from run_config.run_cfg import RunCfg


def get(seed: int) -> RunCfg[PNCBFCfg, LoopCfg]:
    # lam = ExpDecay(1.0, 5_000, 0.7, 25_000, staircase=False, end_value=1e-4)
    dt = DoubleIntWall.DT

    sched1_steps = 35_000
    sched1_warmup = 15_000

    # horizon_end = 200
    # sched1 = Lin(10, 25, sched1_steps, warmup=sched1_warmup)
    # lam = SchedCtsHorizon(sched1, dt)
    lam = Constant(0.4)
    # final_lam = horizon_to_lam(horizon_end, dt=dt)

    # sched2_steps = 10_000
    # lam = JoinSched(lam, Lin(final_lam, 0.0, sched2_steps), sched1.total_steps)
    # lam = Constant(0.3)

    lr = Constant(3e-4)
    # lr = Constant(8e-4)
    # wd = Constant(1e-1)
    wd = Constant(4e-3)

    # act = "tanh"
    act = "softplus"
    # act = "gelu"

    # tgt_rhs = Constant(0.9)
    tgt_rhs = Lin(0.0, 0.9, steps=35_000, warmup=15_000)

    # lam = SchedCtsHorizon(Lin(10, 200, 75_000, warmup=25_000), 0.1)
    # lam = SchedCtsHorizon(ExpDecay(10, 200, 75_000, warmup=25_000), 0.1)
    # lam = Constant(0.1)
    # lam = Constant(0.01)
    collect_size = 8192
    rollout_dt = dt
    train_cfg = PNCBFTrainCfg(
        collect_size,
        rollout_dt,
        rollout_T=24,
        batch_size=8192,
        lam=lam,
        tau=0.005,
        tgt_rhs=tgt_rhs,
    )
    eval_cfg = PNCBFEvalCfg(eval_rollout_T=64)
    alg_cfg = PNCBFCfg(
        act=act,
        lr=lr,
        wd=wd,
        hids=[256, 256],
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        n_Vs=2,
        n_min_tgt=2,
    )
    loop_cfg = LoopCfg(n_iters=lam.total_steps + 5_000, ckpt_every=5_000, log_every=100, eval_every=5_000)
    return RunCfg(seed, alg_cfg, loop_cfg)


def get_pi(seed: int) -> RunCfg[PNCBFCfg, LoopCfg]:
    dt = DoubleIntWall.DT

    sched1_steps = 20_000
    sched1_warmup = 5_000
    final_horizon = 50
    sched1 = Lin(10, final_horizon, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)
    final_lam = horizon_to_lam(final_horizon, dt=dt)

    sched2_steps = 40_000
    lam = JoinSched(lam, Lin(final_lam, 0.03, sched2_steps), sched1.total_steps)

    tgt_rhs = Lin(0.0, 0.95, steps=20_000, warmup=5_000)

    cfg = get(seed)
    cfg.alg_cfg.train_cfg.lam = lam
    cfg.alg_cfg.train_cfg.tgt_rhs = tgt_rhs
    cfg.loop_cfg.n_iters = lam.total_steps + 5_000

    return cfg
