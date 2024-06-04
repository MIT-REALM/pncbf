from pncbf.dyn.segway import Segway
from pncbf.pncbf.pncbf import PNCBFCfg, PNCBFEvalCfg, PNCBFTrainCfg
from pncbf.utils.schedules import Constant, JoinSched, Lin, SchedCtsHorizon, horizon_to_lam
from run_config.loop_cfg import LoopCfg
from run_config.run_cfg import RunCfg


def get(seed: int) -> RunCfg[PNCBFCfg, LoopCfg]:
    dt = Segway.DT
    sched1_steps = 50_000
    sched1_warmup = 20_000
    sched1 = Lin(60, 150, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)

    lr = Constant(3e-4)
    wd = Constant(1e-1)

    tgt_rhs = Lin(0.0, 0.9, steps=50_000, warmup=20_000)

    collect_size = 16_384
    rollout_dt = Segway.DT
    train_cfg = PNCBFTrainCfg(
        collect_size, rollout_dt, rollout_T=99, batch_size=8192, lam=lam, tau=0.005, tgt_rhs=tgt_rhs
    )
    eval_cfg = PNCBFEvalCfg(eval_rollout_T=64)
    alg_cfg = PNCBFCfg(
        act="softplus",
        lr=lr,
        wd=wd,
        hids=[256, 256, 256],
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        n_Vs=2,
        n_min_tgt=2,
    )
    loop_cfg = LoopCfg(n_iters=lam.total_steps + 10_000, ckpt_every=5_000, log_every=100, eval_every=2_500)
    return RunCfg(seed, alg_cfg, loop_cfg)


def get_pi(seed: int) -> RunCfg[PNCBFCfg, LoopCfg]:
    dt = Segway.DT

    sched1_steps = 50_000
    sched1_warmup = 20_000
    sched1 = Lin(60, 150, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)
    final_lam = horizon_to_lam(150, dt=dt)

    sched2_steps = 50_000
    lam = JoinSched(lam, Lin(final_lam, 0.0, sched2_steps), sched1.total_steps)

    tgt_rhs = Lin(0.0, 0.9, steps=50_000, warmup=20_000)

    cfg = get(seed)
    cfg.alg_cfg.train_cfg.lam = lam
    cfg.alg_cfg.train_cfg.tgt_rhs = tgt_rhs
    cfg.loop_cfg.n_iters = lam.total_steps + 5_000

    return cfg
