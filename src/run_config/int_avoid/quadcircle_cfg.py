from pncbf.dyn.quadcircle import QuadCircle
from pncbf.pncbf.pncbf import PNCBFCfg, PNCBFEvalCfg, PNCBFTrainCfg
from pncbf.utils.schedules import (
    Constant,
    ExpDecay,
    JoinSched,
    Lin,
    LinDecay,
    SchedCtsHorizon,
    SchedEffHorizon,
    horizon_to_lam,
)
from run_config.loop_cfg import LoopCfg
from run_config.run_cfg import RunCfg


def get(seed: int) -> RunCfg[PNCBFCfg, LoopCfg]:
    # 3s is probably a long enough horizon.
    dt = QuadCircle.DT
    sched1_steps = 70_000
    sched1_warmup = 100_000
    # sched1_warmup = 300_000
    horizon_final = 100
    # horizon_final = 50
    sched1 = Lin(20, horizon_final, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)

    final_lam = horizon_to_lam(horizon_final, dt=dt)
    sched2_steps = 100_000
    lam = JoinSched(lam, Lin(final_lam, 0.05, sched2_steps), sched1.total_steps)

    lr = Constant(3e-4)
    wd = Constant(1e-1)
    # act = "softplus"
    act = "tanh"

    tgt_rhs = Lin(0.0, 0.9, steps=70_000, warmup=100_000)
    # tgt_rhs = Lin(0.0, 0.9, steps=70_000, warmup=300_000)

    # collect_size = 16_384
    # collect_size = 8192
    collect_size = 2048
    rollout_dt = dt
    train_cfg = PNCBFTrainCfg(
        collect_size,
        rollout_dt,
        rollout_T=49,
        # batch_size=8192,
        batch_size=6144,
        lam=lam,
        tau=0.005,
        tgt_rhs=tgt_rhs,
    )
    eval_cfg = PNCBFEvalCfg(eval_rollout_T=100)
    alg_cfg = PNCBFCfg(
        act=act,
        lr=lr,
        wd=wd,
        hids=[256, 256, 256],
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        n_Vs=2,
        n_min_tgt=2,
    )
    loop_cfg = LoopCfg(n_iters=lam.total_steps + 50_000, ckpt_every=5_000, log_every=100, eval_every=5_000)
    return RunCfg(seed, alg_cfg, loop_cfg)


def get_pi(seed: int) -> RunCfg[PNCBFCfg, LoopCfg]:
    return get(seed)
