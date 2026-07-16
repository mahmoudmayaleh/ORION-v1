#!/usr/bin/env python3
"""DIAGNOSTIC — can the critic fit the GAE-over-stream returns AT ALL?

Collects ONE canary episode (constant-best substrate, sample mode), computes
the exact gae_over_arrivals returns the gate uses, then regresses the same
CentralisedCritic (same width/depth) offline with plain MSE for many epochs,
from each candidate input:
  - o^MDO (the gate's actual critic input)
  - s_t   (the designed global state, incl. stream-position counters)

Reports EV vs epochs. Distinguishes "input carries the information but the
gate's critic budget/loss cannot reach it" from "unfittable from this input".
"""
import logging
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import five_arm_runner as R
import wp7_runner as W
from orion.mdo.coordinator import MDOConfig
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.training.buffer import PPORolloutBuffer
from orion.training.critic import CentralisedCritic
from orion.training.global_state import GlobalStateStats, encode_global_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("criticfit")

FAMILY = "C+_T+_B+"
GOOD_DOMAIN = 2
TINY = 0.05
SEED = 42
ARRIVALS = 45
MU5 = dict(mu=5.0, alpha=0.1, xi=0.1, eta=0.1)

_orig_gen = R.generate_family_instance

def _canary_gen(family, seed, **kw):
    sub = _orig_gen(family, seed, **kw)
    for n, d in sub.graph.nodes(data=True):
        if d.get("domain_id") != GOOD_DOMAIN:
            d["cpu_capacity"] = TINY
            d["ram_capacity"] = TINY
            d["cpu_residual"] = min(d["cpu_residual"], TINY)
            d["ram_residual"] = min(d["ram_residual"], TINY)
    return sub

R.generate_family_instance = _canary_gen


class StateCaptureRunner(EpisodeRunner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.s_t_by_request = {}

    def _handle_arrival(self, slice_req, mdo_mode, rollout, mdo_results, stats,
                        arrival_trace=None):
        gs = GlobalStateStats(total_arrivals=stats.total_arrivals,
                              admitted=stats.admitted,
                              rejected_by_mdo=stats.rejected_by_mdo,
                              max_arrivals=ARRIVALS)
        self.s_t_by_request[slice_req.request_id] = encode_global_state(
            self.substrate, gs).detach()
        super()._handle_arrival(slice_req, mdo_mode, rollout, mdo_results, stats,
                                arrival_trace)


def fit(name, X, y, epochs=2000, lr=3e-4):
    torch.manual_seed(0)
    critic = CentralisedCritic(input_dim=X.shape[1], hidden_dim=128, num_layers=2)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    checkpoints = {}
    for e in range(1, epochs + 1):
        pred = critic(X).squeeze(-1)
        loss = 0.5 * ((pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        opt.step()
        if e in (80, 200, 500, 1000, 2000):
            with torch.no_grad():
                p = critic(X).squeeze(-1)
            var = float(torch.var(y))
            ev = 1.0 - float(torch.var(y - p)) / var if var > 1e-12 else float("nan")
            checkpoints[e] = ev
    log.info("%-6s EV by MSE epochs: %s  (gate budget ~= 80 epochs total, w/ clip + moving targets)",
             name, {k: round(v, 3) for k, v in checkpoints.items()})
    return checkpoints


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fam = {f.short_name: f for f in R.ALL_FAMILIES}[FAMILY]
    sub0 = R.generate_family_instance(fam, seed=0)
    delays = W.build_delays(sub0)
    policy, coord, critic, opt_mdo, opt_critic, obs_dim, num_domains = \
        W.build_stack(sub0, SEED, 3e-3, actors=None, mdo_cfg=MDOConfig(n_part=3, **MU5))

    sub = R.generate_family_instance(fam, seed=0)
    rng = np.random.default_rng(SEED + 1)
    ap = ArrivalProcess(sub, ARRIVALS, W.ARRIVAL_RATE, 2.0, rng)
    ap.generate()
    runner = StateCaptureRunner(sub, ap, coord, delays, plan_builder=W.greedy_plan_builder)
    runner.reset()
    ep = runner.run_episode(mdo_mode="sample")

    buffer = PPORolloutBuffer()
    st_list = []
    for t in ep.rollout.mdo:
        oc = t.obs if t.obs.numel() > 0 else torch.zeros(obs_dim)
        buffer.append_mdo(mdo_obs=t.obs, action=t.action, log_prob=t.log_probs,
                          entropy=t.entropy, aux_value=t.value_estimate,
                          global_state=oc, critic_value=0.0,
                          reward=t.terminal_reward, done=t.committed,
                          tier_mask=t.tier_mask, num_vnfs=t.num_vnfs)
        st_list.append(runner.s_t_by_request[t.request_id])

    vals = torch.zeros(len(buffer))
    _, returns = W.gae_over_arrivals(buffer, vals, gamma=0.99, lam=0.95)
    y = returns.detach()
    log.info("episode: %d transitions, returns head=%.1f tail=%.1f mean=%.1f std=%.1f",
             len(buffer), float(y[0]), float(y[-1]), float(y.mean()), float(y.std()))

    X_obs = torch.stack([o if o.numel() > 0 else torch.zeros(obs_dim)
                         for o in buffer.mdo_obs])
    X_st = torch.stack(st_list)
    fit("o^MDO", X_obs, y)
    fit("s_t", X_st, y)


if __name__ == "__main__":
    main()
