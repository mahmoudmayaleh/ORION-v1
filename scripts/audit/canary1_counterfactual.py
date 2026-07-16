#!/usr/bin/env python3
"""DIAGNOSTIC ONLY — counterfactual for the Canary-1 fault. NOT A FIX. NOT MERGED.

Hypothesis under test (fault report): the gate trainer feeds the centralised
critic the LOCAL MDO observation (wp7_runner.py:394 `global_state=oc` with
`oc = t.obs`) instead of the designed s_t (training/global_state.py, Choice
A1: per-domain load + inter-link state + total_arrivals/admitted/rejected
counters). Under GAE-over-the-arrival-stream (PREREG §N.1) returns are
position-dependent (head ~77 -> tail ~5 in the canary), o^MDO carries no
stream-position information, so the critic cannot fit (EV ~= 0.000) and
advantages are position noise (corr(adv, position) ~ -0.9).

This script reruns Canary 1 (constant-best, no saturation) with the trainer
round-loop copied VERBATIM from wp7_runner.train_arm except the lines marked
# COUNTERFACTUAL: the critic consumes encode_global_state(substrate, stats)
captured at decision time (per arrival, pre-decision), exactly as the design
and mappo_trainer specify. PPO update, GAE, policy, masking, buffer, config:
byte-identical logic.

Expected if the fault report is right: EV >> 0, corr(adv, position) ~ 0,
selection converges high and stays there.
"""
import json
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
from orion.mdo.observation import build_domain_summaries
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.training.buffer import PPORolloutBuffer
from orion.training.config import MAPPOConfig
from orion.training.critic import CentralisedCritic
from orion.training.global_state import (
    GlobalStateStats,
    encode_global_state,
    probe_global_state_dim,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("canary1cf")

FAMILY = "C+_T+_B+"
GOOD_DOMAIN = 2
TINY = 0.05
SEED = 42
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CRITIC_INPUT = sys.argv[2] if len(sys.argv) > 2 else "st"     # "st" | "obs"
VALUE_LOSS = sys.argv[3] if len(sys.argv) > 3 else "clip"     # "clip" | "mse"
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
SERVICE_RATE = 2.0  # same as the fixed Canary 1


class StateCaptureRunner(EpisodeRunner):
    """Gate runner + pre-decision s_t capture per MDO-reaching arrival."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.s_t_by_request: dict[str, torch.Tensor] = {}
        self._stats_ref = None

    def run_episode(self, mdo_mode: str = "sample"):
        self.s_t_by_request.clear()
        return super().run_episode(mdo_mode)

    def _handle_arrival(self, slice_req, mdo_mode, rollout, mdo_results, stats,
                        arrival_trace=None):
        # s_t at decision time: substrate is post-(t-1) allocations/departures,
        # stats reflect arrivals 0..t-1 (this arrival not yet counted).
        gs = GlobalStateStats(total_arrivals=stats.total_arrivals,
                              admitted=stats.admitted,
                              rejected_by_mdo=stats.rejected_by_mdo,
                              max_arrivals=ARRIVALS)
        self.s_t_by_request[slice_req.request_id] = encode_global_state(
            self.substrate, gs).detach()
        super()._handle_arrival(slice_req, mdo_mode, rollout, mdo_results, stats,
                                arrival_trace)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fam = {f.short_name: f for f in R.ALL_FAMILIES}[FAMILY]
    sub0 = R.generate_family_instance(fam, seed=0)
    delays = W.build_delays(sub0)
    canonical_to_domain = [s.domain_id for s in build_domain_summaries(sub0)]
    log.info("canonical_to_domain=%s good=%d", canonical_to_domain, GOOD_DOMAIN)

    policy, coord, _unused_critic, opt_mdo, _unused_opt, obs_dim, num_domains = \
        W.build_stack(sub0, SEED, 3e-3, actors=None,
                      mdo_cfg=MDOConfig(n_part=3, **MU5))

    # COUNTERFACTUAL: critic sized for and fed the designed s_t (or the
    # untouched-path o^MDO input when CRITIC_INPUT == "obs").
    critic_dim = probe_global_state_dim(sub0) if CRITIC_INPUT == "st" else obs_dim
    log.info("matrix cell: critic_input=%s value_loss=%s", CRITIC_INPUT, VALUE_LOSS)
    critic = CentralisedCritic(input_dim=critic_dim,
                               hidden_dim=128, num_layers=2)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)

    cfg = MAPPOConfig(kl_beta_initial=0.0, kl_beta_final=0.0,
                      update_epochs=4, clip_eps=0.2, entropy_coef=0.01,
                      gamma=0.99, gae_lambda=0.95)

    diag, sel_curve = [], []
    for rnd in range(ROUNDS):
        ent_coef_t = cfg.entropy_coef
        sub = R.generate_family_instance(fam, seed=0)
        rng = np.random.default_rng(SEED + rnd * 1_000_000 + 1)
        ap = ArrivalProcess(sub, ARRIVALS, W.ARRIVAL_RATE, SERVICE_RATE, rng)
        ap.generate()
        runner = StateCaptureRunner(sub, ap, coord, delays,
                                    plan_builder=W.greedy_plan_builder)
        runner.reset()
        ep = runner.run_episode(mdo_mode="sample")

        # ── verbatim from wp7_runner.train_arm except the marked lines ──
        buffer = PPORolloutBuffer()
        with torch.no_grad():
            for t in ep.rollout.mdo:
                if CRITIC_INPUT == "st":                      # COUNTERFACTUAL
                    st = runner.s_t_by_request[t.request_id]
                else:                                         # untouched-path input
                    st = t.obs if t.obs.numel() > 0 else torch.zeros(obs_dim)
                cv = float(critic(st.unsqueeze(0)).item())
                buffer.append_mdo(
                    mdo_obs=t.obs, action=t.action, log_prob=t.log_probs,
                    entropy=t.entropy, aux_value=t.value_estimate,
                    global_state=st, critic_value=cv,          # COUNTERFACTUAL (st not obs)
                    reward=t.terminal_reward, done=t.committed,
                    tier_mask=t.tier_mask, num_vnfs=t.num_vnfs)

        if len(buffer) > 0:
            with torch.no_grad():
                vals = torch.tensor(
                    [float(critic(gs.unsqueeze(0)).item())     # COUNTERFACTUAL (gs = s_t)
                     for gs in buffer.global_states], dtype=torch.float32)
            advantages, returns = W.gae_over_arrivals(
                buffer, vals, gamma=cfg.gamma, lam=cfg.gae_lambda)
            buffer.set_gae(advantages, returns)

            # diagnostics (same statistics as canary1_diagnose)
            dones = buffer.dones
            term_idx = [i for i in range(len(dones)) if dones[i] >= 0.5]
            ar_ret = np.array([float(returns[i]) for i in term_idx])
            ar_val = np.array([float(vals[i]) for i in term_idx])
            ar_adv = np.array([float(advantages[i]) for i in term_idx])
            var_ret = float(np.var(ar_ret))
            ev = 1.0 - float(np.var(ar_ret - ar_val)) / var_ret if var_ret > 1e-12 else float("nan")
            ordn = np.arange(len(term_idx), dtype=float)
            corr_pos = (float(np.corrcoef(ar_adv, ordn)[0, 1])
                        if len(ordn) > 2 and np.std(ar_adv) > 1e-12 else float("nan"))

            for _ in range(cfg.update_epochs):
                gs = torch.stack(buffer.global_states)
                old_v = torch.tensor(buffer.critic_values, dtype=torch.float32)
                new_v = critic(gs).squeeze(-1)
                if VALUE_LOSS == "clip":  # untouched-path value loss
                    v_clip = old_v + torch.clamp(new_v - old_v, -cfg.clip_eps, cfg.clip_eps)
                    v_loss = 0.5 * torch.max((new_v - returns) ** 2,
                                             (v_clip - returns) ** 2).mean()
                else:                     # COUNTERFACTUAL: plain MSE
                    v_loss = 0.5 * ((new_v - returns) ** 2).mean()
                opt_critic.zero_grad(); (0.5 * v_loss).backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt_critic.step()

            for _ in range(cfg.update_epochs):
                epoch_loss = torch.tensor(0.0); cnt = 0
                for i in range(len(buffer.mdo_obs)):
                    obs_i = buffer.mdo_obs[i]
                    if obs_i.numel() == 0:
                        continue
                    k = buffer.mdo_num_vnfs[i]
                    if k == 0 or i >= len(buffer.mdo_tier_masks):
                        continue
                    action_i = torch.tensor(buffer.mdo_actions[i], dtype=torch.long)
                    tm_i = buffer.mdo_tier_masks[i]
                    adv_i = float(advantages[i]) if i < len(advantages) else 0.0
                    new_lp, new_ent, new_logits = policy.evaluate_actions(
                        obs_i, tm_i, action_i, k)
                    ratio = torch.exp(new_lp.sum() - buffer.mdo_log_probs[i].sum())
                    step_loss = -torch.min(
                        ratio * adv_i,
                        torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_i)
                    epoch_loss = epoch_loss + step_loss - ent_coef_t * new_ent
                    cnt += 1
                if cnt > 0:
                    opt_mdo.zero_grad(); (epoch_loss / cnt).backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt_mdo.step()
        else:
            ev = corr_pos = float("nan")

        # selection metric
        tot = hit = 0
        for t in ep.rollout.mdo:
            for c in t.action:
                tot += 1
                hit += int(canonical_to_domain[c] == GOOD_DOMAIN)
        sel = 100.0 * hit / tot if tot else 0.0
        sel_curve.append(sel)
        diag.append({"round": rnd + 1, "train_admit": ep.stats.admitted,
                     "total": ep.stats.total_arrivals, "selection_pct": sel,
                     "explained_variance": ev, "corr_adv_vs_position": corr_pos})
        log.info("R%-2d admit=%2d/%2d sel=%5.1f%%  EV=%6.3f corr_pos=%6.3f",
                 rnd + 1, ep.stats.admitted, ep.stats.total_arrivals, sel, ev, corr_pos)

    out = HERE / "out" / f"canary1_counterfactual_{CRITIC_INPUT}_{VALUE_LOSS}.json"
    json.dump(diag, open(out, "w"), indent=2)
    tail_sel = float(np.mean(sel_curve[-3:]))
    log.info("tail-3 selection: %.1f%%  (untouched path plateaued ~70-87%% and collapsed)",
             tail_sel)
    log.info("saved: %s", out)


if __name__ == "__main__":
    main()
