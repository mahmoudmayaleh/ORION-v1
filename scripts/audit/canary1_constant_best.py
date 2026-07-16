#!/usr/bin/env python3
"""Verification battery — Phase 1, Canary 1: constant-best domain.

Directive: M domains, one always admits (reward ~ +5 admission-dominant), all
others always reject. Same MDO policy class, same masking, same buffer, same
PPO config, same trainer path as the gate. Pass: >95% selection of the
admitting domain within a small fraction of one gate's budget (gate = 20
rounds x 45 arrivals; canary budget = 8 rounds max, pass at first round
crossing 95%).

Implementation: the ONLY intervention is substrate surgery — every domain
except GOOD_DOMAIN has node CPU/RAM capacities shrunk below any VNF demand,
so its (real, frozen) GreedyDomainActor is genuinely infeasible for every
fragment. Everything else is the untouched gate path: wp7_runner.train_arm
with mock=True (greedy m~ plan builder), default frozen GreedyDomainActors,
MDOConfig(n_part=3, mu=5, alpha=xi=eta=0.1), lr=3e-3, update_epochs=4,
clip=0.2, entropy_coef=0.01, gamma=0.99, lambda=0.95, GAE-over-arrivals.

Family C+_T+_B+ (tier-friendly: every domain supports every tier, so the
tier mask never hides the signal; capacity-friendly: chains fit in one
domain).

Fail implicates: core update loop, gradient sign, buffer, or action plumbing.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # scripts/
sys.path.insert(0, str(HERE.parent.parent / "src"))

import five_arm_runner as R
import wp7_runner as W
from orion.mdo.coordinator import MDOConfig
from orion.mdo.observation import build_domain_summaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("canary1")

FAMILY = "C+_T+_B+"
GOOD_DOMAIN = 2          # raw domain id that keeps its capacity
TINY = 0.05              # below every template's min CPU/RAM demand
SEED = 42
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8  # gate budget = 20 rounds
ARRIVALS = 45
MU5 = dict(mu=5.0, alpha=0.1, xi=0.1, eta=0.1)

# ── Substrate surgery: shrink every domain except GOOD_DOMAIN ────────────────
_orig_gen = R.generate_family_instance

def _canary_gen(family, seed, **kw):
    sub = _orig_gen(family, seed, **kw)
    g = sub.graph
    for n, d in g.nodes(data=True):
        if d.get("domain_id") != GOOD_DOMAIN:
            d["cpu_capacity"] = TINY
            d["ram_capacity"] = TINY
            d["cpu_residual"] = min(d["cpu_residual"], TINY)
            d["ram_residual"] = min(d["ram_residual"], TINY)
    return sub

R.generate_family_instance = _canary_gen  # both R and W resolve through this module attr

# Fast departures so the good domain never saturates: with the default
# (lambda=4, mu=0.02) ~200 concurrent slices pile into the single good domain
# and "always admits" breaks mid-stream (reward 0 for every choice on
# saturated arrivals -> no gradient and an unfair 95% bar). lambda=4, mu=2.0
# keeps ~2 concurrent, so the admitting domain genuinely always admits.
W.SERVICE_RATE = 2.0


def main():
    fam = {f.short_name: f for f in R.ALL_FAMILIES}[FAMILY]
    # Sanity on the surgered instance: canonical order + which domain is good.
    sub = R.generate_family_instance(fam, seed=0)
    summaries = build_domain_summaries(sub)
    canonical_to_domain = [s.domain_id for s in summaries]
    good_canonical = canonical_to_domain.index(GOOD_DOMAIN)
    log.info("canonical_to_domain=%s  good_domain=%d -> canonical index %d",
             canonical_to_domain, GOOD_DOMAIN, good_canonical)
    caps = {s.domain_id: round(s.cpu_capacity, 2) for s in summaries}
    log.info("per-domain CPU capacity after surgery: %s", caps)

    # Oracle reference: follow_prior with the greedy m~ — on the surgered
    # substrate FFD can only place in the good domain, so m~ == all-good.
    # Bounds what "always choose the good domain" admits on the eval stream.
    from orion.mdo.coordinator import MDOCoordinator
    from orion.actors.greedy_domain_actor import GreedyDomainActor
    from orion.sim.arrival_process import ArrivalProcess
    from orion.sim.episode_runner import EpisodeRunner as _ER
    sub_o = R.generate_family_instance(fam, seed=0)
    actors_o = {d: GreedyDomainActor(d) for d in range(sub_o.num_domains)}
    coord_o = MDOCoordinator(None, actors_o, MDOConfig(n_part=1, **MU5))
    rng_o = np.random.default_rng(SEED + 777)
    ap_o = ArrivalProcess(sub_o, ARRIVALS, W.ARRIVAL_RATE, W.SERVICE_RATE, rng_o)
    ap_o.generate()
    run_o = _ER(sub_o, ap_o, coord_o, W.build_delays(sub_o),
                plan_builder=W.greedy_plan_builder)
    run_o.reset()
    ep_o = run_o.run_episode(mdo_mode="follow_prior")
    log.info("ORACLE (all-good via follow_prior): admit=%d/%d",
             ep_o.stats.admitted, ep_o.stats.total_arrivals)

    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    trace_path = out_dir / "canary1_train_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    curve, coord = W.train_arm(
        "RL-alone", FAMILY, SEED, ROUNDS, ARRIVALS,
        lr=3e-3, beta_start=0.0, beta_end=0.0, agent_b=None, kb=None, mock=True,
        actors=None,                      # default frozen GreedyDomainActors — real path
        mdo_cfg=MDOConfig(n_part=3, **MU5),
        eval_with_train_builder=False, return_coord=True,
        entropy_schedule=None,
        train_trace_path=str(trace_path),
    )

    # ── Metric: per-round selection rate of the good domain ────────────────
    # Train trace partitions are CANONICAL indices (coordinator stores
    # attempt.partition = partition_canonical); map via canonical_to_domain.
    per_round = []
    with open(trace_path) as f:
        for line in f:
            rec = json.loads(line)
            tot = hit = 0
            for pair in rec["pairs"]:
                part = pair.get("partition") or []
                for c in part:
                    tot += 1
                    hit += int(canonical_to_domain[c] == GOOD_DOMAIN)
            per_round.append({
                "round": rec["round"],
                "train_selection_pct": 100.0 * hit / tot if tot else None,
                "n_vnf_choices": tot,
            })

    # Deterministic eval selection on the held-out stream (gate protocol).
    ev = W.instrumented_eval(coord, FAMILY, SEED, ARRIVALS,
                             W.build_delays(R.generate_family_instance(fam, seed=0)),
                             plan_builder=None, mode="deterministic")
    tot = hit = 0
    for pair in ev["pairs"]:
        for c in (pair.get("partition") or []):
            tot += 1
            hit += int(canonical_to_domain[c] == GOOD_DOMAIN)
    eval_sel = 100.0 * hit / tot if tot else 0.0

    # ── L1 pass bar (§O, PINNED at ratification) ───────────────────────────
    # (a) >95% selection of the admitting domain (deterministic eval);
    # (b) walk-away forbidden: no round after FIRST reaching 95% train
    #     selection drops below 90%;
    # (c) EV >= 0.5 averaged over the final 5 rounds (§O.8 telemetry).
    sels = [p["train_selection_pct"] for p in per_round]
    first_95 = next((i for i, s in enumerate(sels) if s is not None and s >= 95.0), None)
    walk_away = (first_95 is not None and
                 any(s is not None and s < 90.0 for s in sels[first_95 + 1:]))
    evs = [c.get("ev") for c in curve if c.get("ev") == c.get("ev")]  # drop NaN
    ev_tail5 = float(np.mean(evs[-5:])) if len(evs) >= 5 else float("nan")

    result = {
        "canary": "constant-best",
        "family": FAMILY, "seed": SEED, "rounds": ROUNDS, "arrivals": ARRIVALS,
        "good_domain": GOOD_DOMAIN, "good_canonical": good_canonical,
        "canonical_to_domain": canonical_to_domain,
        "per_round_selection": per_round,
        "final_eval_selection_pct": eval_sel,
        "eval_admit": ev["admit"], "eval_total": ev["total"],
        "oracle_admit": ep_o.stats.admitted, "oracle_total": ep_o.stats.total_arrivals,
        "curve_foc": [c["eval_foc"] for c in curve],
        "curve_entropy": [c["mdo_entropy"] for c in curve],
        "curve_param_motion": [c["param_motion"] for c in curve],
        "curve_ev": [c.get("ev") for c in curve],
        "first_round_at_95": None if first_95 is None else per_round[first_95]["round"],
        "walk_away": walk_away,
        "ev_tail5_mean": ev_tail5,
        "pass": (eval_sel > 95.0) and (not walk_away)
                and (ev_tail5 == ev_tail5 and ev_tail5 >= 0.5),
    }
    out = out_dir / "canary1_result.json"
    json.dump(result, open(out, "w"), indent=2)
    log.info("per-round train selection of good domain: %s",
             [round(p["train_selection_pct"], 1) if p["train_selection_pct"] is not None
              else None for p in per_round])
    log.info("final deterministic-eval selection: %.1f%%  walk_away=%s  EV(tail5)=%.3f -> %s",
             eval_sel, walk_away, ev_tail5, "PASS" if result["pass"] else "FAIL")
    log.info("saved: %s", out)


if __name__ == "__main__":
    main()
