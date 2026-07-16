#!/usr/bin/env python3
"""Verification battery — Phase 1, Canary 3: saturating domain (temporal credit).

Directive: one domain admits only its first k selections per stream, then
rejects; another admits always at lower reward. With conformant GAE, pass:
policy learns to front-load then switch, or avoid the trap. Fail implicates:
temporal credit path (GAE wiring under the real done/stream semantics).

Implementation: domains A and B keep capacity, all others are shrunk to
unplaceable (same surgery as canary 1). A's actor is the real
GreedyDomainActor wrapped in a stream-level saturation counter: after k
feasible fragment placements it returns infeasible for the rest of the
episode (counter reset at EpisodeRunner.reset via a runner subclass, so
"per stream" is exact). B's actor is the real GreedyDomainActor with its
reported resource_cost scaled x4, which flows through the real efficiency +
quality-shaping reward terms, making B admissions strictly lower-reward than
A admissions. A's real allocations debit residuals, so saturation is
correlated with A's observed load. Departures disabled (SERVICE_RATE ~ 0).

Everything else is the untouched gate trainer path (train_arm, mock=True,
MU5, n_part=3, gate PPO config, GAE-over-arrivals).

Pass criteria (final 3 rounds, averaged):
  - trap avoidance: selection of A among LATE arrivals (stream index >= 2k)
    < 30% (a trapped policy stays >> 50%: A is cheaper AND was rewarded early)
  - admission stays high: train admits >= 80% of MDO-reaching arrivals.
Front-loading (A-selection among first k arrivals) is reported but not
required — the reward margin (~0.2 vs 5.0) is second-order by construction.
"""
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import five_arm_runner as R
import wp7_runner as W
from orion.actors.greedy_domain_actor import GreedyDomainActor
from orion.actors.types import DomainResponse
from orion.mdo.coordinator import MDOConfig
from orion.mdo.observation import build_domain_summaries
from orion.sim.episode_runner import EpisodeRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("canary3")

FAMILY = "C+_T+_B+"
DOM_A, DOM_B = 1, 3          # A saturates after K_SAT; B always admits, costlier
K_SAT = 8
COST_FACTOR_B = 4.0
TINY = 0.05
SEED = 42
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ARRIVALS = 45
MU5 = dict(mu=5.0, alpha=0.1, xi=0.1, eta=0.1)

_orig_gen = R.generate_family_instance

def _canary_gen(family, seed, **kw):
    sub = _orig_gen(family, seed, **kw)
    for n, d in sub.graph.nodes(data=True):
        if d.get("domain_id") not in (DOM_A, DOM_B):
            d["cpu_capacity"] = TINY
            d["ram_capacity"] = TINY
            d["cpu_residual"] = min(d["cpu_residual"], TINY)
            d["ram_residual"] = min(d["ram_residual"], TINY)
        elif d.get("domain_id") == DOM_B:
            # "B always admits" must hold PHYSICALLY: with departures disabled
            # a stock domain genuinely fills mid-stream (first L3 run: admission
            # ceiling ~50-60%, making the 80% bar unreachable). x20 capacity
            # keeps the premise true; A keeps stock capacity so its stub
            # saturation stays correlated with visible load.
            d["cpu_capacity"] *= 20.0
            d["ram_capacity"] *= 20.0
            d["cpu_residual"] *= 20.0
            d["ram_residual"] *= 20.0
    return sub

R.generate_family_instance = _canary_gen
W.SERVICE_RATE = 1e-7  # no departures within an episode


class SaturatingActor:
    """Real greedy actor for A; rejects after K_SAT feasible placements per stream."""

    def __init__(self, domain_id: int, k: int) -> None:
        self.inner = GreedyDomainActor(domain_id)
        self.domain_id = domain_id
        self.k = k
        self.count = 0

    def act(self, substrate, fragment):
        if self.count >= self.k:
            return DomainResponse(domain_id=self.domain_id, feasible=False)
        resp = self.inner.act(substrate, fragment)
        if resp.feasible:
            self.count += 1
        return resp


class CostlyActor:
    """Real greedy actor for B; reports x4 resource cost (feeds real reward terms)."""

    def __init__(self, domain_id: int, factor: float) -> None:
        self.inner = GreedyDomainActor(domain_id)
        self.domain_id = domain_id
        self.factor = factor

    def act(self, substrate, fragment):
        resp = self.inner.act(substrate, fragment)
        if resp.feasible:
            try:
                resp = replace(resp, resource_cost=resp.resource_cost * self.factor)
            except TypeError:
                resp.resource_cost = resp.resource_cost * self.factor
        return resp


ACTORS = {}


class CounterResetRunner(EpisodeRunner):
    """Gate EpisodeRunner + per-stream saturation counter reset. No other changes."""

    def reset(self) -> None:
        super().reset()
        a = ACTORS.get(DOM_A)
        if a is not None:
            a.count = 0


W.EpisodeRunner = CounterResetRunner


def main():
    fam = {f.short_name: f for f in R.ALL_FAMILIES}[FAMILY]
    sub = R.generate_family_instance(fam, seed=0)
    canonical_to_domain = [s.domain_id for s in build_domain_summaries(sub)]
    log.info("canonical_to_domain=%s  A(sat,k=%d)=%d B(costly)=%d",
             canonical_to_domain, K_SAT, DOM_A, DOM_B)

    ACTORS.clear()
    for d in range(sub.num_domains):
        if d == DOM_A:
            ACTORS[d] = SaturatingActor(d, K_SAT)
        elif d == DOM_B:
            ACTORS[d] = CostlyActor(d, COST_FACTOR_B)
        else:
            ACTORS[d] = GreedyDomainActor(d)  # unplaceable anyway (surgery)

    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    trace_path = out_dir / "canary3_train_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    curve, coord = W.train_arm(
        "RL-alone", FAMILY, SEED, ROUNDS, ARRIVALS,
        lr=3e-3, beta_start=0.0, beta_end=0.0, agent_b=None, kb=None, mock=True,
        actors=ACTORS, mdo_cfg=MDOConfig(n_part=3, **MU5),
        eval_with_train_builder=False, return_coord=True,
        entropy_schedule=None, train_trace_path=str(trace_path),
    )

    per_round = []
    with open(trace_path) as f:
        for line in f:
            rec = json.loads(line)
            early_a = early_n = late_a = late_n = admit = reach = 0
            for pair in rec["pairs"]:
                reach += 1
                admit += int(pair["admit"])
                sel_a = [int(canonical_to_domain[c] == DOM_A)
                         for c in (pair.get("partition") or [])]
                if pair["index"] < K_SAT:
                    early_a += sum(sel_a); early_n += len(sel_a)
                elif pair["index"] >= 2 * K_SAT:
                    late_a += sum(sel_a); late_n += len(sel_a)
            per_round.append({
                "round": rec["round"],
                "early_a_pct": 100.0 * early_a / early_n if early_n else None,
                "late_a_pct": 100.0 * late_a / late_n if late_n else None,
                "admit_pct": 100.0 * admit / reach if reach else None,
            })

    tail = per_round[-3:]
    tail_late_a = float(np.mean([p["late_a_pct"] for p in tail
                                 if p["late_a_pct"] is not None]))
    tail_admit = float(np.mean([p["admit_pct"] for p in tail
                                if p["admit_pct"] is not None]))
    tail_early_a = float(np.mean([p["early_a_pct"] for p in tail
                                  if p["early_a_pct"] is not None]))

    result = {
        "canary": "saturation",
        "family": FAMILY, "seed": SEED, "rounds": ROUNDS, "arrivals": ARRIVALS,
        "dom_a": DOM_A, "dom_b": DOM_B, "k_sat": K_SAT,
        "canonical_to_domain": canonical_to_domain,
        "per_round": per_round,
        "tail3_late_a_selection_pct": tail_late_a,
        "tail3_early_a_selection_pct": tail_early_a,
        "tail3_admit_pct": tail_admit,
        "pass": tail_late_a < 30.0 and tail_admit >= 80.0,
    }
    out = out_dir / "canary3_result.json"
    json.dump(result, open(out, "w"), indent=2)
    log.info("per-round [early_A%%, late_A%%, admit%%]: %s",
             [[round(p["early_a_pct"], 1) if p["early_a_pct"] is not None else None,
               round(p["late_a_pct"], 1) if p["late_a_pct"] is not None else None,
               round(p["admit_pct"], 1) if p["admit_pct"] is not None else None]
              for p in per_round])
    log.info("tail-3: early_A=%.1f%% late_A=%.1f%% admit=%.1f%% -> %s",
             tail_early_a, tail_late_a, tail_admit,
             "PASS" if result["pass"] else "FAIL")
    log.info("saved: %s", out)


if __name__ == "__main__":
    main()
