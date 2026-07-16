#!/usr/bin/env python3
"""Verification battery — Phase 1, Canary 2: observation-conditional domain.

Directive: the admitting domain is indicated by one observation feature that
switches per arrival. Pass: policy tracks the switch. Fail implicates:
observation tensor not reaching the policy, staleness, or feature-order
scrambling.

Implementation: two candidate domains A and B keep capacity, all others are
shrunk to unplaceable. Before EVERY arrival, a per-arrival toggle stomps
residuals: on even arrivals A is open (full residuals) and B is closed
(residuals ~0); on odd arrivals the reverse. The (real, frozen)
GreedyDomainActor on a zero-residual domain is genuinely infeasible, and the
observation's cpu_res_frac / h^m features for A and B are exactly the
indicator the policy must read. Because the coordinator builds the
observation from the live substrate inside resolve_arrival, a policy that
tracks the switch proves obs -> policy freshness and feature alignment.

The toggle is injected by subclassing EpisodeRunner (toggle applied at the
top of _handle_arrival, i.e. before plan build / obs build / decision) and
patching wp7_runner.EpisodeRunner. Everything else is the untouched gate
trainer path (train_arm, mock=True, MU5, n_part=3, gate PPO config).

Departures are disabled (SERVICE_RATE ~ 0) so residual stomping cannot be
confounded by deallocation bookkeeping.

Pass: selection matches the currently-open domain in >90% of per-VNF choices
on the final rounds (choice between A and B is binary; random = 50%,
constant-side = 50%).
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
from orion.sim.episode_runner import EpisodeRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("canary2")

FAMILY = "C+_T+_B+"
DOM_A, DOM_B = 1, 3
TINY = 0.05
SEED = 42
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 60  # L1 showed ~20-round critic transient
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
    return sub

R.generate_family_instance = _canary_gen
W.SERVICE_RATE = 1e-7  # lifetimes >> episode horizon: no departures


def _open_domain(arrival_index: int) -> int:
    return DOM_A if arrival_index % 2 == 0 else DOM_B


class ToggleRunner(EpisodeRunner):
    """Gate EpisodeRunner + pre-arrival residual toggle. No other changes."""

    def _handle_arrival(self, slice_req, mdo_mode, rollout, mdo_results, stats,
                        arrival_trace=None):
        idx = stats.total_arrivals  # index of THIS arrival (0-based)
        open_dom = _open_domain(idx)
        closed_dom = DOM_B if open_dom == DOM_A else DOM_A
        g = self.substrate.graph
        for n, d in g.nodes(data=True):
            if d.get("domain_id") == open_dom:
                d["cpu_residual"] = d["cpu_capacity"]
                d["ram_residual"] = d["ram_capacity"]
            elif d.get("domain_id") == closed_dom:
                d["cpu_residual"] = TINY
                d["ram_residual"] = TINY
        super()._handle_arrival(slice_req, mdo_mode, rollout, mdo_results, stats,
                                arrival_trace)


W.EpisodeRunner = ToggleRunner


def main():
    fam = {f.short_name: f for f in R.ALL_FAMILIES}[FAMILY]
    sub = R.generate_family_instance(fam, seed=0)
    canonical_to_domain = [s.domain_id for s in build_domain_summaries(sub)]
    log.info("canonical_to_domain=%s  A=%d B=%d", canonical_to_domain, DOM_A, DOM_B)

    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    trace_path = out_dir / "canary2_train_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    curve, coord = W.train_arm(
        "RL-alone", FAMILY, SEED, ROUNDS, ARRIVALS,
        lr=3e-3, beta_start=0.0, beta_end=0.0, agent_b=None, kb=None, mock=True,
        actors=None, mdo_cfg=MDOConfig(n_part=3, **MU5),
        eval_with_train_builder=False, return_coord=True,
        entropy_schedule=None, train_trace_path=str(trace_path),
    )

    # Selection-tracking metric per round: fraction of per-VNF choices equal
    # to the domain that is OPEN for that arrival. Uses the trace's global
    # stream index so structural rejects don't desync the parity.
    per_round = []
    with open(trace_path) as f:
        for line in f:
            rec = json.loads(line)
            tot = hit = 0
            for pair in rec["pairs"]:
                open_dom = _open_domain(pair["index"])
                for c in (pair.get("partition") or []):
                    tot += 1
                    hit += int(canonical_to_domain[c] == open_dom)
            per_round.append({
                "round": rec["round"],
                "track_pct": 100.0 * hit / tot if tot else None,
                "n": tot,
            })

    tail = [p["track_pct"] for p in per_round[-3:] if p["track_pct"] is not None]
    tail_mean = float(np.mean(tail)) if tail else 0.0

    result = {
        "canary": "observation-conditional",
        "family": FAMILY, "seed": SEED, "rounds": ROUNDS, "arrivals": ARRIVALS,
        "dom_a": DOM_A, "dom_b": DOM_B,
        "canonical_to_domain": canonical_to_domain,
        "per_round_tracking": per_round,
        "tail3_mean_tracking_pct": tail_mean,
        "curve_entropy": [c["mdo_entropy"] for c in curve],
        "curve_param_motion": [c["param_motion"] for c in curve],
        "pass": tail_mean > 90.0,
    }
    out = out_dir / "canary2_result.json"
    json.dump(result, open(out, "w"), indent=2)
    log.info("per-round tracking%%: %s",
             [round(p["track_pct"], 1) if p["track_pct"] is not None else None
              for p in per_round])
    log.info("tail-3 mean tracking: %.1f%% -> %s", tail_mean,
             "PASS" if result["pass"] else "FAIL")
    log.info("saved: %s", out)


if __name__ == "__main__":
    main()
