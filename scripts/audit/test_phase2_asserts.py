#!/usr/bin/env python3
"""Verification battery — Phase 2 assertion tests 2.3, 2.5, 2.6, 2.8 (+2.1 runtime).

Run on the UNMODIFIED gate family C+_T-_B- (canonical sort permutes
nontrivially: [1, 3, 2, 0, 4]), through the real EpisodeRunner/MDOCoordinator
path. No training. Instrumentation is recorder-wrappers only; every wrapped
call delegates to the original.

  2.3 Observation freshness: the obs tensor the policy consumed for arrival t
      equals the obs recomputed from the substrate at decision time (post
      allocation of t-1, all departures up to t applied), AND the h^m features
      in the trained input path equal the trace recorder's per-domain hm
      snapshot (cross-frame check via canonical_to_domain).
  2.5 Reward-transition association: per arrival, every retry trial carries
      the identical terminal reward; done marks exactly the last trial;
      transitions are contiguous and in arrival order; admitted arrivals carry
      positive reward, rejected non-positive (catches off-by-one shifts).
  2.6 Departure accounting: after a full stream (all departures drained),
      node CPU/RAM residuals and link BW residuals equal capacities exactly.
  2.8 Random-baseline parity (+2.1 round-trip): in BOTH random and sample
      modes, the partition handed to the dispatcher equals
      [canonical_to_domain[c] for c in stored_canonical_partition] for every
      attempt — same frame mapping, same dispatch path.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import five_arm_runner as R
import wp7_runner as W
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.observation import (
    DOMAIN_FEAT_DIM,
    build_domain_summaries,
    build_mdo_observation,
    observation_to_tensor,
)
from orion.actors.greedy_domain_actor import GreedyDomainActor
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner

FAMILY = "C+_T-_B-"
SEED = 42
ARRIVALS = 45
MU5 = dict(mu=5.0, alpha=0.1, xi=0.1, eta=0.1)
FAM = {f.short_name: f for f in R.ALL_FAMILIES}

results = {}


def build_env(mode_needs_policy: bool):
    sub = R.generate_family_instance(FAM[FAMILY], seed=0)
    delays = W.build_delays(sub)
    if mode_needs_policy:
        policy, coord, *_ = W.build_stack(sub, SEED, 3e-3, actors=None,
                                          mdo_cfg=MDOConfig(n_part=3, **MU5))
    else:
        actors = {d: GreedyDomainActor(d) for d in range(sub.num_domains)}
        coord = MDOCoordinator(None, actors, MDOConfig(n_part=3, **MU5))
    rng = np.random.default_rng(SEED)
    ap = ArrivalProcess(sub, ARRIVALS, W.ARRIVAL_RATE, W.SERVICE_RATE, rng)
    ap.generate()
    runner = EpisodeRunner(sub, ap, coord, delays, plan_builder=W.greedy_plan_builder)
    return sub, coord, runner


# ── 2.3 Observation freshness ────────────────────────────────────────────────
def check_2_3():
    sub, coord, runner = build_env(mode_needs_policy=False)
    canonical_to_domain = [s.domain_id for s in build_domain_summaries(sub)]
    M = len(canonical_to_domain)

    records = []
    orig_resolve = coord.resolve_arrival

    def recording_resolve(substrate, slice_req, plan, inter_domain_delays,
                          mode="sample", cost_greedy=None):
        expected = observation_to_tensor(build_mdo_observation(substrate, plan))
        res = orig_resolve(substrate=substrate, slice_req=slice_req, plan=plan,
                           inter_domain_delays=inter_domain_delays, mode=mode,
                           cost_greedy=cost_greedy)
        records.append((slice_req.request_id, expected, res.obs_tensor))
        return res

    coord.resolve_arrival = recording_resolve
    runner.reset()
    ep = runner.run_episode(mdo_mode="random")

    assert len(records) > 10, "too few MDO-reaching arrivals to test"
    mismatches = [rid for rid, exp, got in records if not torch.equal(exp, got)]
    assert not mismatches, f"obs != decision-time state for {len(mismatches)} arrivals: {mismatches[:3]}"

    # Teeth: obs must actually evolve across the stream (allocations visible).
    distinct = len({tuple(exp[: M * DOMAIN_FEAT_DIM].tolist()) for _, exp, _ in records})
    assert distinct > 1, "domain features never changed across a 45-arrival stream — no teeth"

    # h^m in the trained input path == trace recorder's hm (frame-aware).
    trace_by_rid = {p["rid"]: p for p in ep.arrival_trace}
    checked = 0
    for rid, exp, got in records:
        p = trace_by_rid.get(rid)
        if p is None or p.get("hm") is None:
            continue
        for m in range(M):
            obs_hm = float(got[m * DOMAIN_FEAT_DIM + 5])
            trace_hm = float(p["hm"][canonical_to_domain[m]])
            assert abs(obs_hm - trace_hm) < 1e-6, (
                f"h^m mismatch rid={rid} canonical={m} domain={canonical_to_domain[m]}: "
                f"obs={obs_hm} trace={trace_hm}")
            checked += 1
    assert checked > 0, "no hm snapshots to cross-check"
    return f"PASS ({len(records)} arrivals, {distinct} distinct domain-feature states, {checked} h^m cross-checks)"


# ── 2.5 Reward-transition association ────────────────────────────────────────
def check_2_5():
    sub, coord, runner = build_env(mode_needs_policy=False)
    runner.reset()
    ep = runner.run_episode(mdo_mode="random")

    result_by_rid = {r.request_id: r for r in ep.mdo_results}
    # group transitions by request, tracking stream order + contiguity
    seen_order, groups = [], {}
    for t in ep.rollout.mdo:
        if t.request_id not in groups:
            groups[t.request_id] = []
            seen_order.append(t.request_id)
        else:
            assert seen_order[-1] == t.request_id, (
                f"transitions of {t.request_id} not contiguous")
        groups[t.request_id].append(t)

    # arrival order must match mdo_results order (both MDO-reaching only)
    assert seen_order == [r.request_id for r in ep.mdo_results], "arrival order mismatch"

    n_admit = n_reject = 0
    for rid, ts in groups.items():
        res = result_by_rid[rid]
        rewards = {t.terminal_reward for t in ts}
        assert len(rewards) == 1, f"{rid}: trials carry different rewards {rewards}"
        assert len(ts) == res.retry_history.num_attempts, (
            f"{rid}: {len(ts)} transitions vs {res.retry_history.num_attempts} attempts")
        for j, t in enumerate(ts):
            assert t.trial_index == j, f"{rid}: trial_index disorder"
            assert t.committed == (j == len(ts) - 1), f"{rid}: done flag not on last trial"
        r = ts[0].terminal_reward
        if res.admitted:
            n_admit += 1
            assert r > 0, f"{rid}: admitted but reward {r} <= 0 (off-by-one suspect)"
        else:
            n_reject += 1
            assert r <= 0, f"{rid}: rejected but reward {r} > 0 (off-by-one suspect)"
    assert n_admit > 0 and n_reject > 0, "need both outcomes for the sign association to have teeth"
    return f"PASS ({len(groups)} arrivals: {n_admit} admitted, {n_reject} rejected, all trials consistent)"


# ── 2.6 Departure accounting / resource conservation ────────────────────────
def check_2_6():
    sub, coord, runner = build_env(mode_needs_policy=False)
    runner.reset()
    g = sub.graph
    cap = {n: (d["cpu_capacity"], d["ram_capacity"]) for n, d in g.nodes(data=True)}
    bw_cap = {d["link_id"]: d["bandwidth_capacity"] for _, _, d in g.edges(data=True)}

    ep = runner.run_episode(mdo_mode="random")
    assert ep.stats.admitted > 0, "no admissions — conservation check has no teeth"
    assert ep.stats.departures == ep.stats.admitted, (
        f"departures {ep.stats.departures} != admissions {ep.stats.admitted} "
        f"(stream not fully drained or leak)")
    assert not runner._active_plans, f"{len(runner._active_plans)} plans still active after drain"

    bad = []
    for n, d in g.nodes(data=True):
        if abs(d["cpu_residual"] - cap[n][0]) > 1e-6 or abs(d["ram_residual"] - cap[n][1]) > 1e-6:
            bad.append((n, d["cpu_residual"], cap[n][0], d["ram_residual"], cap[n][1]))
    assert not bad, f"node residuals != capacity after full drain: {bad[:5]}"
    bad_bw = []
    for u, v, d in g.edges(data=True):
        if abs(d["bw_residual"] - bw_cap[d["link_id"]]) > 1e-6:
            bad_bw.append((d["link_id"], d["bw_residual"], bw_cap[d["link_id"]]))
    assert not bad_bw, f"link BW residuals != capacity after full drain: {bad_bw[:5]}"
    return (f"PASS ({ep.stats.admitted} admitted, {ep.stats.departures} departed, "
            f"{g.number_of_nodes()} nodes + {g.number_of_edges()} links conserved to 1e-6)")


# ── 2.8 Random-baseline parity + 2.1 runtime round-trip ─────────────────────
def check_2_8():
    out = {}
    for mode, needs_policy in (("random", False), ("sample", True)):
        sub, coord, runner = build_env(mode_needs_policy=needs_policy)
        canonical_to_domain = [s.domain_id for s in build_domain_summaries(sub)]
        dispatched = []
        orig_bf = coord._build_fragments

        def recording_bf(plan, partition, slice_req, _orig=orig_bf, _rec=dispatched):
            _rec.append(list(partition))
            return _orig(plan, partition, slice_req)

        coord._build_fragments = recording_bf
        runner.reset()
        ep = runner.run_episode(mdo_mode=mode)

        attempts = [a for r in ep.mdo_results for a in r.retry_history.attempts]
        assert len(attempts) == len(dispatched), (
            f"{mode}: {len(attempts)} attempts vs {len(dispatched)} dispatches")
        for a, disp in zip(attempts, dispatched):
            expect = [canonical_to_domain[c] for c in a.partition]
            assert disp == expect, (
                f"{mode}: dispatch {disp} != canonical_to_domain(stored {a.partition}) = {expect}")
        out[mode] = len(attempts)
        assert canonical_to_domain != list(range(len(canonical_to_domain))), (
            "sort did not permute — test would have no teeth on this family")
    return (f"PASS (round-trip holds for every attempt: random n={out['random']}, "
            f"sample n={out['sample']}; permutation nontrivial)")


def main():
    print(f"family={FAMILY} (canonical sort permutes), seed={SEED}, arrivals={ARRIVALS}\n")
    for name, fn in (("2.3 observation freshness", check_2_3),
                     ("2.5 reward-transition association", check_2_5),
                     ("2.6 departure accounting", check_2_6),
                     ("2.8 random parity + 2.1 round-trip", check_2_8)):
        try:
            msg = fn()
        except AssertionError as e:
            msg = f"FAIL — {e}"
        except Exception as e:  # noqa: BLE001
            msg = f"ERROR — {type(e).__name__}: {e}"
        results[name] = msg
        print(f"[{name}] {msg}")
    n_fail = sum(1 for v in results.values() if not v.startswith("PASS"))
    print(f"\n{'ALL GREEN' if n_fail == 0 else f'{n_fail} CHECK(S) NOT GREEN'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
