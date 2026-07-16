#!/usr/bin/env python
"""Track E.1 v2 — R.2(86.6%) vs R45(0.0%) 2x2, run in R.2's OWN tolerant harness.

The v1 attempt (via wp7.eval_foc) CRASHED on an infeasible plan->partition mapping
(coordinator tier-assert). That exposed a THIRD suspect the 2x2 hadn't controlled for:
R.2's run_q_cell projects the LLM plan through _llm_plan_to_greedy_result (feasible-clamped)
then calls coord.resolve_arrival directly and reads mdo.admitted (tolerant); R45's eval_foc
feeds make_llm_plan_builder's PlanSummary straight into follow_prior, which asserts.

So we run ALL FOUR cells through R.2's exact placement path (reproduces 86.6% by construction
for [greedy,train]), varying ONLY actors and the arrival stream:
  {greedy actors, frozen-BC actors} x {arrival_seed 42 (R.2 train), 819 (=42+777 held-out)}
follow_prior, seed-42 RC substrate (gen_seed, bw_override 70), local-8B plans, cache OFF, no M^B.
No Plain/ceiling re-run (run_q_cell does not call _evaluate).

Clean separation of all three suspects:
  [greedy,train] ~= 84 (reproduction) AND [bc,*] low  -> ACTORS
  [*,held] low                                         -> STREAM
  ALL FOUR ~= 84                                        -> the R45 wp7/eval_foc HARNESS (mapping) is the cause
  mixed/none                                            -> STOP AND REPORT
"""
import sys, json, copy, time, logging
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ on box

import numpy as np
import q_pilot_runner as Q
from rc_train_runner import build_rc_bc_actors, prereg_sha256

from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.actors.greedy_domain_actor import GreedyDomainActor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trackE1")

CEIL_42 = 97  # frozen seed-42 ceiling (R.2 refs); R.2 [greedy,train] = 86.6% (~84 admits)


def _build_delays(substrate):
    g = substrate.graph
    delays = {}
    for u, v in g.edges():
        ud, vd = g.nodes[u].get("domain_id", -1), g.nodes[v].get("domain_id", -1)
        if ud != vd and ud >= 0 and vd >= 0:
            k = (min(ud, vd), max(ud, vd))
            delays[k] = min(delays.get(k, 999.0), float(g[u][v]["propagation_delay"]))
    return delays


def run_cell(agent_b, kb, substrate, actors, arrival_seed):
    """Exact replica of run_q_cell's per-arrival loop (cache OFF, M^B off, real LLM),
    with INJECTABLE actors + arrival stream. Returns admits + reject taxonomy."""
    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(substrate, Q.ARRIVALS_PER_INSTANCE, Q.ARRIVAL_RATE, Q.SERVICE_RATE,
                        rng, slice_factory=Q.rc_slice_factory)
    ap.generate()
    coord = MDOCoordinator(None, actors, MDOConfig(n_part=1))
    delays = _build_delays(substrate)
    m = dict(total=0, admitted=0, structural=0, schema_fail=0, api_fail=0,
             reasons=Counter())
    for event in ap.events:
        if event.event_type != EventType.ARRIVAL or event.slice_request is None:
            continue
        sr = event.slice_request
        m["total"] += 1
        try:
            sr_dict = Q._slice_request_to_dict(sr, substrate)
            topo = Q.build_abstract_topology(substrate)
            plan_dict, check = agent_b.generate_with_memory(sr_dict, topo, kb=kb, mb=None,
                                                            max_retries=0)
        except (ValueError, KeyError):
            m["schema_fail"] += 1; m["reasons"]["schema_fail"] += 1; continue
        except Exception as e:  # api-fail etc.
            m["api_fail"] += 1; m["reasons"]["api-fail"] += 1
            log.warning("api-fail: %s", str(e)[:100]); continue
        if not getattr(check, "is_valid", False):
            m["structural"] += 1; m["reasons"]["structural"] += 1; continue
        greedy = Q._llm_plan_to_greedy_result(plan_dict, sr, substrate)
        if greedy is None:
            m["structural"] += 1; m["reasons"]["structural"] += 1; continue
        plan_summary = Q.plan_to_summary(greedy, sr, substrate)
        if plan_summary is None:
            m["reasons"]["plan_conv_fail"] += 1; continue
        mdo = coord.resolve_arrival(copy.deepcopy(substrate), sr, plan_summary, delays,
                                    mode="follow_prior")
        if mdo.admitted:
            m["admitted"] += 1
        else:
            if mdo.retry_history.attempts and mdo.retry_history.attempts[-1].violation:
                v = mdo.retry_history.attempts[-1].violation
                tag = ("actor_infeasible" if v.actor_infeasible else
                       "cross_domain_bw" if v.cross_domain_infeasible else
                       "c7_delay" if v.c7_violated else "other_violation")
            else:
                tag = "coordinator"
            m["reasons"][tag] += 1
    m["reasons"] = dict(m["reasons"])
    return m


def main():
    t0 = time.time()
    log.info("Track E.1 v2 start. prereg=%s", prereg_sha256())

    # Seed-42 RC substrate EXACTLY as the pilot/R.2 built it.
    substrate = Q.generate_rc_instance(seed=Q.RC_GEN_SEED, inter_domain_bw_override=70)
    log.info("substrate: %d domains, bw_override=70, gen_seed=%s",
             substrate.num_domains, Q.RC_GEN_SEED)

    # Local-8B Agent B (real), K^B loaded, cache off, temp 0.
    from orion.llm.llm_backend import LLMBackend, LLMConfig
    from orion.llm.agent_b import AgentB
    from orion.llm.semantic_memory import SemanticMemory
    cfg = LLMConfig(base_url="http://localhost:8000/v1", api_key="EMPTY",
                    model="default", temperature=0.0, max_tokens=2048)
    agent_b = AgentB(LLMBackend(cfg))
    kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
    kb = SemanticMemory.from_json(kb_path) if kb_path.exists() else None

    log.info("building BC actors (seed=42, 2000 scenarios, 6 epochs)...")
    bc_actors = build_rc_bc_actors(42, 2000, 6)

    def greedy_actors():
        return {d: GreedyDomainActor(d) for d in range(substrate.num_domains)}

    STREAMS = {"train(42)": 42, "held-out(819)": 819}
    ACTORS = {"greedy": greedy_actors, "bc": (lambda: bc_actors)}

    cells = {}
    for akind, amk in ACTORS.items():
        for sname, aseed in STREAMS.items():
            m = run_cell(agent_b, kb, substrate, amk(), aseed)
            adm = m["admitted"]; tot = m["total"]
            cells[f"{akind}|{sname}"] = {
                "actors": akind, "stream": sname, "admit": adm, "total": tot,
                "foc_vs_ceil97_pct": round(100.0 * adm / CEIL_42, 1),
                "admit_rate_pct": round(100.0 * adm / max(1, tot), 1),
                "reject_taxonomy": m["reasons"]}
            log.info("CELL %-7s x %-14s -> admit=%d/%d  FoC97=%.1f%%  rej=%s",
                     akind, sname, adm, tot, 100.0 * adm / CEIL_42, m["reasons"])

    out = {"track": "E.1", "seed": 42, "bw_override": 70,
           "frozen_refs": {"R2_seed42_foc_pct": 86.6, "R45_seed42_followprior_pct": 0.0,
                           "ceiling_seed42": CEIL_42},
           "note": "run in R.2's tolerant run_q_cell harness; only actors+stream vary",
           "prereg_sha256_rc_train": prereg_sha256(), "cells": cells,
           "elapsed_s": round(time.time() - t0, 1)}
    Path("data").mkdir(exist_ok=True)
    Path("data/track_e1_results.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 76)
    print("TRACK E.1 READOUT — seed 42, R.2's tolerant harness, only actors+stream vary")
    print("=" * 76)
    print(f"{'actors':<8}{'stream':<16}{'admit':>7}{'FoC/97':>9}{'admit_rate':>12}   reject_taxonomy")
    for k in ("greedy|train(42)", "greedy|held-out(819)", "bc|train(42)", "bc|held-out(819)"):
        c = cells[k]
        print(f"{c['actors']:<8}{c['stream']:<16}{c['admit']:>7}{c['foc_vs_ceil97_pct']:>8}%"
              f"{c['admit_rate_pct']:>11}%   {c['reject_taxonomy']}")
    print("\nfrozen: R.2 [greedy,train] = 86.6% (~84) ; R45 [bc,held-out] = 0.0%")

    g0 = cells["greedy|train(42)"]["admit"]; g7 = cells["greedy|held-out(819)"]["admit"]
    b0 = cells["bc|train(42)"]["admit"];     b7 = cells["bc|held-out(819)"]["admit"]
    print("\n[pre-named verdict]")
    if g0 < 40:
        print(f"  REPRODUCTION FAIL: [greedy,train] admit={g0} != frozen R.2 ~84.")
        print("  Even R.2's own harness does not reproduce here -> deeper drift. STOP AND REPORT.")
    else:
        print(f"  reproduction OK: [greedy,train] admit={g0} ~= frozen R.2 ~84.")
        if b0 >= 40 and b7 >= 40 and g0 >= 40 and g7 >= 40:
            print("  READING 3 (harness): all four cells admit high in R.2's harness.")
            print("  => R45's 0.0% is NOT actors, NOT stream -> it is the wp7/eval_foc follow_prior")
            print("     MAPPING harness. R45 verdict reclassified invalid-execution at the eval harness.")
        elif b0 < 15 and b7 < 15:
            print("  READING 1 (actors): BC actors dead on both streams; greedy high.")
            print("  => ACTOR ARTIFACT. R45 invalid-execution at the actor layer.")
        elif g7 < 15 and b7 < 15 and g0 >= 40:
            print("  READING 2 (stream): held-out collapses for both actor sets; train high.")
            print("  => STREAM/held-out path. Audit memoization; restate per-stream.")
        else:
            print(f"  MIXED: g0={g0} g7={g7} b0={b0} b7={b7}. STOP AND REPORT (battery discipline).")
    print(f"\nresults -> data/track_e1_results.json  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
