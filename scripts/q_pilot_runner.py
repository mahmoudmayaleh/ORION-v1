"""§Q pilot runner — ORION-frontier (Claude Sonnet) vs ORION-local (LLaMA-3-8B),
cache ON, on the routing-critical family C+_T-_B-_RC, seed 42.

Pre-registered by PREREG_AMENDMENT_2026-07-15_Q.md (+ Q.4 Δ RC-v2) and the
pilot ruling (2026-07-15). Deployable stack: follow_prior + GreedyDomainActor +
router, K^B + M^B live, plan cache ON (revalidate + drift-invalidate, §6.3),
cold-start per (seed, family) cell.

Pilot gate (numeric, both must hold):
  (1) ORION-frontier structural-reject rate <= 1/2 * ORION-local's, AND
  (2) FoC(frontier) >= FoC(local) + 5 pp.
Readout order (committed): schema/structural validity (void triggers) →
reject taxonomy frontier vs local → FoC. Raw numbers go to senior before the
band and grid cap are set. Clears → band + grid cap committed → grid. Fails →
stop; §P scoped claim is the paper; RC characterization is the envelope section.

Run on the box (needs the local llama.cpp server on :8000 and
ORION_FRONTIER_API_KEY sourced). --mock swaps both LLMs for FFD to smoke the
wiring without servers or spend.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from orion.llm.abstract_topology import build_abstract_topology
from orion.llm.plan_cache import (
    plan_signature, sfc_template, AbstractPlan, instantiate_plan, revalidate_plan, PlanCache,
)
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.substrate.routing_critical import (
    generate_rc_instance, rc_slice_factory, rc_signature,
    RC_FAMILY_SHORT, RC_GEN_SEED, RC_BW_OVERRIDES,
)

# Reused machinery from the five-arm runner (verifier, coordinator, converters).
from five_arm_runner import (
    _slice_request_to_dict, _llm_plan_to_greedy_result, plan_to_summary,
    _extract_plan_shape, run_static_arm, write_to_mb,
    ARRIVALS_PER_INSTANCE, ARRIVAL_RATE, SERVICE_RATE,
)
from rc_family_validity import _evaluate  # ceiling + Plain over the RC stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("q_pilot")

PILOT_SEED = 42
PILOT_BW_OVERRIDE = RC_BW_OVERRIDES[0]  # 70, paired with seed 42 (Q.4)
MEMORY_CAPACITY_K = 50


def prereg_sha256() -> str:
    p = Path(__file__).resolve().parent.parent / "docs" / "PREREG_AMENDMENT_2026-07-15_Q.md"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


_EDGE_TIERS = {"ran_edge", "mec"}
_CLOUD_TIERS = {"regional_cloud", "central_cloud"}


def _rc_forced(sr, substrate) -> bool:
    """True if this arrival is a forced trap chain: head VNF pinned to edge tiers
    AND tail VNF pinned to cloud tiers (rc_slice_factory's ~60% forced coin).
    Flexible chains use {mec, regional_cloud} at every position, so a flexible head
    admits regional_cloud and fails the edge-subset test."""
    if not sr.vnfs:
        return False
    g = substrate.graph

    def _tiers(vnf):
        return {str(g.nodes[n]["tier"]).split(".")[-1].lower()
                for n in vnf.permitted_nodes if n in g.nodes}

    head, tail = _tiers(sr.vnfs[0]), _tiers(sr.vnfs[-1])
    return bool(head) and bool(tail) and head <= _EDGE_TIERS and tail <= _CLOUD_TIERS


def state_hash(plan_cache, mb) -> str:
    """Cold-start assertion: hash of (cache size, M^B size). Empty at stream start."""
    n_cache = len(plan_cache.entries) if plan_cache is not None else 0
    n_mb = len(getattr(mb, "_entries", [])) if mb is not None else 0
    return f"cache={n_cache},mb={n_mb}", (n_cache == 0 and n_mb == 0)


def _build_arms(mock: bool, snapshot: str | None, cost_cap: float):
    """Return {arm_name: (agent_b_or_None, cost_meter_or_None)} + kb, mock flag."""
    from orion.llm.semantic_memory import SemanticMemory

    kb = None
    kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
    if kb_path.exists():
        kb = SemanticMemory.from_json(kb_path)
        logger.info("K^B loaded: %d entries", len(kb.entries))

    if mock:
        return {"ORION-local": (None, None), "ORION-frontier": (None, None)}, kb

    from orion.llm.llm_backend import LLMBackend, LLMConfig
    from orion.llm.agent_b import AgentB
    from orion.llm.frontier_backend import FrontierBackend, CostMeter, make_frontier_config

    # Local: LLaMA-3-8B on :8000 (deterministic).
    local_cfg = LLMConfig(base_url="http://localhost:8000/v1", api_key="EMPTY",
                          model="default", temperature=0.0, max_tokens=2048)
    local_agent = AgentB(LLMBackend(local_cfg))

    # Frontier: Claude Sonnet, pinned snapshot, hardened backend.
    if not snapshot:
        raise SystemExit("--snapshot <dated-sonnet-id> is required for a real frontier run "
                         "(pin it at pilot time per ruling 4).")
    cost = CostMeter(cap_usd=cost_cap)
    frontier_agent = AgentB(FrontierBackend(make_frontier_config(snapshot), cost))
    logger.info("Frontier arm: Sonnet snapshot=%s cap=$%.0f", snapshot, cost_cap)
    return {"ORION-local": (local_agent, None),
            "ORION-frontier": (frontier_agent, cost)}, kb


def run_q_cell(arm, agent_b, kb, mb, plan_cache, substrate, arrival_seed, mock):
    """Cache-ON instance loop for one (seed, family) cell. Returns a metrics dict."""
    from orion.actors.greedy_domain_actor import GreedyDomainActor
    from orion.mdo.coordinator import MDOConfig, MDOCoordinator
    from orion.llm.frontier_backend import ApiFailError, CostCapExceeded

    # Cold-start assertion (R2): state hash empty at stream start, logged.
    sh, empty = state_hash(plan_cache, mb)
    logger.info("  [%s] cold-start state @ stream start: %s (empty=%s)", arm, sh, empty)
    assert empty, f"cold-start violated for {arm}: {sh}"

    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(substrate, ARRIVALS_PER_INSTANCE, ARRIVAL_RATE, SERVICE_RATE, rng,
                        slice_factory=rc_slice_factory)
    ap.generate()

    actors = {d: GreedyDomainActor(d) for d in range(substrate.num_domains)}
    coord = MDOCoordinator(None, actors, MDOConfig(n_part=1))
    delays = {}
    g = substrate.graph
    for u, v in g.edges():
        ud, vd = g.nodes[u].get("domain_id", -1), g.nodes[v].get("domain_id", -1)
        if ud != vd and ud >= 0 and vd >= 0:
            k = (min(ud, vd), max(ud, vd))
            delays[k] = min(delays.get(k, 999.0), float(g[u][v]["propagation_delay"]))

    topo_sig = rc_signature(substrate)
    m = dict(total=0, admitted=0, structural=0, schema_fail=0, api_fail=0,
             plan_conv_fail=0, cache_hit=0, cache_miss=0, reasons=Counter(), trace=[])

    for event in ap.events:
        if event.event_type != EventType.ARRIVAL or event.slice_request is None:
            continue
        sr = event.slice_request
        m["total"] += 1
        plan_summary = None
        plan_dict = {}
        # Per-arrival trace (paper record, §R): updated in-place through the arrival.
        fe = sr.flow_edges
        tr = {"rid": sr.request_id, "n_vnfs": len(sr.vnfs),
              "bw_head": round(fe[0].bandwidth_demand, 1) if fe else None,
              "bw_tail": round(fe[-1].bandwidth_demand, 1) if fe else None,
              "forced": _rc_forced(sr, substrate),
              "cache_hit": False, "admitted": False, "reject": None,
              # M^B retrieval composition for THIS arrival's prompt (None = M^B
              # off or not consulted). The exemplar-poisoning test needs the
              # retrieved label mix beside the plan's fate, per arrival.
              "mb_retr_n": None, "mb_retr_pos": None, "mb_retr_neg": None}
        m["trace"].append(tr)

        # ── Cache lookup (§6.3): hit + revalidate → reuse, no LLM call ──
        key = plan_signature(sr)
        entry = None if plan_cache is None else plan_cache.get(key)
        if entry is not None:
            abstract = entry.plan
            if revalidate_plan(abstract, substrate):
                try:
                    plan_summary = instantiate_plan(abstract, sr)
                    m["cache_hit"] += 1
                    tr["cache_hit"] = True
                except ValueError:
                    plan_summary = None
            if plan_summary is None and plan_cache is not None:
                plan_cache.mark_stale(key)  # drift → refresh below

        # ── Cache miss → plan builder (mock FFD, or Agent B over K^B+M^B) ──
        if plan_summary is None:
            m["cache_miss"] += 1
            if mock:
                from orion.baselines.greedy_ffd import _run_greedy_ffd
                from orion.baselines.colocation_ffd import GreedyConfig as _GC
                res = _run_greedy_ffd(substrate, sr, _GC())
                if not res.feasible or res.plan is None:
                    m["structural"] += 1
                    tr["reject"] = "structural"
                    if mb is not None:
                        write_to_mb(mb, arm, sr, False, {}, ["structural"], topo_sig, None)
                    continue
                plan_dict = {"vnf_assignments": [
                    {"vnf_id": v.vnf_id,
                     "domain": g.nodes[res.plan.vnf_placements[v.vnf_id]]["domain_id"]}
                    for v in sr.vnfs]}
            else:
                try:
                    sr_dict = _slice_request_to_dict(sr, substrate)
                    topo = build_abstract_topology(substrate)
                    plan_dict, check = agent_b.generate_with_memory(
                        sr_dict, topo, kb=kb, mb=mb, max_retries=0)
                    # What M^B actually fed this prompt. Read-only telemetry:
                    # EpisodicMemory.retrieve() sets _last_retrieval, labelling
                    # via entry.tags["label"] (the correct read -- see the
                    # _mb_composition bug, 2026-07-16).
                    _lr = getattr(mb, "_last_retrieval", None) if mb is not None else None
                    if _lr is not None:
                        tr["mb_retr_n"] = _lr.get("n")
                        tr["mb_retr_pos"] = _lr.get("pos")
                        tr["mb_retr_neg"] = _lr.get("neg")
                except CostCapExceeded:
                    logger.error("  [%s] COST CAP hit — refusing further calls, stopping cell.", arm)
                    m["cost_stopped"] = True
                    break
                except ApiFailError as e:
                    m["api_fail"] += 1
                    m["reasons"]["api-fail"] += 1
                    tr["reject"] = "api-fail"
                    logger.warning("  [%s] api-fail: %s", arm, str(e)[:120])
                    continue
                except (ValueError, KeyError):
                    m["schema_fail"] += 1
                    m["reasons"]["schema_fail"] += 1
                    tr["reject"] = "schema_fail"
                    continue
                if not getattr(check, "is_valid", False):
                    m["structural"] += 1
                    m["reasons"]["structural"] += 1
                    tr["reject"] = "structural"
                    if mb is not None:
                        write_to_mb(mb, arm, sr, False, plan_dict, ["structural"], topo_sig, None)
                    continue

            greedy = _llm_plan_to_greedy_result(plan_dict, sr, substrate)
            if greedy is None:
                m["structural"] += 1
                m["reasons"]["structural"] += 1
                tr["reject"] = "structural"
                if mb is not None:
                    write_to_mb(mb, arm, sr, False, plan_dict, ["structural"], topo_sig, None)
                continue
            plan_summary = plan_to_summary(greedy, sr, substrate)
            if plan_summary is None:
                m["plan_conv_fail"] += 1
                tr["reject"] = "plan_conv_fail"
                continue
            # Cache the request-invariant abstract plan for this signature.
            if plan_cache is not None:
                abstract = AbstractPlan(
                    sfc_template=sfc_template(sr),
                    required_tiers=list(plan_summary.required_tiers),
                    suggested_domains=list(plan_summary.suggested_domains),
                )
                if key in plan_cache.entries:
                    plan_cache.refresh(key, abstract)
                else:
                    plan_cache.put(key, abstract)

        # ── Route through the shared coordinator (identical to every arm) ──
        mdo = coord.resolve_arrival(copy.deepcopy(substrate), sr, plan_summary, delays,
                                    mode="follow_prior")
        if mdo.admitted:
            m["admitted"] += 1
            tr["admitted"] = True
            if mb is not None:
                shape = _extract_plan_shape(
                    _llm_plan_to_greedy_result(plan_dict, sr, substrate) if plan_dict else None,
                    sr, substrate)
                write_to_mb(mb, arm, sr, True, plan_dict, [], topo_sig, shape)
        else:
            if mdo.retry_history.attempts:
                last = mdo.retry_history.attempts[-1]
                if last.violation:
                    v = last.violation
                    tag = ("actor_infeasible" if v.actor_infeasible else
                           "cross_domain_bw" if v.cross_domain_infeasible else
                           "c7_delay" if v.c7_violated else "other_violation")
                else:
                    tag = "unknown"
                m["reasons"][tag] += 1
                tr["reject"] = tag
            else:
                tr["reject"] = "coordinator"
            if mb is not None:
                write_to_mb(mb, arm, sr, False, plan_dict, [], topo_sig, None)

    m["reasons"] = dict(m["reasons"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="FFD stand-in for both LLMs (smoke)")
    ap.add_argument("--snapshot", default=None, help="dated Sonnet snapshot id (real run)")
    ap.add_argument("--cost-cap", type=float, default=10.0, help="pilot $ cap (default 10)")
    args = ap.parse_args()

    from orion.llm.episodic_memory import EpisodicMemory
    from orion.retrieval import RetrievalConfig, RetrievalMode

    logger.info("§Q PILOT — %s seed=%d bw=%g | prereg_sha256=%s",
                RC_FAMILY_SHORT, PILOT_SEED, PILOT_BW_OVERRIDE, prereg_sha256()[:12])

    substrate = generate_rc_instance(seed=RC_GEN_SEED, inter_domain_bw_override=PILOT_BW_OVERRIDE)

    # Context: ceiling + Plain over the identical RC stream.
    total, ceiling, plain = _evaluate(substrate, arrival_seed=PILOT_SEED)
    plain_foc = 100.0 * plain / ceiling if ceiling else float("nan")
    logger.info("Reference: total=%d ceiling=%d Plain=%d (Plain FoC=%.1f%%)",
                total, ceiling, plain, plain_foc)

    arms, kb = _build_arms(args.mock, args.snapshot, args.cost_cap)
    results = {}
    for arm, (agent_b, cost) in arms.items():
        mb = EpisodicMemory(
            config=RetrievalConfig(mode=RetrievalMode.NO_RERANK, apply_recency=True, k_final=3),
            max_entries=MEMORY_CAPACITY_K, write_policy="selective", evict_policy="importance")
        plan_cache = PlanCache(capacity=64)  # per-(seed,family) cold cache
        m = run_q_cell(arm, agent_b, kb, mb, plan_cache, substrate, PILOT_SEED, args.mock)
        m["foc"] = 100.0 * m["admitted"] / ceiling if ceiling else float("nan")
        m["struct_reject_rate"] = 100.0 * m["structural"] / m["total"] if m["total"] else 0.0
        m["schema_fail_rate"] = 100.0 * m["schema_fail"] / m["total"] if m["total"] else 0.0
        if cost is not None:
            m["cost"] = cost.summary()
        results[arm] = m

    _readout(results, ceiling, plain_foc)


def _readout(results, ceiling, plain_foc):
    L, F = results["ORION-local"], results["ORION-frontier"]
    print("\n" + "=" * 72)
    print("Q PILOT READOUT (order: validity -> reject taxonomy -> FoC)")
    print("=" * 72)

    # 1. Validity first (void triggers: malformed/content-error > 10% → arm VOID)
    print("\n[1] VALIDITY (void trigger = schema/api-fail > 10%)")
    for name, m in (("ORION-local", L), ("ORION-frontier", F)):
        print(f"  {name:15s} schema_fail={m['schema_fail']} ({m['schema_fail_rate']:.1f}%)  "
              f"api_fail={m.get('api_fail',0)}  cache_hit={m['cache_hit']} miss={m['cache_miss']}"
              + (f"  cost={m['cost']}" if 'cost' in m else ""))
        if m["schema_fail_rate"] > 10.0:
            print(f"      !! {name} VOID — schema-fail > 10%")

    # 2. Reject taxonomy frontier vs local
    print("\n[2] REJECT TAXONOMY (frontier vs local)")
    keys = sorted(set(L["reasons"]) | set(F["reasons"]))
    print(f"  {'reason':20s} {'local':>8s} {'frontier':>10s}")
    for k in keys:
        print(f"  {k:20s} {L['reasons'].get(k,0):>8d} {F['reasons'].get(k,0):>10d}")
    print(f"  {'structural(builder)':20s} {L['structural']:>8d} {F['structural']:>10d}")

    # 3. FoC + the gate
    print("\n[3] FoC (admitted / ceiling)  [ceiling=%d, Plain FoC=%.1f%%]" % (ceiling, plain_foc))
    print(f"  ORION-local    FoC={L['foc']:.1f}%  struct-reject={L['struct_reject_rate']:.1f}%")
    print(f"  ORION-frontier FoC={F['foc']:.1f}%  struct-reject={F['struct_reject_rate']:.1f}%")

    print("\n[GATE] (both must hold)")
    c1 = F["struct_reject_rate"] <= 0.5 * L["struct_reject_rate"]
    c2 = F["foc"] >= L["foc"] + 5.0
    print(f"  (1) frontier struct-reject ({F['struct_reject_rate']:.1f}%) <= 1/2 local "
          f"({0.5*L['struct_reject_rate']:.1f}%): {'PASS' if c1 else 'FAIL'}")
    print(f"  (2) frontier FoC ({F['foc']:.1f}%) >= local ({L['foc']:.1f}%) + 5pp "
          f"({L['foc']+5:.1f}%): {'PASS' if c2 else 'FAIL'}")
    verdict = ("CLEARS -> set band + grid cap, fire grid" if (c1 and c2)
               else "FAILS -> stop; P scoped claim is the paper; RC = envelope section")
    print(f"  VERDICT: {verdict}")
    print("  (raw numbers above go to senior BEFORE band + grid cap are set.)")


if __name__ == "__main__":
    main()
