#!/usr/bin/env python3
"""Full-path stub test: verify the abstract-to-concrete handoff.

Wraps ColocFB's concrete decisions into Agent B's exact output schema,
then routes through the full real pipeline:
  abstract plan → structural checker → PlanSummary → selector (follow_prior)
  → deterministic placer → router → admission count

Must confirm:
  1. The abstract-to-concrete handoff works (no crashes)
  2. A ColocFB-equivalent abstract plan scores approximately what ColocFB
     scores directly (symmetry test)
  3. M^B writes and retrievals work through the full path
  4. Fraction-of-ceiling computed through the same gate

If the same decisions lose points crossing the abstract-to-concrete boundary,
the LLM arms carry a structural handicap.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.greedy_domain_actor import GreedyDomainActor
from orion.baselines.colocation_ffd import colocation_ffd
from orion.baselines.greedy_ffd import GreedyConfig
from orion.llm.abstract_topology import build_abstract_topology
from orion.llm.episodic_memory import EpisodicMemory
from orion.llm.structural_checker import check_plan
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.types import PlanSummary
from orion.retrieval import RetrievalConfig, RetrievalMode
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.episode_runner import EpisodeRunner
from orion.substrate.topology_families import (
    ALL_FAMILIES, generate_family_instance, compute_signature,
)
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_DOMAINS = 5
ARRIVALS = 100
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02


def coloc_to_abstract_plan(coloc_result, sr, substrate):
    """Wrap ColocFB's concrete plan into Agent B's abstract output schema."""
    if not coloc_result.feasible or coloc_result.plan is None:
        return None

    g = substrate.graph
    plan = coloc_result.plan

    vnf_assignments = []
    for vnf in sr.vnfs:
        nid = plan.vnf_placements[vnf.vnf_id]
        dom = g.nodes[nid]["domain_id"]
        tier = g.nodes[nid]["tier"]

        # Get permitted tiers for this VNF
        permitted_tiers = sorted({g.nodes[n]["tier"] for n in vnf.permitted_nodes if n in g.nodes})

        vnf_assignments.append({
            "vnf_id": vnf.vnf_id,
            "domain": f"d{dom}",
            "required_tier": tier,
            "cpu_demand": vnf.cpu_demand,
            "ram_demand": vnf.ram_demand,
        })

    flow_requirements = []
    for fe in sr.flow_edges:
        src_nid = plan.vnf_placements[fe.source_vnf]
        dst_nid = plan.vnf_placements[fe.target_vnf]
        crosses = g.nodes[src_nid]["domain_id"] != g.nodes[dst_nid]["domain_id"]
        flow_requirements.append({
            "source_vnf": fe.source_vnf,
            "target_vnf": fe.target_vnf,
            "min_bandwidth_mbps": fe.bandwidth_demand,
            "crosses_domain_boundary": crosses,
        })

    return {
        "plan_id": f"{sr.request_id}_stub",
        "vnf_assignments": vnf_assignments,
        "flow_requirements": flow_requirements,
        "rationale": "Stub: ColocFB decisions wrapped in abstract format.",
    }


def abstract_plan_to_plan_summary(abstract_plan, sr, substrate):
    """Convert Agent B's abstract plan to PlanSummary for the coordinator."""
    g = substrate.graph
    assignments = {a["vnf_id"]: a for a in abstract_plan["vnf_assignments"]}

    vnf_ids = []
    required_tiers = []
    suggested_domains = []
    cpu_demands = []
    ram_demands = []
    vcrs = []

    for vnf in sr.vnfs:
        a = assignments[vnf.vnf_id]
        domain_id = int(a["domain"][1:])  # "d0" -> 0

        vnf_ids.append(vnf.vnf_id)
        required_tiers.append(InfrastructureTier(a["required_tier"]))
        suggested_domains.append(domain_id)
        cpu_demands.append(vnf.cpu_demand)
        ram_demands.append(vnf.ram_demand)
        vcrs.append(vnf.vcr)

    bw_demands = [f.bandwidth_demand for f in sr.flow_edges]

    return PlanSummary(
        vnf_ids=vnf_ids,
        required_tiers=required_tiers,
        suggested_domains=suggested_domains,
        cpu_demands=cpu_demands,
        ram_demands=ram_demands,
        vcrs=vcrs,
        bw_demands=bw_demands,
    )


def build_slice_request_dict(sr, substrate):
    """Convert SliceRequest to dict for the structural checker."""
    g = substrate.graph
    vnfs = []
    for v in sr.vnfs:
        permitted_tiers = sorted({g.nodes[n]["tier"] for n in v.permitted_nodes if n in g.nodes})
        vnfs.append({
            "vnf_id": v.vnf_id,
            "cpu_demand": v.cpu_demand,
            "ram_demand": v.ram_demand,
            "permitted_tiers": permitted_tiers,
            "computational_intensity": v.computational_intensity,
            "vcr": v.vcr,
        })
    return {
        "request_id": sr.request_id,
        "slice_type": sr.slice_type.value,
        "vnfs": vnfs,
        "flow_edges": [
            {"source_vnf": f.source_vnf, "target_vnf": f.target_vnf,
             "bandwidth_demand": f.bandwidth_demand}
            for f in sr.flow_edges
        ],
        "qos": {"max_e2e_delay": sr.qos.max_e2e_delay, "min_throughput": sr.qos.min_throughput},
    }


def build_delays(sub):
    delays = {}
    for u, v in sub.graph.edges():
        u_dom = sub.graph.nodes[u].get("domain_id", -1)
        v_dom = sub.graph.nodes[v].get("domain_id", -1)
        if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
            key = (min(u_dom, v_dom), max(u_dom, v_dom))
            delays.setdefault(key, 8.0)
    return delays


def main():
    logger.info("=" * 90)
    logger.info("FULL-PATH STUB TEST: abstract-to-concrete handoff verification")
    logger.info("=" * 90)

    # Test on one instance per family, one seed
    test_families = ["C+_T+_B-", "C-_T-_B-", "C+_T-_B+", "C-_T+_B-"]
    cfg = GreedyConfig()

    total_direct = 0
    total_pipeline = 0
    total_checker_fails = 0
    total_arrivals = 0
    total_mb_writes = 0

    for fname in test_families:
        family = [f for f in ALL_FAMILIES if f.short_name == fname][0]
        sub = generate_family_instance(family, seed=0)
        delays = build_delays(sub)
        topo_sig = compute_signature(sub, fname).to_dict()

        # Build actors (deterministic best-fit)
        actors = {}
        for d in range(NUM_DOMAINS):
            actors[d] = GreedyDomainActor(domain_id=d)

        # Build M^B
        mb = EpisodicMemory(
            config=RetrievalConfig(mode=RetrievalMode.NO_RERANK, apply_recency=True, k_final=3),
            max_entries=50,
        )

        rng = np.random.default_rng(42)
        ap = ArrivalProcess(sub, ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
        ap.generate()

        direct_ok = pipeline_ok = checker_fail = 0
        arrival_count = 0

        for event in ap.events:
            if event.event_type != EventType.ARRIVAL or event.slice_request is None:
                continue
            arrival_count += 1
            sr = event.slice_request

            # ── Direct path (static arm) ──
            coloc_result = colocation_ffd(sub, sr, cfg)
            direct_admitted = coloc_result.feasible
            if direct_admitted:
                direct_ok += 1

            if not coloc_result.feasible:
                continue  # Skip pipeline test for kills (both would fail)

            # ── Full pipeline path (what LLM arm does) ──

            # Step 1: Wrap concrete decisions in abstract format
            abstract_plan = coloc_to_abstract_plan(coloc_result, sr, sub)
            if abstract_plan is None:
                continue

            # Step 2: Structural checker
            abstract_topo = build_abstract_topology(sub)
            sr_dict = build_slice_request_dict(sr, sub)
            check_result = check_plan(abstract_plan, sr_dict, abstract_topo)

            if not check_result.is_valid:
                checker_fail += 1
                if checker_fail <= 3:
                    logger.warning("  Checker FAIL on %s: %s",
                                   sr.request_id, check_result.summary())
                continue

            # Step 3: Convert to PlanSummary
            plan_summary = abstract_plan_to_plan_summary(abstract_plan, sr, sub)

            # Step 4: Coordinator dispatch (follow_prior + deterministic placer)
            coord = MDOCoordinator(None, actors, MDOConfig(n_part=1))
            mdo_result = coord.resolve_arrival(
                copy.deepcopy(sub), sr, plan_summary, delays, mode="follow_prior")

            if mdo_result.admitted:
                pipeline_ok += 1

            # Step 5: M^B write
            violations = []
            if not mdo_result.admitted and mdo_result.retry_history.attempts:
                last = mdo_result.retry_history.attempts[-1]
                if last.violation:
                    if last.violation.actor_infeasible:
                        violations.append("actor_infeasible")
                    elif last.violation.cross_domain_infeasible:
                        violations.append("C5b")
                    elif last.violation.c7_violated:
                        violations.append("C7")

            mb.record(
                slice_spec={"slice_type": sr.slice_type.value, "num_vnfs": len(sr.vnfs)},
                plan=abstract_plan,
                m_committed=mdo_result.total_cost,
                constraints_violated=violations,
                reward=1.0 if mdo_result.admitted else -1.0,
                topology_signature=topo_sig,
                plan_shape={"strategy": "co-locate", "domains_used": [0]},
                violation_tag=violations[0] if violations else None,
            )

        # M^B retrieval test
        if mb._entries:
            query = f"eMBB placement on {fname}"
            retrieved = mb.retrieve(query, top_k=3)
            few_shot = mb.to_few_shot(retrieved)
            total_mb_writes += len(mb._entries)

        total_direct += direct_ok
        total_pipeline += pipeline_ok
        total_checker_fails += checker_fail
        total_arrivals += arrival_count

        gap = direct_ok - pipeline_ok
        logger.info("")
        logger.info("  %s (%d arrivals):", fname, arrival_count)
        logger.info("    Direct (ColocFB):     %d admitted", direct_ok)
        logger.info("    Pipeline (abstract):  %d admitted", pipeline_ok)
        logger.info("    Checker failures:     %d", checker_fail)
        logger.info("    Gap (direct-pipeline): %d arrivals", gap)
        logger.info("    M^B entries:          %d", len(mb._entries))
        logger.info("    M^B retrieval test:   %d entries retrieved, %d few-shot parsed",
                     len(retrieved) if mb._entries else 0,
                     len(few_shot) if mb._entries else 0)

    logger.info("")
    logger.info("=" * 90)
    logger.info("SYMMETRY TEST SUMMARY")
    logger.info("=" * 90)
    logger.info("  Total arrivals:     %d", total_arrivals)
    logger.info("  Direct admitted:    %d", total_direct)
    logger.info("  Pipeline admitted:  %d", total_pipeline)
    logger.info("  Checker failures:   %d", total_checker_fails)
    logger.info("  M^B total writes:   %d", total_mb_writes)

    gap = total_direct - total_pipeline
    if gap == 0 and total_checker_fails == 0:
        logger.info("")
        logger.info("  PASS: Zero gap. Same decisions score identically through both paths.")
        logger.info("  The abstract-to-concrete handoff introduces no handicap.")
    elif total_checker_fails > 0:
        logger.info("")
        logger.info("  WARNING: %d structural checker failures on ColocFB-equivalent plans.",
                     total_checker_fails)
        logger.info("  The checker rejects plans the builder considers valid.")
        logger.info("  FIX BEFORE LAUNCH: the checker is asymmetrically harder than the builder.")
    elif gap > 0:
        logger.info("")
        logger.info("  WARNING: %d arrivals lost crossing the abstract-to-concrete boundary.",
                     gap)
        logger.info("  The deterministic placer or router rejects plans the builder admits.")
        logger.info("  FIX BEFORE LAUNCH: the LLM arms carry a structural handicap.")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
