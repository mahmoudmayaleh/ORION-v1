#!/usr/bin/env python3
"""BC pretrain domain actors from greedy traces, then cold-start admission check.

The acceptance check: BC-pretrained actor in the live episode loop with NO RL.
Greedy gets 46%. If BC actor admits near that, BC worked. If BC loss is tiny
but admission is ~2%, the observation projection doesn't match runtime — debug.

Usage:
    python scripts/bc_pretrain_and_check.py --seed 0 --bc-epochs 10 --bc-scenarios 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.baselines.greedy_ffd import greedy_place_on_substrate, _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.observation import build_inter_domain_links
from orion.mdo.precommit_check import (
    check_c5b_inter,
    compute_e2e_delay,
    count_inter_domain_hops,
    domain_sequence_from_partition,
    inter_domain_demand_by_pair,
    inter_domain_residual_by_pair,
)
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.episode_runner import EpisodeRunner
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.training.bc_dataset import BCDatasetSpec
from orion.training.bc_pretrain import bc_pretrain
from orion.training.config import MAPPOConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bc_check")

NUM_ARRIVALS = 2000
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
HIDDEN_DIM = 64


def build_substrate(seed):
    rng = np.random.default_rng(seed)
    return generate_multi_domain_topology(
        TopologyConfig(num_domains=3, nodes_per_domain=[8, 10, 12],
                       intra_link_density=0.4, inter_domain_links=4), rng)


def run_greedy(substrate, seed):
    rng = np.random.default_rng(seed + 1_000_000)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()
    substrate.reset()
    ap.reset()
    total = admitted = 0
    per_type = {}
    while ap.has_next():
        ev = ap.next_event()
        if ev.event_type == EventType.DEPARTURE:
            p = substrate._active_slices.get(ev.request_id)
            if p:
                substrate.deallocate(p[0], p[1])
            continue
        total += 1
        st = ev.slice_request.slice_type.value
        per_type.setdefault(st, {"total": 0, "admitted": 0})
        per_type[st]["total"] += 1
        r = greedy_place_on_substrate(substrate, ev.slice_request)
        if r.feasible:
            admitted += 1
            per_type[st]["admitted"] += 1
    return total, admitted, per_type


def run_greedy_with_precommit(substrate, seed, max_inter_hops=3):
    """Run greedy, then filter each admission through C5b/C7/C9.

    Greedy's internal routing skips C5b (aggregate inter-domain BW),
    C7 (E2E delay vs QoS budget), and C9 (inter-domain hop count).
    This function applies the same precommit checks the MDO commit gate
    uses, giving the true greedy ceiling in the same feasibility universe
    the RL system is judged in.
    """
    rng = np.random.default_rng(seed + 1_000_000)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()
    substrate.reset()
    ap.reset()

    total = admitted_raw = admitted_gated = 0
    gate_rejects = {"c5b": 0, "c7": 0, "c9": 0}

    while ap.has_next():
        ev = ap.next_event()
        if ev.event_type == EventType.DEPARTURE:
            p = substrate._active_slices.get(ev.request_id)
            if p:
                substrate.deallocate(p[0], p[1])
            continue

        total += 1
        sr = ev.slice_request
        result = _run_greedy_ffd(substrate, sr, GreedyConfig())
        if not result.feasible or result.plan is None:
            continue

        admitted_raw += 1

        # Extract greedy's implicit partition (domain per VNF)
        partition = []
        for vnf in sr.vnfs:
            node_id = result.plan.vnf_placements[vnf.vnf_id]
            domain = int(node_id.split("n")[0][1:])
            partition.append(domain)

        bw_demands = [f.bandwidth_demand for f in sr.flow_edges]

        # C5b: aggregate inter-domain BW demand vs residual
        inter_links = build_inter_domain_links(substrate)
        residuals = inter_domain_residual_by_pair(inter_links)
        demand = inter_domain_demand_by_pair(partition, bw_demands)
        c5b_fail = check_c5b_inter(demand, residuals)

        # C7: E2E delay. Greedy doesn't track intra-domain delay from routing,
        # so approximate with link propagation delays along the placed path.
        dom_seq = domain_sequence_from_partition(partition)
        inter_delays = {}
        for u, v, d in substrate.graph.edges(data=True):
            sd = substrate.graph.nodes[u]["domain_id"]
            dd = substrate.graph.nodes[v]["domain_id"]
            if sd != dd:
                inter_delays[(sd, dd)] = min(
                    inter_delays.get((sd, dd), float("inf")),
                    d["propagation_delay"],
                )

        # Sum inter-domain delays along domain sequence
        total_inter_delay = 0.0
        for i in range(len(dom_seq) - 1):
            if dom_seq[i] != dom_seq[i + 1]:
                total_inter_delay += inter_delays.get(
                    (dom_seq[i], dom_seq[i + 1]), 0.0
                )

        # Approximate intra-domain delay from routes greedy computed
        total_intra_delay = 0.0
        for (src_vnf, dst_vnf), link_ids in result.plan.flow_routes.items():
            for lid in link_ids:
                for u, v, d in substrate.graph.edges(data=True):
                    if d["link_id"] == lid:
                        total_intra_delay += d["propagation_delay"]
                        break

        e2e = total_intra_delay + total_inter_delay
        c7_fail = e2e > sr.qos.max_e2e_delay

        # C9: inter-domain hops
        inter_hops = count_inter_domain_hops(partition)
        c9_fail = inter_hops > max_inter_hops

        if c5b_fail:
            gate_rejects["c5b"] += 1
        if c7_fail:
            gate_rejects["c7"] += 1
        if c9_fail:
            gate_rejects["c9"] += 1

        if c5b_fail or c7_fail or c9_fail:
            continue

        # Passes all gates — admit and allocate
        substrate.allocate(result.plan, sr)
        admitted_gated += 1

    return total, admitted_raw, admitted_gated, gate_rejects


def _greedy_plan_builder(slice_req, substrate):
    """Plan builder that uses greedy FFD's domain assignments as suggested_domains.

    Matches the BC training signal: the actors were trained on observations
    from domains greedy chose, so runtime must send VNFs to those same domains.
    """
    from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
    from orion.mdo.types import PlanSummary
    from orion.types import InfrastructureTier

    result = _run_greedy_ffd(substrate, slice_req, GreedyConfig())
    if not result.feasible or result.plan is None:
        return None

    vnf_ids = []
    required_tiers = []
    suggested_domains = []
    for vnf in slice_req.vnfs:
        node_id = result.plan.vnf_placements[vnf.vnf_id]
        domain = int(node_id.split("n")[0][1:])
        tier_str = substrate.graph.nodes[node_id]["tier"]
        vnf_ids.append(vnf.vnf_id)
        required_tiers.append(InfrastructureTier(tier_str))
        suggested_domains.append(domain)

    return PlanSummary(
        vnf_ids=vnf_ids,
        required_tiers=required_tiers,
        suggested_domains=suggested_domains,
        cpu_demands=[v.cpu_demand for v in slice_req.vnfs],
        ram_demands=[v.ram_demand for v in slice_req.vnfs],
        vcrs=[v.vcr for v in slice_req.vnfs],
        bw_demands=[f.bandwidth_demand for f in slice_req.flow_edges],
    )


def run_bc_actor(substrate, seed, actors):
    """Run episode with BC-pretrained actors, greedy-aligned MDO partition."""
    rng = np.random.default_rng(seed + 1_000_000)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    delays = {}
    for u, v, d in substrate.graph.edges(data=True):
        sd, dd = substrate.graph.nodes[u]["domain_id"], substrate.graph.nodes[v]["domain_id"]
        if sd != dd:
            delays[(sd, dd)] = min(delays.get((sd, dd), float("inf")), d["propagation_delay"])

    coord = MDOCoordinator(None, actors, MDOConfig(n_part=3))
    runner = EpisodeRunner(substrate, ap, coord, delays, plan_builder=_greedy_plan_builder)
    runner.reset()
    # follow_prior uses the plan builder's suggested_domains
    ep = runner.run_episode(mdo_mode="follow_prior")
    return ep


def analyze_reject_reasons(ep_result):
    """Tally per-reject-reason from MDO results.

    For each rejected arrival, inspect the last PartitionAttempt's ViolationInfo
    to determine whether rejection was due to:
      - actor_infeasible: domain actor couldn't place (empty mask or chose NULL or routing fail)
      - c7_violated: E2E delay exceeded budget
      - c5b_violated: inter-domain BW insufficient
      - c9_violated: too many inter-domain hops
      - structural: plan_builder returned None (no MDO result)

    Within actor_infeasible, distinguish:
      - null_action: actor had feasible nodes but chose NULL (unlikely post-BC)
      - empty_mask: no feasible node for some VNF (capacity exhausted)
      - routing_fail: placed VNFs but routing failed
    """
    from collections import Counter

    reasons = Counter()
    actor_detail = Counter()

    for mdo_result in ep_result.mdo_results:
        if mdo_result.admitted:
            reasons["admitted"] += 1
            continue

        # Look at the LAST attempt's violation (the one that sealed rejection)
        last_attempt = mdo_result.retry_history.attempts[-1] if mdo_result.retry_history.attempts else None
        if last_attempt is None or last_attempt.violation is None:
            reasons["unknown"] += 1
            continue

        v = last_attempt.violation

        # Tally all violated constraints (can be multiple)
        if v.actor_infeasible:
            reasons["actor_infeasible"] += 1

            # Dig into domain responses to classify the infeasibility
            for did, resp in last_attempt.domain_responses.items():
                if resp.feasible:
                    continue
                # Check if actor had step_records (VNFs it attempted)
                steps = getattr(resp, "step_records", [])
                if not steps:
                    # No VNFs attempted — routing failure on first flow or empty fragment
                    actor_detail["routing_fail"] += 1
                else:
                    last_step = steps[-1]
                    # NULL_ACTION = -1 in DomainPolicy
                    if last_step.action_idx == -1:
                        # Actor chose NULL — check if mask was empty (forced) or chosen
                        if not last_step.action_mask.any():
                            actor_detail["empty_mask"] += 1
                        else:
                            actor_detail["null_chosen"] += 1
                    else:
                        # Actor placed a node but routing failed after
                        actor_detail["routing_fail"] += 1

        if v.c7_violated:
            reasons["c7_e2e_delay"] += 1
        if v.c5b_violated:
            reasons["c5b_inter_bw"] += 1
        if v.c9_violated:
            reasons["c9_hops"] += 1

        # If no specific violation flagged
        if not v.has_violation:
            reasons["no_violation_flagged"] += 1

    # Also count structural rejections (plan_builder returned None)
    structural = ep_result.stats.rejected_structural
    reasons["structural"] = structural

    return dict(reasons), dict(actor_detail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-scenarios", type=int, default=5000)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)

    topo_config = TopologyConfig(
        num_domains=3, nodes_per_domain=[8, 10, 12],
        intra_link_density=0.4, inter_domain_links=4,
    )

    # ── Greedy baseline (raw — no precommit gate) ─────────────────
    logger.info("Running greedy baseline (raw, no precommit)...")
    greedy_sub = build_substrate(seed)
    g_total, g_admitted, g_per_type = run_greedy(greedy_sub, seed)
    logger.info("Greedy (raw): %d/%d (%.1f%%)", g_admitted, g_total, g_admitted / g_total * 100)
    for st, v in sorted(g_per_type.items()):
        logger.info("  %s: %d/%d", st, v["admitted"], v["total"])

    # ── Greedy baseline (gated — same C5b/C7/C9 as commit gate) ──
    logger.info("Running greedy baseline (with precommit gate)...")
    gated_sub = build_substrate(seed)
    gg_total, gg_raw, gg_gated, gg_rejects = run_greedy_with_precommit(gated_sub, seed)
    logger.info("Greedy (gated): %d/%d (%.1f%%) — raw was %d/%d (%.1f%%)",
                gg_gated, gg_total, gg_gated / gg_total * 100,
                gg_raw, gg_total, gg_raw / gg_total * 100)
    logger.info("  Gate rejections: C5b=%d, C7=%d, C9=%d",
                gg_rejects["c5b"], gg_rejects["c7"], gg_rejects["c9"])
    if gg_raw > 0:
        pct_lost = (gg_raw - gg_gated) / gg_raw * 100
        logger.info("  %.1f%% of greedy admissions fail the commit gate", pct_lost)

    # ── Build actors ─────────────────────────────────────────────────
    actors = {}
    for d in range(topo_config.num_domains):
        torch.manual_seed(seed + 3_000_000 + d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=HIDDEN_DIM))

    # ── Pre-BC cold-start check ──────────────────────────────────────
    logger.info("Pre-BC cold-start check (untrained actors)...")
    pre_sub = build_substrate(seed)
    pre_ep = run_bc_actor(pre_sub, seed, actors)
    pre_stats = pre_ep.stats
    logger.info("Pre-BC: %d/%d (%.1f%%)", pre_stats.admitted, pre_stats.total_arrivals,
                pre_stats.admission_rate * 100)

    # ── BC pretraining ───────────────────────────────────────────────
    logger.info("BC pretraining: %d scenarios, %d epochs, lr=%g...",
                args.bc_scenarios, args.bc_epochs, args.bc_lr)

    bc_config = MAPPOConfig(
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        bc_entropy_coef=0.01,
        bc_seed=seed,
    )
    bc_spec = BCDatasetSpec(
        seed=seed,
        num_scenarios=args.bc_scenarios,
        topology_config=topo_config,
    )

    t0 = time.time()
    logs, meta = bc_pretrain(
        domain_actors=actors,
        spec=bc_spec,
        dataset_path=Path("data/bc_demonstrations.pt"),
        config=bc_config,
    )
    bc_time = time.time() - t0
    logger.info("BC done in %.1fs. Dataset hash: %s", bc_time, meta.get("dataset_hash"))

    # Log final BC losses
    for d, epoch_logs in logs.items():
        if epoch_logs:
            last = epoch_logs[-1]
            logger.info("  Domain %d: final loss=%.4f, entropy=%.4f, samples=%d",
                        d, last.imitation_loss, last.entropy_bonus, last.num_samples)

    # ── Post-BC acceptance check ─────────────────────────────────────
    logger.info("Post-BC acceptance check (BC actors, no RL)...")
    post_sub = build_substrate(seed)
    post_ep = run_bc_actor(post_sub, seed, actors)
    post_stats = post_ep.stats
    logger.info("Post-BC: %d/%d (%.1f%%)", post_stats.admitted, post_stats.total_arrivals,
                post_stats.admission_rate * 100)
    logger.info("  Per-type admitted: %s", dict(post_stats.per_slice_type_admitted))
    logger.info("  Per-type total:    %s", dict(post_stats.per_slice_type_total))

    # ── Per-reject-reason analysis ─────────────────────────────────
    logger.info("")
    logger.info("=== REJECT REASON ANALYSIS (Post-BC) ===")
    reasons, actor_detail = analyze_reject_reasons(post_ep)
    total_rejected = post_stats.total_arrivals - post_stats.admitted
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / max(total_rejected, 1) * 100
        logger.info("  %-25s %4d  (%.1f%% of rejections)", reason, count, pct)
    if actor_detail:
        logger.info("  --- Actor infeasibility breakdown ---")
        for detail, count in sorted(actor_detail.items(), key=lambda x: -x[1]):
            logger.info("    %-23s %4d", detail, count)

    # Also analyze pre-BC for comparison
    logger.info("")
    logger.info("=== REJECT REASON ANALYSIS (Pre-BC) ===")
    pre_reasons, pre_actor_detail = analyze_reject_reasons(pre_ep)
    pre_rejected = pre_stats.total_arrivals - pre_stats.admitted
    for reason, count in sorted(pre_reasons.items(), key=lambda x: -x[1]):
        pct = count / max(pre_rejected, 1) * 100
        logger.info("  %-25s %4d  (%.1f%% of rejections)", reason, count, pct)
    if pre_actor_detail:
        logger.info("  --- Actor infeasibility breakdown ---")
        for detail, count in sorted(pre_actor_detail.items(), key=lambda x: -x[1]):
            logger.info("    %-23s %4d", detail, count)

    # ── Verdict ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== VERDICT ===")
    logger.info("Greedy (raw):   %d/%d (%.1f%%) — INFLATED, skips C5b/C7/C9",
                g_admitted, g_total, g_admitted / g_total * 100)
    logger.info("Greedy (gated): %d/%d (%.1f%%) — true ceiling under commit gate",
                gg_gated, gg_total, gg_gated / gg_total * 100)
    logger.info("Pre-BC:         %d/%d (%.1f%%)", pre_stats.admitted, pre_stats.total_arrivals,
                pre_stats.admission_rate * 100)
    logger.info("Post-BC:        %d/%d (%.1f%%)", post_stats.admitted, post_stats.total_arrivals,
                post_stats.admission_rate * 100)

    true_baseline = max(gg_gated, 1)
    ratio_pre = pre_stats.admitted / true_baseline
    ratio_post = post_stats.admitted / true_baseline
    logger.info("")
    logger.info("Ratio vs gated greedy: Pre-BC=%.0f%%, Post-BC=%.0f%%",
                ratio_pre * 100, ratio_post * 100)

    if ratio_pre > ratio_post:
        logger.info("BC is a NET NEGATIVE: untrained actors outperform BC actors (%.1f%% vs %.1f%%).",
                    pre_stats.admission_rate * 100, post_stats.admission_rate * 100)
        logger.info("  BC's only remaining justification: faster early RL convergence (warm start).")
        logger.info("  Measure with cold-start-RL vs BC-start-RL, not cold placement.")
    elif ratio_post > 0.8:
        logger.info("BC PASSED: %.0f%% of gated greedy. Actors are competent.", ratio_post * 100)
    elif ratio_post > 0.3:
        logger.info("BC PARTIAL: %.0f%% of gated greedy.", ratio_post * 100)
    else:
        logger.info("BC FAILED: %.0f%% of gated greedy.", ratio_post * 100)

    # ── Diagnosis guidance ─────────────────────────────────────────
    if total_rejected > 0:
        actor_pct = reasons.get("actor_infeasible", 0) / total_rejected
        commit_gate_pct = (reasons.get("c7_e2e_delay", 0) + reasons.get("c5b_inter_bw", 0)
                          + reasons.get("c9_hops", 0)) / total_rejected
        logger.info("")
        if actor_pct > 0.7:
            logger.info("DIAGNOSIS: %.0f%% actor infeasible → actor competence problem.", actor_pct * 100)
            empty = actor_detail.get("empty_mask", 0)
            null = actor_detail.get("null_chosen", 0)
            route = actor_detail.get("routing_fail", 0)
            if empty > null + route:
                logger.info("  Dominated by empty masks → scale BC data (more episodes, capacity diversity).")
            elif null > empty + route:
                logger.info("  Dominated by NULL choices → BC didn't converge, check loss/lr.")
            else:
                logger.info("  Mixed actor failures → check routing + mask distribution.")
        elif commit_gate_pct > 0.5:
            logger.info("DIAGNOSIS: %.0f%% commit-gate (C5b/C7/C9) → partition infeasible under current model.",
                        commit_gate_pct * 100)
            logger.info("  Greedy's partition is stale vs current routing. Regenerate greedy, retrain BC.")


if __name__ == "__main__":
    main()
