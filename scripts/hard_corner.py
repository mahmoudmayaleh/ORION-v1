#!/usr/bin/env python3
"""Hard corner test: scarce bottleneck tier + tight inter-domain links.

Tests whether a gap opens between random-feasible and locality-aware
heuristics when chains are forced to span domains.

Topology: 5 domains × 3 nodes, 10% edge/10% MEC distribution
  → Most domains lack MEC/edge → forced cross-domain splits
  → Tight inter-domain BW (200 Mbps) and delays (8ms)
  → Cross-domain splits are costly and violating

Sweep: arrival_rate ∈ {2, 4, 8, 12} at fixed capacity
Track: admission, violations, cross-domain fraction, cost, delay
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.sim.reward import RewardWeights
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_ARRIVALS = 200
SERVICE_RATE = 0.02
K_EVAL = 3
NUM_DOMAINS = 5
INTER_DOMAIN_BW = 200.0   # tight BW on cross-domain links
INTER_DOMAIN_DELAY = 8.0  # high delay on cross-domain links


def build_substrate(seed):
    rng = np.random.default_rng(seed)
    sub = generate_multi_domain_topology(
        TopologyConfig(
            num_domains=NUM_DOMAINS,
            nodes_per_domain=[3, 3, 3, 3, 3],
            intra_link_density=0.5,
            inter_domain_links=3,
            tier_distribution={
                "ran_edge": 0.10,
                "mec": 0.10,
                "regional_cloud": 0.40,
                "central_cloud": 0.40,
            },
        ), rng,
    )
    # Tighten inter-domain links
    for u, v, d in sub.graph.edges(data=True):
        u_dom = sub.graph.nodes[u].get("domain_id", -1)
        v_dom = sub.graph.nodes[v].get("domain_id", -1)
        if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
            d["bw_capacity"] = INTER_DOMAIN_BW
            d["bw_residual"] = INTER_DOMAIN_BW
    return sub


def build_delays(sub):
    delays = {}
    for u, v in sub.graph.edges():
        u_dom = sub.graph.nodes[u].get("domain_id", -1)
        v_dom = sub.graph.nodes[v].get("domain_id", -1)
        if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
            key = (min(u_dom, v_dom), max(u_dom, v_dom))
            delays.setdefault(key, INTER_DOMAIN_DELAY)
    return delays


def _greedy_plan_builder(slice_req, substrate):
    result = _run_greedy_ffd(substrate, slice_req, GreedyConfig())
    if not result.feasible or result.plan is None:
        return None
    vnf_ids, required_tiers, suggested_domains = [], [], []
    for vnf in slice_req.vnfs:
        node_id = result.plan.vnf_placements[vnf.vnf_id]
        domain = int(node_id.split("n")[0][1:])
        tier_str = substrate.graph.nodes[node_id]["tier"]
        vnf_ids.append(vnf.vnf_id)
        required_tiers.append(InfrastructureTier(tier_str))
        suggested_domains.append(domain)
    return PlanSummary(
        vnf_ids=vnf_ids, required_tiers=required_tiers,
        suggested_domains=suggested_domains,
        cpu_demands=[v.cpu_demand for v in slice_req.vnfs],
        ram_demands=[v.ram_demand for v in slice_req.vnfs],
        vcrs=[v.vcr for v in slice_req.vnfs],
        bw_demands=[f.bandwidth_demand for f in slice_req.flow_edges],
    )


def _cost_aware_lb_plan_builder(slice_req, substrate):
    g = substrate.graph
    domain_cpu = {}
    domain_nodes_by_tier = {}
    for node_id, d in g.nodes(data=True):
        dom = d.get("domain_id", -1)
        if dom < 0:
            continue
        domain_cpu[dom] = domain_cpu.get(dom, 0.0) + float(d["cpu_residual"])
        tier = d["tier"]
        domain_nodes_by_tier.setdefault(dom, {}).setdefault(tier, []).append(node_id)

    vnf_ids, required_tiers, suggested_domains = [], [], []
    domain_counts = {}

    for vnf in slice_req.vnfs:
        vnf_ids.append(vnf.vnf_id)
        req_tier = None
        if vnf.permitted_nodes:
            for nid in vnf.permitted_nodes:
                if nid in g.nodes:
                    req_tier = g.nodes[nid]["tier"]
                    break
        if req_tier is None:
            req_tier = "mec"
        required_tiers.append(InfrastructureTier(req_tier))

        feasible_domains = []
        for dom, nodes_by_tier in domain_nodes_by_tier.items():
            tier_nodes = nodes_by_tier.get(req_tier, [])
            for nid in tier_nodes:
                nd = g.nodes[nid]
                if float(nd["cpu_residual"]) >= vnf.cpu_demand and \
                   float(nd["ram_residual"]) >= vnf.ram_demand:
                    feasible_domains.append(dom)
                    break

        if not feasible_domains:
            feasible_domains = list(domain_cpu.keys())

        best = min(
            feasible_domains,
            key=lambda d: (-domain_cpu.get(d, 0.0), -domain_counts.get(d, 0), d),
        )
        suggested_domains.append(best)
        domain_counts[best] = domain_counts.get(best, 0) + 1

    return PlanSummary(
        vnf_ids=vnf_ids, required_tiers=required_tiers,
        suggested_domains=suggested_domains,
        cpu_demands=[v.cpu_demand for v in slice_req.vnfs],
        ram_demands=[v.ram_demand for v in slice_req.vnfs],
        vcrs=[v.vcr for v in slice_req.vnfs],
        bw_demands=[f.bandwidth_demand for f in slice_req.flow_edges],
    )


@dataclass
class Metrics:
    admission_rate: float = 0.0
    hard_violations: int = 0
    cost_per_slice: float = 0.0
    avg_e2e_delay: float = 0.0
    cross_domain_frac: float = 0.0  # fraction of admitted slices spanning >1 domain
    domains_per_slice: float = 0.0
    structural_rejects: int = 0
    total_admitted: int = 0


def extract(ep) -> Metrics:
    m = Metrics()
    m.admission_rate = 100 * ep.stats.admitted / max(ep.stats.total_arrivals, 1)
    m.hard_violations = ep.stats.hard_penalty_fires
    m.structural_rejects = ep.stats.rejected_structural
    m.total_admitted = ep.stats.admitted

    costs, delays, dom_counts, xd = [], [], [], 0
    for r in ep.mdo_results:
        if not r.admitted:
            continue
        costs.append(r.total_cost)
        delays.append(r.e2e_delay)
        if r.partition is not None:
            n_doms = len(set(r.partition))
            dom_counts.append(n_doms)
            if n_doms > 1:
                xd += 1

    m.cost_per_slice = np.mean(costs) if costs else 0.0
    m.avg_e2e_delay = np.mean(delays) if delays else 0.0
    m.domains_per_slice = np.mean(dom_counts) if dom_counts else 0.0
    m.cross_domain_frac = 100 * xd / max(len(dom_counts), 1) if dom_counts else 0.0
    return m


def eval_config(seed, arrival_rate, mode, plan_builder):
    all_m = []
    for k in range(K_EVAL):
        eval_seed = seed * 10000 + k
        sub = build_substrate(seed)
        delays = build_delays(sub)
        rng = np.random.default_rng(eval_seed)
        ap = ArrivalProcess(sub, NUM_ARRIVALS, arrival_rate, SERVICE_RATE, rng)
        ap.generate()

        actors = {}
        for d in range(NUM_DOMAINS):
            torch.manual_seed(seed + d)
            actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=64))

        coord = MDOCoordinator(
            None, actors,
            MDOConfig(n_part=1, mu=1.0, alpha=0.1, xi=0.1, eta=0.1),
        )
        runner = EpisodeRunner(
            sub, ap, coord, delays,
            plan_builder=plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        ep = runner.run_episode(mdo_mode=mode)
        all_m.append(extract(ep))

    avg = Metrics()
    avg.admission_rate = np.mean([m.admission_rate for m in all_m])
    avg.hard_violations = float(np.mean([m.hard_violations for m in all_m]))
    avg.cost_per_slice = np.mean([m.cost_per_slice for m in all_m])
    avg.avg_e2e_delay = np.mean([m.avg_e2e_delay for m in all_m])
    avg.cross_domain_frac = np.mean([m.cross_domain_frac for m in all_m])
    avg.domains_per_slice = np.mean([m.domains_per_slice for m in all_m])
    avg.structural_rejects = float(np.mean([m.structural_rejects for m in all_m]))
    avg.total_admitted = float(np.mean([m.total_admitted for m in all_m]))
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    arrival_rates = [2.0, 4.0, 8.0, 12.0]
    configs = [
        ("RANDOM",        "random",       _greedy_plan_builder),
        ("GREEDY",        "follow_prior", _greedy_plan_builder),
        ("COST_AWARE_LB", "follow_prior", _cost_aware_lb_plan_builder),
    ]
    seeds = list(range(args.seeds))

    # Print topology info
    sub = build_substrate(0)
    g = sub.graph
    logger.info("=" * 90)
    logger.info("HARD CORNER TEST: tier-scarce topology + tight inter-domain links")
    logger.info("  %d domains × 3 nodes, 10%% edge/10%% MEC", NUM_DOMAINS)
    logger.info("  Inter-domain BW: %.0f Mbps, delay: %.0f ms", INTER_DOMAIN_BW, INTER_DOMAIN_DELAY)
    logger.info("  %d seeds × %d eval streams × %d rates × %d configs", len(seeds), K_EVAL, len(arrival_rates), len(configs))

    # Show per-domain tier info
    for dom in range(NUM_DOMAINS):
        tiers = {}
        for nid, d in g.nodes(data=True):
            if d["domain_id"] == dom:
                t = d["tier"]
                tiers[t] = tiers.get(t, 0) + 1
        tier_str = ", ".join(f"{t}:{c}" for t, c in sorted(tiers.items()))
        logger.info("  Domain %d: %s", dom, tier_str)
    logger.info("=" * 90)

    results = {}

    for rate in arrival_rates:
        logger.info("")
        logger.info("--- arrival_rate=%.1f  (λ/μ=%.0f) ---", rate, rate / SERVICE_RATE)

        for cfg_name, cfg_mode, cfg_pb in configs:
            all_seed = []
            for s in seeds:
                m = eval_config(s, rate, cfg_mode, cfg_pb)
                all_seed.append(m)

            avg = Metrics()
            avg.admission_rate = np.mean([m.admission_rate for m in all_seed])
            avg.hard_violations = np.mean([m.hard_violations for m in all_seed])
            avg.cost_per_slice = np.mean([m.cost_per_slice for m in all_seed])
            avg.avg_e2e_delay = np.mean([m.avg_e2e_delay for m in all_seed])
            avg.cross_domain_frac = np.mean([m.cross_domain_frac for m in all_seed])
            avg.domains_per_slice = np.mean([m.domains_per_slice for m in all_seed])
            avg.structural_rejects = np.mean([m.structural_rejects for m in all_seed])

            results[(rate, cfg_name)] = avg

            logger.info("  %-14s  admit=%5.1f%%  viols=%4.1f  xd_frac=%5.1f%%  doms=%.2f  cost=%6.1f  delay=%5.1f  struct=%4.0f",
                        cfg_name,
                        avg.admission_rate, avg.hard_violations,
                        avg.cross_domain_frac, avg.domains_per_slice,
                        avg.cost_per_slice, avg.avg_e2e_delay,
                        avg.structural_rejects)

    # ── Summary table ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 90)
    logger.info("HARD CORNER SUMMARY")
    logger.info("=" * 90)
    logger.info("%-6s  %-14s  %7s  %6s  %7s  %6s  %7s  %6s  %6s",
                "λ", "CONFIG", "ADMIT%", "VIOLS", "XD_FRAC", "DOMS", "COST", "DELAY", "STRUCT")
    logger.info("-" * 85)

    for rate in arrival_rates:
        for cfg_name, _, _ in configs:
            r = results[(rate, cfg_name)]
            logger.info("%-6.1f  %-14s  %6.1f%%  %5.1f  %6.1f%%  %5.2f  %6.1f  %5.1f  %5.0f",
                        rate, cfg_name,
                        r.admission_rate, r.hard_violations,
                        r.cross_domain_frac, r.domains_per_slice,
                        r.cost_per_slice, r.avg_e2e_delay,
                        r.structural_rejects)
        logger.info("")

    # ── Gap analysis ────────────────────────────────────────────────────────
    logger.info("GAP ANALYSIS: RANDOM vs GREEDY")
    logger.info("%-6s  %8s  %8s  %8s  %8s  %8s",
                "λ", "ΔADMIT", "ΔVIOLS", "ΔXD_FRAC", "ΔCOST", "ΔDELAY")
    logger.info("-" * 55)

    for rate in arrival_rates:
        r = results[(rate, "RANDOM")]
        g = results[(rate, "GREEDY")]
        logger.info("%-6.1f  %+7.1f%%  %+7.1f  %+7.1f%%  %+7.1f  %+6.1f",
                    rate,
                    r.admission_rate - g.admission_rate,
                    r.hard_violations - g.hard_violations,
                    r.cross_domain_frac - g.cross_domain_frac,
                    r.cost_per_slice - g.cost_per_slice,
                    r.avg_e2e_delay - g.avg_e2e_delay)

    logger.info("")
    logger.info("GAP ANALYSIS: COST_AWARE_LB vs RANDOM")
    logger.info("%-6s  %8s  %8s  %8s  %8s  %8s",
                "λ", "ΔADMIT", "ΔVIOLS", "ΔXD_FRAC", "ΔCOST", "ΔDELAY")
    logger.info("-" * 55)

    for rate in arrival_rates:
        r = results[(rate, "RANDOM")]
        lb = results[(rate, "COST_AWARE_LB")]
        logger.info("%-6.1f  %+7.1f%%  %+7.1f  %+7.1f%%  %+7.1f  %+6.1f",
                    rate,
                    lb.admission_rate - r.admission_rate,
                    lb.hard_violations - r.hard_violations,
                    lb.cross_domain_frac - r.cross_domain_frac,
                    lb.cost_per_slice - r.cost_per_slice,
                    lb.avg_e2e_delay - r.avg_e2e_delay)

    # ── Verdict ─────────────────────────────────────────────────────────────
    logger.info("")
    max_viol_gap = max(
        results[(rate, "RANDOM")].hard_violations - results[(rate, "GREEDY")].hard_violations
        for rate in arrival_rates
    )
    max_admit_gap = max(
        results[(rate, "RANDOM")].admission_rate - results[(rate, "GREEDY")].admission_rate
        for rate in arrival_rates
    )
    logger.info("VERDICT:")
    logger.info("  Max admission gap (random - greedy): %+.1f%%", max_admit_gap)
    logger.info("  Max violation gap (random - greedy): %+.1f", max_viol_gap)

    if max_viol_gap > 3:
        logger.info("  GAP OPENS: violations diverge under tier scarcity.")
        logger.info("  → The Pareto corner (high admission + low violations) is real.")
        logger.info("  → Reparameterize the benchmark to this regime and test ORION.")
    elif max_admit_gap > 5 and max_viol_gap > 1:
        logger.info("  MODERATE GAP: admission and violations both separate.")
        logger.info("  → Some discrimination exists. Tighten further or test ORION here.")
    else:
        logger.info("  NO CLEAR GAP: even the hard corner does not discriminate.")
        logger.info("  → Partition decision may not reward intelligence on this problem.")


if __name__ == "__main__":
    main()
