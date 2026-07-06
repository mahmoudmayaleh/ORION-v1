#!/usr/bin/env python3
"""Load × heterogeneity sweep: find the regime where partitioning matters.

Sweeps arrival_rate (load) and slice-size mix (heterogeneity) for three
heuristics: RANDOM_FEAS, GREEDY, COST_AWARE_LB. No training, no LLM.

Tracks the admission-vs-violation frontier at each grid point to find
whether an exploitable gap opens under contention.

Load knob: arrival_rate ∈ {2, 4, 6, 8, 12}
  At service_rate=0.02, steady-state λ/μ = {100, 200, 300, 400, 600}

Heterogeneity knob:
  - "uniform": standard 5-type weighted mix (current default)
  - "heavy":   70% XR (big chains, 3-4 VNFs, CPU 2-16) + 30% URLLC (small, tight delay)
  This forces large/demanding slices competing with tiny/latency-critical ones.

Capacity knob (complementary to load):
  - "normal":  [8,10,12] nodes per domain (30 total)
  - "scarce":  [4,5,6] nodes per domain (15 total)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.policy import MDOPolicy
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.sim.reward import RewardWeights
from orion.sim.slice_generator import generate_slice_request
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier, SliceType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_ARRIVALS = 200
SERVICE_RATE = 0.02
K_EVAL = 3  # streams per (seed, config, grid-point)
NUM_DOMAINS = 3


# ── Substrate configs ───────────────────────────────────────────────────────

CAPACITY_CONFIGS = {
    "normal": [8, 10, 12],
    "scarce": [4, 5, 6],
}


# ── Heterogeneity configs ──────────────────────────────────────────────────

def _make_arrival_process(substrate, num_arrivals, arrival_rate, rng,
                          hetero="uniform"):
    """Build an arrival process, optionally forcing a heavy/mixed workload."""
    ap = ArrivalProcess(substrate, num_arrivals, arrival_rate, SERVICE_RATE, rng)

    if hetero == "uniform":
        ap.generate()
        return ap

    # "heavy" mix: 70% XR + 30% URLLC
    inter_arrivals = rng.exponential(1.0 / arrival_rate, size=num_arrivals)
    lifetimes = rng.exponential(1.0 / SERVICE_RATE, size=num_arrivals)

    events = []
    time = 0.0
    for i in range(num_arrivals):
        time += inter_arrivals[i]
        # Force slice type
        if rng.random() < 0.7:
            stype = SliceType.XR
        else:
            stype = SliceType.URLLC

        request = generate_slice_request(
            request_id=f"req_{i:04d}",
            substrate=substrate,
            rng=rng,
            slice_type=stype,
            arrival_time=time,
            lifetime=lifetimes[i],
        )
        from orion.sim.arrival_process import Event, EventType
        events.append(Event(
            time=time,
            event_type=EventType.ARRIVAL,
            request_id=request.request_id,
            slice_request=request,
        ))
        events.append(Event(
            time=time + lifetimes[i],
            event_type=EventType.DEPARTURE,
            request_id=request.request_id,
            slice_request=None,
        ))

    events.sort(key=lambda e: (e.time, e.event_type == EventType.ARRIVAL))
    ap.events = events
    ap._event_idx = 0
    return ap


# ── Plan builders ───────────────────────────────────────────────────────────

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


# ── Metrics ─────────────────────────────────────────────────────────────────

@dataclass
class GridMetrics:
    admission_rate: float = 0.0
    cost_per_slice: float = 0.0
    cross_domain_splits: float = 0.0
    avg_e2e_delay: float = 0.0
    hard_violations: int = 0
    structural_rejects: int = 0


def extract_metrics(ep) -> GridMetrics:
    m = GridMetrics()
    m.admission_rate = 100 * ep.stats.admitted / max(ep.stats.total_arrivals, 1)
    m.hard_violations = ep.stats.hard_penalty_fires
    m.structural_rejects = ep.stats.rejected_structural

    costs, delays, dom_counts = [], [], []
    for r in ep.mdo_results:
        if not r.admitted:
            continue
        costs.append(r.total_cost)
        delays.append(r.e2e_delay)
        if r.partition is not None:
            dom_counts.append(len(set(r.partition)))

    m.cost_per_slice = np.mean(costs) if costs else 0.0
    m.avg_e2e_delay = np.mean(delays) if delays else 0.0
    m.cross_domain_splits = np.mean(dom_counts) if dom_counts else 0.0
    return m


# ── Evaluation ──────────────────────────────────────────────────────────────

def build_substrate(seed, capacity="normal"):
    rng = np.random.default_rng(seed)
    return generate_multi_domain_topology(
        TopologyConfig(
            num_domains=NUM_DOMAINS,
            nodes_per_domain=CAPACITY_CONFIGS[capacity],
            intra_link_density=0.4,
            inter_domain_links=4,
        ), rng,
    )


def build_delays(sub):
    delays = {}
    for u, v in sub.graph.edges():
        u_dom = sub.graph.nodes[u].get("domain_id", -1)
        v_dom = sub.graph.nodes[v].get("domain_id", -1)
        if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
            key = (min(u_dom, v_dom), max(u_dom, v_dom))
            delays.setdefault(key, 2.0)
    return delays


def eval_config(seed, arrival_rate, capacity, hetero, mode, plan_builder):
    """Run K_EVAL episodes and return averaged GridMetrics."""
    all_m = []
    for k in range(K_EVAL):
        eval_seed = seed * 10000 + k
        sub = build_substrate(seed, capacity)
        delays = build_delays(sub)
        rng = np.random.default_rng(eval_seed)
        ap = _make_arrival_process(sub, NUM_ARRIVALS, arrival_rate, rng, hetero)

        # Build a simple coordinator with untrained actors (heuristic-only)
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
        all_m.append(extract_metrics(ep))

    avg = GridMetrics()
    avg.admission_rate = np.mean([m.admission_rate for m in all_m])
    avg.cost_per_slice = np.mean([m.cost_per_slice for m in all_m])
    avg.cross_domain_splits = np.mean([m.cross_domain_splits for m in all_m])
    avg.avg_e2e_delay = np.mean([m.avg_e2e_delay for m in all_m])
    avg.hard_violations = int(np.mean([m.hard_violations for m in all_m]))
    avg.structural_rejects = int(np.mean([m.structural_rejects for m in all_m]))
    return avg


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    arrival_rates = [2.0, 4.0, 6.0, 8.0, 12.0]
    capacities = ["normal", "scarce"]
    heteros = ["uniform", "heavy"]

    configs = [
        ("RANDOM",       "random",       _greedy_plan_builder),
        ("GREEDY",       "follow_prior", _greedy_plan_builder),
        ("COST_AWARE_LB","follow_prior", _cost_aware_lb_plan_builder),
    ]

    seeds = list(range(args.seeds))

    logger.info("=" * 90)
    logger.info("REGIME SWEEP: load × heterogeneity × capacity")
    logger.info("  %d seeds × %d eval streams", len(seeds), K_EVAL)
    logger.info("  Arrival rates: %s", arrival_rates)
    logger.info("  Capacities: %s", capacities)
    logger.info("  Heterogeneity: %s", heteros)
    logger.info("  Heuristics: %s", [c[0] for c in configs])
    logger.info("=" * 90)

    # Store results for final summary
    results = {}  # (capacity, hetero, rate, config_name) -> list of GridMetrics

    for cap in capacities:
        for het in heteros:
            logger.info("")
            logger.info("━" * 90)
            logger.info("REGIME: capacity=%s  heterogeneity=%s", cap, het)
            logger.info("━" * 90)

            for rate in arrival_rates:
                logger.info("")
                logger.info("  arrival_rate=%.1f  (λ/μ=%.0f)", rate, rate / SERVICE_RATE)

                for cfg_name, cfg_mode, cfg_pb in configs:
                    all_seed_metrics = []
                    for s in seeds:
                        m = eval_config(s, rate, cap, het, cfg_mode, cfg_pb)
                        all_seed_metrics.append(m)

                    # Average across seeds
                    avg_admit = np.mean([m.admission_rate for m in all_seed_metrics])
                    avg_cost = np.mean([m.cost_per_slice for m in all_seed_metrics])
                    avg_doms = np.mean([m.cross_domain_splits for m in all_seed_metrics])
                    avg_delay = np.mean([m.avg_e2e_delay for m in all_seed_metrics])
                    avg_viols = np.mean([m.hard_violations for m in all_seed_metrics])
                    avg_struct = np.mean([m.structural_rejects for m in all_seed_metrics])

                    key = (cap, het, rate, cfg_name)
                    results[key] = {
                        "admit": avg_admit, "cost": avg_cost, "doms": avg_doms,
                        "delay": avg_delay, "viols": avg_viols, "struct": avg_struct,
                    }

                    logger.info("    %-14s  admit=%5.1f%%  cost=%6.1f  doms=%.2f  delay=%5.1f  viols=%4.1f  struct=%4.0f",
                                cfg_name, avg_admit, avg_cost, avg_doms, avg_delay, avg_viols, avg_struct)

    # ── Final summary table ─────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 90)
    logger.info("REGIME SWEEP SUMMARY")
    logger.info("=" * 90)

    for cap in capacities:
        for het in heteros:
            logger.info("")
            logger.info("REGIME: %s / %s", cap, het)
            logger.info("%-6s  %-14s  %7s  %8s  %6s  %7s  %6s  %6s",
                        "λ", "CONFIG", "ADMIT%", "COST", "DOMS", "DELAY", "VIOLS", "STRUCT")
            logger.info("-" * 70)

            for rate in arrival_rates:
                for cfg_name, _, _ in configs:
                    key = (cap, het, rate, cfg_name)
                    r = results[key]
                    logger.info("%-6.1f  %-14s  %6.1f%%  %7.1f  %5.2f  %6.1f  %5.1f  %5.0f",
                                rate, cfg_name,
                                r["admit"], r["cost"], r["doms"],
                                r["delay"], r["viols"], r["struct"])

    # ── Gap analysis at each grid point ─────────────────────────────────────
    logger.info("")
    logger.info("=" * 90)
    logger.info("GAP ANALYSIS: RANDOM vs GREEDY  (positive = random wins)")
    logger.info("=" * 90)
    logger.info("%-8s %-8s %-6s  %8s  %8s  %8s  %8s  %8s",
                "CAP", "HETERO", "λ", "ΔADMIT", "ΔCOST", "ΔDOMS", "ΔDELAY", "ΔVIOLS")
    logger.info("-" * 75)

    gap_opens = False
    for cap in capacities:
        for het in heteros:
            for rate in arrival_rates:
                rk = (cap, het, rate, "RANDOM")
                gk = (cap, het, rate, "GREEDY")
                r, g = results[rk], results[gk]
                da = r["admit"] - g["admit"]
                dc = r["cost"] - g["cost"]
                dd = r["doms"] - g["doms"]
                dl = r["delay"] - g["delay"]
                dv = r["viols"] - g["viols"]
                logger.info("%-8s %-8s %5.1f  %+7.1f%%  %+7.1f  %+6.2f  %+6.1f  %+6.1f",
                            cap, het, rate, da, dc, dd, dl, dv)
                # Flag if greedy starts winning on admission OR violation gap widens
                if g["admit"] > r["admit"] or dv > 5:
                    gap_opens = True

    logger.info("")
    if gap_opens:
        logger.info("GAP DETECTED: greedy overtakes or violations diverge in at least one regime.")
        logger.info("→ The benchmark CAN discriminate. Find the sweet spot and test ORION there.")
    else:
        logger.info("NO GAP: random dominates across the entire sweep including violations.")
        logger.info("→ Problem does not reward intelligent partitioning at any tested load.")


if __name__ == "__main__":
    main()
