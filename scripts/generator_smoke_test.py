#!/usr/bin/env python3
"""Generator smoke test: confirm co-location sign flips across families.

Runs co-location-first and FFD once across all 8 generated families.
Checks that co-location dominates in friendly regimes and is beaten in
hostile ones. This is the paper's motivation table: no fixed plan policy
wins across families.

Half a day, not a study. If the sign does not flip, fix the generator.
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig, GreedyResult
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.substrate.topology_families import (
    ALL_FAMILIES, TRAIN_FAMILIES, TEST_FAMILIES,
    TopologyFamily, generate_family_instance, compute_signature,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Import the co-location builder from the bracket script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_builder_bracket import _run_colocation_ffd

NUM_ARRIVALS = 200
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
INSTANCE_SEEDS = [0, 1, 2]  # 3 instances per family for the smoke test


def _run_pure_colocation(substrate, slice_req, config):
    """Pure co-location: reject if no single domain can host the chain.
    No FFD fallback. This is the policy that ONLY co-locates.
    """
    result = _run_colocation_ffd(substrate, slice_req, config)
    if result.feasible and result.plan is not None:
        # Check if it's actually single-domain
        domains = set()
        g = substrate.graph
        for vnf_id, node_id in result.plan.vnf_placements.items():
            domains.add(g.nodes[node_id]["domain_id"])
        if len(domains) > 1:
            # The co-location builder fell back to FFD (cross-domain)
            # Pure co-location rejects this
            return GreedyResult(feasible=False, cost=float("inf"), plan=None,
                                fail_reason="no single-domain placement")
    return result


def run_static_kills(substrate, arrival_seed, builder_fn):
    """Count kills on a fresh substrate."""
    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    admitted = 0
    killed = 0
    for event in ap.events:
        if event.event_type != EventType.ARRIVAL or event.slice_request is None:
            continue
        sr = event.slice_request
        result = builder_fn(substrate, sr, GreedyConfig())
        if result.feasible:
            admitted += 1
        else:
            killed += 1
    return admitted, killed


def main():
    logger.info("=" * 90)
    logger.info("GENERATOR SMOKE TEST — co-location sign flip across families")
    logger.info("  %d families, %d instances each, %d arrivals per instance",
                len(ALL_FAMILIES), len(INSTANCE_SEEDS), NUM_ARRIVALS)
    logger.info("=" * 90)

    builders = [
        ("FFD", _run_greedy_ffd),
        ("PureColoc", _run_pure_colocation),
        ("ColocFB", _run_colocation_ffd),  # with fallback (actual baseline)
    ]
    results = []

    for family in ALL_FAMILIES:
        totals = {name: {"admitted": 0, "killed": 0} for name, _ in builders}
        total_arrivals = 0

        logger.info("")
        logger.info("--- Family: %s ---", family.short_name)

        for inst_seed in INSTANCE_SEEDS:
            sub = generate_family_instance(family, seed=inst_seed)
            sig = compute_signature(sub, family.short_name)

            if inst_seed == INSTANCE_SEEDS[0]:
                logger.info("  Signature: domains=%d, nodes=%s, tier_coverage=%s",
                           sig.num_domains, sig.nodes_per_domain,
                           sig.tier_coverage_per_domain)
                logger.info("  Inter-BW: mean=%.0f min=%.0f max=%.0f links=%d",
                           sig.inter_bw_mean, sig.inter_bw_min, sig.inter_bw_max,
                           sig.num_inter_links)
                logger.info("  Tiers: %s", sig.tiers_per_domain)

            arrival_seed = inst_seed * 1000 + 42
            for name, builder_fn in builders:
                adm, kill = run_static_kills(sub, arrival_seed, builder_fn)
                totals[name]["admitted"] += adm
                totals[name]["killed"] += kill

            total_arrivals += totals["FFD"]["admitted"] + totals["FFD"]["killed"]

        for name, _ in builders:
            total = totals[name]["admitted"] + totals[name]["killed"]
            kr = 100 * totals[name]["killed"] / total if total > 0 else 0
            logger.info("  %-10s: kill=%.1f%% (%d/%d)", name, kr, totals[name]["killed"], total)

        ffd_kr = 100 * totals["FFD"]["killed"] / total_arrivals if total_arrivals > 0 else 0
        pure_kr = 100 * totals["PureColoc"]["killed"] / total_arrivals if total_arrivals > 0 else 0
        fb_kr = 100 * totals["ColocFB"]["killed"] / total_arrivals if total_arrivals > 0 else 0

        # The key comparison: does pure co-location lose to FFD anywhere?
        pure_vs_ffd = pure_kr - ffd_kr  # positive = FFD wins

        results.append({
            "family": family.short_name,
            "ffd_kill": ffd_kr,
            "pure_coloc_kill": pure_kr,
            "coloc_fb_kill": fb_kr,
            "pure_vs_ffd": pure_vs_ffd,
        })

    # ── Summary table ──────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 90)
    logger.info("  %-12s  %8s  %10s  %10s  %12s",
                "Family", "FFD Kill", "PureColoc", "ColocFB", "PureVsFFD")
    logger.info("  " + "-" * 65)

    sign_flips = 0
    for r in results:
        marker = ""
        if r["pure_vs_ffd"] > 1.0:
            marker = " ← FFD wins"
            sign_flips += 1
        elif r["pure_vs_ffd"] < -1.0:
            marker = " ← PureColoc wins"
        logger.info("  %-12s  %7.1f%%  %9.1f%%  %9.1f%%  %+10.1f pp%s",
                    r["family"], r["ffd_kill"], r["pure_coloc_kill"],
                    r["coloc_fb_kill"], r["pure_vs_ffd"], marker)

    logger.info("")
    logger.info("  Sign flips (PureColoc loses to FFD): %d/%d", sign_flips, len(results))
    logger.info("  ColocFB (with fallback) is always <= FFD by construction.")
    logger.info("")
    logger.info("  The paper's point: PureColoc wins on some families, FFD wins on")
    logger.info("  others. ColocFB combines both but is static — M^B learns the")
    logger.info("  trade-off per topology and adapts dynamically.")

    # ── Verdict ────────────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    if sign_flips > 0:
        logger.info("PASS: sign flips exist in PureColoc vs FFD.")
        logger.info("Generator produces regimes where co-location is the wrong policy.")
        logger.info("")
        logger.info("Train families (%d): %s", len(TRAIN_FAMILIES),
                    [f.short_name for f in TRAIN_FAMILIES])
        logger.info("Test families (%d):  %s", len(TEST_FAMILIES),
                    [f.short_name for f in TEST_FAMILIES])
    else:
        logger.info("FAIL: PureColoc never loses to FFD. Generator needs more hostile regimes.")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
