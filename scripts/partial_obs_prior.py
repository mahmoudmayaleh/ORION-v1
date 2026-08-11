"""Partial-obs colocation-first partition builder (2026-07-24).

The LLM-free m̃ source that sees ONLY the MDO's observation surface:
DomainSummary aggregates + the K x M node-based feasibility mask. No node
residuals, no full-substrate FFD. Shared by probe_partialobs_baseline.py
(follow_prior approach) and grid_runner.py (Plain-partial approach + RL-poprior's
KL prior / obs m̃).

Import-light on purpose: orion.* only, no runner imports (grid_runner and the
probe both import this module, so it must not import them back).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.config import MDO_HEADROOM_CPU_REF, MDO_HEADROOM_RAM_REF  # noqa: E402
from orion.mdo.observation import build_domain_summaries  # noqa: E402
from orion.mdo.types import PlanSummary  # noqa: E402
from orion.types import InfrastructureTier  # noqa: E402


def _required_tiers(sr, substrate):
    """Per-VNF modal tier of permitted_nodes — request-side info, obs-legal."""
    g = substrate.graph
    tiers = []
    for v in sr.vnfs:
        counts = {}
        for n in v.permitted_nodes:
            if n in g.nodes:
                t = g.nodes[n]["tier"]
                counts[t] = counts.get(t, 0) + 1
        tiers.append(InfrastructureTier(max(counts, key=counts.get)) if counts
                     else InfrastructureTier.MEC)
    return tiers


def partial_obs_builder(sr, substrate):
    """Colocation-first partition from DomainSummary aggregates + the K x M
    node-based feasibility mask ONLY (exactly the MDO's observation surface).

    Discipline: touches substrate only through build_domain_summaries() and
    per-domain permitted-node intersection (the same two inputs the MDO obs
    exposes as summary features and tier_mask). Never reads node residuals.
    """
    summaries = build_domain_summaries(substrate)
    M = len(summaries)
    K = len(sr.vnfs)
    dom_nodes = [set(substrate.nodes_in_domain(s.domain_id)) for s in summaries]
    feas = [[bool(set(v.permitted_nodes) & dom_nodes[m]) for m in range(M)]
            for v in sr.vnfs]  # == build_tier_masks node-based rows
    if any(not any(row) for row in feas):
        return None  # structural reject: some VNF feasible nowhere
    cpu = [v.cpu_demand for v in sr.vnfs]
    ram = [v.ram_demand for v in sr.vnfs]

    # 1) colocation-first: a single domain feasible for ALL VNFs, best residual
    #    slack after the whole chain; require the summary headroom signal to fit
    #    the largest VNF (single-node fragmentation proxy, h^m).
    best, best_slack = None, 0.0
    for m in range(M):
        if not all(feas[k][m] for k in range(K)):
            continue
        s = summaries[m]
        slack = min(s.cpu_residual - sum(cpu), s.ram_residual - sum(ram))
        if slack <= 0:
            continue
        if s.max_node_headroom * min(MDO_HEADROOM_CPU_REF, MDO_HEADROOM_RAM_REF) \
                < max(max(cpu), max(ram)):
            continue
        if best is None or slack > best_slack:
            best, best_slack = m, slack
    if best is not None:
        chosen = [best] * K
    else:
        # 2) split fallback: per-VNF best-fit on running aggregate estimates.
        est_cpu = [s.cpu_residual for s in summaries]
        est_ram = [s.ram_residual for s in summaries]
        chosen = []
        for k in range(K):
            cands = [(min(est_cpu[m] - cpu[k], est_ram[m] - ram[k]), m)
                     for m in range(M) if feas[k][m]]
            slack, m = max(cands)
            if slack <= 0:  # no summary-feasible domain left
                return None
            chosen.append(m)
            est_cpu[m] -= cpu[k]
            est_ram[m] -= ram[k]
    doms = [summaries[m].domain_id for m in chosen]
    return PlanSummary(
        vnf_ids=[v.vnf_id for v in sr.vnfs],
        required_tiers=_required_tiers(sr, substrate),
        suggested_domains=doms,
        cpu_demands=cpu, ram_demands=ram,
        vcrs=[v.vcr for v in sr.vnfs],
        bw_demands=[fl.bandwidth_demand for fl in sr.flow_edges])
