"""Shared QoS admission gate for plans produced OUTSIDE the coordinator.

The coordinator's commits are verified post-commit by the ground-truth
verifier (M/M/1 sojourn delay, hop budget), and violating admissions are
deallocated. Plans admitted by static baselines (Plain's co-location FFD) did not pass
through that machinery. This module gives them the SAME check, with the
verifier's delay model, evaluated on the candidate plan against the current
(pre-allocation) substrate state.

C7: sum of node sojourns (per placed VNF, including this slice's own load)
plus link sojourns (per routed flow-edge link) must fit the delay budget.
C9: inter-domain link traversals across all flow routes must fit the hop
budget.
"""
from __future__ import annotations

import math

from orion.sim.delay_model import link_sojourn, node_sojourn


def plan_qos_ok(substrate, slice_req, plan, max_inter_domain_hops: int = 3) -> bool:
    """True iff `plan` satisfies C7 (sojourn model) and C9 on `substrate` now."""
    return plan_qos_reason(substrate, slice_req, plan, max_inter_domain_hops) is None


def plan_qos_reason(substrate, slice_req, plan,
                    max_inter_domain_hops: int = 3) -> str | None:
    """The constraint this plan fails, or None if it passes.

    §Y.13 — same checks, same order, same thresholds as `plan_qos_ok`, which is
    now a wrapper over this. One implementation rather than two: the gate runs on
    the coordinator's commit path as well as Plain's, so a second copy that
    drifted would move acceptance everywhere and be attributed to the taxonomy.

    Returns "C9" or "C7". Plain has no pre-commit / post-commit distinction,
    because it checks before allocating rather than after; a "C7" here is the
    same load-dependent sojourn model that produces `post_commit_c7_delay` for
    the coordinator approaches.
    """
    g = substrate.graph
    link_by_id = {d["link_id"]: (u, v, d) for u, v, d in g.edges(data=True)}

    # C9 — inter-domain traversals over all routed flows.
    hops = 0
    for lids in plan.flow_routes.values():
        for lid in lids:
            u, v, _ = link_by_id[lid]
            if g.nodes[u]["domain_id"] != g.nodes[v]["domain_id"]:
                hops += 1
    if hops > max_inter_domain_hops:
        return "C9"

    vmap = {v.vnf_id: v for v in slice_req.vnfs}

    # C7 — node sojourns, with this slice's own CPU load applied.
    extra_cpu: dict[str, float] = {}
    for fid, nid in plan.vnf_placements.items():
        extra_cpu[nid] = extra_cpu.get(nid, 0.0) + vmap[fid].cpu_demand
    total = 0.0
    for fid, nid in plan.vnf_placements.items():
        nd = g.nodes[nid]
        cpu_used = float(nd["cpu_capacity"]) - float(nd["cpu_residual"]) + extra_cpu[nid]
        s = node_sojourn(float(nd["processing_delay"]),
                         vmap[fid].computational_intensity,
                         float(nd["cpu_capacity"]), cpu_used)
        if math.isinf(s):
            return "C7"
        total += s

    # C7 — link sojourns, with this slice's own bandwidth load applied.
    extra_bw: dict[str, float] = {}
    for per_link in plan.bw_allocations.values():
        for lid, bw in per_link.items():
            extra_bw[lid] = extra_bw.get(lid, 0.0) + bw
    for lids in plan.flow_routes.values():
        for lid in lids:
            _, _, d = link_by_id[lid]
            bw_cap = float(d.get("bandwidth_capacity", d.get("bw_capacity", 0.0)))
            bw_used = bw_cap - float(d["bw_residual"]) + extra_bw.get(lid, 0.0)
            s = link_sojourn(float(d["propagation_delay"]), bw_cap, bw_used)
            if math.isinf(s):
                return "C7"
            total += s

    return None if total <= slice_req.qos.max_e2e_delay else "C7"
