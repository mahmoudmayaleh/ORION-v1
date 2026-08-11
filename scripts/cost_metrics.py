"""Secondary cost metrics — per-admission placement-footprint aggregates.

FoC stays the primary metric; these fields are banked alongside it so cost can
be reported as a SECONDARY table. Two things are exposed, deliberately kept
apart:

  (a) admission selection — WHICH requests an approach admits. `demand_mean` (sum of
      cpu+ram over the request's VNFs) and `k_mean` (chain length) are properties
      of the admitted REQUEST, identical whichever placement is chosen, so
      differences between approaches here are selection bias, not placement quality.
      An approach that admits fewer slices usually admits the easy ones; these two
      columns make that visible instead of letting a per-admission cost average
      silently benefit from it.
  (b) placement footprint — what the chosen embedding consumes beyond the
      request's own demand. `inter_hops_mean` (inter-domain link traversals) and
      `inter_bw_mean` (sum over flows of bandwidth x inter-domain links, Mbps-hop)
      are properties of the PLACEMENT and are the discriminative cost axis in
      this substrate (node cpu/ram consumed is fixed by the request).

All aggregates are over ADMITTED arrivals of the held-out stream only. The same
link-classification rule (an edge whose endpoints carry different domain_ids is
inter-domain) is applied to every approach, so Plain/MILP plan routes and coordinator
cross-domain routes are measured identically.
"""
from __future__ import annotations

import numpy as np


def link_endpoint_map(sub):
    """link_id -> (u, v) for every substrate edge that carries a link_id."""
    return {d["link_id"]: (u, v)
            for u, v, d in sub.graph.edges(data=True) if "link_id" in d}


class CostAccumulator:
    """Per-admission collector; one instance per eval episode / approach cell."""

    def __init__(self, sub):
        self.g = sub.graph
        self.lmap = link_endpoint_map(sub)
        self.rows: list[dict] = []

    def _inter_links(self, links) -> int:
        n = 0
        for lid in links or []:
            uv = self.lmap.get(lid)
            if uv is not None and (
                    self.g.nodes[uv[0]].get("domain_id")
                    != self.g.nodes[uv[1]].get("domain_id")):
                n += 1
        return n

    def add_plan(self, sr, plan) -> None:
        """Admitted PlacementPlan (Plain / MILP): routes from plan.flow_routes."""
        bw_by_flow = {(fe.source_vnf, fe.target_vnf): float(fe.bandwidth_demand)
                      for fe in sr.flow_edges}
        hops, bwh = 0, 0.0
        for flow, links in (plan.flow_routes or {}).items():
            h = self._inter_links(links)
            hops += h
            bwh += h * bw_by_flow.get(tuple(flow), 0.0)
        self._push(sr, hops, bwh)

    def add_mdo(self, sr, res) -> None:
        """Admitted MDOResult: routes from the coordinator's committed
        cross-domain routes (colocated flows have no route and contribute 0,
        matching the plan-based path where such flows are routeless)."""
        hops, bwh = 0, 0.0
        for flow, links in (res.cross_domain_routes or {}).items():
            h = self._inter_links(links)
            hops += h
            bwh += h * float((res.cross_domain_bw or {}).get(flow, 0.0))
        self._push(sr, hops, bwh)

    def _push(self, sr, hops, bwh) -> None:
        self.rows.append({
            "demand": float(sum(v.cpu_demand + v.ram_demand for v in sr.vnfs)),
            "k": len(sr.vnfs),
            "inter_hops": hops,
            "inter_bw": bwh,
        })

    def summary(self) -> dict:
        """Banked into the cell under key "cost"."""
        if not self.rows:
            return {"n_admitted": 0}
        def mean(key):
            return round(float(np.mean([r[key] for r in self.rows])), 2)
        return {
            "n_admitted": len(self.rows),
            "demand_mean": mean("demand"),
            "k_mean": mean("k"),
            "inter_hops_mean": mean("inter_hops"),
            "inter_bw_mean": mean("inter_bw"),
        }
