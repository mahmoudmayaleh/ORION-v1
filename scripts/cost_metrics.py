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

  (c) substrate occupancy — how loaded the NETWORK is while the approach runs, added
      2026-08-12: node CPU and RAM utilisation, per tier and overall, and inter-domain
      link bandwidth utilisation. This is a property of neither the request nor a
      single placement but of the whole trajectory, so it is SAMPLED at every arrival
      (pre-decision, pre-allocation) and time-averaged, not read once at the end. An
      end-of-episode reading would report whatever the last few departures left behind.

      Read (c) together with acceptance, never alone. Utilisation is not a quantity to
      maximise or minimise on its own: an approach that admits nothing has near-zero
      utilisation and an approach that packs badly has high utilisation with low
      acceptance. The informative comparisons are utilisation at EQUAL acceptance
      (packing efficiency, where lower is better) and the spread across nodes
      (`node_cpu_p95` against `node_cpu_mean`, where a large gap means hotspotting,
      i.e. the substrate is full where it matters while the mean looks comfortable).

All aggregates in (a) and (b) are over ADMITTED arrivals of the held-out stream only.
The same link-classification rule (an edge whose endpoints carry different domain_ids
is inter-domain) is applied to every approach, so Plain's plan routes and
coordinator cross-domain routes are measured identically. Group (c) is over every
arrival, admitted or not, because occupancy is a state of the network rather than an
outcome of a request.
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

        # ── occupancy sampling state ─────────────────────────────────────────
        # Capacities are constant for the episode, so they are read once. The
        # denominators are therefore identical across approaches on a given
        # (seed, instance) and the utilisation figures are directly comparable.
        self._nodes = list(self.g.nodes)
        self._tier_of = {n: self.g.nodes[n].get("tier") for n in self._nodes}
        self._tiers = sorted({t for t in self._tier_of.values() if t})
        self._cpu_cap = {n: float(self.g.nodes[n].get("cpu_capacity", 0.0))
                         for n in self._nodes}
        self._ram_cap = {n: float(self.g.nodes[n].get("ram_capacity", 0.0))
                         for n in self._nodes}
        self._cpu_cap_total = sum(self._cpu_cap.values())
        self._ram_cap_total = sum(self._ram_cap.values())
        self._cpu_cap_tier = {t: sum(self._cpu_cap[n] for n in self._nodes
                                     if self._tier_of[n] == t) for t in self._tiers}
        self._ram_cap_tier = {t: sum(self._ram_cap[n] for n in self._nodes
                                     if self._tier_of[n] == t) for t in self._tiers}
        # Links split by whether they cross an administrative boundary. Inter-domain
        # bandwidth is the scarce, discriminative one; intra-domain is reported too so
        # "the network is busy" and "the boundaries are busy" stay distinguishable.
        self._inter_edges, self._intra_edges = [], []
        for u, v, d in self.g.edges(data=True):
            cap = float(d.get("bandwidth_capacity", 0.0))
            if cap <= 0:
                continue
            tgt = (self._inter_edges
                   if self.g.nodes[u].get("domain_id") != self.g.nodes[v].get("domain_id")
                   else self._intra_edges)
            tgt.append((u, v, cap))
        self._inter_bw_cap = sum(c for _, _, c in self._inter_edges)
        self._intra_bw_cap = sum(c for _, _, c in self._intra_edges)
        self._samples: list[dict] = []

    def sample_utilization(self) -> None:
        """Record substrate occupancy at this instant. Call once per arrival,
        BEFORE the decision, so every approach samples at the same points of the
        same stream and no approach is measured after its own allocation."""
        g = self.g
        cpu_used, ram_used = {}, {}
        per_node_cpu, per_node_ram = [], []
        for n in self._nodes:
            cu = self._cpu_cap[n] - float(g.nodes[n].get("cpu_residual", 0.0))
            ru = self._ram_cap[n] - float(g.nodes[n].get("ram_residual", 0.0))
            cpu_used[n], ram_used[n] = cu, ru
            if self._cpu_cap[n] > 0:
                per_node_cpu.append(cu / self._cpu_cap[n])
            if self._ram_cap[n] > 0:
                per_node_ram.append(ru / self._ram_cap[n])

        def edge_util(edges, cap_total):
            if cap_total <= 0:
                return 0.0
            used = sum(c - float(g.edges[u, v].get("bw_residual", 0.0))
                       for u, v, c in edges)
            return used / cap_total

        row = {
            "cpu_used": sum(cpu_used.values()),
            "ram_used": sum(ram_used.values()),
            "node_cpu_p95": float(np.percentile(per_node_cpu, 95)) if per_node_cpu else 0.0,
            "node_ram_p95": float(np.percentile(per_node_ram, 95)) if per_node_ram else 0.0,
            # A node is saturated when it cannot take the largest VNF in the
            # workload; 0.90 is a plain high-water mark, not that threshold.
            "node_cpu_frac_over_90": (float(np.mean([u >= 0.90 for u in per_node_cpu]))
                                      if per_node_cpu else 0.0),
            "inter_bw_util": edge_util(self._inter_edges, self._inter_bw_cap),
            "intra_bw_util": edge_util(self._intra_edges, self._intra_bw_cap),
        }
        for t in self._tiers:
            row[f"cpu_used@{t}"] = sum(cpu_used[n] for n in self._nodes
                                       if self._tier_of[n] == t)
            row[f"ram_used@{t}"] = sum(ram_used[n] for n in self._nodes
                                       if self._tier_of[n] == t)
        self._samples.append(row)

    def utilization_summary(self) -> dict:
        """Time-averaged occupancy. Absolute means are in CPU cores and GB, so the
        network cost of an approach can be read directly and not only as a ratio."""
        if not self._samples:
            return {"n_samples": 0}
        def m(key):
            return float(np.mean([s[key] for s in self._samples]))
        cpu_used, ram_used = m("cpu_used"), m("ram_used")
        out = {
            "n_samples": len(self._samples),
            "cpu_allocated_mean": round(cpu_used, 2),
            "ram_allocated_mean": round(ram_used, 2),
            "cpu_capacity": round(self._cpu_cap_total, 2),
            "ram_capacity": round(self._ram_cap_total, 2),
            "node_cpu_util_mean": round(cpu_used / self._cpu_cap_total, 4)
                                  if self._cpu_cap_total else None,
            "node_ram_util_mean": round(ram_used / self._ram_cap_total, 4)
                                  if self._ram_cap_total else None,
            "node_cpu_util_p95": round(m("node_cpu_p95"), 4),
            "node_ram_util_p95": round(m("node_ram_p95"), 4),
            "node_cpu_frac_over_90": round(m("node_cpu_frac_over_90"), 4),
            "inter_domain_bw_util": round(m("inter_bw_util"), 4),
            "intra_domain_bw_util": round(m("intra_bw_util"), 4),
        }
        out["node_cpu_util_by_tier"] = {
            t: round(m(f"cpu_used@{t}") / self._cpu_cap_tier[t], 4)
            for t in self._tiers if self._cpu_cap_tier.get(t)}
        out["node_ram_util_by_tier"] = {
            t: round(m(f"ram_used@{t}") / self._ram_cap_tier[t], 4)
            for t in self._tiers if self._ram_cap_tier.get(t)}
        # Cost PER ADMITTED SLICE. Without this an approach that admits little looks
        # cheap; with it, "CPU-cores held per slice carried" is comparable across rows.
        if self.rows:
            out["cpu_allocated_per_admit"] = round(cpu_used / len(self.rows), 3)
            out["ram_allocated_per_admit"] = round(ram_used / len(self.rows), 3)
        return out

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
        """Admitted PlacementPlan (Plain): routes from plan.flow_routes."""
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
