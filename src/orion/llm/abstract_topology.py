"""Generate the abstract topology view that Agent B receives.

Agent B never sees the raw per-node substrate graph. It receives per-domain
aggregates and inter-domain link summaries, matching v6 Section 5.2.

The abstract view is the SAME surface the MDO policy observes
(`mdo.observation.build_domain_summaries`) and the same one the partial-observability
heuristic plans from (`scripts/partial_obs_prior`). Every quantity emitted here has a
counterpart in that tensor, and `test_agent_b_sees_the_mdo_surface` fails if one of
the three drifts. This equality is load-bearing for the whole comparison: ORION's plan
layer and the LLM-free partition baselines must be choosing between domains on the
same evidence, or the result measures who was told more rather than who planned better.

Two quantities were missing until 2026-08-12 and are restored here.

Capacities. The view carried residuals with nothing to divide them by, so Agent B
could not tell 500 free CPU out of 5000 from 500 out of 600. The MDO tensor carries
`cpu_cap_norm` / `ram_cap_norm`, and the plan cache's own condition key is expressed
in residual FRACTIONS, so the cache knew the utilisation of a state that the planner
it was caching for could not compute. Per-tier capacities are included for the same
reason: the MDO can learn a fixed composition across episodes, whereas Agent B is
stateless per arrival and must be told.

Largest free node per tier (h^m). See `mdo.types.DomainSummary` for what it is and why
it is per tier. Agent B was the only partitioner without it, including after 2026-08-11
when it was removed from the others: a tier residual is a sum, and a chain is placed on
individual nodes.
"""

from __future__ import annotations

from orion.config import MDO_HEADROOM_CPU_REF, MDO_HEADROOM_RAM_REF
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import TIER_ORDER


def build_abstract_topology(substrate: SubstrateNetwork) -> dict:
    """Convert a SubstrateNetwork into Agent B's abstract topology dict.

    Returns a dict with:
      - domains: list of {domain_id, label, dominant_tiers,
            cpu_residual, ram_residual, cpu_capacity, ram_capacity,
            cpu_residual_by_tier, ram_residual_by_tier,
            cpu_capacity_by_tier, ram_capacity_by_tier,
            largest_free_node_by_tier}
      - inter_domain_links: list of {link_id, source_domain, target_domain,
            bandwidth_residual_mbps, bandwidth_capacity_mbps, propagation_delay_ms}
    """
    domains = []
    for domain_id in range(substrate.num_domains):
        nodes = substrate.nodes_in_domain(domain_id)
        if not nodes:
            continue

        cpu_res = sum(substrate.graph.nodes[n]["cpu_residual"] for n in nodes)
        ram_res = sum(substrate.graph.nodes[n]["ram_residual"] for n in nodes)
        cpu_cap = sum(substrate.graph.nodes[n]["cpu_capacity"] for n in nodes)
        ram_cap = sum(substrate.graph.nodes[n]["ram_capacity"] for n in nodes)

        # Collect unique tiers in this domain, in canonical order rather than
        # alphabetical, so "the first tier" means the same thing here as everywhere
        # else. Alphabetical put central_cloud before edge, which read as a
        # hierarchy inversion in the prompt.
        present = {substrate.graph.nodes[n]["tier"] for n in nodes}
        tiers = [t.value for t in TIER_ORDER if t.value in present]

        # Per-tier residuals (§Y.1e). The planner is choosing a domain per VNF and
        # every VNF is tier-restricted, so a domain's aggregate residual is the
        # wrong quantity: it cannot say "this domain's edge tier is full but its
        # regional tier is not", which is the common case once domains hold
        # different tier sets. Reported for EVERY tier including absent ones, which
        # read 0.0, so the planner sees one consistent shape per domain.
        tier_cpu = {t.value: 0.0 for t in TIER_ORDER}
        tier_ram = {t.value: 0.0 for t in TIER_ORDER}
        tier_cpu_cap = {t.value: 0.0 for t in TIER_ORDER}
        tier_ram_cap = {t.value: 0.0 for t in TIER_ORDER}
        # h^m per tier, computed by the SAME rule as DomainSummary.tier_max_node_*:
        # the node maximising min(cpu/CPU_REF, ram/RAM_REF), reported as that one node's
        # residuals so the pair always describes a node that exists.
        best_fit = {t.value: -1.0 for t in TIER_ORDER}
        tier_hcpu = {t.value: 0.0 for t in TIER_ORDER}
        tier_hram = {t.value: 0.0 for t in TIER_ORDER}
        for n in nodes:
            t = substrate.graph.nodes[n]["tier"]
            n_cpu = substrate.graph.nodes[n]["cpu_residual"]
            n_ram = substrate.graph.nodes[n]["ram_residual"]
            tier_cpu[t] += n_cpu
            tier_ram[t] += n_ram
            tier_cpu_cap[t] += substrate.graph.nodes[n]["cpu_capacity"]
            tier_ram_cap[t] += substrate.graph.nodes[n]["ram_capacity"]
            fit = min(n_cpu / MDO_HEADROOM_CPU_REF, n_ram / MDO_HEADROOM_RAM_REF)
            if fit > best_fit[t]:
                best_fit[t] = fit
                tier_hcpu[t] = n_cpu
                tier_hram[t] = n_ram

        tier_labels = {
            "edge": "Edge",
            "regional_cloud": "Regional Cloud",
            "central_cloud": "Central Cloud",
        }
        # ALL tiers, not tiers[:2]. Truncating made a domain holding all three
        # read as "Edge/Regional Cloud", identical to a domain holding only two,
        # so the label contradicted dominant_tiers in the same object. Two of the
        # five domains were mislabelled that way on every prompt.
        label = "/".join(tier_labels.get(t, t) for t in tiers)

        domains.append({
            "domain_id": f"d{domain_id}",
            "label": label,
            "dominant_tiers": tiers,
            "cpu_residual": round(cpu_res, 1),
            "ram_residual": round(ram_res, 1),
            "cpu_capacity": round(cpu_cap, 1),
            "ram_capacity": round(ram_cap, 1),
            "cpu_residual_by_tier": {k: round(v, 1) for k, v in tier_cpu.items()},
            "ram_residual_by_tier": {k: round(v, 1) for k, v in tier_ram.items()},
            "cpu_capacity_by_tier": {k: round(v, 1) for k, v in tier_cpu_cap.items()},
            "ram_capacity_by_tier": {k: round(v, 1) for k, v in tier_ram_cap.items()},
            "largest_free_node_by_tier": {
                k: {"cpu": round(tier_hcpu[k], 1), "ram": round(tier_hram[k], 1)}
                for k in tier_hcpu
            },
        })

    # Inter-domain links: aggregate by (src_domain, dst_domain)
    # Multiple physical links between the same domain pair are summed for BW
    # and min'd for delay (best-case path).
    link_agg: dict[tuple[int, int], dict] = {}
    for u, v, d in substrate.graph.edges(data=True):
        src_dom = substrate.graph.nodes[u]["domain_id"]
        dst_dom = substrate.graph.nodes[v]["domain_id"]
        if src_dom == dst_dom:
            continue

        key = (src_dom, dst_dom)
        if key not in link_agg:
            link_agg[key] = {"bw": 0.0, "cap": 0.0, "delay": float("inf")}

        link_agg[key]["bw"] += d["bw_residual"]
        link_agg[key]["cap"] += d["bandwidth_capacity"]
        link_agg[key]["delay"] = min(link_agg[key]["delay"], d["propagation_delay"])

    inter_domain_links = []
    for (src_dom, dst_dom), agg in sorted(link_agg.items()):
        inter_domain_links.append({
            "link_id": f"l_d{src_dom}_d{dst_dom}",
            "source_domain": f"d{src_dom}",
            "target_domain": f"d{dst_dom}",
            "bandwidth_residual_mbps": round(agg["bw"], 1),
            "bandwidth_capacity_mbps": round(agg["cap"], 1),
            "propagation_delay_ms": round(agg["delay"], 1),
        })

    return {
        "domains": domains,
        "inter_domain_links": inter_domain_links,
    }
