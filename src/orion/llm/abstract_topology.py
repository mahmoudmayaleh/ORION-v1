"""Generate the abstract topology view that Agent B receives.

Agent B never sees the raw per-node substrate graph. It receives per-domain
aggregates and inter-domain link summaries, matching v6 Section 5.2.
"""

from __future__ import annotations

from orion.substrate.graph_model import SubstrateNetwork


def build_abstract_topology(substrate: SubstrateNetwork) -> dict:
    """Convert a SubstrateNetwork into Agent B's abstract topology dict.

    Returns a dict with:
      - domains: list of {domain_id, label, dominant_tiers, cpu_residual, ram_residual}
      - inter_domain_links: list of {link_id, source_domain, target_domain,
            bandwidth_residual_mbps, propagation_delay_ms}
    """
    domains = []
    for domain_id in range(substrate.num_domains):
        nodes = substrate.nodes_in_domain(domain_id)
        if not nodes:
            continue

        cpu_res = sum(substrate.graph.nodes[n]["cpu_residual"] for n in nodes)
        ram_res = sum(substrate.graph.nodes[n]["ram_residual"] for n in nodes)

        # Collect unique tiers in this domain
        tiers = sorted({substrate.graph.nodes[n]["tier"] for n in nodes})

        # Build a human-readable label from the dominant tiers
        tier_labels = {
            "ran_edge": "RAN/Edge",
            "mec": "MEC",
            "regional_cloud": "Regional Cloud",
            "central_cloud": "Central Cloud",
        }
        label = "/".join(tier_labels.get(t, t) for t in tiers[:2])

        domains.append({
            "domain_id": f"d{domain_id}",
            "label": label,
            "dominant_tiers": tiers,
            "cpu_residual": round(cpu_res, 1),
            "ram_residual": round(ram_res, 1),
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
            link_agg[key] = {"bw": 0.0, "delay": float("inf")}

        link_agg[key]["bw"] += d["bw_residual"]
        link_agg[key]["delay"] = min(link_agg[key]["delay"], d["propagation_delay"])

    inter_domain_links = []
    for (src_dom, dst_dom), agg in sorted(link_agg.items()):
        inter_domain_links.append({
            "link_id": f"l_d{src_dom}_d{dst_dom}",
            "source_domain": f"d{src_dom}",
            "target_domain": f"d{dst_dom}",
            "bandwidth_residual_mbps": round(agg["bw"], 1),
            "propagation_delay_ms": round(agg["delay"], 1),
        })

    return {
        "domains": domains,
        "inter_domain_links": inter_domain_links,
    }
