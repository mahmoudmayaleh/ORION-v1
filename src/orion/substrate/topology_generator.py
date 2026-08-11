"""Synthetic multi-domain 6G substrate topology generator.

Produces realistic substrate graphs for training, evaluation, and prompt
development. All randomness goes through the explicit rng parameter.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from orion.config import TopologyConfig
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import InfrastructureTier, LinkType

# Tier-to-capacity lookup tables.
# §Y.1e: three tiers. `edge` spans the union of the former ran_edge (8-16 CPU) and
# mec (16-32) ranges rather than either endpoint, because the merged tier really
# does hold both small access sites and larger MEC hosts; the within-tier spread
# represents that heterogeneity instead of hiding it.
_TIER_CPU_RANGE: dict[InfrastructureTier, tuple[float, float]] = {
    InfrastructureTier.EDGE:            (8.0,  32.0),
    InfrastructureTier.REGIONAL_CLOUD:  (32.0, 48.0),
    InfrastructureTier.CENTRAL_CLOUD:   (48.0, 64.0),
}
_TIER_RAM_RANGE: dict[InfrastructureTier, tuple[float, float]] = {
    InfrastructureTier.EDGE:            (16.0,  128.0),
    InfrastructureTier.REGIONAL_CLOUD:  (128.0, 192.0),
    InfrastructureTier.CENTRAL_CLOUD:   (192.0, 256.0),
}
_TIER_PROC_DELAY_RANGE: dict[InfrastructureTier, tuple[float, float]] = {
    InfrastructureTier.EDGE:            (0.5, 1.5),
    InfrastructureTier.REGIONAL_CLOUD:  (1.0, 2.5),
    InfrastructureTier.CENTRAL_CLOUD:   (1.5, 3.0),
}


def generate_multi_domain_topology(
    config: TopologyConfig,
    rng: np.random.Generator,
) -> SubstrateNetwork:
    """Generate a synthetic multi-domain 6G substrate topology.

    Algorithm:
      1. For each domain m in 0..M-1:
         a. Create nodes_per_domain[m] nodes, assign tiers by weighted sampling.
         b. Assign capacities and processing delay by tier.
         c. Generate intra-domain edges via Erdos-Renyi G(n, p=intra_link_density).
         d. Ensure full intra-domain connectivity by adding MST edges if needed.
         e. Assign link attributes: BW ~ Uniform(bw_range),
            delay ~ Uniform(delay_intra_range).
      2. Add inter-domain links between adjacent domain pairs (m, m+1):
         pick random border nodes and connect them.
         delay ~ Uniform(delay_inter_range).
      3. Convert each undirected physical link to two directed edges (both directions).

    Args:
        config: TopologyConfig instance with generation parameters.
        rng: Seeded NumPy random generator. Never call np.random.*() directly here.

    Returns:
        SubstrateNetwork with residuals initialized to total capacities.
    """
    graph = nx.DiGraph()

    # Sorted tier names for reproducible weighted sampling
    tier_names = [t.value for t in InfrastructureTier]
    tier_weights = [config.tier_distribution.get(t, 0.0) for t in tier_names]
    total_weight = sum(tier_weights)
    tier_probs = [w / total_weight for w in tier_weights]

    # Track node IDs per domain for inter-domain link generation
    domain_nodes: dict[int, list[str]] = {}

    # Step 1: Build per-domain subgraphs
    for domain_id in range(config.num_domains):
        n_nodes = config.nodes_per_domain[domain_id]
        node_ids = [f"d{domain_id}n{i}" for i in range(n_nodes)]
        domain_nodes[domain_id] = node_ids

        # Sample tiers
        tier_indices = rng.choice(len(tier_names), size=n_nodes, p=tier_probs)

        for i, node_id in enumerate(node_ids):
            tier = InfrastructureTier(tier_names[tier_indices[i]])
            cpu = float(rng.uniform(*_TIER_CPU_RANGE[tier]))
            ram = float(rng.uniform(*_TIER_RAM_RANGE[tier]))
            proc_delay = float(rng.uniform(*_TIER_PROC_DELAY_RANGE[tier]))

            graph.add_node(
                node_id,
                domain_id=domain_id,
                tier=tier.value,
                cpu_capacity=round(cpu, 2),
                ram_capacity=round(ram, 2),
                processing_delay=round(proc_delay, 3),
                cpu_residual=round(cpu, 2),
                ram_residual=round(ram, 2),
            )

        # Generate intra-domain edges using Erdos-Renyi on undirected graph
        undirected = _erdos_renyi_connected(
            node_ids=node_ids,
            density=config.intra_link_density,
            rng=rng,
        )

        for u, v in undirected.edges():
            bw = float(rng.uniform(*config.bw_range))
            delay = float(rng.uniform(*config.delay_intra_range))
            _add_directed_link_pair(
                graph, u, v,
                bandwidth_capacity=round(bw, 1),
                propagation_delay=round(delay, 3),
                link_type=LinkType.INTRA,
            )

    # Step 2: Add inter-domain links between adjacent domain pairs
    for domain_id in range(config.num_domains - 1):
        src_domain_nodes = domain_nodes[domain_id]
        dst_domain_nodes = domain_nodes[domain_id + 1]

        n_links = config.inter_domain_links
        # Sample distinct border node pairs
        src_indices = rng.choice(len(src_domain_nodes), size=n_links, replace=True)
        dst_indices = rng.choice(len(dst_domain_nodes), size=n_links, replace=True)

        for k in range(n_links):
            u = src_domain_nodes[int(src_indices[k])]
            v = dst_domain_nodes[int(dst_indices[k])]
            # Inter-domain BW is proportionally higher (2x mean of bw_range)
            bw_lo = config.bw_range[1] * 0.5
            bw_hi = config.bw_range[1] * 1.5
            bw = float(rng.uniform(bw_lo, bw_hi))
            delay = float(rng.uniform(*config.delay_inter_range))
            _add_directed_link_pair(
                graph, u, v,
                bandwidth_capacity=round(bw, 1),
                propagation_delay=round(delay, 3),
                link_type=LinkType.INTER,
            )

    return SubstrateNetwork(graph=graph, num_domains=config.num_domains)


def _erdos_renyi_connected(
    node_ids: list[str],
    density: float,
    rng: np.random.Generator,
) -> nx.Graph:
    """Generate a connected undirected Erdos-Renyi graph on node_ids.

    If the generated graph is not connected, spanning tree edges are added
    to ensure full connectivity within the domain.

    Args:
        node_ids: List of node identifier strings.
        density: Edge probability p for the G(n, p) model.
        rng: Seeded random generator.

    Returns:
        A connected undirected NetworkX Graph with node_ids as vertices.
    """
    n = len(node_ids)
    # Build adjacency matrix via random sampling
    undirected = nx.Graph()
    undirected.add_nodes_from(node_ids)

    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                undirected.add_edge(node_ids[i], node_ids[j])

    # Ensure connectivity: connect all components via a chain
    components = list(nx.connected_components(undirected))
    if len(components) > 1:
        for k in range(len(components) - 1):
            u = next(iter(components[k]))
            v = next(iter(components[k + 1]))
            undirected.add_edge(u, v)

    return undirected


def _add_directed_link_pair(
    graph: nx.DiGraph,
    u: str,
    v: str,
    bandwidth_capacity: float,
    propagation_delay: float,
    link_type: LinkType,
) -> None:
    """Add two directed edges (u->v and v->u) for one physical link.

    The link_id for u->v is "l_{u}_{v}" and for v->u is "l_{v}_{u}".
    If either directed edge already exists (e.g. from duplicate inter-domain
    sampling), it is skipped to avoid overwriting.

    Args:
        graph: The DiGraph to add edges to.
        u: Source node identifier.
        v: Target node identifier.
        bandwidth_capacity: Total bandwidth in Mbps.
        propagation_delay: One-way propagation delay in ms.
        link_type: INTRA or INTER domain classification.
    """
    for src, dst in [(u, v), (v, u)]:
        if not graph.has_edge(src, dst):
            graph.add_edge(
                src, dst,
                link_id=f"l_{src}_{dst}",
                bandwidth_capacity=bandwidth_capacity,
                bw_residual=bandwidth_capacity,
                propagation_delay=propagation_delay,
                link_type=link_type.value,
            )
