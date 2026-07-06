"""Build PyG Data objects from a domain subgraph and plan fragment.

Node features [N_m x 15]:
  0: residual_cpu_frac        (cpu_residual / cpu_capacity)
  1: residual_ram_frac        (ram_residual / ram_capacity)
  2: cpu_capacity_norm        (cpu_capacity / max_cpu_in_domain)
  3: ram_capacity_norm        (ram_capacity / max_ram_in_domain)
  4-7: tier_onehot            (4-dim: ran_edge, mec, regional_cloud, central_cloud)
  8: processing_delay_norm    (delay / max_delay_in_domain)
  9: node_degree_norm         (degree / max_degree_in_domain)
  10: mean_incident_bw_frac   (mean residual_bw_frac on incident intra-domain edges)
  11: hop_to_nearest_border   (shortest path hops to nearest border node, normalized)
  12: is_border_node          (1 if node has inter-domain links, 0 otherwise)
  13: border_to_target        (1 if node borders a domain in fragment's target set)
  14: has_placed_vnf          (1 if a VNF from this slice is already placed here)

Edge features [E_m x 2]:
  0: residual_bw_frac         (bw_residual / bandwidth_capacity)
  1: propagation_delay_norm   (delay / max_delay_in_domain)
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from orion.substrate.graph_model import SubstrateNetwork
from orion.types import InfrastructureTier

_TIER_ORDER = [
    InfrastructureTier.RAN_EDGE,
    InfrastructureTier.MEC,
    InfrastructureTier.REGIONAL_CLOUD,
    InfrastructureTier.CENTRAL_CLOUD,
]
_TIER_TO_IDX = {t: i for i, t in enumerate(_TIER_ORDER)}

NODE_FEAT_DIM = 15
EDGE_FEAT_DIM = 2


def build_domain_observation(
    substrate: SubstrateNetwork,
    domain_id: int,
    target_domain_ids: set[int] | None = None,
    placed_node_ids: set[str] | None = None,
) -> tuple[Data, list[str]]:
    """Build a PyG Data object for a single domain subgraph.

    Args:
        substrate: The full substrate network (digital twin state).
        domain_id: Which domain to extract.
        target_domain_ids: Domains that the current plan fragment connects to.
            Used for the border_to_target feature.
        placed_node_ids: Nodes that already have a VNF placed from this slice.

    Returns:
        Tuple of (PyG Data with x, edge_index, edge_attr) and the ordered list
        of node_ids (so action indices can be mapped back to node IDs).
    """
    if target_domain_ids is None:
        target_domain_ids = set()
    if placed_node_ids is None:
        placed_node_ids = set()

    g = substrate.graph
    node_ids = sorted(substrate.nodes_in_domain(domain_id))
    n_nodes = len(node_ids)

    if n_nodes == 0:
        return Data(
            x=torch.zeros(0, NODE_FEAT_DIM),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, EDGE_FEAT_DIM),
        ), []

    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    node_set = set(node_ids)

    # Pre-compute domain-level normalization constants
    max_cpu = max(g.nodes[n]["cpu_capacity"] for n in node_ids)
    max_ram = max(g.nodes[n]["ram_capacity"] for n in node_ids)
    max_delay = max(g.nodes[n]["processing_delay"] for n in node_ids)
    if max_cpu == 0:
        max_cpu = 1.0
    if max_ram == 0:
        max_ram = 1.0
    if max_delay == 0:
        max_delay = 1.0

    # Pre-compute border node info
    border_nodes: set[str] = set()
    border_to_target_nodes: set[str] = set()
    for u, v, d in g.edges(data=True):
        if u in node_set and v not in node_set:
            border_nodes.add(u)
            v_domain = g.nodes[v]["domain_id"]
            if v_domain in target_domain_ids:
                border_to_target_nodes.add(u)
        elif v in node_set and u not in node_set:
            border_nodes.add(v)
            u_domain = g.nodes[u]["domain_id"]
            if u_domain in target_domain_ids:
                border_to_target_nodes.add(v)

    # Compute hop distance to nearest border node via BFS on domain subgraph
    domain_subgraph = g.subgraph(node_ids).to_undirected()
    hop_to_border = _compute_hop_to_border(domain_subgraph, node_ids, border_nodes)
    max_hop = max(hop_to_border.values()) if hop_to_border else 1.0
    if max_hop == 0:
        max_hop = 1.0

    # Build edge_index and edge_attr (intra-domain directed edges only)
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_feats: list[list[float]] = []
    max_edge_delay = 1.0
    intra_edges = []
    for u, v, d in g.edges(data=True):
        if u in node_set and v in node_set:
            intra_edges.append((u, v, d))
            max_edge_delay = max(max_edge_delay, d["propagation_delay"])

    for u, v, d in intra_edges:
        src_list.append(node_id_to_idx[u])
        dst_list.append(node_id_to_idx[v])
        bw_cap = d["bandwidth_capacity"]
        bw_frac = d["bw_residual"] / bw_cap if bw_cap > 0 else 1.0
        delay_norm = d["propagation_delay"] / max_edge_delay
        edge_feats.append([bw_frac, delay_norm])

    # Compute per-node degree and mean incident BW fraction
    degree = [0] * n_nodes
    incident_bw_sum = [0.0] * n_nodes
    incident_bw_count = [0] * n_nodes
    for u, v, d in intra_edges:
        u_idx = node_id_to_idx[u]
        v_idx = node_id_to_idx[v]
        degree[u_idx] += 1
        bw_cap = d["bandwidth_capacity"]
        bw_frac = d["bw_residual"] / bw_cap if bw_cap > 0 else 1.0
        incident_bw_sum[u_idx] += bw_frac
        incident_bw_count[u_idx] += 1
        # incoming edge contributes to v's incident BW
        incident_bw_sum[v_idx] += bw_frac
        incident_bw_count[v_idx] += 1

    max_degree = max(degree) if degree else 1
    if max_degree == 0:
        max_degree = 1

    # Build node features [N x 15]
    node_feats = np.zeros((n_nodes, NODE_FEAT_DIM), dtype=np.float32)
    for i, nid in enumerate(node_ids):
        nd = g.nodes[nid]
        cpu_cap = nd["cpu_capacity"]
        ram_cap = nd["ram_capacity"]
        tier = InfrastructureTier(nd["tier"])

        node_feats[i, 0] = nd["cpu_residual"] / cpu_cap if cpu_cap > 0 else 1.0
        node_feats[i, 1] = nd["ram_residual"] / ram_cap if ram_cap > 0 else 1.0
        node_feats[i, 2] = cpu_cap / max_cpu
        node_feats[i, 3] = ram_cap / max_ram
        node_feats[i, 4 + _TIER_TO_IDX[tier]] = 1.0
        node_feats[i, 8] = nd["processing_delay"] / max_delay
        node_feats[i, 9] = degree[i] / max_degree
        count = incident_bw_count[i]
        node_feats[i, 10] = incident_bw_sum[i] / count if count > 0 else 1.0
        node_feats[i, 11] = hop_to_border.get(nid, max_hop) / max_hop
        node_feats[i, 12] = 1.0 if nid in border_nodes else 0.0
        node_feats[i, 13] = 1.0 if nid in border_to_target_nodes else 0.0
        node_feats[i, 14] = 1.0 if nid in placed_node_ids else 0.0

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32) if edge_feats else torch.zeros(0, EDGE_FEAT_DIM)

    data = Data(
        x=torch.from_numpy(node_feats),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return data, node_ids


def _compute_hop_to_border(
    undirected_subgraph: nx.Graph,
    node_ids: list[str],
    border_nodes: set[str],
) -> dict[str, float]:
    """BFS from all border nodes to compute min hop distance for each node."""
    if not border_nodes:
        return {nid: 0.0 for nid in node_ids}

    hop_dist: dict[str, float] = {}
    for nid in node_ids:
        hop_dist[nid] = float("inf")

    # Multi-source BFS from all border nodes
    queue: list[tuple[str, int]] = []
    for bn in border_nodes:
        if bn in undirected_subgraph:
            hop_dist[bn] = 0.0
            queue.append((bn, 0))

    visited: set[str] = set(border_nodes)
    head = 0
    while head < len(queue):
        node, dist = queue[head]
        head += 1
        for neighbor in undirected_subgraph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                hop_dist[neighbor] = float(dist + 1)
                queue.append((neighbor, dist + 1))

    # Nodes unreachable from any border get max distance
    for nid in node_ids:
        if hop_dist[nid] == float("inf"):
            hop_dist[nid] = float(len(node_ids))

    return hop_dist
