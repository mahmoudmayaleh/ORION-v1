"""SubstrateNetwork: directed-graph wrapper for the multi-domain 6G substrate.

G^P = (N, L) multi-domain substrate.
Each physical undirected link is stored as two directed edges to support the
flow conservation constraint C6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from orion.types import (
    InfrastructureTier,
    LinkAttributes,
    LinkType,
    NodeAttributes,
    PlacementPlan,
    SliceRequest,
)


class SubstrateNetwork:
    """Multi-domain physical substrate G^P = (N, L).

    Wraps a NetworkX DiGraph. Each node and edge stores both total capacity
    and residual capacity as graph attributes. The residual state is modified
    by allocate()/deallocate() and reset by reset(), making this class the
    digital twin during RL training.

    Node attributes stored in graph:
        domain_id, tier, cpu_capacity, ram_capacity, processing_delay,
        cpu_residual, ram_residual

    Edge attributes stored in graph:
        link_id, bandwidth_capacity, bw_residual, propagation_delay, link_type

    Attributes:
        graph: The underlying directed graph.
        num_domains: Number of administrative domains.
    """

    def __init__(self, graph: nx.DiGraph, num_domains: int) -> None:
        """Initialize with a pre-built DiGraph.

        Args:
            graph: Directed graph with node and edge attributes set.
            num_domains: Number of administrative domains in the substrate.
        """
        self.graph = graph
        self.num_domains = num_domains
        # Maps request_id -> (PlacementPlan, SliceRequest) for active slices
        self._active_slices: dict[str, tuple[PlacementPlan, SliceRequest]] = {}

    # ── Topology queries ───────────────────────────────────────────────────────

    def nodes_in_domain(self, domain_id: int) -> list[str]:
        """Return all node IDs belonging to domain_id.

        Args:
            domain_id: Zero-based domain index.

        Returns:
            List of node_id strings.
        """
        return [
            n for n, d in self.graph.nodes(data=True) if d["domain_id"] == domain_id
        ]

    def nodes_by_tier(self, tier: InfrastructureTier) -> list[str]:
        """Return all node IDs of a given infrastructure tier.

        Args:
            tier: The InfrastructureTier to filter by.

        Returns:
            List of node_id strings.
        """
        return [n for n, d in self.graph.nodes(data=True) if d["tier"] == tier]

    def inter_domain_links(self) -> list[str]:
        """Return link IDs whose source and target belong to different domains.

        Returns:
            List of link_id strings for inter-domain directed edges.
        """
        result = []
        for u, v, d in self.graph.edges(data=True):
            if self.graph.nodes[u]["domain_id"] != self.graph.nodes[v]["domain_id"]:
                result.append(d["link_id"])
        return result

    def get_node_attrs(self, node_id: str) -> NodeAttributes:
        """Return a NodeAttributes dataclass for the given node.

        Args:
            node_id: Node identifier.

        Returns:
            NodeAttributes with total (not residual) capacity values.

        Raises:
            KeyError: If node_id is not in the graph.
        """
        d = self.graph.nodes[node_id]
        return NodeAttributes(
            node_id=node_id,
            domain_id=d["domain_id"],
            tier=InfrastructureTier(d["tier"]),
            cpu_capacity=d["cpu_capacity"],
            ram_capacity=d["ram_capacity"],
            processing_delay=d["processing_delay"],
        )

    def get_link_attrs(self, link_id: str) -> LinkAttributes:
        """Return a LinkAttributes dataclass for the given link.

        Args:
            link_id: Link identifier string.

        Returns:
            LinkAttributes with total (not residual) capacity values.

        Raises:
            KeyError: If link_id does not correspond to any edge.
        """
        for u, v, d in self.graph.edges(data=True):
            if d["link_id"] == link_id:
                return LinkAttributes(
                    link_id=link_id,
                    source=u,
                    target=v,
                    bandwidth_capacity=d["bandwidth_capacity"],
                    propagation_delay=d["propagation_delay"],
                    link_type=LinkType(d["link_type"]),
                )
        raise KeyError(f"Link '{link_id}' not found in substrate graph.")

    # ── Residual capacity queries ──────────────────────────────────────────────

    def get_residual_cpu(self, node_id: str) -> float:
        """Return current residual CPU capacity of a node in vCPUs.

        Args:
            node_id: Node identifier.

        Returns:
            Residual vCPUs remaining.
        """
        return float(self.graph.nodes[node_id]["cpu_residual"])

    def get_residual_ram(self, node_id: str) -> float:
        """Return current residual RAM capacity of a node in GB.

        Args:
            node_id: Node identifier.

        Returns:
            Residual GB RAM remaining.
        """
        return float(self.graph.nodes[node_id]["ram_residual"])

    def get_residual_bw(self, link_id: str) -> float:
        """Return current residual bandwidth of a link in Mbps.

        Args:
            link_id: Link identifier string.

        Returns:
            Residual Mbps remaining.

        Raises:
            KeyError: If link_id does not correspond to any edge.
        """
        for _, _, d in self.graph.edges(data=True):
            if d["link_id"] == link_id:
                return float(d["bw_residual"])
        raise KeyError(f"Link '{link_id}' not found in substrate graph.")

    def shortest_path_delay(self, src_node: str, dst_node: str) -> float:
        """Compute shortest propagation delay path between two nodes (Dijkstra).

        Args:
            src_node: Source node identifier.
            dst_node: Destination node identifier.

        Returns:
            Total propagation delay in ms along the shortest path.

        Raises:
            nx.NetworkXNoPath: If no path exists between the nodes.
        """
        return float(
            nx.shortest_path_length(
                self.graph, src_node, dst_node, weight="propagation_delay"
            )
        )

    def incident_links(self, node_id: str) -> tuple[list[str], list[str]]:
        """Return outgoing and incoming link IDs for a node.

        Used for the flow conservation constraint C6:
          sum_{l in delta+(n)} y_l - sum_{l in delta-(n)} y_l = b_n

        Args:
            node_id: Node identifier.

        Returns:
            Tuple (delta_plus, delta_minus) where:
              delta_plus:  link IDs of outgoing edges from node_id
              delta_minus: link IDs of incoming edges to node_id
        """
        delta_plus = [d["link_id"] for _, _, d in self.graph.out_edges(node_id, data=True)]
        delta_minus = [d["link_id"] for _, _, d in self.graph.in_edges(node_id, data=True)]
        return delta_plus, delta_minus

    # ── Resource management (digital twin) ────────────────────────────────────

    def allocate(self, plan: PlacementPlan, request: SliceRequest) -> None:
        """Decrement residual capacities for a placed slice.

        Applies the allocation from plan to the residual capacities:
          cpu_residual[n] -= cpu_allocations[f]   for each VNF f on node n
          ram_residual[n] -= ram_allocations[f]
          bw_residual[l]  -= bw_allocations[(f,f')][l]  for each flow

        Also registers the slice in the active slice registry.

        Args:
            plan: The accepted placement plan for the slice.
            request: The slice request being admitted.

        Raises:
            ValueError: If the request_id is already active.
        """
        if request.request_id in self._active_slices:
            raise ValueError(
                f"Slice '{request.request_id}' is already active. "
                "Call deallocate() before re-allocating."
            )

        for vnf_id, node_id in plan.vnf_placements.items():
            cpu = plan.cpu_allocations[vnf_id]
            ram = plan.ram_allocations[vnf_id]
            self.graph.nodes[node_id]["cpu_residual"] -= cpu
            self.graph.nodes[node_id]["ram_residual"] -= ram

        for flow_key, per_link_bw in plan.bw_allocations.items():
            for link_id, bw in per_link_bw.items():
                for u, v, d in self.graph.edges(data=True):
                    if d["link_id"] == link_id:
                        d["bw_residual"] -= bw
                        break

        self._active_slices[request.request_id] = (plan, request)

    def deallocate(self, plan: PlacementPlan, request: SliceRequest) -> None:
        """Restore residual capacities when a slice departs.

        Reverses the allocation applied by allocate().

        Args:
            plan: The placement plan to reverse.
            request: The departing slice request.

        Raises:
            KeyError: If request_id is not currently active.
        """
        if request.request_id not in self._active_slices:
            raise KeyError(
                f"Slice '{request.request_id}' is not active and cannot be deallocated."
            )

        for vnf_id, node_id in plan.vnf_placements.items():
            cpu = plan.cpu_allocations[vnf_id]
            ram = plan.ram_allocations[vnf_id]
            self.graph.nodes[node_id]["cpu_residual"] += cpu
            self.graph.nodes[node_id]["ram_residual"] += ram

        for flow_key, per_link_bw in plan.bw_allocations.items():
            for link_id, bw in per_link_bw.items():
                for u, v, d in self.graph.edges(data=True):
                    if d["link_id"] == link_id:
                        d["bw_residual"] += bw
                        break

        del self._active_slices[request.request_id]

    def reset(self) -> None:
        """Restore all residual capacities to total values (episode reset).

        Clears the active slice registry. Used at the start of each RL episode.
        """
        for node_id, d in self.graph.nodes(data=True):
            d["cpu_residual"] = d["cpu_capacity"]
            d["ram_residual"] = d["ram_capacity"]

        for u, v, d in self.graph.edges(data=True):
            d["bw_residual"] = d["bandwidth_capacity"]

        self._active_slices.clear()

    # ── RL state aggregation ───────────────────────────────────────────────────

    def domain_residual_summary(self) -> dict[int, dict[str, float]]:
        """Compute per-domain aggregated features for the RL global state.

        Returns:
            Dict mapping domain_id to a feature dict with keys:
              cpu_residual_frac:   sum(residual_cpu) / sum(total_cpu)
              ram_residual_frac:   sum(residual_ram) / sum(total_ram)
              bw_residual_frac:    sum(residual_bw on intra-links) / sum(total_bw)
              active_slice_count:  number of active slices with VNFs in this domain
              utilization:         1 - mean(cpu_residual_frac, ram_residual_frac)
        """
        summary: dict[int, dict[str, float]] = {}

        for domain_id in range(self.num_domains):
            nodes = self.nodes_in_domain(domain_id)
            total_cpu = sum(self.graph.nodes[n]["cpu_capacity"] for n in nodes)
            total_ram = sum(self.graph.nodes[n]["ram_capacity"] for n in nodes)
            res_cpu = sum(self.graph.nodes[n]["cpu_residual"] for n in nodes)
            res_ram = sum(self.graph.nodes[n]["ram_residual"] for n in nodes)

            node_set = set(nodes)
            total_bw = 0.0
            res_bw = 0.0
            for u, v, d in self.graph.edges(data=True):
                if u in node_set and v in node_set:
                    total_bw += d["bandwidth_capacity"]
                    res_bw += d["bw_residual"]

            cpu_frac = res_cpu / total_cpu if total_cpu > 0 else 1.0
            ram_frac = res_ram / total_ram if total_ram > 0 else 1.0
            bw_frac = res_bw / total_bw if total_bw > 0 else 1.0

            active_count = sum(
                1
                for plan, req in self._active_slices.values()
                if any(
                    self.graph.nodes[n]["domain_id"] == domain_id
                    for n in plan.vnf_placements.values()
                )
            )

            summary[domain_id] = {
                "cpu_residual_frac": cpu_frac,
                "ram_residual_frac": ram_frac,
                "bw_residual_frac": bw_frac,
                "active_slice_count": float(active_count),
                "utilization": 1.0 - (cpu_frac + ram_frac) / 2.0,
            }

        return summary

    # ── LLM prompt serialization ───────────────────────────────────────────────

    def to_prompt_text(self) -> str:
        """Compact, LLM-friendly representation of current substrate state.

        Format:
          === DOMAIN 0 (ran_edge) ===
            n0 [ran_edge] CPU: 12/32 vCPU  RAM: 24/64 GB  delay: 1.0ms
            links:
              n0->n1: BW  450/1000 Mbps, delay 1.2ms [intra]
          === INTER-DOMAIN LINKS ===
              n3->n10: BW 800/2000 Mbps, delay 8.5ms [inter]

        Returns:
            Multi-line string representation of the substrate state.
        """
        lines: list[str] = []

        for domain_id in range(self.num_domains):
            nodes = sorted(self.nodes_in_domain(domain_id))
            if not nodes:
                continue

            # Pick a representative tier label for the domain header
            first_tier = self.graph.nodes[nodes[0]]["tier"]
            lines.append(f"=== DOMAIN {domain_id} ({first_tier}) ===")

            for node_id in nodes:
                d = self.graph.nodes[node_id]
                lines.append(
                    f"  {node_id} [{d['tier']}] "
                    f"CPU: {d['cpu_residual']:.0f}/{d['cpu_capacity']:.0f} vCPU  "
                    f"RAM: {d['ram_residual']:.0f}/{d['ram_capacity']:.0f} GB  "
                    f"proc_delay: {d['processing_delay']:.1f}ms"
                )

            node_set = set(nodes)
            intra_edges = [
                (u, v, ed)
                for u, v, ed in self.graph.edges(data=True)
                if u in node_set and v in node_set
            ]
            if intra_edges:
                lines.append("  links:")
                for u, v, ed in sorted(intra_edges, key=lambda x: x[2]["link_id"]):
                    lines.append(
                        f"    {u}->{v}: BW {ed['bw_residual']:.0f}/{ed['bandwidth_capacity']:.0f}"
                        f" Mbps, delay {ed['propagation_delay']:.1f}ms [intra]"
                    )

        # Inter-domain links
        inter_edges = [
            (u, v, d)
            for u, v, d in self.graph.edges(data=True)
            if self.graph.nodes[u]["domain_id"] != self.graph.nodes[v]["domain_id"]
        ]
        if inter_edges:
            lines.append("=== INTER-DOMAIN LINKS ===")
            for u, v, ed in sorted(inter_edges, key=lambda x: x[2]["link_id"]):
                lines.append(
                    f"  {u}->{v}: BW {ed['bw_residual']:.0f}/{ed['bandwidth_capacity']:.0f}"
                    f" Mbps, delay {ed['propagation_delay']:.1f}ms [inter]"
                )

        return "\n".join(lines)

    def to_compact_prompt_text(self) -> str:
        """Minimal LLM-friendly substrate representation for CPU inference.

        Omits per-link bandwidth/delay values (checked by FeasibilityValidator,
        not by the LLM). Uses dense single-line node formatting and adjacency-list
        link encoding to reduce prompt length from ~1400 words to ~300 words.

        Format:
          DOMAIN 0 (mec, 8 nodes):
            d0n0[mec] 10/32C 24/64R  d0n1[mec] 16/32C 12/64R  ...
            adjacency: d0n0->d0n1,d0n3  d0n1->d0n0,d0n2  ...
          INTER-DOMAIN: d0n0->d1n5  d1n5->d0n0  ...
          Link ID rule: l_{src}_{dst}  (e.g. d0n0->d1n5  =>  l_d0n0_d1n5)

        Returns:
            Compact multi-line string representation of the substrate.
        """
        lines: list[str] = []

        for domain_id in range(self.num_domains):
            nodes = sorted(self.nodes_in_domain(domain_id))
            if not nodes:
                continue

            first_tier = self.graph.nodes[nodes[0]]["tier"]
            lines.append(f"DOMAIN {domain_id} ({first_tier}, {len(nodes)} nodes):")

            # Dense node line: up to 4 nodes per line
            node_parts = []
            for node_id in nodes:
                d = self.graph.nodes[node_id]
                node_parts.append(
                    f"{node_id}[{d['tier']}] "
                    f"{d['cpu_residual']:.0f}/{d['cpu_capacity']:.0f}C "
                    f"{d['ram_residual']:.0f}/{d['ram_capacity']:.0f}R"
                )
            chunk_size = 4
            for i in range(0, len(node_parts), chunk_size):
                lines.append("  " + "  ".join(node_parts[i : i + chunk_size]))

            # Adjacency list: one entry per node, intra-domain only
            node_set = set(nodes)
            adj_parts = []
            for node_id in nodes:
                neighbors = [
                    v
                    for _, v in self.graph.out_edges(node_id)
                    if v in node_set
                ]
                if neighbors:
                    adj_parts.append(f"{node_id}->" + ",".join(sorted(neighbors)))
            if adj_parts:
                chunk_size = 3
                lines.append("  adjacency:")
                for i in range(0, len(adj_parts), chunk_size):
                    lines.append("    " + "  ".join(adj_parts[i : i + chunk_size]))

        # Inter-domain adjacency
        inter_pairs = [
            f"{u}->{v}"
            for u, v in self.graph.edges()
            if self.graph.nodes[u]["domain_id"] != self.graph.nodes[v]["domain_id"]
        ]
        if inter_pairs:
            lines.append("INTER-DOMAIN:")
            chunk_size = 6
            for i in range(0, len(inter_pairs), chunk_size):
                lines.append("  " + "  ".join(inter_pairs[i : i + chunk_size]))

        lines.append("Link ID rule: l_{src}_{dst}  (e.g. d0n0->d1n5 => l_d0n0_d1n5)")

        return "\n".join(lines)

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the substrate to a JSON-compatible dictionary.

        Returns:
            Dict with 'num_domains' and 'graph' keys, suitable for JSON export.
        """
        return {
            "num_domains": self.num_domains,
            "graph": nx.node_link_data(self.graph, edges="edges"),
        }

    def to_json(self, path: Path) -> None:
        """Serialize the substrate to a JSON file.

        Args:
            path: Destination file path. Parent directory must exist.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubstrateNetwork:
        """Deserialize a substrate from a dictionary.

        Args:
            data: Dictionary produced by to_dict().

        Returns:
            Reconstructed SubstrateNetwork with residuals set to capacities
            if not present in the serialized data.
        """
        graph: nx.DiGraph = nx.node_link_graph(data["graph"], directed=True, edges="edges")
        # Ensure residuals are present (backward-compat for serialized graphs without them)
        for _, d in graph.nodes(data=True):
            d.setdefault("cpu_residual", d["cpu_capacity"])
            d.setdefault("ram_residual", d["ram_capacity"])
        for _, _, d in graph.edges(data=True):
            d.setdefault("bw_residual", d["bandwidth_capacity"])
        return cls(graph=graph, num_domains=data["num_domains"])

    @classmethod
    def from_json(cls, path: Path) -> SubstrateNetwork:
        """Deserialize a substrate from a JSON file.

        Args:
            path: Path to a JSON file produced by to_json().

        Returns:
            Reconstructed SubstrateNetwork.
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
