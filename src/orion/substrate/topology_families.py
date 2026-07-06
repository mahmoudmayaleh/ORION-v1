"""Parametrized topology family generator for memory experiments.

Three hostility axes, crossed to produce named regimes:

  Capacity axis:
    friendly  — nodes_per_domain large enough for whole chains (6-8 nodes)
    hostile   — nodes_per_domain small relative to chain length (2-3 nodes)

  Tier axis:
    friendly  — every domain has all four tier types
    hostile   — tiers split across domains (edge/MEC in some, cloud in others)

  BW axis:
    generous  — high inter-domain BW (spreading is cheap)
    scarce    — low inter-domain BW (spreading is fatal)

Crossing gives 8 families. Some reward co-location, some punish it,
some depend on load — the third category is where memory earns its keep.

The topology signature exposes features that determine which plan shape
survives: per-domain tier coverage, capacity relative to demand,
inter-domain BW quantiles, connectivity summary. A human given this
vector can decide co-locate vs spread.

Design decisions documented:
  - Family-level train/test split (hold out parameter-space regions)
  - Arrival process and slice-mix FROZEN across families (one axis at a time)
  - Instance sizes kept enumerator-tractable (15-40 nodes)
  - Signature is part of the generator, not an afterthought
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from orion.config import TopologyConfig
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier


# ── Regime axes ─────────────────────────────────────────────────────────────


class CapacityRegime(str, Enum):
    FRIENDLY = "capacity_friendly"
    HOSTILE = "capacity_hostile"


class TierRegime(str, Enum):
    FRIENDLY = "tier_friendly"
    HOSTILE = "tier_hostile"


class BWRegime(str, Enum):
    GENEROUS = "bw_generous"
    SCARCE = "bw_scarce"


# ── Family definition ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class TopologyFamily:
    """A named topology regime defined by three axes."""
    capacity: CapacityRegime
    tier: TierRegime
    bw: BWRegime

    @property
    def name(self) -> str:
        return f"{self.capacity.value}__{self.tier.value}__{self.bw.value}"

    @property
    def short_name(self) -> str:
        c = "C+" if self.capacity == CapacityRegime.FRIENDLY else "C-"
        t = "T+" if self.tier == TierRegime.FRIENDLY else "T-"
        b = "B+" if self.bw == BWRegime.GENEROUS else "B-"
        return f"{c}_{t}_{b}"

    @property
    def colocation_prediction(self) -> str:
        """Predict whether co-location-first wins on this family."""
        if self.capacity == CapacityRegime.HOSTILE:
            return "hostile"  # chains can't fit single domain
        if self.tier == TierRegime.HOSTILE:
            return "hostile"  # tier requirements force cross-domain
        if self.bw == BWRegime.GENEROUS:
            return "friendly"  # co-location wins but spreading is also safe
        return "friendly"  # co-location wins, spreading is punished


ALL_FAMILIES = [
    TopologyFamily(c, t, b)
    for c in CapacityRegime
    for t in TierRegime
    for b in BWRegime
]


# ── Train/test split (family-level, fixed by design) ────────────────────────


# Training families: regimes where the answer is clear
TRAIN_FAMILIES = [
    TopologyFamily(CapacityRegime.FRIENDLY, TierRegime.FRIENDLY, BWRegime.GENEROUS),
    TopologyFamily(CapacityRegime.FRIENDLY, TierRegime.FRIENDLY, BWRegime.SCARCE),
    TopologyFamily(CapacityRegime.HOSTILE, TierRegime.HOSTILE, BWRegime.GENEROUS),
    TopologyFamily(CapacityRegime.HOSTILE, TierRegime.HOSTILE, BWRegime.SCARCE),
    TopologyFamily(CapacityRegime.FRIENDLY, TierRegime.HOSTILE, BWRegime.GENEROUS),
]

# Test families: held-out regions — includes the mixed/load-dependent cases
TEST_FAMILIES = [
    TopologyFamily(CapacityRegime.HOSTILE, TierRegime.FRIENDLY, BWRegime.SCARCE),
    TopologyFamily(CapacityRegime.HOSTILE, TierRegime.FRIENDLY, BWRegime.GENEROUS),
    TopologyFamily(CapacityRegime.FRIENDLY, TierRegime.HOSTILE, BWRegime.SCARCE),
]


# ── Topology signature ─────────────────────────────────────────────────────


@dataclass
class TopologySignature:
    """Feature vector exposing what determines which plan shape survives.

    Designed so a human given this vector can decide co-locate vs spread.
    M^B retrieval keys on this.
    """
    num_domains: int
    nodes_per_domain: list[int]

    # Per-domain tier coverage (fraction of 4 tiers present)
    tier_coverage_per_domain: list[float]
    # Per-domain: which tiers are present
    tiers_per_domain: list[list[str]]

    # Per-domain capacity relative to typical chain demand
    cpu_per_domain: list[float]
    ram_per_domain: list[float]

    # Inter-domain BW stats
    inter_bw_mean: float
    inter_bw_min: float
    inter_bw_max: float
    num_inter_links: int

    # Connectivity
    avg_intra_degree: float
    inter_connectivity: float  # inter_links / (num_domains * (num_domains-1) / 2)

    # Regime labels (for analysis, not for retrieval)
    family: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_domains": self.num_domains,
            "nodes_per_domain": self.nodes_per_domain,
            "tier_coverage": self.tier_coverage_per_domain,
            "tiers_per_domain": self.tiers_per_domain,
            "cpu_per_domain": self.cpu_per_domain,
            "ram_per_domain": self.ram_per_domain,
            "inter_bw_mean": self.inter_bw_mean,
            "inter_bw_min": self.inter_bw_min,
            "inter_bw_max": self.inter_bw_max,
            "num_inter_links": self.num_inter_links,
            "avg_intra_degree": self.avg_intra_degree,
            "inter_connectivity": self.inter_connectivity,
            "family": self.family,
        }


def compute_signature(sub: SubstrateNetwork, family_name: str = "") -> TopologySignature:
    """Compute the topology signature from a generated substrate."""
    g = sub.graph
    num_domains = sub.num_domains
    all_tiers = {t.value for t in InfrastructureTier}

    nodes_per_domain = []
    tier_coverage = []
    tiers_per_domain = []
    cpu_per_domain = []
    ram_per_domain = []

    for dom in range(num_domains):
        dom_nodes = [n for n, d in g.nodes(data=True) if d.get("domain_id") == dom]
        nodes_per_domain.append(len(dom_nodes))

        tiers = set()
        cpu_total = 0.0
        ram_total = 0.0
        for nid in dom_nodes:
            d = g.nodes[nid]
            tiers.add(d["tier"])
            cpu_total += float(d["cpu_capacity"])
            ram_total += float(d["ram_capacity"])

        tier_coverage.append(len(tiers) / len(all_tiers))
        tiers_per_domain.append(sorted(tiers))
        cpu_per_domain.append(round(cpu_total, 1))
        ram_per_domain.append(round(ram_total, 1))

    # Inter-domain BW stats
    inter_bws = []
    inter_links = 0
    for u, v, d in g.edges(data=True):
        u_dom = g.nodes[u].get("domain_id", -1)
        v_dom = g.nodes[v].get("domain_id", -1)
        if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
            inter_bws.append(float(d.get("bw_capacity", d.get("bandwidth_capacity", 0))))
            inter_links += 1

    # Intra-domain degree
    intra_edges = 0
    total_nodes = sum(nodes_per_domain)
    for u, v, d in g.edges(data=True):
        u_dom = g.nodes[u].get("domain_id", -1)
        v_dom = g.nodes[v].get("domain_id", -1)
        if u_dom == v_dom and u_dom >= 0:
            intra_edges += 1

    avg_intra_degree = intra_edges / total_nodes if total_nodes > 0 else 0
    n_pairs = num_domains * (num_domains - 1) / 2 if num_domains > 1 else 1
    inter_conn = (inter_links / 2) / n_pairs if n_pairs > 0 else 0  # /2 for directed→undirected

    return TopologySignature(
        num_domains=num_domains,
        nodes_per_domain=nodes_per_domain,
        tier_coverage_per_domain=[round(c, 2) for c in tier_coverage],
        tiers_per_domain=tiers_per_domain,
        cpu_per_domain=cpu_per_domain,
        ram_per_domain=ram_per_domain,
        inter_bw_mean=round(np.mean(inter_bws), 1) if inter_bws else 0.0,
        inter_bw_min=round(min(inter_bws), 1) if inter_bws else 0.0,
        inter_bw_max=round(max(inter_bws), 1) if inter_bws else 0.0,
        num_inter_links=inter_links // 2,  # undirected count
        avg_intra_degree=round(avg_intra_degree, 2),
        inter_connectivity=round(inter_conn, 2),
        family=family_name,
    )


# ── Family → TopologyConfig mapping ────────────────────────────────────────


def family_to_config(family: TopologyFamily, num_domains: int = 5) -> TopologyConfig:
    """Map a family to a TopologyConfig.

    This is the core design: each axis controls specific parameters.
    """

    # ── Capacity axis ──
    if family.capacity == CapacityRegime.FRIENDLY:
        nodes_per_domain = [6] * num_domains  # 6 nodes → chains of 3-5 fit easily
    else:
        nodes_per_domain = [3] * num_domains  # 3 nodes → chains of 3+ can't fit

    # ── Tier axis ──
    if family.tier == TierRegime.FRIENDLY:
        # Every domain gets all tiers (balanced distribution)
        tier_dist = {
            "ran_edge": 0.25,
            "mec": 0.25,
            "regional_cloud": 0.25,
            "central_cloud": 0.25,
        }
    else:
        # Tiers split: will be overridden per-domain below
        # Use a distribution that creates splits naturally
        tier_dist = {
            "ran_edge": 0.10,
            "mec": 0.10,
            "regional_cloud": 0.40,
            "central_cloud": 0.40,
        }

    # ── BW axis ──
    if family.bw == BWRegime.GENEROUS:
        bw_range = (200, 1000)  # Enough for spreading, not so much that nothing fails
        inter_domain_links = 4
    else:
        bw_range = (50, 200)  # Spreading is expensive/fatal
        inter_domain_links = 2

    return TopologyConfig(
        num_domains=num_domains,
        nodes_per_domain=nodes_per_domain,
        intra_link_density=0.5,
        inter_domain_links=inter_domain_links,
        tier_distribution=tier_dist,
        bw_range=bw_range,
        delay_intra_range=(0.5, 5.0),
        delay_inter_range=(5.0, 20.0),
    )


def generate_family_instance(
    family: TopologyFamily,
    seed: int,
    num_domains: int = 5,
    inter_domain_bw_override: float | None = None,
) -> SubstrateNetwork:
    """Generate one topology instance from a family.

    For tier-hostile families, post-processes tier assignments to ensure
    tiers are split across domains (the generator's random sampling may
    not produce clean splits).
    """
    rng = np.random.default_rng(seed)
    config = family_to_config(family, num_domains)
    sub = generate_multi_domain_topology(config, rng)
    g = sub.graph

    # ── Tier-hostile post-processing ──
    if family.tier == TierRegime.HOSTILE:
        _enforce_tier_split(sub, rng)

    # ── BW override (for precise control) ──
    if inter_domain_bw_override is not None:
        for u, v, d in g.edges(data=True):
            u_dom = g.nodes[u].get("domain_id", -1)
            v_dom = g.nodes[v].get("domain_id", -1)
            if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
                d["bandwidth_capacity"] = inter_domain_bw_override
                d["bw_capacity"] = inter_domain_bw_override
                d["bw_residual"] = inter_domain_bw_override

    return sub


def _enforce_tier_split(sub: SubstrateNetwork, rng: np.random.Generator) -> None:
    """Force tier-hostile distribution: edge/MEC tiers in odd domains,
    cloud tiers in even domains. Ensures no single domain has all tiers.
    """
    g = sub.graph
    edge_tiers = [InfrastructureTier.RAN_EDGE.value, InfrastructureTier.MEC.value]
    cloud_tiers = [InfrastructureTier.REGIONAL_CLOUD.value, InfrastructureTier.CENTRAL_CLOUD.value]

    for nid, d in g.nodes(data=True):
        dom = d.get("domain_id", -1)
        if dom < 0:
            continue

        if dom % 2 == 0:
            # Even domains: cloud only
            if d["tier"] in edge_tiers:
                new_tier = rng.choice(cloud_tiers)
                _reassign_tier(d, new_tier, rng)
        else:
            # Odd domains: edge/MEC only
            if d["tier"] in cloud_tiers:
                new_tier = rng.choice(edge_tiers)
                _reassign_tier(d, new_tier, rng)


def _reassign_tier(node_data: dict, new_tier_value: str, rng: np.random.Generator) -> None:
    """Reassign a node's tier and update capacity ranges to match."""
    tier = InfrastructureTier(new_tier_value)
    from orion.substrate.topology_generator import (
        _TIER_CPU_RANGE, _TIER_RAM_RANGE, _TIER_PROC_DELAY_RANGE,
    )
    cpu = float(rng.uniform(*_TIER_CPU_RANGE[tier]))
    ram = float(rng.uniform(*_TIER_RAM_RANGE[tier]))
    proc_delay = float(rng.uniform(*_TIER_PROC_DELAY_RANGE[tier]))

    node_data["tier"] = new_tier_value
    node_data["cpu_capacity"] = round(cpu, 2)
    node_data["ram_capacity"] = round(ram, 2)
    node_data["processing_delay"] = round(proc_delay, 3)
    node_data["cpu_residual"] = round(cpu, 2)
    node_data["ram_residual"] = round(ram, 2)


# ── Convenience: generate a batch for one family ───────────────────────────


def generate_family_batch(
    family: TopologyFamily,
    n_instances: int = 5,
    base_seed: int = 0,
    num_domains: int = 5,
) -> list[tuple[SubstrateNetwork, TopologySignature]]:
    """Generate n_instances from a family, each with a different seed."""
    results = []
    for i in range(n_instances):
        sub = generate_family_instance(family, seed=base_seed + i, num_domains=num_domains)
        sig = compute_signature(sub, family_name=family.short_name)
        results.append((sub, sig))
    return results
