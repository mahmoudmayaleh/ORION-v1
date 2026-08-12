r"""§Y.1e - heterogeneous multi-domain 6G substrate generator.

The single generator on the §Y path. Two properties carry the design.

**Tier is a consequence of position in the hierarchy**, not a draw from a
probability vector. Leaves are `edge`, the layer above is `regional_cloud`, the
top is `central_cloud`. That is what makes edge-anchored and cloud-anchored VNFs
structurally separated rather than nominally labelled, and it is why the
intra-domain graph is a tree-like partial mesh rather than Erdos-Renyi: a random
graph deletes the rule that assigns the tier and leaves nothing to replace it.

**Domains differ in which tiers they hold.** A domain is an operator with a
particular infrastructure footprint, not a copy of its neighbours. There is no
core domain and no star.

        D0 ----------- D1              partial mesh, 8 adjacencies,
        |  \         /  |              INTER_LINKS_PER_PAIR links each
        |    \     /    |
        |      D3       |              D3 = the central-cloud domain
        |    /     \    |
        |  /         \  |
        D4 ----------- D2

    D0, D2   edge + regional + central, 20 nodes, 3 levels
    D1, D4   edge + regional,           15 nodes, 2 levels
    D3       central only,              10 nodes, flat

Why not one core domain plus four identical regional ones, which is what this
replaced (measured on that substrate, 2026-07-31):

  * 15 of the core's 20 nodes carried the `regional_cloud` tier, so 15 of the
    substrate's 19 regional-cloud nodes sat INSIDE the core and 4 sat in the
    regions. Under the Plain heuristic the core absorbed 35-40% of all placed
    VNFs against 13-20% for each regional domain. A tier named "regional" was
    79% located outside the regions.
  * No VNF template required `central_cloud`, so the core was an overflow pool
    that produced no forced cross-domain placement. Heterogeneous composition
    produces it directly: D3 can host no complete chain (see DOMAIN_TIERS).
  * The choice had no recorded justification, while every other structural
    choice here does.

**Generator fixed, instances sampled.** Structure, composition, tier assignment,
link counts and inter-domain adjacency are IDENTICAL in every instance. An
instance varies capacities, bandwidths, delays, and WHICH same-tier pairs receive
the lateral links.

The size is FIXED. There is no size axis: the MDO observation has no padding over
M or L, so a policy trained at one shape cannot even be fed an observation from
another.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import (
    _TIER_CPU_RANGE,
    _TIER_PROC_DELAY_RANGE,
    _TIER_RAM_RANGE,
    _add_directed_link_pair,
)
from orion.mdo.observation import DOMAIN_FEAT_DIM as _DOMAIN_FEAT_DIM
from orion.types import InfrastructureTier, LinkType, TIER_ORDER

_EDGE = InfrastructureTier.EDGE
_REGIONAL = InfrastructureTier.REGIONAL_CLOUD
_CENTRAL = InfrastructureTier.CENTRAL_CLOUD

# --------------------------------------------------------------------------
# Committed structural constants (§Y.1e). Changing any of these is an amendment.
# --------------------------------------------------------------------------

#: Branching factor 4, NOT 2. Depth is not a free parameter: a hierarchy has
#: exactly one transport hop per tier boundary, so a 3-tier domain is two hops
#: from an access node to the domain root. Inserting extra levels charges
#: propagation delay for hops the architecture does not have. Measured cost of
#: getting this wrong on the superseded substrate: 6.05 ms of intra-domain link
#: delay on V2X and mMTC paths that never left their own domain, against a 5-20 ms
#: budget. Branching 4 also gives the realistic 15:4:1 fan-out instead of 10:9:1.
BRANCHING_FACTOR = 4

#: Tiers held by each domain, ordered TOP FIRST. len(tiers) == number of levels.
#:
#: D3 holds `central_cloud` only, and that has a consequence worth knowing before
#: reading any result: no VNF template permits central cloud EXCLUSIVELY, and
#: every chain contains at least one edge-only or regional-only VNF, so **D3 can
#: host no complete chain**. Its nodes are unreachable to a colocation-first
#: placer and can only ever receive a fragment of a split partition. That is
#: realistic for a national data centre hosting only cloud-anchored functions, and
#: it is what makes cross-domain placement genuinely forced here. Registered as a
#: prediction in §Y.1e so it is checked rather than rationalised afterwards.
DOMAIN_TIERS: dict[int, tuple[InfrastructureTier, ...]] = {
    0: (_CENTRAL, _REGIONAL, _EDGE),
    1: (_REGIONAL, _EDGE),
    2: (_CENTRAL, _REGIONAL, _EDGE),
    3: (_CENTRAL,),
    4: (_REGIONAL, _EDGE),
}

#: Nodes per domain, scaled to how many tiers it holds. Frozen up front.
DOMAIN_SIZES: dict[int, int] = {0: 20, 1: 15, 2: 20, 3: 10, 4: 15}

NUM_DOMAINS = len(DOMAIN_TIERS)                     # 5
TOTAL_NODES = sum(DOMAIN_SIZES.values())            # 80

#: Inter-domain adjacencies: a PARTIAL MESH, not a star.
#:
#: D3 connects to all four others; D0, D1, D2, D4 form a ring among themselves.
#: Degrees are 4 for D3 and 3 for everyone else.
#:
#: Justification of record, which is what matters rather than the count. D1 and D4
#: hold no central cloud at all, so any chain of theirs containing a cloud-anchored
#: VNF must leave the domain: reachability to a central-bearing domain is a
#: CORRECTNESS property. D3 holds 10 of the 12 central-cloud nodes, so it is
#: reachable directly from every domain. The ring means peer traffic among the
#: other four never transits D3, which is what stops this collapsing back into the
#: star it replaces. Guarantee: any single adjacency can fail with every domain
#: still reachable AND every domain still holding a path to central cloud. That is
#: asserted as a property in tests/test_y_topology_and_load.py, by deleting each
#: adjacency in turn, rather than as a link count that could drift from the
#: argument it exists to support.
#:
#: A UNIFORM degree of 3 is not realizable over five domains: five nodes each of
#: odd degree gives an odd degree sum. This is the nearest realizable design, and
#: D3's higher degree is the point rather than a workaround, since it holds the
#: scarce tier.
INTER_DOMAIN_ADJACENCIES: tuple[tuple[int, int], ...] = (
    (3, 0), (3, 1), (3, 2), (3, 4),     # the central-cloud domain, reachable from all
    (0, 1), (1, 2), (2, 4), (4, 0),     # ring, so peer traffic need not transit D3
)

#: Parallel links per inter-domain adjacency. More than one so an adjacency is not
#: a single cable, and they attach to DIFFERENT gateway nodes where a domain has
#: more than one top-tier node, so the pair survives a node failure and not only a
#: link failure.
INTER_LINKS_PER_PAIR = 2

#: Same-tier lateral links, as a fraction of the domain's node count, sampled ON
#: TOP of the structural backbone (see `_backbone_edges`). 0.30 is the figure put
#: to the supervisor on 2026-07-30 and approved.
#:
#: The budget is split ACROSS layers with a floor of one per eligible layer rather
#: than sampled from the pooled candidate set. Pooling is dominated by the widest
#: layer: at 20 nodes the edge layer offers 105 candidate pairs against the
#: regional layer's 6, so instances came out with every lateral link at the edge
#: and the layer above left a bare tree. Layer COUNTS are deterministic; which
#: pairs receive the links still varies by instance.
EXTRA_LINK_FRAC = 0.30

# Bandwidth (Mbps) and propagation delay (ms) by the TIER PAIR a link connects,
# not by a depth index: with variable domain depth, keying on depth would attach
# different physical meanings to the same number in different domains.
#
# Delay figures are grounded, not chosen. Single-mode fibre propagates at
# ~4.9 us/km (refractive index ~1.47). The operator target is that the end-to-end
# latency of the WHOLE 5G transport network stays within 2-4 ms, which is what
# rules out the 5.0-15.0 ms inter-domain range this replaced: a single link then
# exceeded the entire transport budget and made every chain reaching another
# domain structurally infeasible for XR and V2X at any load.
_BW_ACCESS = (1_000.0, 2_000.0)
_BW_AGGREGATION = (2_500.0, 5_000.0)
_BW_CORE_UPLINK = (6_000.0, 10_000.0)
_BW_DATACENTRE = (6_000.0, 10_000.0)
_BW_INTER = (8_000.0, 12_000.0)

_DELAY_ACCESS = (0.2, 0.8)              # published metro access figure, 10-40 km
_DELAY_AGGREGATION = (0.8, 1.6)         # published aggregation-layer figure
_DELAY_DATACENTRE = (0.05, 0.2)         # links inside one central site
_DELAY_INTER = (0.5, 2.0)               # ~100-400 km of fibre at 4.9 us/km

#: (tier, tier) -> (bandwidth range, delay range). Unordered: keyed on the sorted
#: pair so link direction cannot change a link's class.
_LINK_PROFILE: dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]] = {
    (_EDGE, _EDGE): (_BW_ACCESS, _DELAY_ACCESS),
    (_EDGE, _REGIONAL): (_BW_AGGREGATION, _DELAY_AGGREGATION),
    (_REGIONAL, _REGIONAL): (_BW_AGGREGATION, _DELAY_AGGREGATION),
    (_REGIONAL, _CENTRAL): (_BW_CORE_UPLINK, _DELAY_AGGREGATION),
    (_CENTRAL, _CENTRAL): (_BW_DATACENTRE, _DELAY_DATACENTRE),
    # Not produced by any committed composition (edge never parents central), but
    # defined so a future composition change fails a test rather than a KeyError
    # at generation time.
    (_EDGE, _CENTRAL): (_BW_CORE_UPLINK, _DELAY_AGGREGATION),
}

#: Directed inter-domain pairs. The observation AGGREGATES inter-domain links by
#: ordered domain pair, so each ADJACENCY contributes 2 entries regardless of how
#: many physical links it carries.
NUM_DOMAIN_PAIRS = 2 * len(INTER_DOMAIN_ADJACENCIES)                  # 16

#: Per-domain observation features. IMPORTED, not restated: the observation builder
#: owns the layout, and a second definition here would be a number that must agree
#: with the emitter and would fail silently if it ever stopped agreeing.
NUM_TIERS = len(TIER_ORDER)
DOMAIN_FEAT_DIM = _DOMAIN_FEAT_DIM                                    # 17

#: MDO observation width at this substrate. Constant across every §Y cell.
MAX_VNFS = 10
OBS_DIM = (DOMAIN_FEAT_DIM * NUM_DOMAINS
           + 3 * NUM_DOMAIN_PAIRS
           + (5 + NUM_DOMAINS) * MAX_VNFS   # per-VNF features + one-hot of m̃_k
           + 5)                                                       # 238

# The adjacency list is written down explicitly, so it must name real domains and
# contain no duplicate or self adjacency. A silent bad entry would be dropped by
# the generator and quietly reduce connectivity.
assert all(0 <= a < NUM_DOMAINS and 0 <= b < NUM_DOMAINS and a != b
           for a, b in INTER_DOMAIN_ADJACENCIES), \
    "INTER_DOMAIN_ADJACENCIES must name distinct existing domains"
assert len({frozenset(p) for p in INTER_DOMAIN_ADJACENCIES}) == len(INTER_DOMAIN_ADJACENCIES), \
    "INTER_DOMAIN_ADJACENCIES contains a duplicate adjacency"
assert set(DOMAIN_TIERS) == set(DOMAIN_SIZES), \
    "DOMAIN_TIERS and DOMAIN_SIZES must cover the same domains"

#: Instance seed split (§Y.1). Disjoint by construction, not by chance.
TRAIN_INSTANCES = tuple(range(20))
HELDOUT_INSTANCES = (100, 101, 102, 103, 104)


# --------------------------------------------------------------------------
# Intra-domain structure
# --------------------------------------------------------------------------

def _profile_for(tier_a: InfrastructureTier, tier_b: InfrastructureTier):
    """Bandwidth and delay ranges for a link joining two tiers, order-independent."""
    key = (tier_a, tier_b)
    if key not in _LINK_PROFILE:
        key = (tier_b, tier_a)
    return _LINK_PROFILE[key]


def _levels_and_parents(n: int, num_levels: int) -> tuple[list[int], list[int | None], int]:
    """Lay out `n` nodes as a forest of `num_levels` levels, branching BRANCHING_FACTOR.

    Returns `(level_of_index, parent_of_index, num_roots)`.

    The number of roots follows from the size rather than being chosen: one root
    holds `(b^L - 1) / (b - 1)` nodes, so a domain that needs more than that gets
    more roots. Concretely at branching 4: a 3-level domain of 20 is ONE root with
    4 children and 15 grandchildren; a 2-level domain of 15 is THREE roots of four,
    not one root of fourteen. A single wide root would make the domain a star with
    one point of failure and would put its entire top tier on one node.

    A 1-level domain is flat: every node is a root and there is no hierarchy,
    which is the whole meaning of a domain holding a single tier.
    """
    if num_levels < 1:
        raise ValueError("a domain must hold at least one tier")
    per_root = sum(BRANCHING_FACTOR ** d for d in range(num_levels))
    num_roots = max(1, -(-n // per_root))       # ceil division

    level: list[int] = [0] * num_roots
    parent: list[int | None] = [None] * num_roots
    frontier = list(range(num_roots))
    idx = num_roots
    depth = 0
    while idx < n and depth < num_levels - 1:
        nxt: list[int] = []
        for p in frontier:
            for _ in range(BRANCHING_FACTOR):
                if idx >= n:
                    break
                level.append(depth + 1)
                parent.append(p)
                nxt.append(idx)
                idx += 1
            if idx >= n:
                break
        if not nxt:
            break
        frontier = nxt
        depth += 1

    if len(level) != n:
        raise RuntimeError(
            f"layout produced {len(level)} nodes for a {num_levels}-level domain of "
            f"{n}: the size does not fit the branching factor")
    return level, parent, num_roots


def _backbone_edges(level: list[int], parent: list[int | None],
                    num_roots: int) -> list[tuple[int, int]]:
    """Structural links: every parent-child edge, plus a ring over the top tier.

    The ring is NOT optional and is NOT part of the sampled lateral budget. Three
    roots of four children are three separate components until the roots are
    joined, and a flat domain has no tree at all: at 10 nodes the 30% lateral
    budget is 3 links, which cannot connect it. Connectivity must therefore be
    structural, and it is asserted per domain in the tests.
    """
    edges = [(p, i) for i, p in enumerate(parent) if p is not None]
    if num_roots > 1:
        # `_add_directed_link_pair` skips an existing edge, so a 2-root "ring"
        # correctly yields one link rather than a duplicate.
        edges += [(r, (r + 1) % num_roots) for r in range(num_roots)]
    return edges


def _lateral_candidates(level: list[int], node_ids: list[str],
                        graph: nx.DiGraph) -> list[tuple[int, list[tuple[int, int]]]]:
    """Same-level node pairs not already joined, grouped by level."""
    by_level: dict[int, list[int]] = {}
    for i, d in enumerate(level):
        by_level.setdefault(d, []).append(i)
    layers = [
        (d, [(a, b)
             for pos, a in enumerate(peers)
             for b in peers[pos + 1:]
             if not graph.has_edge(node_ids[a], node_ids[b])])
        for d, peers in sorted(by_level.items())
    ]
    return [(d, cands) for d, cands in layers if cands]


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def generate_hierarchical_topology(
    instance_seed: int = 0,
    extra_link_frac: float = EXTRA_LINK_FRAC,
) -> SubstrateNetwork:
    """Build one instance of the §Y substrate.

    There is no size or composition argument: both are fixed for all of §Y (see
    DOMAIN_TIERS / DOMAIN_SIZES for why).

    Args:
        instance_seed: Selects the instance. Structure is identical for every
            seed; only capacities, bandwidths, delays and which same-tier pairs
            receive the lateral links vary. Use `TRAIN_INSTANCES` /
            `HELDOUT_INSTANCES` to stay on the split.
        extra_link_frac: Same-tier lateral links added on top of the backbone.

    Returns:
        A `SubstrateNetwork` with residuals initialised to capacities.
    """
    rng = np.random.default_rng(instance_seed)
    graph = nx.DiGraph()

    domain_roots: dict[int, list[str]] = {}

    for domain_id in sorted(DOMAIN_TIERS):
        tiers = DOMAIN_TIERS[domain_id]
        n = DOMAIN_SIZES[domain_id]
        level, parent, num_roots = _levels_and_parents(n, len(tiers))

        node_ids = [f"d{domain_id}n{i}" for i in range(n)]
        domain_roots[domain_id] = node_ids[:num_roots]
        tier_of = [tiers[d] for d in level]

        for i, node_id in enumerate(node_ids):
            tier = tier_of[i]
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

        # Structural backbone. Deterministic: the hierarchy is never perturbed.
        for a, b in _backbone_edges(level, parent, num_roots):
            bw_range, delay_range = _profile_for(tier_of[a], tier_of[b])
            _add_directed_link_pair(
                graph, node_ids[a], node_ids[b],
                bandwidth_capacity=round(float(rng.uniform(*bw_range)), 1),
                propagation_delay=round(float(rng.uniform(*delay_range)), 3),
                link_type=LinkType.INTRA,
            )

        # Instance-varying lateral links, between nodes at the same LEVEL, which
        # by construction means the same TIER. This is the form the supervisor
        # accepted the tree-like topology on, and it leaves the hierarchy, hence
        # the tier of every node, untouched.
        n_extra = int(round(extra_link_frac * n))
        layers = _lateral_candidates(level, node_ids, graph)
        if layers and n_extra:
            total_cands = sum(len(c) for _, c in layers)
            quota = {d: max(1, round(n_extra * len(c) / total_cands))
                     for d, c in layers}
            # Trim from the widest layer down until the budget is met exactly, so
            # EXTRA_LINK_FRAC still means what it says.
            while sum(quota.values()) > n_extra:
                widest = max(quota, key=lambda d: quota[d])
                if quota[widest] <= 1:
                    break
                quota[widest] -= 1
            for depth, cands in layers:
                k = min(quota[depth], len(cands))
                if k <= 0:
                    continue
                chosen = rng.choice(len(cands), size=k, replace=False)
                for c in np.atleast_1d(chosen):
                    a, b = cands[int(c)]
                    bw_range, delay_range = _profile_for(tier_of[a], tier_of[b])
                    _add_directed_link_pair(
                        graph, node_ids[a], node_ids[b],
                        bandwidth_capacity=round(float(rng.uniform(*bw_range)), 1),
                        propagation_delay=round(float(rng.uniform(*delay_range)), 3),
                        link_type=LinkType.INTRA,
                    )

    # Inter-domain: the partial mesh. Parallel links attach to DIFFERENT top-tier
    # nodes where a domain has more than one, so an adjacency survives a gateway
    # node failure and not only a link failure.
    for a, b in INTER_DOMAIN_ADJACENCIES:
        roots_a, roots_b = domain_roots[a], domain_roots[b]
        for k in range(INTER_LINKS_PER_PAIR):
            _add_directed_link_pair(
                graph, roots_a[k % len(roots_a)], roots_b[k % len(roots_b)],
                bandwidth_capacity=round(float(rng.uniform(*_BW_INTER)), 1),
                propagation_delay=round(float(rng.uniform(*_DELAY_INTER)), 3),
                link_type=LinkType.INTER,
            )

    return SubstrateNetwork(graph=graph, num_domains=NUM_DOMAINS)


def describe() -> str:
    """Structural summary of the committed substrate, for the pre-registration."""
    sub = generate_hierarchical_topology(instance_seed=0)
    counts: dict[str, int] = {}
    per_domain: dict[int, dict[str, int]] = {}
    for node_id in sub.graph.nodes:
        attrs = sub.graph.nodes[node_id]
        tier = attrs["tier"]
        counts[tier] = counts.get(tier, 0) + 1
        row = per_domain.setdefault(attrs["domain_id"], {})
        row[tier] = row.get(tier, 0) + 1
    lines = [
        f"M={NUM_DOMAINS}  total nodes={TOTAL_NODES}",
        f"  obs_dim={OBS_DIM}   ({DOMAIN_FEAT_DIM}M + 3L + 5*MAX_VNFS + 5, "
        f"L={NUM_DOMAIN_PAIRS})",
        f"  directed links: {sub.graph.number_of_edges()} "
        f"({len(sub.inter_domain_links())} inter-domain)",
        "  tiers: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    ]
    for domain_id in sorted(per_domain):
        row = per_domain[domain_id]
        lines.append(f"    D{domain_id} ({DOMAIN_SIZES[domain_id]} nodes): "
                     + ", ".join(f"{k}={v}" for k, v in sorted(row.items())))
    return "\n".join(lines)
