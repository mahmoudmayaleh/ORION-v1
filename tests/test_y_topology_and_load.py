"""§Y.1 / §Y.2 guards for the hierarchical generator and the load model.

These pin the properties the §Y design rests on, including the failure modes found
while building it, so none can be reintroduced silently:

  * the topology size is a CONSTANT, not a parameter (§Y.11 descoped 2026-07-29),
    and cross-size evaluation is a shape error rather than a degradation,
  * a depth-cutoff core tier rule that pinned central_cloud at 3 nodes,
  * a lambda-space calibration bracket, which at the committed 100-node substrate
    never reaches saturation at all.
"""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pytest

from orion.config import MDO_HEADROOM_CPU_REF, MDO_HEADROOM_RAM_REF

from orion.sim.load_levels import (
    CALIBRATED_LEVELS,
    NUM_ARRIVALS,
    RHO_SWEEP,
    SEEDS,
    SERVICE_RATE,
    WARMUP_ARRIVALS,
    acceptance_ratio,
    acceptance_ratio_steady,
    arrival_rate_for_rho,
    capacity_by_tier,
    expected_slice_demand,
    get_level,
    make_arrival_process,
    offered_load_fraction,
    steady_state,
    substrate_capacity,
)
from orion.mdo.observation import build_inter_domain_links
from orion.sim.slice_generator import generate_slice_request
from orion.substrate.hierarchical_topology import (
    DOMAIN_FEAT_DIM,
    DOMAIN_SIZES,
    DOMAIN_TIERS,
    HELDOUT_INSTANCES,
    INTER_DOMAIN_ADJACENCIES,
    NUM_DOMAIN_PAIRS,
    NUM_DOMAINS,
    OBS_DIM,
    TOTAL_NODES,
    TRAIN_INSTANCES,
    generate_hierarchical_topology,
)
from orion.types import TIER_ORDER, InfrastructureTier


# --------------------------------------------------------------------------
# §Y.1 structure
# --------------------------------------------------------------------------

def test_substrate_is_connected():
    assert nx.is_connected(generate_hierarchical_topology(0).graph.to_undirected())


def test_tier_is_position_not_draw():
    """Tier assignment must be identical across instances; only capacities vary."""
    a = generate_hierarchical_topology(instance_seed=0)
    b = generate_hierarchical_topology(instance_seed=7)
    tiers_a = {n: a.graph.nodes[n]["tier"] for n in a.graph.nodes}
    tiers_b = {n: b.graph.nodes[n]["tier"] for n in b.graph.nodes}
    assert tiers_a == tiers_b
    caps_a = {n: a.graph.nodes[n]["cpu_capacity"] for n in a.graph.nodes}
    caps_b = {n: b.graph.nodes[n]["cpu_capacity"] for n in b.graph.nodes}
    assert caps_a != caps_b, "instances must differ in capacity, or they are one draw"


def test_generation_is_deterministic_in_instance_seed():
    assert (generate_hierarchical_topology(3).to_dict()
            == generate_hierarchical_topology(3).to_dict())


def test_every_tier_is_populated():
    """A tier that exists in the enum but on no node is a dead branch in every mask,
    every template restriction and every per-tier observation slot."""
    sub = generate_hierarchical_topology(0)
    counts = {}
    for n in sub.graph.nodes:
        counts[sub.graph.nodes[n]["tier"]] = counts.get(sub.graph.nodes[n]["tier"], 0) + 1
    assert set(counts) == {t.value for t in TIER_ORDER}
    assert counts == {"edge": 54, "regional_cloud": 14, "central_cloud": 12}
    assert sub.graph.number_of_nodes() == TOTAL_NODES == 80


def test_domains_hold_different_tier_sets():
    """Y.1e. The point of the composition change: a domain is an operator with a
    particular footprint, not a copy of its neighbours. If every domain held every
    tier, colocation would always be available and the partition decision would
    collapse to "find a domain with room"."""
    sub = generate_hierarchical_topology(0)
    held = {d: {sub.graph.nodes[n]["tier"] for n in sub.nodes_in_domain(d)}
            for d in range(NUM_DOMAINS)}
    assert len({frozenset(v) for v in held.values()}) > 1, "all domains are identical"
    for d, tiers in DOMAIN_TIERS.items():
        assert held[d] == {t.value for t in tiers}
        assert len(sub.nodes_in_domain(d)) == DOMAIN_SIZES[d]


def test_no_domain_can_host_every_chain_alone():
    """The registered Y.1e prediction, asserted so a template edit cannot silently
    break it: the central-only domain can host NO complete chain, because no template
    permits central cloud exclusively and every chain contains at least one edge-only
    or regional-only VNF. That is what makes cross-domain placement forced here
    rather than optional."""
    sub = generate_hierarchical_topology(0)
    rng = np.random.default_rng(0)
    central_only = [d for d, t in DOMAIN_TIERS.items()
                    if set(t) == {InfrastructureTier.CENTRAL_CLOUD}]
    assert central_only, "no central-only domain in the committed composition"

    colocatable = {d: 0 for d in range(NUM_DOMAINS)}
    for i in range(300):
        sr = generate_slice_request(f"r{i}", sub, rng, arrival_time=0.0, lifetime=20.0)
        assert all(v.permitted_nodes for v in sr.vnfs), "a VNF is placeable nowhere"
        for d in range(NUM_DOMAINS):
            nodes = set(sub.nodes_in_domain(d))
            if all(set(v.permitted_nodes) & nodes for v in sr.vnfs):
                colocatable[d] += 1
    for d in central_only:
        assert colocatable[d] == 0, (
            f"D{d} is central-only but colocated {colocatable[d]} chains")
    assert any(colocatable[d] > 0 for d in range(NUM_DOMAINS) if d not in central_only)


def test_levels_follow_tier_count_and_the_top_tier_is_plural_where_it_must_be():
    """One level per tier: a hierarchy has exactly one transport hop per tier
    boundary, so depth is not free. Extra levels charge propagation delay for hops
    the architecture does not have (measured: 6.05 ms of intra-domain delay on paths
    that never left their domain, against a 5-20 ms budget).

    Also pins that a 2-level domain is three roots of four rather than one root of
    fourteen: a single wide root would be a star with one point of failure and would
    put the domain's whole top tier on one node.
    """
    from orion.substrate.hierarchical_topology import _levels_and_parents

    for d, tiers in DOMAIN_TIERS.items():
        level, _parent, num_roots = _levels_and_parents(DOMAIN_SIZES[d], len(tiers))
        assert max(level) == len(tiers) - 1
        assert len(level) == DOMAIN_SIZES[d]
        if len(tiers) == 1:
            assert num_roots == DOMAIN_SIZES[d], "a one-tier domain must be flat"
        elif len(tiers) == 2:
            assert num_roots > 1, "a 2-level domain must not hang off a single root"


def test_every_domain_is_internally_connected():
    """Connectivity is STRUCTURAL, not sampled. Three roots of four children are
    three separate components until the roots are joined, and a flat domain has no
    tree at all: at 10 nodes the 30% lateral budget is 3 links, which cannot connect
    it. Relying on the sampled budget would leave some instances disconnected and
    others not, which is the worst failure mode available."""
    for inst in (0, 1, 2, HELDOUT_INSTANCES[0]):
        sub = generate_hierarchical_topology(inst)
        u = sub.graph.to_undirected()
        for d in range(NUM_DOMAINS):
            nodes = [n for n in sub.graph.nodes if sub.graph.nodes[n]["domain_id"] == d]
            assert nx.is_connected(u.subgraph(nodes)), (
                f"instance {inst}: domain {d} is internally disconnected")


def test_transport_delays_match_published_figures():
    """The per-segment delay ranges follow a published parameterisation.

    Source (verified 2026-08-03): Krolikowski et al., "Joint routing and
    scheduling for large-scale deterministic IP networks", Computer Communications
    165:33-42, 2021, section 7.1. Access is U(0.2, 0.8) ms for 10-40 km between
    elements and aggregation is U(0.8, 1.6) ms. Those are that paper's SIMULATION
    SETUP parameters over a synthetic IPRAN topology, not measurements and not
    standardised values, so the claim this pins is "consistent with a published
    parameterisation", never "characteristic of real metro networks".

    The end-to-end bound below is anchored on the same paper's per-class delay
    constraints, which give 4-6 ms for demands staying inside one domain. An
    earlier version of this test justified the bound by a "2-4 ms end-to-end 5G
    transport budget" attributed to vendor material; that figure could not be
    sourced and has been withdrawn rather than left as an unattributed anchor.

    Regression it still guards: inter-domain was once 5-15 ms, so a single link
    exceeded the whole intra-domain budget and any chain leaving its domain was
    structurally infeasible for XR/V2X regardless of load.
    """
    from orion.substrate.hierarchical_topology import (
        _DELAY_ACCESS, _DELAY_AGGREGATION, _DELAY_DATACENTRE, _DELAY_INTER,
    )

    assert _DELAY_ACCESS == (0.2, 0.8)          # Krolikowski et al. 7.1, access
    assert _DELAY_AGGREGATION == (0.8, 1.6)     # Krolikowski et al. 7.1, aggregation
    assert _DELAY_DATACENTRE[1] <= 0.5          # intra-datacentre
    assert _DELAY_INTER[1] <= 2.0               # inter-domain fibre, ~100-400 km
    # Assert on the TYPICAL path (segment midpoints); the all-maxima path is a
    # tail, not the quantity a per-class budget describes, so it is bounded
    # separately and more loosely.
    mid = lambda r: (r[0] + r[1]) / 2
    typical = (mid(_DELAY_ACCESS) + mid(_DELAY_AGGREGATION)
               + mid(_DELAY_INTER) + mid(_DELAY_DATACENTRE))
    assert typical <= 4.0, f"typical edge-to-central transport path is {typical:.2f} ms"
    worst = (_DELAY_ACCESS[1] + _DELAY_AGGREGATION[1]
             + _DELAY_INTER[1] + _DELAY_DATACENTRE[1])
    assert worst <= 6.0, f"worst-case edge-to-central transport path is {worst:.2f} ms"


def _domain_graph(sub) -> nx.Graph:
    dg = nx.Graph()
    dg.add_nodes_from(range(NUM_DOMAINS))
    for link_id in sub.inter_domain_links():
        attrs = sub.get_link_attrs(link_id)
        dg.add_edge(sub.graph.nodes[attrs.source]["domain_id"],
                    sub.graph.nodes[attrs.target]["domain_id"])
    return dg


def test_inter_domain_graph_survives_any_single_adjacency_failure():
    """Y.1e: partial mesh, no star, no core domain.

    The supervisor requires every connectivity choice to be justifiable, so the
    JUSTIFICATION is what gets asserted rather than a link count, which could drift
    away from the argument it exists to support. Two properties, both of which the
    superseded pure star failed: no single adjacency failure disconnects the domain
    graph, and no single adjacency failure cuts any domain off from central cloud.
    The second matters because D1 and D4 hold no central cloud at all, so a chain of
    theirs with a cloud-anchored VNF MUST leave the domain.
    """
    sub = generate_hierarchical_topology(0)
    dg = _domain_graph(sub)
    assert ({frozenset(e) for e in dg.edges()}
            == {frozenset(p) for p in INTER_DOMAIN_ADJACENCIES})

    central = {d for d, t in DOMAIN_TIERS.items()
               if InfrastructureTier.CENTRAL_CLOUD in t}
    for edge in list(dg.edges()):
        cut = nx.restricted_view(dg, [], [edge])
        assert nx.is_connected(cut), f"losing adjacency {edge} disconnects the domain graph"
        for d in range(NUM_DOMAINS):
            assert any(nx.has_path(cut, d, c) for c in central), (
                f"losing adjacency {edge} cuts domain {d} off from central cloud")


def test_parallel_inter_domain_links_do_not_share_one_gateway_node():
    """An adjacency must survive a gateway NODE failure, not only a link failure,
    wherever the domain has more than one top-tier node to attach to."""
    sub = generate_hierarchical_topology(0)
    g = sub.graph
    by_pair = {}
    for link_id in sub.inter_domain_links():
        a = sub.get_link_attrs(link_id)
        key = frozenset((g.nodes[a.source]["domain_id"], g.nodes[a.target]["domain_id"]))
        by_pair.setdefault(key, set()).add(frozenset((a.source, a.target)))
    for pair, links in by_pair.items():
        doms = sorted(pair)
        if all(len([n for n in sub.nodes_in_domain(d)
                    if g.nodes[n]["tier"] == DOMAIN_TIERS[d][0].value]) > 1
               for d in doms):
            endpoints = {n for lk in links for n in lk}
            assert len(endpoints) > 2, (
                f"adjacency {doms} lands every link on the same node pair")


def test_lateral_intra_domain_links_are_same_tier():
    """The supervisor accepted the tree-like topology specifically BECAUSE the extra
    links are lateral links between nodes of the same tier. If they ever spanned
    tiers, the tier-by-position property the whole substrate rests on would be
    perturbed and the ratification would no longer describe the code.

    A cross-tier intra link is therefore only legal as a backbone parent-child edge,
    i.e. between ADJACENT tiers of that domain, never skipping one.
    """
    for inst in (0, 1, 2, HELDOUT_INSTANCES[0]):
        sub = generate_hierarchical_topology(inst)
        g = sub.graph
        same_tier = [(u, v) for u, v, a in g.edges(data=True)
                     if a["link_type"] == "intra"
                     and g.nodes[u]["tier"] == g.nodes[v]["tier"]]
        assert same_tier, f"instance {inst} has no same-tier intra links at all"
        for u, v, a in g.edges(data=True):
            if a["link_type"] != "intra":
                continue
            tu, tv = g.nodes[u]["tier"], g.nodes[v]["tier"]
            if tu == tv:
                continue
            tiers = [t.value for t in DOMAIN_TIERS[g.nodes[u]["domain_id"]]]
            assert abs(tiers.index(tu) - tiers.index(tv)) == 1, (
                f"intra link {u}-{v} skips a tier: {tu}/{tv}")
        for d in range(NUM_DOMAINS):
            if len({g.nodes[n]["tier"] for n in sub.nodes_in_domain(d)}) > 1:
                assert any(g.nodes[u]["domain_id"] == d for u, v in same_tier), (
                    f"instance {inst}: domain {d} has no same-tier links")


def test_obs_dim_matches_the_realised_substrate():
    """Checked against the real builder rather than a restated formula.

    The observation AGGREGATES inter-domain links by ordered domain pair, so each
    ADJACENCY contributes 2 entries and INTER_LINKS_PER_PAIR does not widen it. The
    Y.1e change moves L 12 -> 16 and the per-domain block 6 -> 12 (per-tier
    residuals), so obs_dim goes 121 -> 163 and every earlier checkpoint is invalid.

    2026-08-11: h^m (largest single-node headroom) removed from the domain block,
    12 -> 11, obs_dim 163 -> 158.

    2026-08-12: h^m restored and generalised, per tier and as the best-fitting
    node's own residual CPU and RAM rather than one blended ratio, so the domain
    block carries 4 numbers per tier instead of 2 and goes 11 -> 17, obs_dim
    158 -> 188. Every checkpoint written before this date is invalid. Two reasons
    the removal was wrong. A domain publishing "I can seat one VNF of up to this
    size" is advertising a capability, not its internal layout, which is what the
    boundary argument actually forbids. And it made the comparison unequal, since
    the full-observability baselines read node residuals directly and could always
    answer the question h^m answers, while the partial-observability arms could not
    even ask it. It is now published on the same surface to all three consumers of
    the abstract view: this tensor, Agent B's topology dict, and the
    partial-observability heuristic.
    """
    sub = generate_hierarchical_topology(0)
    n_pairs = len(build_inter_domain_links(sub))
    assert n_pairs == NUM_DOMAIN_PAIRS == 2 * len(INTER_DOMAIN_ADJACENCIES) == 16
    assert DOMAIN_FEAT_DIM == 5 + 4 * len(TIER_ORDER) == 17
    assert OBS_DIM == (DOMAIN_FEAT_DIM * NUM_DOMAINS + 3 * n_pairs
                       + (5 + NUM_DOMAINS) * 10 + 5) == 238


def test_the_size_and_composition_are_constants_not_parameters():
    """Y.11 descoped 2026-07-29. The generator takes no size or composition argument,
    so there is no code path that produces a substrate Y cannot compare against its
    own results."""
    import inspect

    assert (NUM_DOMAINS, TOTAL_NODES, OBS_DIM) == (5, 80, 238)
    params = inspect.signature(generate_hierarchical_topology).parameters
    assert not {"size", "num_domains", "composition", "tiers"} & set(params)
    sub = generate_hierarchical_topology(0)
    assert sub.num_domains == 5
    assert sub.graph.number_of_nodes() == 80


def test_a_policy_cannot_be_fed_another_sizes_observation():
    """Pins WHY the size is fixed: cross-size transfer is a shape error, not a
    degradation. Nothing in Y crosses this boundary, and no Y result may be presented
    as evidence about another topology size."""
    import torch
    from orion.mdo.policy import AutoregMDOPolicy

    policy = AutoregMDOPolicy(obs_dim=OBS_DIM - 12, num_domains=3)
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        policy.forward(torch.zeros(OBS_DIM),
                       torch.ones(4, NUM_DOMAINS, dtype=torch.bool), 4)


def test_train_and_heldout_instances_are_disjoint():
    assert not set(TRAIN_INSTANCES) & set(HELDOUT_INSTANCES)


# --------------------------------------------------------------------------
# §Y.2 load model
# --------------------------------------------------------------------------

def test_lifetimes_are_live_within_an_episode():
    """The pre-§Y setup had mean lifetime 50 against a ~25 t.u. episode, so nothing
    departed. Departures must actually occur inside the measured window."""
    sub = generate_hierarchical_topology(0)
    ap = make_arrival_process(sub, arrival_rate=1.0,
                              rng=np.random.default_rng(42),
                              slice_factory=generate_slice_request)
    span = ap.events[-1].time
    mean_lifetime = 1.0 / SERVICE_RATE
    assert mean_lifetime < span / 4, "lifetimes are long relative to the episode"
    last_arrival = max(e.time for e in ap.events if e.slice_request is not None)
    departures_before_end = sum(
        1 for e in ap.events if e.slice_request is None and e.time < last_arrival)
    assert departures_before_end > NUM_ARRIVALS // 2


def test_offered_load_uses_the_binding_resource():
    """rho must be max(cpu, ram), not the mean: a substrate saturates on whichever
    resource runs out first, and averaging hides a bound resource behind a free one."""
    sub = generate_hierarchical_topology(0)
    cpu_cap, ram_cap = substrate_capacity(sub)
    # CPU-bound workload: huge cpu demand, negligible ram.
    rho = offered_load_fraction(sub, arrival_rate=1.0,
                                expected_cpu=cpu_cap * SERVICE_RATE,
                                expected_ram=1e-9)
    assert rho == pytest.approx(1.0, rel=1e-6)


def test_rho_is_the_calibration_coordinate_not_lambda():
    """Regression: the pre-§Y bracket was written in lambda for a 32-node substrate.
    At the committed 100-node substrate lambda=4.0 is only ~0.3 of capacity, so the
    L3/L4 acceptance targets were unreachable from it. The sweep is specified in rho
    and converted, and the conversion must round-trip."""
    sub = generate_hierarchical_topology(0)
    ecpu, eram = expected_slice_demand(sub, generate_slice_request,
                                       np.random.default_rng(42), num_samples=200)

    for rho in RHO_SWEEP:
        lam = arrival_rate_for_rho(sub, rho, ecpu, eram)
        assert offered_load_fraction(sub, lam, ecpu, eram) == pytest.approx(rho, rel=1e-6)

    assert offered_load_fraction(sub, 4.0, ecpu, eram) < 0.5
    assert arrival_rate_for_rho(sub, 1.0, ecpu, eram) > 4.0


def test_tier_capacity_is_uneven_so_aggregate_rho_cannot_measure_difficulty():
    """Per-tier capacity must be materially uneven, because VNF templates are
    tier-restricted and a whole-substrate rho therefore cannot represent difficulty.

    Measured on the committed substrate: regional_cloud 563 < central_cloud 667 <
    edge 1166 CPU. The ratio is under 2x, so the assertion is on the SHARE a single
    tier holds rather than on a max/min ratio: what matters is that no tier is close
    to interchangeable with the whole substrate, not that one is tiny.

    Note that the SCARCEST tier need not be the BINDING one. On the superseded
    four-tier substrate the calibration saturated `regional_cloud` (0.94 at L4)
    because eMBB is 30% of arrivals and carries the largest chains, even though that
    tier held the most capacity. Which tier binds is a property of the WORKLOAD's
    tier demands, not of capacity alone, which is exactly why the calibration records
    per-tier utilisation next to the aggregate instead of inferring it. Merging MEC
    into edge concentrates demand further, so this has to be re-measured rather than
    assumed to carry over.
    """
    tiers = capacity_by_tier(generate_hierarchical_topology(0))
    cpu = {k: v["cpu"] for k, v in tiers.items()}
    assert set(cpu) == {t.value for t in TIER_ORDER}
    total = sum(cpu.values())
    assert max(cpu.values()) / total < 0.75, (
        f"one tier holds most of the substrate, so aggregate rho ~ that tier: {cpu}")
    assert min(cpu.values()) / total > 0.05, (
        f"a tier is vestigial and cannot bind on its own: {cpu}")
    # Edge holds the most CPU: it is 54 of 80 nodes under a realistic RAN fan-out,
    # even though it has the smallest per-node capacity range.
    assert cpu[InfrastructureTier.EDGE.value] == max(cpu.values())


def test_calibrated_levels_are_frozen_and_ordered():
    """The ladder must be either FROZEN and ordered, or EMPTY and refusing.

    It is empty as of 2026-07-30: the §Y.1d substrate change and the ratified
    acceptance definition both move the number the levels are pinned on, so the
    2026-07-29 freeze was invalidated rather than carried forward. The guard that
    matters in that state is that `get_level` refuses instead of handing out a
    stale lambda, which is what would silently corrupt every downstream cell.
    """
    if not CALIBRATED_LEVELS:
        with pytest.raises(RuntimeError, match="calibration has not been run"):
            get_level("L2")
        return

    assert set(CALIBRATED_LEVELS) == {"L1", "L2", "L3", "L4"}
    lams = [CALIBRATED_LEVELS[n].arrival_rate for n in ("L1", "L2", "L3", "L4")]
    accs = [CALIBRATED_LEVELS[n].plain_acceptance for n in ("L1", "L2", "L3", "L4")]
    assert lams == sorted(lams), "lambda must rise L1 -> L4"
    assert accs == sorted(accs, reverse=True), "acceptance must fall L1 -> L4"
    # A is the offered concurrency; it must be well under N or the episode cannot
    # reach the steady state the level claims to describe.
    for n in CALIBRATED_LEVELS:
        assert CALIBRATED_LEVELS[n].erlangs / NUM_ARRIVALS <= 0.25


def test_steady_state_window_drops_the_transient():
    records = [True] * NUM_ARRIVALS
    assert len(steady_state(records)) == NUM_ARRIVALS - WARMUP_ARRIVALS


def test_steady_state_refuses_an_episode_it_would_empty():
    with pytest.raises(ValueError, match="warm-up window"):
        steady_state([True] * WARMUP_ARRIVALS)


def test_acceptance_ratio_counts_every_generated_request():
    """Ratified definition (2026-07-30): accepted / total GENERATED, no window.

    Regression: `acceptance_ratio` windowed by WARMUP_ARRIVALS, so it disagreed
    with the number `grid_runner` and `wp7_runner` actually report by up to 3.9
    points at L4, and the load levels were frozen against the windowed value.
    """
    records = [False] * WARMUP_ARRIVALS + [True] * (NUM_ARRIVALS - WARMUP_ARRIVALS)
    expected = (NUM_ARRIVALS - WARMUP_ARRIVALS) / NUM_ARRIVALS
    assert acceptance_ratio(records) == pytest.approx(expected)
    # The windowed value is still available, but only under a name that says so.
    assert acceptance_ratio_steady(records) == 1.0


def test_reported_acceptance_matches_the_runners_denominator():
    """Pins the definition against the runners rather than against a restatement:
    `eval_plain` computes admitted / total arrivals seen, with no window."""
    import ast
    import inspect
    import textwrap

    from orion.sim import load_levels

    # Parse and drop the docstring and comments. A line filter is not enough: the
    # docstring here NAMES the windowing it warns against, so a substring search
    # over the raw source matches the warning and not the code.
    fn = ast.parse(textwrap.dedent(
        inspect.getsource(load_levels.acceptance_ratio))).body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)):
        fn.body = fn.body[1:]
    body = ast.unparse(fn)
    assert "WARMUP" not in body and "steady_state" not in body, (
        "the primary acceptance metric must not window; that is the bug this "
        "amendment fixed")


def test_five_seeds_minimum():
    assert len(SEEDS) >= 5


# --------------------------------------------------------------------------
# §Y.1e per-tier observation
# --------------------------------------------------------------------------

def test_observation_width_is_measured_not_declared():
    """OBS_DIM must equal what the builder actually emits, checked by BUILDING one.

    The two used to be independent numbers: `hierarchical_topology` declared a width
    and `observation.py` emitted one, with nothing forcing them to agree. A mismatch
    surfaces as a matmul shape error deep in training, long after the run started, or
    worse it does not surface at all if the policy was constructed from the same
    wrong constant.
    """
    import numpy as np

    from orion.mdo.observation import build_mdo_observation, observation_to_tensor
    from orion.mdo.types import PlanSummary

    sub = generate_hierarchical_topology(0)
    rng = np.random.default_rng(0)
    sr = generate_slice_request("r0", sub, rng, arrival_time=0.0, lifetime=20.0)
    first_tier = lambda v: InfrastructureTier(
        sub.graph.nodes[sorted(v.permitted_nodes)[0]]["tier"])
    plan = PlanSummary(
        vnf_ids=[v.vnf_id for v in sr.vnfs],
        required_tiers=[first_tier(v) for v in sr.vnfs],
        suggested_domains=[0] * len(sr.vnfs),
        cpu_demands=[v.cpu_demand for v in sr.vnfs],
        ram_demands=[v.ram_demand for v in sr.vnfs],
        vcrs=[v.vcr for v in sr.vnfs],
        bw_demands=[f.bandwidth_demand for f in sr.flow_edges])

    tensor = observation_to_tensor(build_mdo_observation(sub, plan), max_vnfs=10)
    assert tensor.shape[0] == OBS_DIM == 238


def test_per_tier_residuals_distinguish_domains_that_lack_a_tier():
    """The whole point of §Y.1e's observation change, asserted on the substrate.

    A domain that holds no central cloud and a domain that holds nothing BUT central
    cloud must be distinguishable from the observation alone, otherwise the
    orchestrator keeps choosing domains that cannot take the chain. Measured cost of
    not having this: `actor_infeasible` rejections ran 59/250/494/711 out of 2000 at
    L1-L4 for the partial-obs heuristic, against zero when node residuals were read.
    """
    from orion.mdo.observation import build_domain_summaries

    sub = generate_hierarchical_topology(0)
    by_domain = {s.domain_id: s for s in build_domain_summaries(sub)}
    assert set(by_domain) == set(DOMAIN_TIERS)

    for d, tiers in DOMAIN_TIERS.items():
        s = by_domain[d]
        assert set(s.tier_cpu_residual) == set(TIER_ORDER), "a tier slot is missing"
        for t in TIER_ORDER:
            if t in tiers:
                assert s.tier_cpu_residual[t] > 0, f"D{d} holds {t} but reports 0 CPU"
            else:
                assert s.tier_cpu_residual[t] == 0, f"D{d} lacks {t} but reports CPU"
        # The per-tier block must decompose the aggregate, not restate or double it.
        assert sum(s.tier_cpu_residual.values()) == pytest.approx(s.cpu_residual)
        assert sum(s.tier_ram_residual.values()) == pytest.approx(s.ram_residual)


def test_llm_sees_the_same_per_tier_state_as_the_mdo():
    """The planner and the selector must not disagree about what a domain holds.

    They are separate code paths (`abstract_topology` builds the LLM's view,
    `build_domain_summaries` the MDO's), so nothing structural stops one being
    updated without the other. That divergence would be invisible: both produce
    well-formed output and the plan would simply be built against a different
    network than the one the selector scores.
    """
    from orion.llm.abstract_topology import build_abstract_topology
    from orion.mdo.observation import build_domain_summaries

    sub = generate_hierarchical_topology(0)
    mdo = {s.domain_id: s for s in build_domain_summaries(sub)}
    for entry in build_abstract_topology(sub)["domains"]:
        d = int(entry["domain_id"].lstrip("d"))
        for t in TIER_ORDER:
            # The LLM view rounds to 1 dp for prompt readability, so the tolerance
            # is half an ulp of THAT, not of the underlying float.
            assert entry["cpu_residual_by_tier"][t.value] == pytest.approx(
                mdo[d].tier_cpu_residual[t], abs=0.1), (
                f"LLM and MDO disagree on D{d} {t.value} residual")

# ---------------------------------------------------------------------------
# §Y.10 workload guards (2026-08-01)
# ---------------------------------------------------------------------------


def _draw(scenario, n=3000, instance=100, seed=0):
    from orion.sim.scenario_slices import make_scenario_slice_factory
    from orion.substrate.hierarchical_topology import generate_hierarchical_topology
    sub = generate_hierarchical_topology(instance)
    factory = make_scenario_slice_factory(scenario)
    rng = np.random.default_rng(seed)
    return [factory(f"r{i}", sub, rng) for i in range(n)]


def test_no_chain_is_truncated():
    """Every request carries its slice type's COMPLETE chain.

    The generator used to keep templates[:n_vnfs], which removed the tail of the
    chain: an eMBB request without its vEPC, an mMTC request without Analytics.
    That is not a shorter valid slice of that type, it is one missing the function
    that defines the service, so the admission decision about it means nothing. It
    also made 61% of arrivals two-VNF chains, which fit any domain trivially and
    never pose the partitioning question. See §Y.10.
    """
    from orion.sim.slice_generator import _VNF_TEMPLATES
    for scenario in ("conventional", "complex", "stress"):
        for sr in _draw(scenario, n=1500):
            full = len(_VNF_TEMPLATES[sr.slice_type])
            assert len(sr.vnfs) == full, (
                f"{scenario}: {sr.slice_type.value} chain has {len(sr.vnfs)} VNFs, "
                f"template has {full}. Chain truncation has come back."
            )


def test_complexity_levels_are_distinct_and_ordered():
    """S2 is a harder workload than S1, in chain length AND per-slice demand."""
    k = {}
    cpu = {}
    for scenario in ("conventional", "complex"):
        reqs = _draw(scenario)
        k[scenario] = float(np.mean([len(r.vnfs) for r in reqs]))
        cpu[scenario] = float(np.mean([sum(v.cpu_demand for v in r.vnfs) for r in reqs]))

    # Declared in §Y.10; a drift here means the pre-registered table is stale.
    assert abs(k["conventional"] - 2.86) < 0.10, k
    assert abs(k["complex"] - 3.26) < 0.10, k
    assert abs(cpu["conventional"] - 11.35) < 0.60, cpu
    assert abs(cpu["complex"] - 15.59) < 0.80, cpu
    assert k["complex"] > k["conventional"]
    assert cpu["complex"] > cpu["conventional"]


def test_complexity_axis_varies_only_the_mix():
    """S1 and S2 differ in the slice-type mix and in NOTHING else.

    §Y.4 requires one axis at a time. `complex` holds the VCR at 1.0, exactly as
    `conventional` does, so bandwidth behaviour along the chain is unchanged and
    the mix is the single varying factor. If this fails, the complexity axis has
    silently become a bandwidth axis too and its cells are confounded.
    """
    from orion.sim.scenario_slices import make_scenario_slice_factory
    assert make_scenario_slice_factory("conventional").rho == 1.0
    assert make_scenario_slice_factory("complex").rho == 1.0

    per_type_cpu = {}
    for scenario in ("conventional", "complex"):
        by_type = {}
        for r in _draw(scenario, n=3000):
            by_type.setdefault(r.slice_type, []).append(
                sum(v.cpu_demand for v in r.vnfs))
        per_type_cpu[scenario] = {t: float(np.mean(v)) for t, v in by_type.items()}

    shared = set(per_type_cpu["conventional"]) & set(per_type_cpu["complex"])
    assert len(shared) == 5, "both levels must still draw all five slice types"
    for t in shared:
        a = per_type_cpu["conventional"][t]
        b = per_type_cpu["complex"][t]
        assert abs(a - b) / a < 0.05, (
            f"{t.value} demand differs between levels ({a:.2f} vs {b:.2f}); the "
            f"axis is supposed to reweight types, not change them"
        )


# --------------------------------------------------------------------------
# The abstract surface is ONE surface (2026-08-12)
# --------------------------------------------------------------------------
#
# ORION's plan layer, the RL policy and the partial-observability baseline all
# choose between domains. If they choose on different evidence, the comparison
# between them measures who was told more, not who planned better. Three defects
# of exactly that kind were found on 2026-08-12: Agent B had no capacities and so
# could not compute utilisation at all, only Agent B lacked h^m, and the LLM plan
# builder derived a VNF's required tier alphabetically while the heuristic used
# the modal tier, which handed the SAME policy two different action masks. These
# guards exist so none of the three can come back quietly.

def test_agent_b_sees_the_mdo_surface():
    """Every quantity in the MDO's DomainSummary has a counterpart in Agent B's
    abstract topology, with the same value."""
    from orion.llm.abstract_topology import build_abstract_topology
    from orion.mdo.observation import build_domain_summaries

    sub = generate_hierarchical_topology(0)
    summaries = {s.domain_id: s for s in build_domain_summaries(sub)}
    topo = build_abstract_topology(sub)
    assert len(topo["domains"]) == len(summaries) == NUM_DOMAINS

    for d in topo["domains"]:
        s = summaries[int(d["domain_id"][1:])]
        assert d["cpu_residual"] == pytest.approx(s.cpu_residual, abs=0.1)
        assert d["ram_residual"] == pytest.approx(s.ram_residual, abs=0.1)
        # Capacity: absent from the view until 2026-08-12, so Agent B could not
        # tell 500 free CPU out of 5000 from 500 out of 600, while the policy read
        # cpu_cap_norm and the plan cache keyed on residual FRACTIONS.
        assert d["cpu_capacity"] == pytest.approx(s.cpu_capacity, abs=0.1)
        assert d["ram_capacity"] == pytest.approx(s.ram_capacity, abs=0.1)
        for t in TIER_ORDER:
            assert d["cpu_residual_by_tier"][t.value] == pytest.approx(
                s.tier_cpu_residual[t], abs=0.1)
            assert d["ram_residual_by_tier"][t.value] == pytest.approx(
                s.tier_ram_residual[t], abs=0.1)
            # h^m, per tier.
            assert d["largest_free_node_by_tier"][t.value]["cpu"] == pytest.approx(
                s.tier_max_node_cpu[t], abs=0.1)
            assert d["largest_free_node_by_tier"][t.value]["ram"] == pytest.approx(
                s.tier_max_node_ram[t], abs=0.1)

    links = {(l.source_domain, l.target_domain): l
             for l in build_inter_domain_links(sub)}
    assert len(topo["inter_domain_links"]) == len(links)
    for l in topo["inter_domain_links"]:
        ref = links[(int(l["source_domain"][1:]), int(l["target_domain"][1:]))]
        assert l["bandwidth_residual_mbps"] == pytest.approx(ref.bw_residual, abs=0.1)
        assert l["bandwidth_capacity_mbps"] == pytest.approx(ref.bw_capacity, abs=0.1)


def test_largest_free_node_describes_one_real_node():
    """The reported (cpu, ram) pair must be ONE node's residuals. Taking the max of
    each independently would advertise a node that does not exist, and the domain
    actor would then fail to place against a summary that promised it could."""
    from orion.mdo.observation import build_domain_summaries

    sub = generate_hierarchical_topology(0)
    for s in build_domain_summaries(sub):
        nodes = sub.nodes_in_domain(s.domain_id)
        for t in TIER_ORDER:
            in_tier = [n for n in nodes if sub.graph.nodes[n]["tier"] == t.value]
            pair = (s.tier_max_node_cpu[t], s.tier_max_node_ram[t])
            if not in_tier:
                assert pair == (0.0, 0.0), "an absent tier must read zero"
                continue
            real = {(sub.graph.nodes[n]["cpu_residual"],
                     sub.graph.nodes[n]["ram_residual"]) for n in in_tier}
            assert pair in real
            # And it must be the BEST-fitting one under the frozen references.
            best = max(min(c / MDO_HEADROOM_CPU_REF, r / MDO_HEADROOM_RAM_REF)
                       for c, r in real)
            assert min(pair[0] / MDO_HEADROOM_CPU_REF,
                       pair[1] / MDO_HEADROOM_RAM_REF) == pytest.approx(best)


def test_headroom_refs_still_bound_templates():
    """The h^m references are frozen literals, by design, so they cannot drift with
    the generator. That only stays safe if a template outgrowing them FAILS here:
    a VNF larger than the reference makes the fit score exceed 1.0 and the
    'best-fitting node' stops meaning what its docstring says."""
    from orion.sim.slice_generator import _VNF_TEMPLATES

    max_cpu = max(t["cpu"][1] for ts in _VNF_TEMPLATES.values() for t in ts)
    max_ram = max(t["ram"][1] for ts in _VNF_TEMPLATES.values() for t in ts)
    assert max_cpu <= MDO_HEADROOM_CPU_REF
    assert max_ram <= MDO_HEADROOM_RAM_REF


def test_both_plan_builders_derive_the_same_required_tier():
    """`required_tier` is the MDO's hard action mask (USE_NODE_BASED_TIER_MASK is
    False), so if the LLM builder and the heuristic builder compute it differently
    the same policy gets two different action spaces and the LLM rows stop being
    comparable with the baselines.

    They did. The LLM builder used sorted(permitted_tiers)[0], alphabetically first,
    which puts central_cloud ahead of edge and regional_cloud, so any VNF permitting
    central_cloud at all was pinned to it. Measured over 400 L3 requests, domains
    able to host the whole chain: 2.00 under that rule against 4.00 under the modal
    rule for eMBB, mMTC and XR, which are 60% of arrivals.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "scripts"))
    import wp7_runner  # noqa: F401  (import path check only)
    from partial_obs_prior import _required_tiers

    # Comments stripped: the fix's own comment quotes the rule it replaced.
    code = "\n".join(ln for ln in inspect.getsource(
        wp7_runner.make_llm_plan_builder).splitlines()
        if not ln.strip().startswith("#"))
    assert "sorted(" not in code, (
        "the alphabetical-tier rule is back in the LLM plan builder")

    sub = generate_hierarchical_topology(0)
    rng = np.random.default_rng(0)
    for i in range(50):
        sr = generate_slice_request(f"r{i}", sub, rng, arrival_time=0.0, lifetime=20.0)
        heuristic = _required_tiers(sr, sub)
        llm_rule = []
        for v in sr.vnfs:
            counts: dict = {}
            for n in v.permitted_nodes:
                if n in sub.graph.nodes:
                    tt = sub.graph.nodes[n]["tier"]
                    counts[tt] = counts.get(tt, 0) + 1
            llm_rule.append(InfrastructureTier(max(counts, key=counts.get))
                            if counts else InfrastructureTier.EDGE)
        assert llm_rule == heuristic
