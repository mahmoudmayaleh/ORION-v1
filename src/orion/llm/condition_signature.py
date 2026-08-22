"""Network-condition signature — the §Y retrieval state key for M^B.

Why this exists (PREREG_AMENDMENT_2026-07-27_Y §Y.6): M^B retrieval scores
episodes with REMEMBERER's two-term form, S = lambda*f_task + (1-lambda)*f_state,
where f_state was substrate similarity over the TOPOLOGY signature. Under §Y the
topology is fixed, so every stored episode has a byte-identical topology
signature and f_state == 1.0 for every candidate. Two consequences, both fatal:

  1. The state term stops discriminating — ranking silently collapses to slice
     text alone, the exact negative-transfer case the two-term form exists to
     prevent.
  2. Abstention becomes impossible. combined = 0.5*f_task + 0.5*1.0 >= 0.5 =
     RETRIEVAL_FLOOR for EVERY entry, so the floor never filters and
     abstain_rate is pinned at 0 by construction.

The fix is not to drop the state term but to point it at the state that
actually varies once the substrate is fixed: network CONDITION. What decides
whether a past plan shape survives now is how loaded the network is and where
the scarcity sits, not what the graph looks like.

Observability (RULE, verified in code): this signature is built from per-domain
AGGREGATES and the operating point only — the same abstraction level as
DomainSummary and the abstract topology Agent B already receives. It never
reads node-level residuals, so it does not smuggle full observability into the
plan layer.
"""

from __future__ import annotations

import os

from typing import Any

from orion.substrate.graph_model import SubstrateNetwork
from orion.types import InfrastructureTier

# Occupancy buckets on the RESIDUAL fraction (1.0 = empty, 0.0 = full). Coarse
# on purpose: the bucket is a filter/tag, the numeric fields do the ranking.
_BUCKETS = (
    (0.75, "free"),
    (0.50, "moderate"),
    (0.25, "tight"),
    (0.00, "saturated"),
)


def occupancy_bucket(residual_fraction: float) -> str:
    """Coarse congestion label from an overall residual fraction."""
    for lo, name in _BUCKETS:
        if residual_fraction >= lo:
            return name
    return "saturated"


def _safe_frac(residual: float, capacity: float) -> float:
    return float(residual / capacity) if capacity > 0 else 0.0


def compute_condition_signature(
    substrate: SubstrateNetwork,
    load_level: str | None = None,
) -> dict[str, Any]:
    """Per-domain / per-tier congestion snapshot at decision time.

    Args:
        substrate: The substrate, read at its CURRENT residual state.
        load_level: Operating-point label ("L1".."L4", §Y.2). None when the
            caller has no load level to declare (legacy paths, probes).

    Returns:
        JSON-serializable dict. All fractions are residual/capacity in [0,1],
        so 1.0 means empty and 0.0 means exhausted.
    """
    g = substrate.graph

    dom_cpu: list[float] = []
    dom_ram: list[float] = []
    tier_res: dict[str, list[float]] = {t.value: [] for t in InfrastructureTier}

    cpu_res_tot = cpu_cap_tot = ram_res_tot = ram_cap_tot = 0.0

    for domain_id in range(substrate.num_domains):
        nodes = substrate.nodes_in_domain(domain_id)
        if not nodes:
            continue
        c_res = sum(g.nodes[n]["cpu_residual"] for n in nodes)
        c_cap = sum(g.nodes[n]["cpu_capacity"] for n in nodes)
        r_res = sum(g.nodes[n]["ram_residual"] for n in nodes)
        r_cap = sum(g.nodes[n]["ram_capacity"] for n in nodes)

        dom_cpu.append(round(_safe_frac(c_res, c_cap), 3))
        dom_ram.append(round(_safe_frac(r_res, r_cap), 3))

        cpu_res_tot += c_res
        cpu_cap_tot += c_cap
        ram_res_tot += r_res
        ram_cap_tot += r_cap

    # Per-tier scarcity: which LAYER of the hierarchy is exhausted. Under a
    # tier-restricted slice mix this is what decides admissibility, and it is
    # invisible in a whole-network average.
    tier_cap: dict[str, float] = {t.value: 0.0 for t in InfrastructureTier}
    tier_cpu: dict[str, float] = {t.value: 0.0 for t in InfrastructureTier}
    for _, d in g.nodes(data=True):
        t = d["tier"]
        if t in tier_cap:
            tier_cap[t] += float(d["cpu_capacity"])
            tier_cpu[t] += float(d["cpu_residual"])
    for t in tier_res:
        tier_res[t] = round(_safe_frac(tier_cpu[t], tier_cap[t]), 3)

    # Inter-domain bandwidth headroom (aggregated per directed domain pair, the
    # same aggregation the MDO observation uses).
    pair_res: dict[tuple[int, int], list[float]] = {}
    for u, v, d in g.edges(data=True):
        su, sv = g.nodes[u]["domain_id"], g.nodes[v]["domain_id"]
        if su == sv:
            continue
        agg = pair_res.setdefault((su, sv), [0.0, 0.0])
        agg[0] += float(d["bw_residual"])
        agg[1] += float(d["bandwidth_capacity"])
    inter_fracs = [_safe_frac(r, c) for r, c in pair_res.values()]

    cpu_frac = _safe_frac(cpu_res_tot, cpu_cap_tot)
    ram_frac = _safe_frac(ram_res_tot, ram_cap_tot)
    overall = min(cpu_frac, ram_frac)  # the binding resource, not the average

    # An empty load_level is not benign. condition_similarity guards the
    # exact-match term with `if la and lb`, so "" makes the term VANISH rather
    # than score zero -- a store written without a level silently loses the one
    # term that separates operating regimes, and nothing downstream can tell.
    # eval_memory did exactly this until 2026-08-04. Probes may legitimately pass
    # None; anything writing a store must not.
    if load_level is not None and not str(load_level).strip():
        raise ValueError(
            "compute_condition_signature: load_level is empty. Pass a real level "
            "(L1..L4) or None; an empty string disables the exact-match term.")

    return {
        "load_level": load_level or "",
        "cpu_residual_frac": round(cpu_frac, 3),
        "ram_residual_frac": round(ram_frac, 3),
        "domain_cpu_residual": dom_cpu,
        "domain_ram_residual": dom_ram,
        "tier_cpu_residual": tier_res,
        "inter_bw_residual_mean": round(
            sum(inter_fracs) / len(inter_fracs), 3) if inter_fracs else 0.0,
        "inter_bw_residual_min": round(min(inter_fracs), 3) if inter_fracs else 0.0,
        "active_slices": int(getattr(substrate, "num_active_slices", 0) or 0),
        "bucket": occupancy_bucket(overall),
    }


# Realised range of each field over the §Y substrate across L1..L4
# (probe_condrange, instance 100, 1600 conditions). A field's difference is
# divided by its own range, so each term contributes in proportion to how much
# that field actually moves. Without this the average is dominated by fields
# that barely vary: every term reads ~1.0, f_state lands at 0.99 for every
# candidate, and the state term gates without ranking. Measured 2026-08-05 by
# the paired copy test: M^B changed 33% of Agent B's plans and moved paired
# feasibility by +0.0 pp (3 better, 3 worse, McNemar p=1.0), because the
# exemplar was selected on slice text with the condition term near-constant.
#
# These are absolute constants, not pool min-max, so the abstain floor keeps a
# fixed meaning across queries.
_SCALE = {
    "cpu_residual_frac": 0.664,
    "ram_residual_frac": 0.318,
    "domain_cpu_residual": 0.907,
    "domain_ram_residual": 0.460,
}

_TIER_SCALE = {
    "edge": 0.841,
    "regional_cloud": 0.954,
    "central_cloud": 0.188,
}

# Range 0.003 and 0.020 across the whole load ladder: inter-domain bandwidth is
# never the binding resource on this substrate, so these two contributed a fixed
# 1.0 to every comparison and diluted the six terms that carry signal. Dropped
# rather than rescaled, since dividing noise by a 0.003 range amplifies noise.
_DEAD_FIELDS = ("inter_bw_residual_mean", "inter_bw_residual_min")


# §AB (2026-08-18): whether the load-level term is a CLIFF or a ramp.
#
# The term is `1.0 if same level else 0.0` -- categorical, on a variable that is
# ordered (L1 < L2 < L3 < L4 by offered load). `condition_similarity` averages six
# terms and retrieval scores `0.5*f_task + 0.5*f_state` against a 0.60 floor, so
# that one label is worth exactly 0.0833 of the combined score.
#
# Measured consequence, from data/parity_cells (store warmed at L2, frozen):
# abstain runs .954/.903/.981 at L1, .027/.000/.019 at L2, and .99+ at L3/L4. It
# fires at exactly one level. At L1 the near-miss quantiles are p50 .5805, p90
# .5939, max .5999 against a floor of .6000 -- add the 0.0833 back and ALL of them
# clear it, so L1's 95% abstain is entirely this label and not any real
# dissimilarity. At L3/L4 the label is worth about a decile on top of a genuine
# congestion mismatch.
#
# Graded uses ordinal distance over the ladder, so adjacent levels score 2/3
# rather than 0. This keeps the signal the categorical form was there for -- L2
# and L4 really are different regimes -- while removing the cliff. It is
# deliberately NOT a drop of the term: the other five terms measure congestion
# directly, so dropping it entirely is also defensible, but that is a larger
# change and this one is enough to make retrieval reachable.
#
# Default False so every banked cell reproduces. Retrieval scores move when True,
# so bank to a scratch directory.
GRADED_LOAD_LEVEL = os.environ.get("ORION_GRADED_LEVEL", "0") != "0"

_LEVEL_ORDER = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}


def _level_similarity(la: str, lb: str) -> float:
    """1.0 for the same level; a ramp or a cliff off it, per GRADED_LOAD_LEVEL."""
    if la == lb:
        return 1.0
    if not GRADED_LOAD_LEVEL:
        return 0.0
    ia, ib = _LEVEL_ORDER.get(la), _LEVEL_ORDER.get(lb)
    if ia is None or ib is None:
        return 0.0            # an unknown label is not a near neighbour of anything
    span = len(_LEVEL_ORDER) - 1
    return max(0.0, 1.0 - abs(ia - ib) / span)


def condition_similarity(a: dict, b: dict) -> float:
    """Congestion similarity in [0,1] over the numeric condition fields.

    Same construction as the topology term it replaces: numeric, absolute (not
    min-max normalized over the pool, so the abstain floor stays reachable), and
    averaged over whichever features both sides carry. Each difference is scaled
    by that field's realised range (`_SCALE`, `_TIER_SCALE`) so a field that
    moves by 0.02 over the whole load ladder cannot outvote one that moves by
    0.95; `_DEAD_FIELDS` names the two that never move at all.

    The load level is an EXACT-MATCH term, not a distance: L2 and L4 are
    different operating regimes, not near neighbours on a line.
    """
    if not a or not b:
        return 0.0

    sims: list[float] = []

    def _sim(diff: float, scale: float) -> float:
        return max(0.0, 1.0 - diff / scale) if scale > 0 else 1.0

    def _scalar(key: str) -> None:
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            sims.append(_sim(abs(float(va) - float(vb)), _SCALE[key]))

    for key in ("cpu_residual_frac", "ram_residual_frac"):
        _scalar(key)

    def _vector(key: str) -> None:
        va, vb = a.get(key), b.get(key)
        if isinstance(va, list) and isinstance(vb, list) and va and len(va) == len(vb):
            sims.append(_sim(sum(abs(float(x) - float(y))
                                 for x, y in zip(va, vb)) / len(va), _SCALE[key]))

    for key in ("domain_cpu_residual", "domain_ram_residual"):
        _vector(key)

    ta, tb = a.get("tier_cpu_residual"), b.get("tier_cpu_residual")
    if isinstance(ta, dict) and isinstance(tb, dict):
        shared = sorted(set(ta) & set(tb))
        if shared:
            # Per-tier scale: edge and regional_cloud drain, central_cloud
            # barely does, so one shared denominator would flatten the two tiers
            # that decide admissibility under a tier-restricted slice mix.
            sims.append(sum(_sim(abs(float(ta[k]) - float(tb[k])),
                                 _TIER_SCALE.get(k, 1.0))
                            for k in shared) / len(shared))

    la, lb = a.get("load_level"), b.get("load_level")
    if la and lb:
        sims.append(_level_similarity(la, lb))

    return sum(sims) / len(sims) if sims else 0.0


def condition_query_terms(condition: dict | None) -> list[str]:
    """Text terms so the lexical stage sees the operating point too.

    Kept deliberately small: the numeric term does the ranking, these only stop
    a saturated-network episode from winning on text alone.
    """
    if not condition:
        return []
    terms = [f"congestion {condition.get('bucket', 'unknown')}"]
    if condition.get("load_level"):
        terms.append(f"load {condition['load_level']}")
    tiers = condition.get("tier_cpu_residual")
    if isinstance(tiers, dict):
        scarce = [t for t, v in tiers.items() if isinstance(v, (int, float)) and v < 0.25]
        terms.extend(f"{t} exhausted" for t in sorted(scarce))
    return terms
