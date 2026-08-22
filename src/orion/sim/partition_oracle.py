"""§Y.3b — the partition oracle: a calibration instrument, not an approach.

WHY THIS EXISTS. The load ladder was pinned on the acceptance of ONE policy, the
`Plain` greedy. That works only while the policy's failures track load, and after
the §Y.1f amendment they stop doing so, in two independent ways:

  * `Plain`'s FFD fallback sorts by decreasing CPU, which discards chain position,
    so C10 refuses it. That costs a LOAD-INDEPENDENT 604 arrivals per 2000 at L1
    and 602 at L4. Its measured range collapses to .486-.384, every acceptance
    target sits above the ceiling, and the ladder goes degenerate.
  * The first-fit builders walk into the emptiest domain and strand the chain, and
    an empty domain is emptiest at LOW load, so their give-ups run 299 per 1000 at
    L1 against 87 at L3. Difficulty then ANTI-correlates with offered load and the
    ladder inverts.

Both are properties of a heuristic, not of the problem. A ladder pinned on either
measures the baseline's blind spot and calls it difficulty.

So the reference here answers the policy-free question instead: of the arrivals
offered, how many are admissible AT ALL? It enumerates contiguous, h^m-admissible,
aggregate-feasible partitions and takes the best one, then places through the same
coordinator and actors as every approach. The partition stops being a source of
difficulty, so what the curve measures is the substrate and the workload.

BOUNDED BY CONSTRUCTION, unlike the pre-§Y `compute_ceiling` that §Y.5 deleted.
That one searched NODE assignments, so it was |feasible nodes|^K, unbounded at §Y
scale, and it never allocated, so it was contention-blind. This searches DOMAIN
assignments: M**K, 625 on the committed composition, pruned further by tier
feasibility and contiguity. It allocates, so contention is real. Measured cost is
5-7 s per 1000 arrivals, against 4-5 s for the greedy builder it replaces.

NOT A CEILING, and it must not be called one. It is exhaustive within ONE arrival,
but admission is a SEQUENTIAL problem: minimising this arrival's delay can consume
capacity a later arrival needed, so a per-arrival optimum is not a stream optimum.
Measured: `MDO-fullobs` exceeds it by 1.4 pp at L2 while trailing it at L1, L3 and
L4. A real ceiling would need offline optimisation over the whole stream and is not
affordable here.

What it IS: a fully-informed REFERENCE POLICY -- monotone in load, low variance,
and able to load the substrate -- which is all a difficulty ladder requires. Report
it as a reference, never as a bound, and expect an arm to edge past it somewhere.
"""
from __future__ import annotations

import itertools

#: Refuse rather than hang if a future composition makes the search unbounded.
MAX_PARTITIONS = 20_000


def _contiguous(idx) -> bool:
    runs = sum(1 for i, d in enumerate(idx) if i == 0 or idx[i - 1] != d)
    return runs == len(set(idx))


def best_feasible_partition(view, substrate=None, slice_req=None):
    """The best contiguous admissible partition for one arrival, or None.

    Args:
        view: a `_PartitionView` over the arrival and the CURRENT substrate.
        substrate, slice_req: supply BOTH to get the delay-aware pick. Omitting
            them falls back to the capacity-only tiebreak, which undercounts
            feasibility and must not be used to calibrate anything.

    Ordering: fewest domains first, then greatest remaining headroom in the
    tightest domain, then lexicographic for determinism. Fewest-domains first
    because every extra domain buys inter-domain flows that C5/C9 and the delay
    budget then have to absorb; maximum headroom because the aggregate test is
    necessary and not sufficient, so the actor is likelier to seat the chain when
    the domain is not left at zero.
    """
    M, K = view.M, view.K
    if K == 0 or M == 0:
        return None
    choices = [[m for m in range(M) if view.admissible(k, m)] for k in range(K)]
    if any(not c for c in choices):
        return None
    space = 1
    for c in choices:
        space *= len(c)
        if space > MAX_PARTITIONS:
            raise RuntimeError(
                f"partition oracle refused: search space exceeds {MAX_PARTITIONS}. "
                "The oracle is bounded on purpose; widen MAX_PARTITIONS only with "
                "a measured cost, and never let it become the pre-§Y node-level "
                "enumeration that §Y.5 deleted.")

    cands = feasible_partitions(view)
    if not cands:
        return None
    if substrate is None or slice_req is None:
        return cands[0][1]      # capacity-only pick, for callers without context

    # DELAY-AWARE, and exhaustively so: score EVERY feasible candidate and keep the
    # one that fits the budget with the fewest domains. The capacity-only tiebreak
    # this replaces discarded arrivals a delay-aware choice admits, which is how a
    # full-observability arm came to score .8810 at L1 against a "ceiling" of .7910.
    budget = getattr(getattr(slice_req, "qos", None), "max_e2e_delay", None)
    dom_nodes = [set(substrate.nodes_in_domain(s.domain_id)) for s in view.summaries]
    perm = [set(x.permitted_nodes) for x in slice_req.vnfs]
    fallback = cands[0][1]
    best_key = best = None
    for key, cand in cands:
        d = predicted_e2e(view, cand, substrate, slice_req, dom_nodes, perm)
        if d == float("inf"):
            continue
        if budget is not None and d > budget:
            continue
        k = (key[0], d, cand)
        if best_key is None or k < best_key:
            best_key, best = k, cand
    return best if best is not None else fallback


# ── predicted admission: the full-substrate view of what a partition will cost ──
#
# Shared with `partial_obs_prior.fullobs_builder` so the oracle and the
# full-observability arm score a partition by exactly the same model. That keeps
# the relationship between them honest: they differ ONLY in how many candidates
# they are allowed to score (the arm stops at its top-K by a cheap key, the oracle
# scores every feasible one), so the oracle dominates by construction and can be
# read as a ceiling rather than as another policy.
#
# WHY THIS EXISTS. The first version of this module ranked candidates by (fewest
# domains, most headroom) and committed the winner. That is a capacity tiebreak,
# and capacity is not what refuses most arrivals: at L3 seed 42 per 2000,
# `cross_domain_infeasible` 305 + `c9_hops` 49 + `post_commit_c7_delay` 285 against
# `actor_infeasible` 73. So the "oracle" was discarding arrivals that a delay-aware
# choice admits, and a full-observability arm scoring .8810 at L1 beat the ceiling
# that was supposed to bound it. An instrument that undercounts feasibility cannot
# calibrate a difficulty ladder.


def predicted_e2e(view, cand, substrate, slice_req, dom_nodes=None, perm=None):
    """Predicted end-to-end delay for `cand`, or inf if it cannot be served.

    Uses the SAME M/M/1 model the verifier applies post-commit:

      node  the sojourn at the node each VNF would actually occupy, from that
            node's real capacity and current load. Colocation inflates this
            superlinearly, because the sojourn is charged per VNF at the shared
            node.
      link  for each chain edge crossing a domain boundary, the sojourn on the
            best inter-domain link between those two domains, from its real
            residual bandwidth. Zero for an edge that stays inside a domain.

    Both terms need node residuals and link state, so neither is computable on the
    MDO's partial observation surface. That is the whole content of the
    observability claim this pair measures.
    """
    from orion.sim.delay_model import link_sojourn, node_sojourn

    g = substrate.graph
    if dom_nodes is None:
        dom_nodes = [set(substrate.nodes_in_domain(s.domain_id))
                     for s in view.summaries]
    if perm is None:
        perm = [set(x.permitted_nodes) for x in slice_req.vnfs]

    res = {n: [float(g.nodes[n]["cpu_residual"]), float(g.nodes[n]["ram_residual"])]
           for m in set(cand) for n in dom_nodes[m]}
    total = 0.0
    for k, m in enumerate(cand):
        pick = None
        for n in sorted(perm[k] & dom_nodes[m]):
            slack = min(res[n][0] - view.cpu[k], res[n][1] - view.ram[k])
            if slack >= 0 and (pick is None or slack < pick[0]):
                pick = (slack, n)
        if pick is None:
            return float("inf")
        n = pick[1]
        res[n][0] -= view.cpu[k]
        res[n][1] -= view.ram[k]
        node = g.nodes[n]
        capacity = float(node["cpu_capacity"])
        used = capacity - float(node["cpu_residual"]) + view.cpu[k]
        t = node_sojourn(base_processing_delay=float(node["processing_delay"]),
                         intensity=slice_req.vnfs[k].computational_intensity,
                         cpu_capacity=capacity, cpu_used=used)
        if t == float("inf"):
            return float("inf")
        total += t

    best_link = _inter_domain_links(g)
    for i, fl in enumerate(slice_req.flow_edges):
        if i + 1 >= len(cand):
            break
        da = view.summaries[cand[i]].domain_id
        db = view.summaries[cand[i + 1]].domain_id
        if da == db:
            continue
        link = best_link.get((min(da, db), max(da, db)))
        if link is None:
            return float("inf")
        cap, res_bw, prop = link
        if res_bw < fl.bandwidth_demand:
            return float("inf")
        t = link_sojourn(propagation_delay=prop, bandwidth_capacity=cap,
                         bandwidth_used=cap - res_bw + fl.bandwidth_demand)
        if t == float("inf"):
            return float("inf")
        total += t
    return total


def _inter_domain_links(g):
    """(capacity, residual, propagation) of the best link between each domain pair."""
    best = {}
    for u, w, d in g.edges(data=True):
        du, dw = g.nodes[u]["domain_id"], g.nodes[w]["domain_id"]
        if du == dw:
            continue
        key = (min(du, dw), max(du, dw))
        res = float(d["bw_residual"])
        prev = best.get(key)
        if prev is None or res > prev[1]:
            best[key] = (float(d["bandwidth_capacity"]), res,
                         float(d["propagation_delay"]))
    return best


def feasible_partitions(view):
    """Every contiguous, admissible, aggregate-feasible partition, best-first.

    Ordering is the cheap capacity key (fewest domains, then greatest headroom in
    the tightest domain). Callers that can afford it rescore with `predicted_e2e`;
    callers that cannot take the prefix.
    """
    M, K = view.M, view.K
    if K == 0 or M == 0:
        return []
    choices = [[m for m in range(M) if view.admissible(k, m)] for k in range(K)]
    if any(not c for c in choices):
        return []
    space = 1
    for c in choices:
        space *= len(c)
        if space > MAX_PARTITIONS:
            raise RuntimeError(
                f"partition oracle refused: search space exceeds {MAX_PARTITIONS}. "
                "The oracle is bounded on purpose; widen MAX_PARTITIONS only with "
                "a measured cost, and never let it become the pre-§Y node-level "
                "enumeration that §Y.5 deleted.")
    out = []
    for cand in itertools.product(*choices):
        if not _contiguous(cand):
            continue
        acc_c: dict[int, float] = {}
        acc_r: dict[int, float] = {}
        for k, m in enumerate(cand):
            acc_c[m] = acc_c.get(m, 0.0) + view.cpu[k]
            acc_r[m] = acc_r.get(m, 0.0) + view.ram[k]
        ok = True
        tightest = float("inf")
        for m in acc_c:
            s = view.summaries[m]
            if s.cpu_residual < acc_c[m] or s.ram_residual < acc_r[m]:
                ok = False
                break
            tightest = min(tightest, s.cpu_residual - acc_c[m])
        if ok:
            out.append(((len(set(cand)), -tightest), cand))
    out.sort()
    return out
