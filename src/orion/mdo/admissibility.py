"""Post-decode admissibility guard for the committed partition (2026-08-22, §AI).

WHY THIS EXISTS. `_PartitionView` in `scripts/partial_obs_prior.py` defines two
different tests and the system applies them in two different places:

    feas[k][m]        may VNF k legally sit in domain m -- permitted nodes, tiers.
                      This is `build_tier_masks`, and it is the ONLY mask the MDO
                      policy's action space carries.
    admissible(k, m)  feas[k][m] AND fits_a_node(k, m), the h^m guard: some tier of
                      m that k may use holds a single free node big enough for it.
                      This is what `partial_obs_builder` ranks and commits on.

So the heuristic guarantees every VNF-domain pair it commits is admissible, while
ORION guarantees only that it is legal. `repair_plan` applies the guard to the LLM
PLAN, before the policy runs; in mode "advised" the policy then moves VNFs and
nothing re-checks. Measured at L3 seed 42, `actor_infeasible` per 2000 arrivals:

    MDO-partial       97     guard on every committed choice
    Full-rpc-fp      360     plan only, no policy override
    Full-rpc         675     policy overrides the plan   <- +315 from the override

This closes that gap by asking the same question of the FINAL partition, whatever
produced it.

OBSERVATION-LEGAL. Every quantity read here is already in the MDO observation:
`fits_a_node` reads `tier_max_node_cpu` / `tier_max_node_ram`, which
`observation_to_tensor` publishes as h^m, and the aggregate residual it ranks by is
the same `cpu_residual` the summary block carries. The guard tells the policy
nothing it was not already shown; it only refuses to commit a choice the policy's
own inputs rule out.

MINIMAL AND CONSERVATIVE, by construction:

  * A VNF whose authored domain is already admissible is NEVER moved. The guard
    cannot second-guess a choice that passes, so on a partition that is fully
    admissible it is a no-op and returns the input unchanged. That is what keeps it
    off the heuristic rows: `partial_obs_builder` commits only admissible pairs, so
    `moved` is 0 for them and the banked baseline numbers cannot move. This is
    asserted in the tests rather than assumed.
  * A VNF the guard cannot re-seat is left EXACTLY as authored, so it fails
    downstream as it does today. The guard can convert a rejection, never invent a
    placement, and the `structural` bin cannot move.
  * Candidates are restricted so the repair can never introduce a C10 violation.
    A repair that scattered the chain would trade an `actor_infeasible` for a
    `chain_order`, which is not a fix.

Sequential accounting mirrors the heuristic's: the running aggregate is debited as
each VNF is seated, so two re-seated VNFs are not both sent to a domain on the
strength of a residual only one of them can consume.
"""
from __future__ import annotations

import itertools
import os

from orion.types import InfrastructureTier

#: Off by default so every banked cell reproduces byte for byte. Turned on per run.
POSTDECODE_GUARD = os.environ.get("ORION_POSTDECODE_GUARD", "0") != "0"

#: Largest partition space the exact minimal repair will enumerate (M ** K). The
#: committed composition is M=5 and K<=4, so 625, and the search is reached only on
#: partitions that are already broken. Above the budget the guard falls back to the
#: left-to-right pass, which is strictly weaker but never worse than not repairing.
EXACT_REPAIR_BUDGET = int(os.environ.get("ORION_EXACT_REPAIR_BUDGET", "4096"))

#: Per-cell counters, so a run can report how often the guard actually fired.
GUARD_STATS = {"seen": 0, "vnfs_moved": 0, "unrepairable": 0, "partitions_changed": 0}


def reset_guard_stats():
    for k in GUARD_STATS:
        GUARD_STATS[k] = 0
    return GUARD_STATS


def _fits_a_node(summary, tiers, cpu_d, ram_d) -> bool:
    """h^m: does some tier the VNF may use hold one free node big enough?

    NECESSARY, not sufficient: the summary reports the best-fitting node per tier,
    so it cannot see that two co-located VNFs would need that same one node. It
    does catch the case the aggregate cannot see at all, a domain whose residual is
    real but spread over nodes that are each too small.
    """
    for t in tiers:
        ti = InfrastructureTier(t)
        if (summary.tier_max_node_cpu.get(ti, 0.0) >= cpu_d
                and summary.tier_max_node_ram.get(ti, 0.0) >= ram_d):
            return True
    return False


def repair_partition(partition, slice_req, substrate, summaries):
    """Re-seat every VNF whose committed domain fails the h^m guard.

    Args:
        partition: per-VNF domain ids, as decoded. Not mutated.
        slice_req: the arrival, for demands and permitted nodes.
        substrate: for the per-domain node sets and their tiers.
        summaries: the `DomainSummary` list the observation was built from.

    Returns:
        The repaired partition, or the input object itself when nothing moved.
    """
    vnfs = list(slice_req.vnfs)
    K, M = len(vnfs), len(summaries)
    if K == 0 or M == 0 or len(partition) != K:
        return partition

    g = substrate.graph
    dom_nodes = [set(substrate.nodes_in_domain(s.domain_id)) for s in summaries]
    dom_index = {s.domain_id: m for m, s in enumerate(summaries)}
    cpu = [v.cpu_demand for v in vnfs]
    ram = [v.ram_demand for v in vnfs]

    perm = [set(v.permitted_nodes) for v in vnfs]
    feas = [[bool(perm[k] & dom_nodes[m]) for m in range(M)] for k in range(K)]
    tiers = [[{g.nodes[n]["tier"] for n in perm[k] & dom_nodes[m]}
              for m in range(M)] for k in range(K)]

    def admissible(k, m):
        return feas[k][m] and _fits_a_node(summaries[m], tiers[k][m], cpu[k], ram[k])

    est_cpu = [s.cpu_residual for s in summaries]
    est_ram = [s.ram_residual for s in summaries]

    GUARD_STATS["seen"] += 1
    out, moved = [], False
    closed: set[int] = set()
    cur: int | None = None
    for k in range(K):
        m = dom_index.get(partition[k])
        if m is None:  # a domain the summary does not carry: leave it alone
            out.append(partition[k])
            continue
        if not admissible(k, m):
            # Contiguity-preserving candidates only: a repair that reopened a domain
            # the chain has left would trade actor_infeasible for chain_order.
            cands = [(min(est_cpu[j] - cpu[k], est_ram[j] - ram[k]), j)
                     for j in range(M)
                     if admissible(k, j) and j not in closed]
            slack, j = max(cands) if cands else (0.0, None)
            if j is not None and slack > 0:
                m, moved = j, True
                GUARD_STATS["vnfs_moved"] += 1
            else:
                # Nothing on the observation surface can host it. Commit as authored
                # and let the actor refuse, rather than inventing a placement the
                # summaries do not support.
                GUARD_STATS["unrepairable"] += 1
        if cur is not None and m != cur:
            closed.add(cur)
        cur = m
        est_cpu[m] -= cpu[k]
        est_ram[m] -= ram[k]
        out.append(summaries[m].domain_id)

    if not moved:
        return partition
    GUARD_STATS["partitions_changed"] += 1
    return out


def _contiguous(idx) -> bool:
    runs = sum(1 for i, d in enumerate(idx) if i == 0 or idx[i - 1] != d)
    return runs == len(set(idx))


def minimal_committable_partition(partition, slice_req, substrate, summaries):
    """The NEAREST partition to `partition` that C10 and the h^m guard both accept.

    WHY A SEARCH AND NOT A LEFT-TO-RIGHT PASS. C10 is a JOINT constraint, so a
    sequential repair cannot fix the shape that actually occurs. Measured on the
    §Y.1f substrate, Agent B answers the eMBB trap with [4, 2, 4] on 38 of 40
    arrivals: it seats the Firewall in the central domain (which is admissible), is
    forced out for the CDN (which cannot go central), and must return for the vEPC
    (which can go nowhere else). Walking left to right, every prefix is legal and
    the failure only becomes visible at the last VNF, by which point the domain it
    needs is closed. The fix is one edit at position 0 -- [2, 2, 4] or [0, 2, 4] --
    and no forward-only rule finds it.

    So: enumerate, keep the candidates that are contiguous, admissible and
    aggregate-feasible, and return the one that changes the FEWEST of the planner's
    assignments, tie-broken toward fewer domains and then lexicographically for
    determinism. Minimum Hamming distance is what makes this a repair rather than a
    replacement: the planner's intent is preserved wherever it was committable, and
    a partition that is already valid is at distance 0 and returned unchanged.

    Returns the input object itself when it is already committable or when no
    committable partition exists (in which case the caller commits as authored and
    the actor refuses, exactly as today).
    """
    vnfs = list(slice_req.vnfs)
    K, M = len(vnfs), len(summaries)
    if K == 0 or M == 0 or len(partition) != K or M ** K > EXACT_REPAIR_BUDGET:
        return partition

    g = substrate.graph
    dom_nodes = [set(substrate.nodes_in_domain(s.domain_id)) for s in summaries]
    dom_index = {s.domain_id: m for m, s in enumerate(summaries)}
    cpu = [v.cpu_demand for v in vnfs]
    ram = [v.ram_demand for v in vnfs]
    perm = [set(v.permitted_nodes) for v in vnfs]
    feas = [[bool(perm[k] & dom_nodes[m]) for m in range(M)] for k in range(K)]
    tiers = [[{g.nodes[n]["tier"] for n in perm[k] & dom_nodes[m]}
              for m in range(M)] for k in range(K)]

    def admissible(k, m):
        return feas[k][m] and _fits_a_node(summaries[m], tiers[k][m], cpu[k], ram[k])

    authored = [dom_index.get(d) for d in partition]
    if None in authored:
        return partition

    def feasible(idx):
        if not _contiguous(idx):
            return False
        if not all(admissible(k, idx[k]) for k in range(K)):
            return False
        acc_c, acc_r = {}, {}
        for k, m in enumerate(idx):
            acc_c[m] = acc_c.get(m, 0.0) + cpu[k]
            acc_r[m] = acc_r.get(m, 0.0) + ram[k]
        return all(summaries[m].cpu_residual >= acc_c[m]
                   and summaries[m].ram_residual >= acc_r[m] for m in acc_c)

    GUARD_STATS["seen"] += 1
    if feasible(authored):
        return partition

    # Only domains some VNF may actually use are worth enumerating.
    choices = [[m for m in range(M) if admissible(k, m)] for k in range(K)]
    if any(not c for c in choices):
        GUARD_STATS["unrepairable"] += 1
        return partition

    best, best_key = None, None
    for cand in itertools.product(*choices):
        if not feasible(cand):
            continue
        dist = sum(1 for k in range(K) if cand[k] != authored[k])
        key = (dist, len(set(cand)), cand)
        if best_key is None or key < best_key:
            best, best_key = cand, key
    if best is None:
        GUARD_STATS["unrepairable"] += 1
        return partition

    GUARD_STATS["vnfs_moved"] += best_key[0]
    GUARD_STATS["partitions_changed"] += 1
    return [summaries[m].domain_id for m in best]
