"""Partial-obs per-VNF best-fit partition builder (2026-07-24; colocation-first
removed 2026-08-21).

The LLM-free m̃ source that sees ONLY the MDO's observation surface:
DomainSummary aggregates + the K x M node-based feasibility mask. No node
residuals, no full-substrate FFD. Shared by probe_partialobs_baseline.py
(follow_prior approach) and grid_runner.py (Plain-partial approach + RL-poprior's
KL prior / obs m̃).

Import-light on purpose: orion.* only, no runner imports (grid_runner and the
probe both import this module, so it must not import them back).
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.mdo.observation import build_domain_summaries  # noqa: E402
from orion.mdo.types import PlanSummary  # noqa: E402
from orion.types import InfrastructureTier  # noqa: E402

# How much of the h^m guard the heuristic is allowed to use. Default "full" is the
# behaviour every banked result was produced with, so no existing cell moves.
#
#   "full"   the guard as written: for VNF k in domain m, try EVERY tier of m that
#            k's permitted set covers, and ask whether that tier's best node fits.
#   "modal"  restrict the guard to k's modal required tier, the single scalar the
#            MDO observation actually carries per VNF. The per-(k,m) tier SET the
#            "full" guard iterates is not in the observation at all: the plan block
#            publishes one tier index per VNF, so a policy cannot represent "in this
#            domain, k may use edge OR regional". Parity correction, not a handicap.
#   "off"    no guard, commit on the aggregates alone.
#
# The observation ALSO cannot express the comparison itself, at any setting. In
# `observation_to_tensor` h^m is normalised by `max_cpu_cap`, a per-instance
# constant, while the VNF demand it would be compared against is normalised by
# `max_cpu_d`, the largest demand in THAT arrival's chain. Absolute demand is
# therefore not recoverable from the observation, so "does this tier's best node fit
# this VNF" is not a function of anything the policy is given. That is a defect in
# the encoding rather than in the baseline, and it is not fixed here.
GUARD_MODE = os.environ.get("ORION_PARTIAL_GUARD", "full")

# Whether the split fallback debits its running residual estimate as it assigns each
# VNF. True is the behaviour every banked result was produced with.
#
# This is the heuristic's other structural advantage, and it is a bigger one than the
# guard. Debiting makes the k-th choice conditional on the first k-1, so the partition
# is a JOINT decision: it will not send two large VNFs to the same domain on the
# strength of a residual only one of them can consume. ORION's partition head is
# per-VNF factored, so it cannot represent that dependence at all, whatever it is
# trained on. Setting this False models the heuristic under the same restriction.
SEQ_ACCOUNTING = os.environ.get("ORION_PARTIAL_SEQ", "1") != "0"

#: `fullobs_builder` bounds. The partition space is M**K, 625 on the committed
#: composition; above the first bound the builder degrades to the greedy rule
#: rather than hanging. The delay model is evaluated only on the best
#: DELAY_SCORED_TOP candidates, so cost is flat in substrate size.
MAX_FULLOBS_PARTITIONS = int(os.environ.get("ORION_FULLOBS_MAX_PARTITIONS", "4096"))
DELAY_SCORED_TOP = int(os.environ.get("ORION_FULLOBS_DELAY_TOP", "12"))

# Whether the builder is restricted to CHAIN-CONTIGUOUS partitions (2026-08-21).
# ON by default: this is the same rule the coordinator now enforces on EVERY
# approach as C10 (see orion.mdo.chain_order), and a builder that could author a
# partition the commit path refuses would be losing arrivals to its own output.
# Set ORION_PARTIAL_CONTIG=0 only together with ORION_CHAIN_ORDER=off, to measure
# what the constraint costs.
#
# Nothing in the partition layer reads the chain today. The builder assigns each VNF
# independently, `MDOPolicy` is per-VNF factored, and Agent B's schema pins one
# domain per VNF with no cross-VNF term, so a partition that leaves a domain and
# returns to it, (A, B, A), is representable everywhere and costs nothing at the
# point of choice. It is only charged later: `MDOCoordinator._build_fragments`
# classifies a flow edge as cross-domain iff its two endpoints landed in different
# domains, so a partition with r maximal runs along the chain pays r-1 cross-domain
# flows, and r is minimised exactly when every domain occupies ONE run.
#
# The test is request-side and needs no substrate at all: the chain order is
# `sr.vnfs` and the edges are `sr.flow_edges`, both of which the builder already
# holds. It is therefore observation-legal at any setting of GUARD_MODE, and it is
# strictly a restriction of the candidate set, never a new source of evidence.
#
# Measured effect on this builder, 500 arrivals at L3 (conventional, instance 100,
# seed 42), with colocation-first already removed: 33% of unconstrained partitions
# re-enter a domain they had left, and the restriction removes all of them while
# losing no arrival.
CONTIGUOUS = os.environ.get("ORION_PARTIAL_CONTIG", "1") != "0"

#: One definition of the rule, shared with the coordinator's C10 enforcement.
from orion.mdo.chain_order import is_contiguous  # noqa: E402,F401


def _required_tiers(sr, substrate):
    """Per-VNF modal tier of permitted_nodes — request-side info, obs-legal."""
    g = substrate.graph
    tiers = []
    for v in sr.vnfs:
        counts = {}
        for n in v.permitted_nodes:
            if n in g.nodes:
                t = g.nodes[n]["tier"]
                counts[t] = counts.get(t, 0) + 1
        tiers.append(InfrastructureTier(max(counts, key=counts.get)) if counts
                     else InfrastructureTier.EDGE)
    return tiers


class _PartitionView:
    """The MDO's observation surface for one arrival, plus the two admissibility
    tests the partial-observability heuristic applies to a partition.

    Extracted 2026-08-17 so the LLM plan path can be held to the SAME tests the
    heuristic applies to its own proposal (see `repair_plan`). Nothing here reads
    node residuals: only `build_domain_summaries` aggregates + per-domain
    permitted-node intersection, which is exactly what `observation_to_tensor`
    and `build_tier_masks` expose.
    """

    __slots__ = ("summaries", "M", "K", "feas", "feas_tiers", "cpu", "ram",
                 "req_tiers", "structural_reject", "domain_to_canonical",
                 "_max_cpu_cap", "_max_ram_cap", "_max_cpu_d", "_max_ram_d")

    def __init__(self, sr, substrate):
        g = substrate.graph
        self.summaries = build_domain_summaries(substrate)
        self.M = len(self.summaries)
        self.K = len(sr.vnfs)
        dom_nodes = [set(substrate.nodes_in_domain(s.domain_id))
                     for s in self.summaries]
        self.feas = [[bool(set(v.permitted_nodes) & dom_nodes[m])
                      for m in range(self.M)]
                     for v in sr.vnfs]  # == build_tier_masks node-based rows
        self.structural_reject = any(not any(row) for row in self.feas)
        # Which TIERS of domain m VNF k may use. Needed because h^m is published per
        # tier and a VNF may only use the tiers its permitted set covers, so the right
        # question is "does one of MY tiers here hold a node big enough", not "does
        # this domain".
        self.feas_tiers = [[{g.nodes[n]["tier"]
                             for n in set(v.permitted_nodes) & dom_nodes[m]}
                            for m in range(self.M)] for v in sr.vnfs]
        self.cpu = [v.cpu_demand for v in sr.vnfs]
        self.ram = [v.ram_demand for v in sr.vnfs]
        self.req_tiers = _required_tiers(sr, substrate)
        self.domain_to_canonical = {s.domain_id: m
                                    for m, s in enumerate(self.summaries)}
        # The exact normalisers `observation_to_tensor` uses, for GUARD_MODE="obsenc".
        self._max_cpu_cap = max((s.cpu_capacity for s in self.summaries),
                                default=1.0) or 1.0
        self._max_ram_cap = max((s.ram_capacity for s in self.summaries),
                                default=1.0) or 1.0
        self._max_cpu_d = max(self.cpu, default=1.0) or 1.0
        self._max_ram_d = max(self.ram, default=1.0) or 1.0

    def fits_a_node(self, k, m):
        """h^m guard: some tier of domain m that VNF k may use has a single free node
        big enough for it (restored 2026-08-12).

        NECESSARY, not sufficient. It cannot see that two co-located VNFs would need
        the same one node, since the summary reports only the best-fitting node per
        tier. It does catch the case the aggregate cannot see at all, a domain whose
        residual is real but spread over nodes that are each too small, which is what
        the aggregate-only version turned into `actor_infeasible` rejections.

        Which tiers it may consult is set by GUARD_MODE; see the note there for why
        "modal" is the observation-legal one.
        """
        if GUARD_MODE == "off":
            return True
        s = self.summaries[m]
        if GUARD_MODE == "obsenc":
            # The guard as the POLICY would have to compute it: both sides in the
            # units `observation_to_tensor` emits. h^m is divided by the per-instance
            # max domain capacity, the demand by the largest demand in THIS chain.
            # The two denominators differ, which is the defect; this mode does not
            # correct it, it reproduces it, so the heuristic is held to exactly the
            # evidence the policy gets.
            for t in self.feas_tiers[k][m]:
                ti = InfrastructureTier(t)
                if (s.tier_max_node_cpu.get(ti, 0.0) / self._max_cpu_cap
                        >= self.cpu[k] / self._max_cpu_d
                        and s.tier_max_node_ram.get(ti, 0.0) / self._max_ram_cap
                        >= self.ram[k] / self._max_ram_d):
                    return True
            return False
        if GUARD_MODE == "modal":
            # Only the one tier index the observation publishes for this VNF, and
            # only if this domain actually offers it to k. A domain that is in the
            # hard mask but holds none of k's modal tier reads h^m = 0 in the
            # observation, which is what the policy would see.
            tiers = ({self.req_tiers[k].value} & self.feas_tiers[k][m])
        else:
            tiers = self.feas_tiers[k][m]
        for t in tiers:
            ti = InfrastructureTier(t)
            if (s.tier_max_node_cpu.get(ti, 0.0) >= self.cpu[k]
                    and s.tier_max_node_ram.get(ti, 0.0) >= self.ram[k]):
                return True
        return False

    def colocation_candidate(self):
        """(m, slack) of the best single domain that can hold the WHOLE chain, or
        (None, 0.0). This is step 1 of `partial_obs_builder`, verbatim."""
        best, best_slack = None, 0.0
        for m in range(self.M):
            if not all(self.feas[k][m] for k in range(self.K)):
                continue
            if not all(self.fits_a_node(k, m) for k in range(self.K)):
                continue
            s = self.summaries[m]
            slack = min(s.cpu_residual - sum(self.cpu),
                        s.ram_residual - sum(self.ram))
            if slack <= 0:
                continue
            if best is None or slack > best_slack:
                best, best_slack = m, slack
        return best, best_slack

    def admissible(self, k, m):
        """Is domain m an admissible host for VNF k on the observation surface?"""
        return self.feas[k][m] and self.fits_a_node(k, m)

    def hosts_whole_chain(self, m):
        """Can domain m still take the ENTIRE chain, as the substrate is NOW?

        The same three tests `colocation_candidate` applies, asked of one nominated
        domain instead of searched over all of them. Used by `repair_plan` mode
        "colo" to decide whether a cached single-host plan is still valid.
        """
        if not all(self.admissible(k, m) for k in range(self.K)):
            return False
        s = self.summaries[m]
        return min(s.cpu_residual - sum(self.cpu),
                   s.ram_residual - sum(self.ram)) > 0


def partial_obs_builder(sr, substrate):
    """Per-VNF best-fit partition from DomainSummary aggregates + the K x M
    node-based feasibility mask ONLY (exactly the MDO's observation surface).

    Discipline: touches substrate only through build_domain_summaries() and
    per-domain permitted-node intersection (the same two inputs the MDO obs
    exposes as summary features and tier_mask). Never reads node residuals.

    COLOCATION-FIRST REMOVED (2026-08-21, user directive). The builder used to
    search for a single domain that could hold the WHOLE chain and commit to it,
    falling back to per-VNF assignment only when no such domain existed. On the
    banked instances that branch fired on 100% of arrivals, so the heuristic
    answered every partitioning question with "one domain" and the multi-domain
    problem the orchestrator exists to solve was never posed. The branch is gone
    rather than flagged off: a rule that decides every arrival is not an ablation
    setting. What is left is the rule that was previously the fallback, applied
    uniformly, so the builder now places each function where the summaries say it
    fits best and colocation is an OUTCOME when the aggregates favour it rather
    than a policy imposed ahead of the evidence.

    `_PartitionView.colocation_candidate` and `hosts_whole_chain` survive because
    `repair_plan` modes "full" and "colo" are defined in terms of them; those modes
    collapse an LLM plan onto one host and are themselves in tension with this
    directive, which is noted at REPAIR_MODE and left for a separate decision.

    Every banked cell produced before this change is superseded: this is a change
    of behaviour on the default path, not a flag.
    """
    v = _PartitionView(sr, substrate)
    if v.structural_reject:
        return None  # structural reject: some VNF feasible nowhere
    M, K, cpu, ram = v.M, v.K, v.cpu, v.ram

    # per-VNF best-fit on running aggregate estimates, over the domains that
    # both permit the VNF and can hold it on one node.
    est_cpu = [s.cpu_residual for s in v.summaries]
    est_ram = [s.ram_residual for s in v.summaries]
    chosen = []
    # CONTIGUOUS: a domain that the chain has already LEFT is closed to it. The
    # current domain always stays open, so the rule never forces a cut, it only
    # forbids coming back, which is exactly the (A, B, A) shape that pays two
    # cross-domain flows where one would do.
    closed: set[int] = set()
    cur = None
    for k in range(K):
        cands = [(min(est_cpu[m] - cpu[k], est_ram[m] - ram[k]), m)
                 for m in range(M) if v.admissible(k, m)]
        if CONTIGUOUS:
            cands = [(s, m) for s, m in cands if m not in closed]
        if not cands:
            return None  # no domain can hold this VNF on any single node it may use
        slack, m = max(cands)
        if slack <= 0:  # no summary-feasible domain left
            return None
        if CONTIGUOUS and cur is not None and m != cur:
            closed.add(cur)  # the run in `cur` has ended; it cannot be reopened
        cur = m
        chosen.append(m)
        if SEQ_ACCOUNTING:
            est_cpu[m] -= cpu[k]
            est_ram[m] -= ram[k]
    doms = [v.summaries[m].domain_id for m in chosen]
    return PlanSummary(
        vnf_ids=[x.vnf_id for x in sr.vnfs],
        required_tiers=_required_tiers(sr, substrate),
        suggested_domains=doms,
        cpu_demands=cpu, ram_demands=ram,
        vcrs=[x.vcr for x in sr.vnfs],
        bw_demands=[fl.bandwidth_demand for fl in sr.flow_edges])


def oracle_builder(sr, substrate):
    """§Y.3b CALIBRATION INSTRUMENT -- not an approach, never a results row.

    The partition-exhaustive reference the load ladder is pinned on. It shares this
    module only because it needs `_PartitionView`, i.e. exactly the same
    admissibility tests the heuristic applies to itself; the search itself lives in
    `orion.sim.partition_oracle`. See that module for why the ladder cannot be
    pinned on any single policy's acceptance after §Y.1f.
    """
    from orion.sim.partition_oracle import best_feasible_partition

    v = _PartitionView(sr, substrate)
    if v.structural_reject:
        return None
    cand = best_feasible_partition(v, substrate, sr)
    if cand is None:
        return None
    return PlanSummary(
        vnf_ids=[x.vnf_id for x in sr.vnfs],
        required_tiers=_required_tiers(sr, substrate),
        suggested_domains=[v.summaries[m].domain_id for m in cand],
        cpu_demands=v.cpu, ram_demands=v.ram,
        vcrs=[x.vcr for x in sr.vnfs],
        bw_demands=[fl.bandwidth_demand for fl in sr.flow_edges])


def fullobs_builder(sr, substrate):
    """The FULL-observability twin of `partial_obs_builder` (rewritten 2026-08-22).

    SAME DECISION, DIFFERENT EVIDENCE -- and this time the extra evidence has a
    channel to act through, which the previous two versions did not.

    History, because both earlier attempts failed in instructive ways. v1 ranked
    domains by the slack in their best single NODE, which is not a better-informed
    version of the partial builder's aggregate-slack rule but a DIFFERENT
    objective, and worst-fit at node granularity: it shattered 100% of chains at L1
    and lost to its own partial-observability twin. v2 restored rule parity by
    ranking on the same aggregate and spending node visibility only on exact
    feasibility, and that tied partial obs to the digit at L1-L3. The tie was not
    noise: `PlanSummary` carries `suggested_domains` and NO node field, so the node
    the builder verifies is discarded and the actor re-derives placement. The only
    channels left were the candidate filter and the ranking, and the h^m guard is
    almost never wrong (0 optimistic and 26-86 pessimistic per ~24,700 checks). An
    ablation whose two arms agree is not measuring observability.

    WHAT FULL OBSERVABILITY IS ACTUALLY WORTH. Look at where arrivals die, L3 seed
    42 per 2000: `actor_infeasible` 73, but `cross_domain_infeasible` 305, `c9_hops`
    49 and `post_commit_c7_delay` 285. Node CAPACITY, the only thing v2's extra
    evidence touched, is the SMALLEST bin. The large ones are inter-domain link
    capacity and the end-to-end delay budget, together 32% of arrivals, and those
    are exactly what a partial-observability agent cannot evaluate: the MDO
    observation carries per-domain aggregates, no link state at all, and no way to
    compare a partition against a delay budget (see the GUARD_MODE note on why the
    encoding cannot even express the comparison).

    So this builder scores a WHOLE PARTITION by what the full substrate predicts
    will happen to it:

      node term  the real M/M/1 sojourn at the node each VNF would occupy, from
                 that node's actual capacity and current load. This is the term
                 behind `post_commit_c7_delay`, and colocation inflates it
                 superlinearly because sojourn is per-VNF at the shared node.
      link term  for each chain edge that crosses domains, the sojourn on the best
                 inter-domain link between those two domains, from its real
                 residual bandwidth. Zero when the edge stays inside a domain. This
                 is the term behind `cross_domain_infeasible` and `c9_hops`.

    A partition whose predicted delay exceeds the arrival's budget is rejected here
    rather than committed and refused later, and among those that fit, the fewest
    domains wins, then the greatest headroom.

    The rule difference from `partial_obs_builder` is FORCED BY the information
    difference, and that is the point of the pair: a partial-obs agent cannot
    evaluate delay, so it must use a greedy capacity proxy; a full-obs agent can, so
    it does. That is a different claim from v1's, which changed the proxy without
    improving it.

    Bounded: at most MAX_FULLOBS_PARTITIONS contiguous admissible candidates are
    enumerated (M**K = 625 on the committed composition) and the delay model is
    evaluated only on the DELAY_SCORED_TOP best by the cheap key, so cost is flat in
    substrate size rather than growing with it.
    """
    import itertools

    from orion.sim.delay_model import link_sojourn, node_sojourn

    v = _PartitionView(sr, substrate)
    if v.structural_reject:
        return None
    M, K, cpu, ram = v.M, v.K, v.cpu, v.ram
    g = substrate.graph
    dom_nodes = [set(substrate.nodes_in_domain(s.domain_id)) for s in v.summaries]
    perm = [set(x.permitted_nodes) for x in sr.vnfs]

    choices = [[m for m in range(M) if v.admissible(k, m)] for k in range(K)]
    if any(not c for c in choices):
        return None
    space = 1
    for c in choices:
        space *= len(c)
    if space > MAX_FULLOBS_PARTITIONS:
        return _fullobs_greedy(sr, substrate, v)

    # ---- candidates: contiguous, admissible, aggregate-feasible ----------------
    cands = []
    for cand in itertools.product(*choices):
        runs = sum(1 for i, d in enumerate(cand) if i == 0 or cand[i - 1] != d)
        if runs != len(set(cand)):
            continue
        acc_c, acc_r = {}, {}
        for k, m in enumerate(cand):
            acc_c[m] = acc_c.get(m, 0.0) + cpu[k]
            acc_r[m] = acc_r.get(m, 0.0) + ram[k]
        ok, tight = True, float("inf")
        for m in acc_c:
            sm = v.summaries[m]
            if sm.cpu_residual < acc_c[m] or sm.ram_residual < acc_r[m]:
                ok = False
                break
            tight = min(tight, sm.cpu_residual - acc_c[m])
        if ok:
            cands.append(((len(set(cand)), -tight), cand))
    if not cands:
        return None
    cands.sort()

    # ---- inter-domain link state: the best link between each domain pair -------
    best_link = {}
    for u, w, d in g.edges(data=True):
        du, dw = g.nodes[u]["domain_id"], g.nodes[w]["domain_id"]
        if du == dw:
            continue
        key = (min(du, dw), max(du, dw))
        cap, res = float(d["bandwidth_capacity"]), float(d["bw_residual"])
        prev = best_link.get(key)
        if prev is None or res > prev[1]:
            best_link[key] = (cap, res, float(d["propagation_delay"]))

    budget = getattr(getattr(sr, "qos", None), "max_e2e_delay", None)

    def predicted_delay(cand):
        """Real node sojourn plus real inter-domain link sojourn, or inf."""
        res = {n: [float(g.nodes[n]["cpu_residual"]),
                   float(g.nodes[n]["ram_residual"])]
               for m in set(cand) for n in dom_nodes[m]}
        total = 0.0
        for k, m in enumerate(cand):
            pick = None
            for n in sorted(perm[k] & dom_nodes[m]):
                slack = min(res[n][0] - cpu[k], res[n][1] - ram[k])
                if slack >= 0 and (pick is None or slack < pick[0]):
                    pick = (slack, n)
            if pick is None:
                return float("inf")
            n = pick[1]
            res[n][0] -= cpu[k]
            res[n][1] -= ram[k]
            node = g.nodes[n]
            capacity = float(node["cpu_capacity"])
            used = capacity - float(node["cpu_residual"]) + cpu[k]
            t = node_sojourn(base_processing_delay=float(node["processing_delay"]),
                             intensity=sr.vnfs[k].computational_intensity,
                             cpu_capacity=capacity, cpu_used=used)
            if t == float("inf"):
                return float("inf")
            total += t
        for i, fl in enumerate(sr.flow_edges):
            if i + 1 >= len(cand):
                break
            da = v.summaries[cand[i]].domain_id
            db = v.summaries[cand[i + 1]].domain_id
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

    chosen = None
    for _key, cand in cands[:DELAY_SCORED_TOP]:
        delay = predicted_delay(cand)
        if delay == float("inf"):
            continue
        if budget is None or delay <= budget:
            chosen = cand
            break
    if chosen is None:
        chosen = cands[0][1]   # nothing predicted to fit: commit the safest shape

    return PlanSummary(
        vnf_ids=[x.vnf_id for x in sr.vnfs],
        required_tiers=_required_tiers(sr, substrate),
        suggested_domains=[v.summaries[m].domain_id for m in chosen],
        cpu_demands=cpu, ram_demands=ram,
        vcrs=[x.vcr for x in sr.vnfs],
        bw_demands=[fl.bandwidth_demand for fl in sr.flow_edges])


def _fullobs_greedy(sr, substrate, v):
    """Fallback when the partition space is too large to enumerate: rank on the
    aggregate with exact node feasibility. Never reached on the committed
    composition (M=5, K<=4); present so a larger workload degrades rather than
    hangs."""
    M, K, cpu, ram = v.M, v.K, v.cpu, v.ram
    est_cpu = [s.cpu_residual for s in v.summaries]
    est_ram = [s.ram_residual for s in v.summaries]
    chosen, closed, cur = [], set(), None
    for k in range(K):
        cands = [(min(est_cpu[m] - cpu[k], est_ram[m] - ram[k]), m)
                 for m in range(M)
                 if v.admissible(k, m) and (not CONTIGUOUS or m not in closed)]
        if not cands:
            return None
        _slack, m = max(cands)
        if CONTIGUOUS and cur is not None and m != cur:
            closed.add(cur)
        cur = m
        chosen.append(m)
        if SEQ_ACCOUNTING:
            est_cpu[m] -= cpu[k]
            est_ram[m] -= ram[k]
    return PlanSummary(
        vnf_ids=[x.vnf_id for x in sr.vnfs],
        required_tiers=_required_tiers(sr, substrate),
        suggested_domains=[v.summaries[m].domain_id for m in chosen],
        cpu_demands=cpu, ram_demands=ram,
        vcrs=[x.vcr for x in sr.vnfs],
        bw_demands=[fl.bandwidth_demand for fl in sr.flow_edges])


# ── plan repair: hold the LLM's m̃ to the SAME tests the heuristic applies ────────
#
# WHY THIS EXISTS (2026-08-17). `partial_obs_builder` applies three admissibility
# tests to its own proposal: node-based feasibility, the h^m guard, and the
# colocation-first + aggregate-slack structure. `make_llm_plan_builder` applies
# NONE of them: it pins the schema so m̃ names a real VNF and a tier-feasible
# domain, and returns the partition unchecked. `tier_filtered` is the only guard on
# that path, is OFF by default, and is a near-no-op under the node-based mask.
#
# HISTORICAL MOTIVATION, and its premise no longer holds. The guard was built when
# `cross_domain_infeasible` + `c9_hops` ran 236 / 220 / 183 / 146 per 2000 at L1-L4
# for `Full` against 12 / 20 / 7 / 6 for `RL-alone`, and the reading was that those
# splits were ELECTIVE: `partial_obs_builder` split 0.0% of arrivals and a
# whole-chain colocation host was node-feasible on 100% of them.
#
# Both halves of that premise are now false. Colocation-first was removed from the
# builder on 2026-08-21, so it splits 61-88% of arrivals, and the §Y.1f amendment
# means a whole-chain host exists on only 48.5%. A split is no longer elective and
# usually is not avoidable. The guard is kept because per-VNF h^m repair is correct
# on its own terms, not because of the gap decomposition above.
#
# This is a PARITY correction, not a new advantage. Every quantity it reads is
# already published to Agent B in `build_abstract_topology` (per-tier residuals,
# per-tier largest free node, aggregates, inter-domain links), so the guard tells
# the planner nothing it was not already shown; it only refuses to commit a
# proposal that the planner's own inputs say is inadmissible.
#
#   "off"       shipped behaviour: m̃ is committed exactly as authored. DEFAULT, so
#               every banked cell reproduces byte-for-byte.
#   "guard"     per-VNF h^m repair only. A VNF whose proposed domain fails
#               `admissible` is moved to the best-slack domain that passes. Splits
#               the planner authored are PRESERVED. Isolates the h^m guard's value
#               on the LLM path.
#   "full"      "guard", plus: if m̃ spans more than one domain and a whole-chain
#               colocation host exists, collapse to it. Isolates the colocation-first
#               structure, which is the term the measurement above says dominates.
#
# Minimal intervention is deliberate in both modes: an admissible proposal is never
# second-guessed, and a plan the planner already colocated is never moved. If the
# guard overrode admissible proposals too it would converge on
# `partial_obs_builder`, ORION would score the baseline's number by construction,
# and the planner's contribution would stop being measurable. The point is to make
# m̃ admissible, not to replace it.
REPAIR_MODE = os.environ.get("ORION_PLAN_REPAIR", "off")

#: Filled by `repair_plan` so a runner can report how often the guard fired.
REPAIR_STATS = {"seen": 0, "collapsed": 0, "vnfs_moved": 0, "unrepairable": 0}


def reset_repair_stats():
    for k in REPAIR_STATS:
        REPAIR_STATS[k] = 0
    return REPAIR_STATS


def repair_plan(plan, sr, substrate, mode=None):
    """Return `plan` with every inadmissible domain assignment repaired.

    Observation-legal throughout: reads only `_PartitionView`, i.e. per-domain
    summaries and permitted-node intersection.

    NEVER returns None for a non-None input. An arrival the guard cannot repair is
    committed as authored and fails downstream exactly as it does today, so the
    `structural` rejection bin does not move and the repair can only be credited
    with rejections it actually converts.
    """
    mode = REPAIR_MODE if mode is None else mode
    if plan is None or mode == "off":
        return plan
    if mode not in ("guard", "full", "colo"):
        raise ValueError(f"unknown ORION_PLAN_REPAIR mode {mode!r}; "
                         "expected one of off|guard|full|colo")

    v = _PartitionView(sr, substrate)
    if v.structural_reject or v.M == 0:
        return plan
    REPAIR_STATS["seen"] += 1

    # m̃ arrives in raw domain-id space; work in canonical index space and map back.
    proposed = []
    for k in range(v.K):
        d = plan.suggested_domains[k] if k < len(plan.suggested_domains) else None
        proposed.append(v.domain_to_canonical.get(d))
    if any(m is None for m in proposed):
        return plan  # m̃ names a domain that is not in the summary; leave it alone

    # 0) §AD.2 colocation-preserving repair, for the §AD contract.
    #
    # The contract makes the plan substrate-SENSITIVE: one host, chosen from live
    # slack. That is exactly the property the plan cache destroys, and the contract
    # cells run at an 89% hit rate against `MDO-partial`'s no cache at all, which is
    # why band 1.0 scored .5140 while naming the same host the heuristic names on
    # 200/200 arrivals. `guard` cannot fix it here: it moves the offending VNF and
    # re-introduces the split the contract just removed.
    #
    # So: keep the planner's host whenever it is STILL admissible for the whole
    # chain, and only when it is not, move the WHOLE chain to the best current host.
    # Shape preserved, authorship preserved, staleness gone.
    if mode == "colo":
        if len(set(proposed)) == 1 and v.hosts_whole_chain(proposed[0]):
            return plan  # the model's choice still holds; never second-guess it
        best, _slack = v.colocation_candidate()
        if best is not None:
            REPAIR_STATS["collapsed"] += 1
            return replace(plan,
                           suggested_domains=[v.summaries[best].domain_id] * v.K)
        # no whole-chain host exists any more: fall through to the per-VNF guard,
        # which is the legitimate split case.

    # 1) collapse an elective split, when a whole-chain host exists.
    if mode == "full" and len(set(proposed)) > 1:
        best, _slack = v.colocation_candidate()
        if best is not None:
            REPAIR_STATS["collapsed"] += 1
            return replace(
                plan,
                suggested_domains=[v.summaries[best].domain_id] * v.K,
            )

    # 2) per-VNF h^m repair, on running aggregate estimates so two repaired VNFs are
    #    not both sent to the same domain on the strength of a residual only one of
    #    them can consume. Same sequential accounting as the split fallback above.
    est_cpu = [s.cpu_residual for s in v.summaries]
    est_ram = [s.ram_residual for s in v.summaries]
    chosen = []
    moved = False
    for k in range(v.K):
        m = proposed[k]
        if not v.admissible(k, m):
            cands = [(min(est_cpu[j] - v.cpu[k], est_ram[j] - v.ram[k]), j)
                     for j in range(v.M) if v.admissible(k, j)]
            slack, j = max(cands) if cands else (0.0, None)
            if j is not None and slack > 0:
                m = j
                moved = True
                REPAIR_STATS["vnfs_moved"] += 1
            else:
                # Nothing on the observation surface can host this VNF. Commit the
                # planner's choice and let the domain actor reject it, rather than
                # inventing a placement the summaries do not support.
                REPAIR_STATS["unrepairable"] += 1
        chosen.append(m)
        if SEQ_ACCOUNTING:
            est_cpu[m] -= v.cpu[k]
            est_ram[m] -= v.ram[k]

    if not moved:
        return plan
    return replace(plan,
                   suggested_domains=[v.summaries[m].domain_id for m in chosen])


def plan_repaired(inner, mode=None):
    """Wrap a plan builder so its m̃ passes through `repair_plan`."""
    def _builder(sr, substrate):
        return repair_plan(inner(sr, substrate), sr, substrate, mode=mode)
    _builder.repair_stats = REPAIR_STATS
    return _builder
