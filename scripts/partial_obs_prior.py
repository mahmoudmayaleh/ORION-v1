"""Partial-obs colocation-first partition builder (2026-07-24).

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


def partial_obs_builder(sr, substrate):
    """Colocation-first partition from DomainSummary aggregates + the K x M
    node-based feasibility mask ONLY (exactly the MDO's observation surface).

    Discipline: touches substrate only through build_domain_summaries() and
    per-domain permitted-node intersection (the same two inputs the MDO obs
    exposes as summary features and tier_mask). Never reads node residuals.
    """
    v = _PartitionView(sr, substrate)
    if v.structural_reject:
        return None  # structural reject: some VNF feasible nowhere
    M, K, cpu, ram = v.M, v.K, v.cpu, v.ram

    # 1) colocation-first: a single domain feasible for ALL VNFs, best residual
    #    slack after the whole chain, and holding a node big enough for each of them.
    best, _best_slack = v.colocation_candidate()
    if best is not None:
        chosen = [best] * K
    else:
        # 2) split fallback: per-VNF best-fit on running aggregate estimates, over the
        #    domains that both permit the VNF and can hold it on one node.
        est_cpu = [s.cpu_residual for s in v.summaries]
        est_ram = [s.ram_residual for s in v.summaries]
        chosen = []
        for k in range(K):
            cands = [(min(est_cpu[m] - cpu[k], est_ram[m] - ram[k]), m)
                     for m in range(M) if v.admissible(k, m)]
            if not cands:
                return None  # no domain can hold this VNF on any single node it may use
            slack, m = max(cands)
            if slack <= 0:  # no summary-feasible domain left
                return None
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


# ── plan repair: hold the LLM's m̃ to the SAME tests the heuristic applies ────────
#
# WHY THIS EXISTS (2026-08-17). `partial_obs_builder` applies three admissibility
# tests to its own proposal: node-based feasibility, the h^m guard, and the
# colocation-first + aggregate-slack structure. `make_llm_plan_builder` applies
# NONE of them: it pins the schema so m̃ names a real VNF and a tier-feasible
# domain, and returns the partition unchecked. `tier_filtered` is the only guard on
# that path, is OFF by default, and is a near-no-op under the node-based mask.
#
# The measured consequence is not the h^m guard, it is the SPLIT. On the banked
# conventional/i100/2000-arrival cells, `cross_domain_infeasible` + `c9_hops` run
# 236 / 220 / 183 / 146 per 2000 at L1-L4 for `Full`, against 12 / 20 / 7 / 6 for
# `RL-alone`. Both bins are reachable ONLY when partition[src] != partition[dst]
# (MDOCoordinator._build_fragments), so every one of them requires a split
# partition. `partial_obs_builder` splits 0.0% of arrivals at every level, and a
# whole-chain colocation host is node-feasible on 100% of arrivals, so those splits
# are elective and the rejections they cause are avoidable. The deltas, 224 / 200 /
# 176 / 140 per 2000, are 11.2 / 10.0 / 8.8 / 7.0 pp, against a measured Full-minus-
# RL-alone gap of 11.3 / 7.3 / 4.8 / 6.6 pp. The split accounts for the whole gap.
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
    if mode not in ("guard", "full"):
        raise ValueError(f"unknown ORION_PLAN_REPAIR mode {mode!r}; "
                         "expected one of off|guard|full")

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
