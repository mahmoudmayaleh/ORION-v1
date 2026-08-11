"""§Y.5 — acceptance ratio and rejection breakdown. Replaces fraction-of-ceiling.

FoC divided admissions by a separate feasibility oracle's count of "arrivals
admissible in principle". §Y deletes that denominator. Feasibility is decided by
the constraints themselves: a placement violating any of C1-C9 or the QoS gate is
infeasible and is a rejection, exactly as the verifier already decides it. There
is no second oracle sitting next to the constraints.

Two reasons beyond the conceptual one:

  * The ceiling was **contention-blind**. `approach_runner.compute_ceiling`
    evaluated each arrival against the substrate as it stood, but the enumeration
    ran over a stream replayed from scratch rather than the occupied network the
    eval actually faced, so the denominator did not fall as the network filled.
    Under a load ladder that inflates the apparent gap at high load for every
    approach, for a reason unrelated to any approach.
  * It was **unbounded**. The enumerator materialised
    `itertools.product(*feasible_nodes)`; at the §Y Large size (100 nodes, ~40
    feasible per VNF, K=5) that is 40^5 = 10^8 tuples built into a list. This is
    the known hang, and it is fatal rather than slow at §Y scale.

The rejection breakdown replaces the diagnostic value the ceiling carried, and is
strictly more informative: it says *why* an approach failed, not just how many it
missed.

§Y.13 (2026-08-03) splits the bins into PRE-COMMIT and POST-COMMIT. The
distinction is not cosmetic: a pre-commit bin is a decision the coordinator made
with the information it had, while a post-commit bin is an admission it granted
and the ground-truth verifier then took away under the realised load. Reading one
as the other inverts what the table says about a policy. Before the split every
post-commit revocation fell into `unattributed`, which absorbed 24 to 89 percent
of all rejections and made every breakdown unpublishable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Ordered so the breakdown always prints the same way, and so a reader can tell a
#: missing bin from a zero one.
REJECT_BINS = (
    # ── PRE-COMMIT: the coordinator refused before anything was allocated ──
    "structural",            # never reached the MDO: no admissible plan built
    "actor_infeasible",      # a domain actor returned z^m = 0 (capacity / tier)
    "cross_domain_infeasible",  # no route between the assigned domains
    "c5b_bandwidth",         # inter-domain bandwidth
    "c7_delay",              # end-to-end delay budget, coordinator's estimate
    "c9_hops",               # inter-domain hop limit
    "qos_gate",              # admitted by the partition, refused by the QoS gate
    # ── POST-COMMIT: committed and allocated, then revoked (§Y.13) ──
    # These are NOT policy decisions. The coordinator returned COMMIT, the plan
    # was allocated, and the ground-truth verifier then fired against the
    # post-allocation load, so the plan was deallocated and the admission
    # withdrawn. They were previously indistinguishable from `unattributed`.
    "post_commit_c2_cpu",       # node CPU over-allocated
    "post_commit_c3_ram",       # node RAM over-allocated
    "post_commit_c5b_bandwidth",  # throughput floor unmet under load
    "post_commit_c7_delay",     # delay budget blown under M/M/1 sojourn
    "post_commit_plan_build",   # committed partition, no realisable placement
    "unattributed",          # rejected with no violation recorded; see below
)

#: Post-commit verifier codes in the precedence `_classify` applies, matching the
#: order in `GroundTruthVerdict.hard_penalty_fired`. C5 and C9 are absent because
#: they do not fire the hard penalty, so they never revoke an admission.
_POST_COMMIT_BINS = (
    ("C2", "post_commit_c2_cpu"),
    ("C3", "post_commit_c3_ram"),
    ("C5b", "post_commit_c5b_bandwidth"),
    ("C7", "post_commit_c7_delay"),
    ("PLAN_BUILD", "post_commit_plan_build"),
)


@dataclass
class AcceptanceReport:
    """§Y.5 per-episode result."""

    admitted: int = 0
    offered: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    @property
    def acceptance(self) -> float:
        """Primary §Y metric: admitted / offered."""
        return self.admitted / self.offered if self.offered else 0.0

    def to_dict(self) -> dict:
        return {
            "acceptance": round(self.acceptance, 4),
            "admitted": self.admitted,
            "offered": self.offered,
            "rejections": {b: self.rejections.get(b, 0) for b in REJECT_BINS},
        }

    def check_conservation(self) -> None:
        """Admitted plus every rejection must equal offered.

        Guards the silent failure this breakdown is meant to prevent: a rejection
        bin that is never populated looks exactly like a constraint that never
        binds. If they do not sum, the breakdown is lying about where the losses
        went and must not be reported.
        """
        total = self.admitted + sum(self.rejections.values())
        if total != self.offered:
            raise ValueError(
                f"rejection breakdown does not conserve arrivals: admitted "
                f"{self.admitted} + rejections {sum(self.rejections.values())} "
                f"!= offered {self.offered}"
            )


def _classify(result) -> str:
    """Map one rejected MDOResult to a bin.

    A rejection can trip several constraints at once. The bins are checked in the
    order above and the FIRST match wins, so the breakdown counts each rejection
    exactly once and the columns sum to the total. This means the breakdown reads
    as "the binding constraint" in a defined precedence, not as "every constraint
    that was unhappy"; the docstring says so because the two are easy to confuse.
    """
    # Post-commit first: this arrival was COMMITTED and allocated, and the
    # ground-truth verifier revoked it. That outcome is not visible in
    # `decision.violation` (which is None on the commit path), so checking the
    # pre-commit record first would bin every one of these as `unattributed`.
    revoked = getattr(result, "revoked_by", None)
    if revoked:
        for code, bin_name in _POST_COMMIT_BINS:
            if code in revoked:
                return bin_name
        return "unattributed"

    decision = getattr(result, "decision", None)
    violation = getattr(decision, "violation", None) if decision else None
    if violation is None:
        return "unattributed"
    if getattr(violation, "actor_infeasible", False):
        return "actor_infeasible"
    if getattr(violation, "cross_domain_infeasible", False):
        return "cross_domain_infeasible"
    if getattr(violation, "c5b_violated", False):
        return "c5b_bandwidth"
    if getattr(violation, "c7_violated", False):
        return "c7_delay"
    if getattr(violation, "c9_violated", False):
        return "c9_hops"
    return "unattributed"


def build_report(episode, qos_rejections: int = 0) -> AcceptanceReport:
    """Summarise one `EpisodeResult` under §Y.5.

    Args:
        episode: An `EpisodeResult`.
        qos_rejections: Arrivals the partition accepted and the QoS gate refused,
            if the caller tracks them separately from the violation record.

    Returns:
        An `AcceptanceReport`. Conservation is NOT asserted here, because callers
        that measure over a steady-state window legitimately hold a subset; call
        `check_conservation()` when reporting a whole episode.
    """
    stats = episode.stats
    rejections: dict[str, int] = {b: 0 for b in REJECT_BINS}
    rejections["structural"] = int(getattr(stats, "rejected_structural", 0))
    rejections["qos_gate"] = int(qos_rejections)

    for result in episode.mdo_results:
        if not getattr(result, "admitted", False):
            rejections[_classify(result)] += 1

    return AcceptanceReport(
        admitted=int(stats.admitted),
        offered=int(stats.total_arrivals),
        rejections=rejections,
    )
