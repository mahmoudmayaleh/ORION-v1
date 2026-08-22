"""C10 — chain-order contiguity, enforced on every approach (2026-08-21).

THE RULE. Read the committed partition along the service chain. Each domain must
occupy exactly ONE maximal run: (A, A, B) and (A, B, B) are admissible, (A, B, A)
is not. Equivalently, the number of runs equals the number of distinct domains.

WHY IT IS A CONSTRAINT AND NOT A PREFERENCE. `MDOCoordinator._build_fragments`
calls a flow edge cross-domain iff its two endpoints landed in different domains,
so a partition with r runs pays r-1 inter-domain traversals, and r is minimal
exactly when no domain is re-entered. A partition that leaves a domain and comes
back buys nothing: it pays two boundary crossings where one placement of the same
multiset pays one, and each crossing is charged again by C5b (inter-domain
bandwidth), C7 (propagation and sojourn) and C9 (the hop limit). In SFC terms it
is a service chain that hands off, hands back, and hands off again.

Until now NO layer above the domain actors read the chain at all.
`partial_obs_builder` assigned each VNF independently, `MDOPolicy` is per-VNF
factored, and Agent B's schema pins one domain per VNF with no cross-VNF term, so
(A, B, A) was representable in all three and cost nothing at the point of choice.
The check is request-side (the chain order is the request's own VNF order), so it
is observation-legal for every approach, including the partial-observability ones.

MODES, via ORION_CHAIN_ORDER:

    "reject"  DEFAULT. A non-contiguous partition is refused before the actors are
              dispatched and binned as `chain_order`. This is the uniform test:
              every approach is held to the same rule and the ones that cannot
              express it lose the arrivals they scatter.
    "off"     the pre-2026-08-21 behaviour, kept so an ablation can measure what
              the constraint costs each approach rather than asserting it.

There is deliberately no coordinator-side "repair" mode. Moving a VNF to restore
contiguity would move it into a domain that may hold no node it is permitted on,
which the commit-path frame check asserts against; repair belongs where
admissibility is known, i.e. in the plan builders (`ORION_PARTIAL_CONTIG`).
"""

from __future__ import annotations

import os


def _mode() -> str:
    """Read the mode at call time so a runner can flip it between cells."""
    m = os.environ.get("ORION_CHAIN_ORDER", "reject")
    if m not in ("reject", "off"):
        raise ValueError(
            f"unknown ORION_CHAIN_ORDER {m!r}; expected one of reject|off")
    return m


def enabled() -> bool:
    return _mode() != "off"


def run_count(domains) -> int:
    """Number of maximal runs of equal domain along the chain."""
    return sum(1 for i, d in enumerate(domains)
               if i == 0 or domains[i - 1] != d)


def is_contiguous(domains) -> bool:
    """Does each domain occupy exactly one maximal run along the chain?"""
    return run_count(domains) == len(set(domains))


def violates(domains) -> bool:
    """Is this partition refused by C10 as currently configured?"""
    return enabled() and not is_contiguous(domains)
