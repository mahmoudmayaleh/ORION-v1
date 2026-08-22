"""Per-cell metrics beyond acceptance (2026-08-21).

Acceptance and the rejection taxonomy say WHETHER a slice was admitted and which
constraint refused the rest. They say nothing about the three questions a
networking reviewer asks next, and until now nothing in the tree answered them:

  structure  Was the multi-domain problem exercised at all? A system whose chains
             all land in one domain is a single-domain placer with extra steps, so
             the split rate, the number of domains per chain and the balance of
             load across domains are first-class results rather than diagnostics.
  qos        Were the admitted slices comfortable or on the edge? Two approaches
             with equal acceptance are not equal if one admits at 30% of the delay
             budget and the other at 95%: the second is one arrival away from the
             post-commit bin.
  timeseries Is the number a steady state or a transient? The curriculum declares
             a 400-arrival transient and the readout has never separated it.

Everything here is derived from records the episode already produces, so it adds
no decisions and cannot change an outcome. It is attached under `cost["structure"]`
etc. rather than through new parameters, so every caller that already passes
`cost_out` gets it without a signature change.
"""

from __future__ import annotations

import math


def _jain(values) -> float | None:
    """Jain's fairness index over per-domain load, in [1/M, 1].

    1.0 means every domain carries the same share; 1/M means one domain carries
    everything. Reported because "multi-domain" is a claim about spread, and a
    mean domain count of 2 is compatible with one domain doing all the work.
    """
    v = [float(x) for x in values]
    if not v or sum(v) <= 0:
        return None
    n = len(v)
    return round((sum(v) ** 2) / (n * sum(x * x for x in v)), 4)


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return round(float(sorted_vals[i]), 4)


def partition_structure(partitions, domain_ids=None) -> dict:
    """Shape of the committed partitions.

    Args:
        partitions: one list of per-VNF domain ids per ADMITTED arrival.
        domain_ids: the full domain set, so a domain that received nothing is
            still counted in the fairness index rather than silently dropped.
    """
    partitions = [list(p) for p in partitions if p]
    if not partitions:
        return {"n_accepted": 0}

    def runs(p):
        return sum(1 for i, d in enumerate(p) if i == 0 or p[i - 1] != d)

    distinct = [len(set(p)) for p in partitions]
    load: dict[int, int] = {int(d): 0 for d in (domain_ids or [])}
    for p in partitions:
        for d in p:
            load[int(d)] = load.get(int(d), 0) + 1

    hist: dict[str, int] = {}
    for n in distinct:
        hist[str(n)] = hist.get(str(n), 0) + 1

    n = len(partitions)
    return {
        "n_accepted": n,
        "split_rate": round(sum(1 for d in distinct if d > 1) / n, 4),
        "domains_per_chain_mean": round(sum(distinct) / n, 3),
        "domains_per_chain_max": max(distinct),
        "domains_per_chain_hist": hist,
        "runs_mean": round(sum(runs(p) for p in partitions) / n, 3),
        "noncontiguous_accepted": sum(1 for p in partitions
                                      if runs(p) != len(set(p))),
        "vnfs_per_domain": {str(k): v for k, v in sorted(load.items())},
        "domain_load_jain": _jain(load.values()),
    }


def qos_margin(pairs) -> dict:
    """How close the admitted slices ran to their delay budget.

    Args:
        pairs: (realised_e2e_ms, budget_ms) per admitted arrival. Records with a
            non-positive or non-finite budget are dropped rather than clamped, so
            a broken budget shows up as a smaller n and not as a flattering ratio.
    """
    ok = [(float(e), float(b)) for e, b in pairs
          if b and b > 0 and math.isfinite(e) and math.isfinite(b)]
    if not ok:
        return {"n": 0}
    ratios = sorted(e / b for e, b in ok)
    delays = sorted(e for e, _ in ok)
    return {
        "n": len(ok),
        "e2e_ms_mean": round(sum(delays) / len(delays), 4),
        "e2e_ms_p95": _pct(delays, 0.95),
        "budget_ratio_mean": round(sum(ratios) / len(ratios), 4),
        "budget_ratio_p95": _pct(ratios, 0.95),
        # Admitted above budget: the coordinator's own estimate said it fitted and
        # the ground-truth model disagreed. Should be ~0 once the verifier has run;
        # anything else means an admission the verifier did not revoke.
        "over_budget_frac": round(
            sum(1 for r in ratios if r > 1.0) / len(ratios), 4),
    }


def acceptance_windows(outcomes, window: int = 200) -> dict:
    """Acceptance per window of arrivals, in arrival order.

    Separates the declared transient from the steady state, and shows whether an
    approach degrades as the substrate fills rather than reporting one number over
    a trajectory that was never stationary.
    """
    outcomes = [bool(x) for x in outcomes]
    if not outcomes:
        return {"window": window, "acceptance": []}
    series = []
    for i in range(0, len(outcomes), window):
        chunk = outcomes[i:i + window]
        if chunk:
            series.append(round(sum(chunk) / len(chunk), 4))
    tail = series[len(series) // 2:] or series
    return {
        "window": window,
        "acceptance": series,
        "steady_state_mean": round(sum(tail) / len(tail), 4),
        "first_window": series[0],
        "last_window": series[-1],
    }


def from_episode(episode, substrate, slice_by_id=None) -> dict:
    """All three blocks from one `EpisodeResult`, for the coordinator approaches.

    Instrumentation contract: this must never raise into an evaluation. The caller
    wraps it, and every field it cannot compute is reported absent rather than
    guessed.
    """
    results = list(getattr(episode, "mdo_results", []) or [])
    partitions, pairs, outcomes = [], [], []
    for res in results:
        admitted = bool(getattr(res, "admitted", False))
        outcomes.append(admitted)
        if not admitted:
            continue
        part = list(getattr(res, "partition", []) or [])
        if part:
            partitions.append(part)
        sr = (slice_by_id or {}).get(getattr(res, "request_id", None))
        budget = getattr(getattr(sr, "qos", None), "max_e2e_delay", None)
        e2e = getattr(res, "e2e_delay", None)
        if budget and e2e is not None:
            pairs.append((e2e, budget))

    domain_ids = None
    try:
        domain_ids = sorted({d["domain_id"]
                             for _, d in substrate.graph.nodes(data=True)})
    except Exception:  # noqa: BLE001 -- instrumentation only
        domain_ids = None

    return {
        "structure": partition_structure(partitions, domain_ids),
        "qos": qos_margin(pairs),
        "timeseries": acceptance_windows(outcomes),
    }
