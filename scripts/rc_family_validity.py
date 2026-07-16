"""§Q Q.4 family-validity check for the routing-critical family C+_T-_B-_RC.

Pre-registered gate: residual (ceiling - Plain-ColocFB) on the RC family must be
>= 25 pp (materially larger than the existing held-out residuals 3.4-13.7 pp),
else near-optimal FFD on the trap is itself a FINDING — reported, not
regenerated (one-draw rule). Runs the committed one draw: gen_seed=20260715,
seeds 42/43/44 with BW overrides {70,90,110}.

Ceiling = arrivals with SOME placement that places + routes + passes the
verifier (colocation placement first, then exhaustive enumeration). Plain =
arrivals where the colocation-FFD placement itself passes. The gap is exactly
the "admittable but the greedy cut missed it" trap.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))  # for verifier helpers

from orion.baselines.colocation_ffd import colocation_ffd, GreedyConfig
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.substrate.routing_critical import (
    generate_rc_instance,
    rc_slice_factory,
    RC_GEN_SEED,
    RC_BW_OVERRIDES,
)
from five_arm_runner import (
    _check_placement_full,
    _get_feasible_nodes,
    ARRIVAL_RATE,
    SERVICE_RATE,
)

ARRIVALS = 100
SEEDS = [42, 43, 44]
COMBO_CAP = 2000


def _coloc_placement(sr, sub):
    res = colocation_ffd(sub, sr, GreedyConfig())
    if not res.feasible or res.plan is None:
        return None
    try:
        return [res.plan.vnf_placements[v.vnf_id] for v in sr.vnfs]
    except (KeyError, AttributeError):
        return None


def _evaluate(sub, arrival_seed):
    """Return (total, ceiling, plain) over the RC arrival stream."""
    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(sub, ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng,
                        slice_factory=rc_slice_factory)
    ap.generate()

    total = ceiling = plain = 0
    for ev in ap.events:
        if ev.event_type != EventType.ARRIVAL or ev.slice_request is None:
            continue
        sr = ev.slice_request
        total += 1

        place = _coloc_placement(sr, sub)
        plain_ok = place is not None and _check_placement_full(place, sr, sub)
        if plain_ok:
            plain += 1
            ceiling += 1
            continue

        # Plain missed it — does ANY placement work? (the trap check)
        feas = _get_feasible_nodes(sr, sub)
        if any(len(n) == 0 for n in feas):
            continue
        combos = list(itertools.product(*feas))
        if len(combos) > COMBO_CAP:
            rs = np.random.default_rng(hash(sr.request_id) % 2**32)
            idx = rs.choice(len(combos), COMBO_CAP, replace=False)
            combos = [combos[i] for i in idx]
        for combo in combos:
            if _check_placement_full(list(combo), sr, sub):
                ceiling += 1
                break
    return total, ceiling, plain


def main():
    print(f"§Q Q.4 RC family validity — gen_seed={RC_GEN_SEED}, "
          f"seeds={SEEDS}, overrides={RC_BW_OVERRIDES}, arrivals={ARRIVALS}")
    print("-" * 68)
    residuals = []
    plain_focs = []
    for seed, bw in zip(SEEDS, RC_BW_OVERRIDES):
        sub = generate_rc_instance(seed=RC_GEN_SEED + (seed - 42),
                                   inter_domain_bw_override=bw)
        total, ceiling, plain = _evaluate(sub, arrival_seed=seed)
        plain_foc = 100.0 * plain / ceiling if ceiling else float("nan")
        residual = 100.0 - plain_foc if ceiling else float("nan")
        residuals.append(residual)
        plain_focs.append(plain_foc)
        print(f"seed {seed} (bw_override={bw:g}): total={total} ceiling={ceiling} "
              f"plain={plain} | Plain_FoC={plain_foc:.1f}  residual={residual:.1f} pp")

    mean_res = float(np.nanmean(residuals))
    mean_plain = float(np.nanmean(plain_focs))
    print("-" * 68)
    print(f"mean residual (ceiling - Plain) = {mean_res:.1f} pp | mean Plain FoC = {mean_plain:.1f}%")
    # RC-v2 tightened two-sided gate (Q.4 Δ): residual >= 25pp AND Plain FoC >= 25%.
    # Outside the window on EITHER side → STOP (committed now so draw 2 cannot
    # become draw 3): both draws to the paper as the regime characterization,
    # §Q grid runs on the 4 existing families with the envelope claim only.
    ok_residual = mean_res >= 25.0
    ok_plain = mean_plain >= 25.0
    if ok_residual and ok_plain:
        verdict = "PASS — in window (residual>=25pp AND Plain FoC>=25%): RC-v2 validates, proceed"
    elif not ok_plain:
        verdict = ("STOP — Plain FoC < 25% (baseline still degenerate/past its boundary). "
                   "Both draws to paper; §Q grid on 4 existing families, envelope claim only.")
    else:
        verdict = ("STOP — residual < 25pp (Plain near-ceiling, no headroom). "
                   "Both draws to paper; §Q grid on 4 existing families, envelope claim only.")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
