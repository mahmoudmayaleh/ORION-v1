#!/usr/bin/env python3
"""WP8 data pull: Plain-ColocFB (+ RA-ColocFB reference) FoC on the three
result groupings — the C-_T-_B- family, extrap (held-out families), interp
(seen families, unseen instance seed) — averaged across run seeds 42/43/44.

Static arms only: no LLM, no RL, CPU. Reuses the exact instance generation,
ceiling enumeration, and coordinator fill pipeline from five_arm_runner so the
numbers are directly comparable to the main run.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import five_arm_runner as R

ARMS = ["Plain-ColocFB", "RA-ColocFB"]
FAM = {f.short_name: f for f in R.ALL_FAMILIES}


_CEIL_CACHE = {}  # (fname, iseed, run_seed) -> ceiling  (arm-independent, slow)


def _ceiling(fname, sub, iseed, run_seed):
    key = (fname, iseed, run_seed)
    if key not in _CEIL_CACHE:
        _, _CEIL_CACHE[key] = R.compute_ceiling(sub, run_seed)
    return _CEIL_CACHE[key]


def group_focs(specs, arm):
    """Aggregate FoC across (family, iseed) specs, per run seed. The exhaustive
    ceiling is computed once per instance and shared across arms."""
    per_seed = np.zeros(len(R.RUN_SEEDS))
    for fname, iseed in specs:
        sub = R.generate_family_instance(FAM[fname], seed=iseed)
        for si, run_seed in enumerate(R.RUN_SEEDS):
            ceiling = _ceiling(fname, sub, iseed, run_seed)
            adm, _, _ = R.run_instance(sub, run_seed, arm, None, None, None, None, False)
            per_seed[si] += (adm / ceiling if ceiling > 0 else 0.0)
    per_seed /= len(specs)
    return per_seed


def report(label, specs):
    print(f"\n=== {label}  (families/instances: {specs}) ===", flush=True)
    for arm in ARMS:
        ps = group_focs(specs, arm)
        print(f"  {arm:14s}: FoC={100*ps.mean():.1f}% +/- {100*ps.std():.1f}%  "
              f"per-seed={[round(100*x,1) for x in ps]}", flush=True)


if __name__ == "__main__":
    # Row 1: the C-_T-_B- family specifically (its train instance seeds 0,1).
    report("C-_T-_B-", [("C-_T-_B-", s) for s in R.TRAIN_INSTANCE_SEEDS])

    # Row 2: extrap — held-out TEST_FAMILIES at instance seeds 0,1.
    extrap_specs = [(f.short_name, s) for f in R.TEST_FAMILIES
                    for s in R.TEST_INSTANCE_SEEDS]
    report("extrap (held-out families)", extrap_specs)

    # Row 3: interp — seen families at the unseen INTERP instance seed.
    interp_specs = [(fname, R.INTERP_INSTANCE_SEED) for fname in R.WARM_UP_ORDER]
    report("interp (seen families, unseen instance)", interp_specs)

    print("\nDone.")
