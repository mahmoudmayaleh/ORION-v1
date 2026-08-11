"""§R R.1/R.2 — local ORION follow_prior baselines on RC-v2 (2026-07-15).

R.1  ORION-local deployable : Agent B (LLaMA-3-8B) plans, plan-cache ON, follow_prior.
R.2  Local, diverse sampling : same, plan-cache OFF (every arrival re-plans).

Three seeds 42/43/44 (bw sweep 70/90/110), byte-identical 100-arrival RC streams,
per-(seed, approach) cold start (cache + M^B wiped, state hash asserted empty). Reuses
q_pilot_runner.run_q_cell (now emitting a per-arrival trace) so R.1/R.2 are the
SAME deployable stack the pilot measured, just cache ON vs OFF.

Settles (PREREG_AMENDMENT_2026-07-15_R.md):
  R-Primary  : R.1 > Plain (per-seed) in mean AND positive sign all three seeds.
  R-Sampling : |R.2 - R.1| = the cache-thinness measurement (characterization).

Needs the local llama.cpp server on :8000. --mock swaps Agent B for FFD (wiring smoke,
no server). Box-only, minutes-to-hours. Records everything per approach/seed for the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from q_pilot_runner import run_q_cell, _rc_forced  # noqa: F401


def prereg_sha256() -> str:
    """§R provenance: hash of the §R amendment (this run's pre-registration)."""
    p = Path(__file__).resolve().parent.parent / "docs" / "PREREG_AMENDMENT_2026-07-15_R.md"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
from orion.llm.plan_cache import PlanCache
from orion.provenance import git_provenance, serving_provenance  # noqa: E402
from orion.substrate.routing_critical import (
    generate_rc_instance, RC_FAMILY_SHORT, RC_GEN_SEED, RC_BW_OVERRIDES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r_local")

# ── FROZEN RC-v2 validity draw (results/rc_family_validity_RESULT.md) ──────────
# §R rule: Plain-ColocFB and the ceiling are NEVER re-run; cite these verbatim.
# R.1/R.2 build the SAME (substrate, arrival stream) as the validity draw
# (generate_rc_instance(RC_GEN_SEED+(seed-42), bw) + arrival_seed=seed +
# rc_slice_factory), so admits are directly comparable against these ceilings.
# Re-running _evaluate drifts across machines (36.4 vs 37.9 vs frozen 37.1) — the
# reason the prereg freezes it.
FROZEN_RC = {
    42: {"bw": 70.0,  "total": 100, "ceiling": 97,  "plain_admits": 36, "plain_foc": 37.1},
    43: {"bw": 90.0,  "total": 100, "ceiling": 100, "plain_admits": 37, "plain_foc": 37.0},
    44: {"bw": 110.0, "total": 100, "ceiling": 100, "plain_admits": 47, "plain_foc": 47.0},
}
BW_FOR_SEED = {s: FROZEN_RC[s]["bw"] for s in FROZEN_RC}
PLAIN_FOC_RC_MEAN = 40.4        # RC-v2 validity draw (frozen reference, context)
MEMORY_CAPACITY_K = 50


def _build_local_agent(port: int):
    from orion.llm.llm_backend import LLMBackend, LLMConfig
    from orion.llm.agent_b import AgentB
    cfg = LLMConfig(base_url=f"http://localhost:{port}/v1", api_key="EMPTY",
                    model="default", temperature=0.0, max_tokens=2048)
    return AgentB(LLMBackend(cfg))


def _load_kb():
    from orion.llm.semantic_memory import SemanticMemory
    kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
    if kb_path.exists():
        kb = SemanticMemory.from_json(kb_path)
        logger.info("K^B loaded: %d entries", len(kb.entries))
        return kb
    return None


def _fresh_mb():
    from orion.llm.episodic_memory import EpisodicMemory
    from orion.retrieval import RetrievalConfig, RetrievalMode
    return EpisodicMemory(
        config=RetrievalConfig(mode=RetrievalMode.NO_RERANK, apply_recency=True, k_final=3),
        max_entries=MEMORY_CAPACITY_K, write_policy="selective", evict_policy="importance")


def _mb_composition(mb):
    """Composition of M^B at cell end.

    The success/failure label lives in `entry.tags["label"]` -- a LIST, e.g.
    {"label": ["success"]} (see EpisodicMemory.write). MemoryEntry has no
    `admitted` and no `success` attribute, so the previous
    `getattr(e, "admitted", getattr(e, "success", False))` fell through to the
    False default for EVERY entry and reported mb_pos=0 on every run ever
    recorded -- including R.2|42's "mb=50(+0/-50)", which read as "M^B learned
    only from failures" when in fact 84 admissions had written positives and 50
    was merely the capacity cap. A telemetry field that cannot express one of its
    two states is not a measurement, so pin it with a test rather than an idiom:
    tests/test_mb_composition.py fails if this silently returns to all-zero.
    """
    entries = list(getattr(mb, "_entries", []))
    pos = sum(1 for e in entries
              if ((getattr(e, "tags", None) or {}).get("label") or [None])[0] == "success")
    return {"mb_entries": len(entries), "mb_pos": pos, "mb_neg": len(entries) - pos}


def _cell(approach, agent_b, kb, substrate, seed, cache_on, mock):
    mb = _fresh_mb()
    plan_cache = PlanCache(capacity=64) if cache_on else None
    t0 = time.time()
    m = run_q_cell(approach, agent_b, kb, mb, plan_cache, substrate, seed, mock)
    m["wall_s"] = round(time.time() - t0, 1)
    m.update(_mb_composition(mb))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--approaches", nargs="+", default=["R.1", "R.2"], choices=["R.1", "R.2"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mock", action="store_true", help="FFD stand-in for Agent B (no server)")
    ap.add_argument("--tag", default="R12")
    args = ap.parse_args()

    # Provenance guard (hard): refuses on untracked code under scripts/ or src/.
    # Runs BEFORE any LLM call so an unprovenanced run dies in seconds, not hours.
    _prov = git_provenance(serving=serving_provenance(args.port), tag=args.tag,
                           prereg="docs/PREREG_AMENDMENT_2026-07-15_R.md")
    logger.info("provenance: commit=%s dirty=%s serving=%s",
                _prov["git_commit"][:8], _prov["git_dirty"], _prov["serving"])

    logger.info("§R R.1/R.2 — %s seeds=%s mock=%s | prereg_sha256=%s",
                RC_FAMILY_SHORT, args.seeds, args.mock, prereg_sha256()[:12])

    kb = _load_kb()
    agent_b = None if args.mock else _build_local_agent(args.port)

    # cache_on per approach: R.1 = cache ON, R.2 = cache OFF (diverse sampling).
    approach_cache = {"R.1": True, "R.2": False}

    out = {"provenance": _prov, "tag": args.tag, "family": RC_FAMILY_SHORT, "gen_seed": RC_GEN_SEED,
           "prereg_sha256": prereg_sha256(), "seeds": args.seeds, "approaches": args.approaches,
           "mock": args.mock, "plain_foc_ref": PLAIN_FOC_RC_MEAN, "refs": {}, "cells": {}}

    # Per-seed reference: FROZEN ceiling + Plain (validity draw, never re-run).
    for seed in args.seeds:
        fr = FROZEN_RC[seed]
        out["refs"][str(seed)] = {"bw": fr["bw"], "total": fr["total"],
                                  "ceiling": fr["ceiling"], "plain": fr["plain_admits"],
                                  "plain_foc": fr["plain_foc"], "frozen": True}
        sub = generate_rc_instance(seed=RC_GEN_SEED + (seed - 42),
                                   inter_domain_bw_override=fr["bw"])
        logger.info("seed %d (bw=%g): FROZEN ceiling=%d Plain=%d (Plain FoC=%.1f%%)",
                    seed, fr["bw"], fr["ceiling"], fr["plain_admits"], fr["plain_foc"])

        for approach in args.approaches:
            logger.info("### %s seed=%d cache=%s", approach, seed, approach_cache[approach])
            m = _cell(approach, agent_b, kb, sub, seed, approach_cache[approach], args.mock)
            ceiling_s = out["refs"][str(seed)]["ceiling"]
            m["foc"] = 100.0 * m["admitted"] / ceiling_s if ceiling_s else float("nan")
            m["struct_reject_rate"] = 100.0 * m["structural"] / m["total"] if m["total"] else 0.0
            m["schema_fail_rate"] = 100.0 * m["schema_fail"] / m["total"] if m["total"] else 0.0
            out["cells"][f"{approach}|{seed}"] = m
            logger.info("  -> %s seed=%d FoC=%.1f%% admit=%d/%d cache_hit=%d wall=%.0fs",
                        approach, seed, m["foc"], m["admitted"], m["total"], m["cache_hit"], m["wall_s"])

    out_path = Path(f"data/r_local_results_{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("results -> %s", out_path)
    _readout(out)


def _readout(out):
    cells, refs = out["cells"], out["refs"]
    seeds = out["seeds"]
    print("\n" + "=" * 78)
    print(f"§R R.1/R.2 READOUT — {out['family']}  (order: validity -> reject -> FoC)")
    print("=" * 78)

    print("\n[1] VALIDITY (void trigger = schema/api-fail > 10%)")
    for k, m in cells.items():
        void = " !!VOID" if m["schema_fail_rate"] > 10.0 else ""
        print(f"  {k:10s} schema_fail={m['schema_fail']} ({m['schema_fail_rate']:.1f}%)  "
              f"api_fail={m.get('api_fail',0)}  cache_hit={m['cache_hit']} miss={m['cache_miss']}"
              f"  mb={m.get('mb_entries',0)}(+{m.get('mb_pos',0)}/-{m.get('mb_neg',0)}){void}")

    print("\n[2] REJECT TAXONOMY (per cell)")
    all_reasons = sorted({r for m in cells.values() for r in m["reasons"]})
    hdr = "  " + f"{'cell':10s}" + "".join(f"{r[:11]:>13s}" for r in all_reasons) + f"{'struct':>9s}"
    print(hdr)
    for k, m in cells.items():
        row = "  " + f"{k:10s}" + "".join(f"{m['reasons'].get(r,0):>13d}" for r in all_reasons)
        print(row + f"{m['structural']:>9d}")

    print("\n[3] FoC (admitted / ceiling) vs per-seed Plain")
    for approach in out["approaches"] if "approaches" in out else ["R.1", "R.2"]:
        vals, signs = [], []
        for s in seeds:
            m = cells.get(f"{approach}|{s}")
            if not m:
                continue
            pf = refs[str(s)]["plain_foc"]
            d = m["foc"] - pf
            vals.append(m["foc"]); signs.append(d)
            print(f"  {approach} seed {s}: FoC={m['foc']:.1f}%  Plain={pf:.1f}%  ({d:+.1f}pp)  "
                  f"admit={m['admitted']}/{m['total']}")
        if vals:
            mean = float(np.mean(vals))
            all_pos = all(x > 0 for x in signs)
            print(f"  {approach} MEAN FoC={mean:.1f}%  (Plain-mean {out['plain_foc_ref']}%)  "
                  f"all-seeds-positive={all_pos}")

    # R-Primary verdict (R.1), R-Sampling (|R.2-R.1|)
    print("\n[R-Primary] R.1 > Plain per-seed, mean AND all-positive:")
    r1_pos, r1_vals = [], []
    for s in seeds:
        m = cells.get(f"R.1|{s}")
        if m:
            d = m["foc"] - refs[str(s)]["plain_foc"]
            r1_pos.append(d > 0); r1_vals.append(m["foc"])
    if r1_vals:
        verdict = "PASS" if (all(r1_pos) and np.mean(r1_vals) > out["plain_foc_ref"]) else "FAIL"
        print(f"  R.1 mean={np.mean(r1_vals):.1f}%  all-positive={all(r1_pos)}  -> {verdict}")
    print("\n[R-Sampling] |R.2 - R.1| per seed (cache-thinness):")
    for s in seeds:
        a, b = cells.get(f"R.1|{s}"), cells.get(f"R.2|{s}")
        if a and b:
            print(f"  seed {s}: R.1={a['foc']:.1f}%  R.2={b['foc']:.1f}%  |d|={abs(b['foc']-a['foc']):.1f}pp")


if __name__ == "__main__":
    main()
