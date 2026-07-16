#!/usr/bin/env python
"""§M M.1 — does M^B's exemplar block degrade Agent B on RC?

Pre-registration: docs/PREREG_M_2026-07-16.md (prereg_M_sha256=d06745d7761b),
committed BEFORE seeds 43/44 existed. Read it before reading any number here.

Two arms, cache-OFF, byte-identical streams, seeds 42/43/44:
  M.1-live  — M^B live (identical to R.2's configuration), instrumented
  M.1-none  — mb=None, otherwise identical

Why M.1-live re-runs rather than reusing R.2-prime's R.2 cells: the per-arrival
retrieval composition (mb_retr_n/pos/neg) did not exist until 2026-07-16, so
R.2-prime's traces cannot answer "retrieved exemplar labels versus the plan's
fate". M.1-live doubles as a determinism check against R.2-prime (identical
config, temp 0): a discrepancy is itself a finding and gets reported.

Pre-named reading (M.1, sign per seed, no compositing): live < none on BOTH 43
and 44 -> proceed to M.2. live >= none on either -> the seed-42 gap is not a
family property; report as a single-seed observation and STOP. No third reading.
Seed 42's comparison was already observed (live 12/100 vs none 20/100) and can
never count as confirmation.

Local-only, no API. ~2h under the single-slot lock.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from q_pilot_runner import run_q_cell  # noqa: E402
from r_local_runner import (  # noqa: E402
    FROZEN_RC, _build_local_agent, _fresh_mb, _load_kb, _mb_composition, prereg_sha256,
)
from orion.provenance import git_provenance, serving_provenance  # noqa: E402
from orion.substrate.routing_critical import RC_FAMILY_SHORT, RC_GEN_SEED, generate_rc_instance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m_probe")

ARMS = {"M.1-live": True, "M.1-none": False}  # arm -> M^B live?


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tag", default="M1")
    args = ap.parse_args()

    prov = git_provenance(serving=serving_provenance(args.port), tag=args.tag,
                          prereg="docs/PREREG_M_2026-07-16.md")
    logger.info("provenance: commit=%s dirty=%s prereg_M=%s",
                prov["git_commit"][:8], prov["git_dirty"], prov["prereg"]["sha256"][:12])
    logger.info("serving: %s", prov["serving"])
    logger.info("§M M.1 — %s seeds=%s arms=%s", RC_FAMILY_SHORT, args.seeds, args.arms)

    agent_b = _build_local_agent(args.port)
    kb = _load_kb()

    out = {"provenance": prov, "tag": args.tag, "family": RC_FAMILY_SHORT,
           "gen_seed": RC_GEN_SEED, "prereg_R_sha256": prereg_sha256(),
           "prereg_M_sha256": prov["prereg"]["sha256"],
           "seeds": args.seeds, "arms": args.arms, "cells": {}}

    for seed in args.seeds:
        fr = FROZEN_RC[seed]
        sub = generate_rc_instance(seed=RC_GEN_SEED + (seed - 42),
                                   inter_domain_bw_override=fr["bw"])
        for arm in args.arms:
            mb = _fresh_mb() if ARMS[arm] else None
            logger.info("### %s seed=%d (M^B %s, cache OFF)", arm, seed,
                        "LIVE" if ARMS[arm] else "OFF")
            t0 = time.time()
            m = run_q_cell(arm, agent_b, kb, mb, None, sub, seed, False)
            m["wall_s"] = round(time.time() - t0, 1)
            if mb is not None:
                m.update(_mb_composition(mb))
            m["foc"] = 100.0 * m["admitted"] / fr["ceiling"] if fr["ceiling"] else float("nan")
            out["cells"][f"{arm}|{seed}"] = m
            logger.info("  -> %s seed=%d admit=%d/%d FoC=%.1f%% wall=%.0fs",
                        arm, seed, m["admitted"], m["total"], m["foc"], m["wall_s"])

    p = Path(f"data/m_probe_results_{args.tag}.json")
    p.write_text(json.dumps(out, indent=2, default=str))
    logger.info("results -> %s", p)

    # ── M.1 readout: outcome only. M.2 (mechanism) is a separate, gated step. ──
    print("\n" + "=" * 78)
    print("§M M.1 — outcome (pre-named: live < none on BOTH 43 and 44 => proceed to M.2)")
    print("=" * 78)
    print(f"  {'seed':>4} {'live':>8} {'none':>8} {'live-none':>10}  reading")
    for seed in args.seeds:
        lv = out["cells"].get(f"M.1-live|{seed}")
        nn = out["cells"].get(f"M.1-none|{seed}")
        if not (lv and nn):
            continue
        d = lv["admitted"] - nn["admitted"]
        note = "prior observation (not confirmation)" if seed == 42 else \
               ("live < none" if d < 0 else "live >= none -> M.1 FAILS on this seed")
        print(f"  {seed:>4} {lv['admitted']:>8} {nn['admitted']:>8} {d:>+10}  {note}")
    print("\n  Seed 42 was observed before §M was written and cannot confirm anything.")
    print("  M.2 runs ONLY if seeds 43 AND 44 both show live < none.")


if __name__ == "__main__":
    main()
