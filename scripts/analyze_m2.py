#!/usr/bin/env python
"""§M M.2 — mechanism: do retrieved M^B exemplars degrade the plan, or is it load?

Pre-registration: docs/PREREG_M_2026-07-16.md (prereg_M_sha256=d06745d7761b).
Gated behind M.1 passing on seeds 43 AND 44 (both live < none). Offline analysis
of M.1-live traces; no new run.

M.2a  retrieved label mix per arrival.
M.2b  admit rate by retrieval composition: empty (n=0) / majority-positive /
      majority-negative. H-M predicts monotone: empty >= maj-pos > maj-neg.
M.2c  THE CONFOUND, pre-named and binding. M^B fills as the substrate fills, so
      composition correlates with stream position by construction, and load rises
      with position. A raw M.2b gradient is EXPECTED under the null. Two controls
      are required before M.2b may be read as support:
        c1  within-position bands: if the effect vanishes inside bands, load
            explains it and H-M FAILS.
        c2  cache-ON contrast at matched positions: cache-ON stops consulting M^B
            after warm-up while facing the same rising load. If cache-ON's late
            admit rate holds where cache-OFF's collapses, the difference is the
            prompt. If both collapse together, it is load.
      Either control killing it means H-M is reported REFUTED.
"""
from __future__ import annotations

import json
from pathlib import Path

BANDS = [(0, 19), (20, 49), (50, 99)]


def klass(t):
    n = t.get("mb_retr_n")
    if n is None:
        return None
    if n == 0:
        return "empty"
    pos, neg = t.get("mb_retr_pos") or 0, t.get("mb_retr_neg") or 0
    if pos > neg:
        return "maj_pos"
    if neg > pos:
        return "maj_neg"
    return "tie"


def rate(ts):
    return (100.0 * sum(1 for t in ts if t.get("admitted")) / len(ts)) if ts else float("nan")


def main():
    m = json.loads(Path("data/m_probe_results_M1.json").read_text())
    r = json.loads(Path("data/r_local_results_R2PRIME.json").read_text())

    print("=" * 84)
    print("§M M.2 — mechanism (gated on M.1 passing seeds 43 AND 44)")
    print("=" * 84)

    # ---- M.2a: label mix ----
    print("\n[M.2a] retrieved label mix (M.1-live, post-warm-up = arrivals 20-99)")
    print(f"  {'seed':>4} {'empty':>7} {'maj_pos':>8} {'maj_neg':>8} {'tie':>5}   majority-negative?")
    for s in (42, 43, 44):
        tr = m["cells"][f"M.1-live|{s}"]["trace"][20:]
        ks = [klass(t) for t in tr]
        c = {k: ks.count(k) for k in ("empty", "maj_pos", "maj_neg", "tie")}
        verdict = "YES" if c["maj_neg"] > (c["maj_pos"] + c["empty"]) else "no"
        print(f"  {s:>4} {c['empty']:>7} {c['maj_pos']:>8} {c['maj_neg']:>8} {c['tie']:>5}   {verdict}")

    # ---- M.2b: raw dose-response ----
    print("\n[M.2b] admit rate by retrieval composition (M.1-live, all arrivals)")
    print(f"  {'seed':>4} {'empty':>14} {'maj_pos':>14} {'maj_neg':>14}   monotone?")
    for s in (42, 43, 44):
        tr = m["cells"][f"M.1-live|{s}"]["trace"]
        g = {k: [t for t in tr if klass(t) == k] for k in ("empty", "maj_pos", "maj_neg")}
        e, p, n = rate(g["empty"]), rate(g["maj_pos"]), rate(g["maj_neg"])
        mono = "yes" if (e >= p > n or (p != p and e > n)) else "no"
        def f(v, ts):
            return f"{v:5.1f}% (n={len(ts):>2})" if ts else "   --  (n= 0)"
        print(f"  {s:>4} {f(e,g['empty']):>14} {f(p,g['maj_pos']):>14} {f(n,g['maj_neg']):>14}   {mono}")

    # ---- M.2c1: within-position bands ----
    print("\n[M.2c-1] CONTROL — admit rate by composition WITHIN position bands")
    print("          (if the effect vanishes inside bands, load explains it => H-M FAILS)")
    for s in (42, 43, 44):
        tr = m["cells"][f"M.1-live|{s}"]["trace"]
        print(f"  seed {s}:")
        for lo, hi in BANDS:
            w = [(i, t) for i, t in enumerate(tr) if lo <= i <= hi]
            g = {k: [t for _, t in w if klass(t) == k] for k in ("empty", "maj_pos", "maj_neg")}
            parts = []
            for k in ("empty", "maj_pos", "maj_neg"):
                parts.append(f"{k}={rate(g[k]):5.1f}%(n={len(g[k]):>2})" if g[k] else f"{k}=  --  (n= 0)")
            print(f"    arrivals {lo:>2}-{hi:<2}  " + "  ".join(parts))

    # ---- M.2c2: cache-ON contrast at matched positions ----
    print("\n[M.2c-2] CONTROL — cache-ON (R.2-prime R.1, stops consulting M^B) vs")
    print("          cache-OFF (M.1-live) admit rate at MATCHED positions")
    print("          (both collapse together => load; ON holds where OFF collapses => prompt)")
    print(f"  {'seed':>4} {'band':>10} {'cache-ON':>10} {'cache-OFF':>10}   reading")
    for s in (42, 43, 44):
        on = r["cells"][f"R.1|{s}"]["trace"]
        off = m["cells"][f"M.1-live|{s}"]["trace"]
        for lo, hi in BANDS:
            a = rate([t for i, t in enumerate(on) if lo <= i <= hi])
            b = rate([t for i, t in enumerate(off) if lo <= i <= hi])
            note = "both low" if (a < 15 and b < 15) else ("ON holds, OFF collapses" if a - b > 25 else "")
            print(f"  {s:>4} {f'{lo}-{hi}':>10} {a:>9.1f}% {b:>9.1f}%   {note}")

    print("\n" + "=" * 84)
    print("Read M.2b ONLY through M.2c. Either control killing it => H-M REFUTED.")


if __name__ == "__main__":
    main()
