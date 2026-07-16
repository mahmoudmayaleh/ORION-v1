#!/usr/bin/env python3
"""§O.9 overhead check (pre-merge discipline, same as O.8):

Measures the instrumentation's own cost on a synthetic loop:
  - profiled() with an ACTIVE collector (the real per-event cost)
  - profiled() with NO collector (the always-safe no-op cost)
  - ProfileCollector.sample_memory() per call (round-boundary sampling)
Reference: the 6.5 ms MDO decision — instrumentation must stay negligible.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from orion.profiling import ProfileCollector, profiled, set_collector

N = 100_000


def bench(label, fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    per = dt / n
    print(f"{label:38s} {per * 1e6:8.2f} us/event  "
          f"({100.0 * per / 6.5e-3:.4f}% of a 6.5 ms MDO decision)")
    return per


def with_collector():
    with profiled("synthetic", {"k": 1}):
        pass


def no_collector():
    with profiled("synthetic", {"k": 1}):
        pass


coll = ProfileCollector(sampler=None, label="overhead-check")
set_collector(coll)
bench("profiled() ACTIVE collector", with_collector, N)
set_collector(None)
bench("profiled() no collector (no-op)", no_collector, N)

coll2 = ProfileCollector(sampler=None, label="mem")
bench("sample_memory() per call", coll2.sample_memory, 10_000)

print(f"\nevents recorded: {len(coll.events)} (sanity: == {N})")
