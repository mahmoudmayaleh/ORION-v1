"""Does the surface partial_obs_builder reads actually move as the substrate fills?

The builder picks the all-VNF-feasible domain with the largest residual slack. It
was measured never to switch across 2000 arrivals, but that measurement ran an
UNTRAINED policy, so commitments did not follow m~ and the suggested domain may
simply never have been loaded. Two checks, cheapest first:

  A. Allocate directly and re-read build_domain_summaries. If the residual does
     not move, the observation surface is stale and the builder is blameless.
  B. Replay the arrival stream in follow_prior mode, where the commitment IS m~,
     and log the suggested domain per arrival. If it switches, there is no bug.
"""

import os
import sys
from collections import Counter
from pathlib import Path

# Determinism pin. Behind __main__ only: fired on import it replaces ANY importer
# via os.execv, which under pytest ends the session with no traceback and rc 0.
# Same reasoning as grid_runner.py:61.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grid_runner as G       # noqa: E402
import wp7_runner as W        # noqa: E402
from partial_obs_prior import partial_obs_builder   # noqa: E402
from orion.mdo.observation import build_domain_summaries   # noqa: E402

SCEN, LEVEL, INST, SEED = "conventional", None, None, 42


def check_a(sub):
    print("=== A. does build_domain_summaries track allocation ===")
    before = {s.domain_id: (s.cpu_residual, s.ram_residual)
              for s in build_domain_summaries(sub)}
    for d, v in sorted(before.items()):
        print(f"  D{d} cpu_res={v[0]:.1f} ram_res={v[1]:.1f}")

    target = 2
    took = 0
    for n in sorted(sub.nodes_in_domain(target)):
        attrs = sub.graph.nodes[n]
        attrs["cpu_residual"] -= attrs["cpu_residual"] * 0.9
        attrs["ram_residual"] -= attrs["ram_residual"] * 0.9
        took += 1
    print(f"  drew down 90% of every node in D{target} ({took} nodes)")

    after = {s.domain_id: (s.cpu_residual, s.ram_residual)
             for s in build_domain_summaries(sub)}
    for d in sorted(after):
        db = before[d]
        da = after[d]
        moved = "MOVED" if abs(da[0] - db[0]) > 1e-6 else "unchanged"
        print(f"  D{d} cpu_res {db[0]:.1f} -> {da[0]:.1f}  headroom {db[2]:.3f} -> "
              f"{da[2]:.3f}  [{moved}]")


def check_b(level, instance, seed, arrivals, mode):
    print(f"=== B. suggested domain over the episode, mode={mode} ===")
    G._wire(SCEN, level, instance)
    W.CUSTOM_PLAN_BUILDER = None
    sub = G._substrate_fn(instance)(seed)
    delays = W.build_delays(sub)
    _, coord, *_ = W.build_stack(sub, seed, 3e-3)
    acc, adm, tot, agree, ep = W._eval_episode(
        coord, level, seed, arrivals, delays,
        plan_builder=partial_obs_builder, mode=mode)

    trace = []
    for t in ep.rollout.mdo:
        sug = list(t.info.get("suggested_domains", [])) if getattr(t, "info", None) else []
        if sug:
            trace.append(int(sug[0]))
    n = len(trace)
    print(f"  acceptance={acc:.3f} admitted={adm}/{tot} arrivals_with_mtilde={n}")
    if not n:
        return
    print(f"  domains used overall: {dict(Counter(trace))}")
    chunk = max(1, n // 10)
    for i in range(0, n, chunk):
        seg = trace[i:i + chunk]
        print(f"    arrivals {i:>5}-{i + len(seg) - 1:>5}: {dict(Counter(seg))}")
    switches = sum(1 for a, b in zip(trace, trace[1:]) if a != b)
    print(f"  switches between consecutive arrivals: {switches}")


def main():
    level = LEVEL or G.TRAINING_LEVEL
    instance = INST if INST is not None else G.TRAIN_INSTANCES[0]
    arrivals = G.NUM_ARRIVALS
    G._wire(SCEN, level, instance)
    check_a(G._substrate_fn(instance)(SEED))
    print()
    check_b(level, instance, SEED, arrivals, "follow_prior")
    print()
    check_b(level, instance, SEED, arrivals, "deterministic")
    print("CHECK_DONE")


if __name__ == "__main__":
    main()
