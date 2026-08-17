"""How much information is in m~ at all, per plan source.

KLS8 and KLS8b both voided with target_modal = 0.9975618251480324, identical to
every printed digit across two arms that share nothing but the eval episode,
because it is a property of m~ and not of the run: 14 of 857 slots name a domain
other than the modal one. `partial_obs_builder` colocates by construction, so
that number says nothing about Agent B, which is the teacher the paper's claim
actually rests on. This measures the same quantity for both sources on the same
episode, so the comparison is like for like.

No training. The modal concentration of m~ depends only on `suggested_domains`,
not on what the policy does with them, so an untrained policy is sufficient and
the cost is one eval episode per source. The Agent B source runs with M^B off,
which is the Memory-off arm: if the knowledge-only planner is already degenerate,
no retrieval policy over episodic records can un-degenerate it.
"""

import argparse
import json
import logging
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mtilde_support")


def build_agent_b_builder(port):
    """Agent B's m~, wired exactly as wp7_runner wires it for an LLM approach.

    mb=None is the Memory-off arm. The plan cache is ON because it is pipeline,
    not an ablation.
    """
    from orion.llm.llm_backend import LLMBackend, LLMConfig
    from orion.llm.agent_b import AgentB
    from orion.llm.semantic_memory import SemanticMemory
    from orion.llm.plan_cache import PlanCache

    cfg = LLMConfig(base_url=f"http://localhost:{port}/v1", api_key="EMPTY",
                    model="default", temperature=0.05, max_tokens=2048)
    agent_b = AgentB(LLMBackend(cfg))
    kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
    kb = SemanticMemory.from_json(kb_path) if kb_path.exists() else None
    log.info("K^B: %s", f"{len(kb.entries)} entries" if kb else "NOT FOUND")

    pb = W.make_llm_plan_builder(agent_b, kb, lambda: None)
    if W.TIER_FILTER_LLM_PLANS:
        pb = W.tier_filtered(pb)
    W.PLAN_CACHE_STATS.clear()
    return W._cached_plan_builder(pb, PlanCache(capacity=W.PLAN_CACHE_CAPACITY),
                                  stats=W.PLAN_CACHE_STATS)


def measure(source, pb, scenario, level, instance, seed, arrivals, mode):
    G._wire(scenario, level, instance)
    W.CUSTOM_PLAN_BUILDER = None
    sub = G._substrate_fn(instance)(seed)
    delays = W.build_delays(sub)
    _, coord, *_ = W.build_stack(sub, seed, 3e-3)

    acc, adm, tot, agree, ep = W._eval_episode(
        coord, level, seed, arrivals, delays, plan_builder=pb, mode=mode)

    slots = Counter()          # m~ domain -> slot count
    distinct = Counter()       # distinct domains in one chain -> arrivals
    n_arrivals = n_split = 0
    for t in ep.rollout.mdo:
        sug = list(t.info.get("suggested_domains", [])) if getattr(t, "info", None) else []
        if not sug:
            continue
        n_arrivals += 1
        sug = [int(d) for d in sug]
        slots.update(sug)
        k = len(set(sug))
        distinct[k] += 1
        n_split += int(k > 1)

    total = sum(slots.values())
    out = {
        "source": source, "mode": mode,
        "acceptance": acc, "admitted": adm, "offered": tot,
        "n_arrivals_with_mtilde": n_arrivals,
        "n_slots": total,
        # The number both KLS8 arms voided on.
        "target_modal_frac": (max(slots.values()) / total) if total else None,
        "off_modal_slots": (total - max(slots.values())) if total else 0,
        "n_distinct_domains_used": len(slots),
        "domain_share": {str(d): slots[d] / total for d in sorted(slots)} if total else {},
        # The quantity that actually decides whether a partition prior can act:
        # a chain kept in one domain carries no partition information.
        "split_arrival_frac": (n_split / n_arrivals) if n_arrivals else None,
        "distinct_domains_hist": {str(k): distinct[k] for k in sorted(distinct)},
    }
    if source == "agent_b":
        out["plan_cache_stats"] = dict(W.PLAN_CACHE_STATS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="conventional")
    ap.add_argument("--level", default=None)
    ap.add_argument("--instance", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arrivals", type=int, default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sources", nargs="+", default=["partial_obs", "agent_b"])
    ap.add_argument("--mode", default="deterministic",
                    choices=["deterministic", "follow_prior"],
                    help="follow_prior closes the loop: the commitment IS m~, so the "
                         "suggested domain actually gets loaded")
    ap.add_argument("--out", default="results/wp7/mtilde_support.json")
    args = ap.parse_args()

    level = args.level or G.TRAINING_LEVEL
    instance = args.instance if args.instance is not None else G.TRAIN_INSTANCES[0]
    arrivals = args.arrivals if args.arrivals is not None else G.NUM_ARRIVALS
    log.info("level=%s instance=%s seed=%s arrivals=%s sources=%s",
             level, instance, args.seed, arrivals, args.sources)

    results = []
    for source in args.sources:
        log.info("=== %s ===", source)
        pb = (partial_obs_builder if source == "partial_obs"
              else build_agent_b_builder(args.port))
        r = measure(source, pb, args.scenario, level, instance, args.seed,
                    arrivals, args.mode)
        results.append(r)
        log.info("%s", json.dumps(r, indent=2, sort_keys=True))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"level": level, "instance": instance,
                               "seed": args.seed, "arrivals": arrivals,
                               "mode": args.mode,
                               "results": results}, indent=2))
    log.info("wrote %s", out)

    print("\nMTILDE SUPPORT")
    print(f"{'source':<14}{'slots':>8}{'modal':>10}{'off-modal':>11}"
          f"{'domains':>9}{'split':>9}")
    for r in results:
        print(f"{r['source']:<14}{r['n_slots']:>8}"
              f"{(r['target_modal_frac'] or 0):>10.4f}{r['off_modal_slots']:>11}"
              f"{r['n_distinct_domains_used']:>9}"
              f"{(r['split_arrival_frac'] or 0):>9.3f}")
    print("MTILDE_SUPPORT_DONE")


if __name__ == "__main__":
    main()
