"""Where the 6.4 s per Agent B call goes: prompt tokens, completion tokens, latency.

A trained LLM cell costs ~11 min per round in steady state, entirely inside model
calls, which puts the full grid at ~350 h. Single-stream decode is output-bound, so
before touching the server config it is worth knowing what the model is actually
being asked to read and write. max_tokens is 2048 for a plan that is a short JSON
object under a pinned schema; if completions land near that ceiling, capping it is
the cheapest speed-up available.
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grid_runner as G       # noqa: E402
import wp7_runner as W        # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("llm_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="calls to time")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="results/wp7/llm_token_profile.json")
    args = ap.parse_args()

    from orion.llm.llm_backend import LLMBackend, LLMConfig
    from orion.llm.agent_b import AgentB
    from orion.llm.semantic_memory import SemanticMemory

    level = G.TRAINING_LEVEL
    instance = G.TRAIN_INSTANCES[0]
    G._wire("conventional", level, instance)
    sub = G._substrate_fn(instance)(42)

    cfg = LLMConfig(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY",
                    model="default", temperature=0.05, max_tokens=2048)
    backend = LLMBackend(cfg)
    agent_b = AgentB(backend)
    kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
    kb = SemanticMemory.from_json(kb_path) if kb_path.exists() else None

    # Uncached builder: every call must hit the model, otherwise this measures the
    # cache and not the model.
    pb = W.make_llm_plan_builder(agent_b, kb, lambda: None)

    import numpy as np
    rng = np.random.default_rng(42)
    ap_proc = W._make_ap(sub, 200, rng)
    ap_proc.generate()
    from orion.sim.arrival_process import EventType
    reqs = [ev.slice_request for ev in ap_proc.events
            if ev.event_type == EventType.ARRIVAL and ev.slice_request is not None]

    rows = []
    for i, sr in enumerate(reqs[:args.n]):
        t0 = time.time()
        plan = pb(sr, sub)
        dt = time.time() - t0
        rows.append({
            "i": i, "k_vnfs": len(sr.vnfs), "seconds": round(dt, 2),
            "prompt_tokens": backend.last_prompt_tokens,
            "completion_tokens": backend.last_completion_tokens,
            "plan": plan is not None,
        })
        log.info("call %2d  k=%d  %5.2fs  prompt=%s  completion=%s  plan=%s",
                 i, len(sr.vnfs), dt, backend.last_prompt_tokens,
                 backend.last_completion_tokens, plan is not None)

    def col(key):
        return [r[key] for r in rows if r[key] is not None]

    summary = {}
    for key in ("seconds", "prompt_tokens", "completion_tokens"):
        v = col(key)
        if v:
            summary[key] = {"mean": round(statistics.mean(v), 1),
                            "median": round(statistics.median(v), 1),
                            "min": min(v), "max": max(v)}
    comp = col("completion_tokens")
    if comp:
        summary["tokens_per_second_decode"] = round(
            sum(comp) / sum(col("seconds")), 1)
        summary["at_max_tokens"] = sum(1 for c in comp if c >= 2040)
    summary["max_tokens_setting"] = cfg.max_tokens
    summary["n_calls"] = len(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    log.info("summary %s", json.dumps(summary, indent=2))
    print("LLM_TOKEN_PROFILE_DONE")


if __name__ == "__main__":
    main()
