#!/usr/bin/env python3
"""Agent B validity probe (v2) — 3-way split with OPPOSITE fixes.

The v1 "decode" bucket conflated two failures with different fixes. The served
response_format={"type":"json_object","schema":AGENT_B_PLAN_JSON_SCHEMA} enforces the
per-assignment SHAPE, but the schema has NO minItems/maxItems on vnf_assignments and NO enum
on vnf_id/domain -> grammar-valid output can still omit a VNF or name an invalid domain. So:

  MALFORMED  : JSON won't parse, or fails AgentBPlanSchema shape  -> the grammar itself is not
               holding (decode fallback / $ref not converted). Fix = decoding.
  CONTENT    : shape-valid JSON, but structural-checker SCHEMA violations (missing/extra VNF,
               unknown domain). Grammar works; the MODEL picks wrong ids/domains on long
               chains. Fix = pin vnf_id/domain enums + list length in the per-request grammar
               (or ConstrainedAgentB / few-shot). NOT a decode fallback.
  FEASIBILITY: shape-valid, right ids, but C4/C8/C5 infeasible vs residuals. Fix = retry budget
               (max_retries>=1); genuine, not a bug.

All three still VOID the LLM arms if concentrated on 3+ VNF chains (Agent B's m~ is wrong there,
and those are the partition-relevant slices) -- but they name DIFFERENT fixes. Matches the gate
call path (generate_with_memory kb=on mb=None max_retries=0). Standalone; concurrent-safe.
"""
import argparse
import logging
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import five_arm_runner as R
from orion.sim.slice_generator import generate_slice_request
from orion.llm.structural_checker import check_plan
from orion.llm.agent_b import AgentBPlanSchema, PlanTruncationError

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("validity")

FAM = {f.short_name: f for f in R.ALL_FAMILIES}


def classify(agent_b, kb, sr_dict, abstract_topo, pinned=False):
    """EXACT gate path: generate_with_memory(kb, mb=None, max_retries=0). Bucket from the check.
    Also re-run AgentBPlanSchema.model_validate to flag SHAPE (grammar) failures distinctly.
    pinned=True passes the per-request enum+length schema (the 2026-07-10 content fix)."""
    from orion.llm.agent_b import build_pinned_plan_schema
    if pinned:
        ps = build_pinned_plan_schema(sr_dict, abstract_topo)
        if ps is None:
            return "feasibility", ["TIER_INFEASIBLE"], "no tier-feasible domain (genuine)"
    else:
        ps = None
    try:
        plan, check = agent_b.generate_with_memory(
            sr_dict, abstract_topo, kb=kb, mb=None, max_retries=0, plan_schema=ps)
    except Exception as e:  # noqa: BLE001
        return "malformed", ["EXC"], str(e)[:80]
    if getattr(check, "is_valid", False):
        return "valid", [], ""
    viols = getattr(check, "violations", []) or []
    if not viols:
        return "malformed", ["PARSE/TRUNC"], ""   # empty -> parse/truncation (generate_and_check)
    # Did the grammar even hold the assignment shape? (distinguishes grammar-broken from content)
    shape_ok = True
    try:
        AgentBPlanSchema.model_validate(plan)
    except Exception:  # noqa: BLE001
        shape_ok = False
    tags = [getattr(v, "constraint", "?") for v in viols]
    if not shape_ok:
        return "malformed", ["SHAPE"] + tags, ""
    if "SCHEMA" in set(tags):
        return "content", tags, ""                # shape ok, wrong vnf ids / unknown domain
    return "feasibility", tags, ""                # C4/C8/C5 only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="C+_T-_B-")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pinned", action="store_true", help="use per-request enum+length schema (the fix)")
    args = ap.parse_args()

    from orion.llm.llm_backend import LLMBackend, LLMConfig
    from orion.llm.agent_b import AgentB
    from orion.llm.semantic_memory import SemanticMemory

    cfg = LLMConfig(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY",
                    model="default", temperature=0.05, max_tokens=2048)
    backend = LLMBackend(cfg)
    agent_b = AgentB(backend)
    kb = None
    kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
    if kb_path.exists():
        kb = SemanticMemory.from_json(kb_path)

    sub = R.generate_family_instance(FAM[args.family], seed=0)
    abstract_topo = R.build_abstract_topology(sub)
    rng = np.random.default_rng(args.seed)

    by_len = defaultdict(lambda: defaultdict(int))
    overall = defaultdict(int)
    tag_counter = Counter()
    n_ctx = getattr(cfg, "n_ctx", None)
    print(f"Agent B validity probe v2: family={args.family} n={args.n} n_ctx={n_ctx} (gate path)\n")

    for i in range(args.n):
        sr = generate_slice_request(request_id=f"probe_{i:04d}", substrate=sub, rng=rng)
        sr_dict = R._slice_request_to_dict(sr, sub)
        k = len(sr.vnfs)
        b, tags, _ = classify(agent_b, kb, sr_dict, abstract_topo, pinned=args.pinned)
        by_len[k][b] += 1
        overall[b] += 1
        tag_counter.update(tags)
        if (i + 1) % 12 == 0:
            print(f"  ...{i+1}/{args.n}")

    n = max(1, args.n)
    print("\n=== buckets (overall) ===")
    for b in ("valid", "malformed", "content", "feasibility"):
        print(f"  {b:12s} {overall[b]:3d}  ({100*overall[b]/n:5.1f}%)")
    print(f"\n  MALFORMED   {100*overall['malformed']/n:5.1f}%  (grammar not holding -> fix DECODING)")
    print(f"  CONTENT     {100*overall['content']/n:5.1f}%  (grammar ok, wrong ids/domain -> pin ENUMS/length)")
    print(f"  FEASIBILITY {100*overall['feasibility']/n:5.1f}%  (genuine C4/C8/C5 x N_struct=1 -> retry budget)")
    print(f"  VALID       {100*overall['valid']/n:5.1f}%")

    print("\n=== by chain length k ===")
    print(f"  {'k':>2s} {'n':>3s} {'valid':>7s} {'malform':>8s} {'content':>8s} {'feas':>7s}")
    for k in sorted(by_len):
        row = by_len[k]; tot = sum(row.values())
        print(f"  {k:2d} {tot:3d} {100*row['valid']/tot:6.1f}% {100*row['malformed']/tot:7.1f}% "
              f"{100*row['content']/tot:7.1f}% {100*row['feasibility']/tot:6.1f}%")

    print("\n=== violation constraint tally (failures) ===")
    for tag, c in tag_counter.most_common():
        print(f"  {tag:12s} {c}")

    print("\n=== PRE-COMMITTED READ ===")
    mal = overall["malformed"] / n; con = overall["content"] / n
    if mal > 0.10:
        print(f"  MALFORMED {100*mal:.0f}% > 10% -> the grammar is NOT holding (decode fallback /")
        print("  $ref not converted). Fix decoding (explicit GBNF / ConstrainedAgentB), relaunch LLM arms.")
    elif con > 0.10:
        print(f"  CONTENT {100*con:.0f}% > 10% (grammar holds) -> model emits shape-valid plans with")
        print("  wrong vnf_ids / invalid domains on long chains. Fix = per-request schema with")
        print("  vnf_id/domain ENUMS + minItems=maxItems=K. Relaunch LLM arms. Gate VOID until then.")
    else:
        print("  Malformed & content both <=10% -> LLM arms healthy enough; remaining failures are")
        print("  genuine feasibility x retry budget. Report rate next to FoC.")


if __name__ == "__main__":
    main()
