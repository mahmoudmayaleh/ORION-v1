"""Agent B: LLM-based abstract placement plan generator.

Uses the orion.llm module (Agent B + K^B semantic memory + structural checker).

Usage:
    python scripts/llm_eval/agent_b_placement.py --model tele-llm
    python scripts/llm_eval/agent_b_placement.py --model tslam

Output is printed and saved to llm-outputs/agent_b_<model>.md
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "llm-outputs"

SLICE_REQUEST_PATH  = DATA_DIR / "placement_eval" / "slice_request.json"
TOPOLOGY_PATH       = DATA_DIR / "placement_eval" / "abstract_topology.json"
FEW_SHOT_PATH       = DATA_DIR / "placement_eval" / "few_shot_examples.json"
KB_PATH             = DATA_DIR / "memory" / "kb_agent_b.json"

from orion.llm.agent_b import build_user_prompt, SYSTEM_PROMPT
from orion.llm.structural_checker import check_plan
from orion.llm.semantic_memory import SemanticMemory, build_query_from_slice


def run(model: str) -> dict:
    slice_request = json.loads(SLICE_REQUEST_PATH.read_text())
    topology      = json.loads(TOPOLOGY_PATH.read_text())
    few_shot      = json.loads(FEW_SHOT_PATH.read_text())

    # Retrieve K^B reference knowledge
    kb = SemanticMemory.from_json(KB_PATH)
    query = build_query_from_slice(slice_request)
    kb_entries = kb.retrieve(
        query,
        slice_type=slice_request.get("slice_type"),
        top_k=5,
    )
    ref_text = kb.format_for_prompt(kb_entries)

    # Build prompt using the module
    prompt = build_user_prompt(
        slice_request, topology,
        few_shot_examples=few_shot,
        reference_knowledge=ref_text,
    )

    print(f"Running Agent B with model: {model}")
    print(f"Slice: {slice_request['request_id']} ({slice_request['slice_type']}, "
          f"{len(slice_request['vnfs'])} VNFs)")
    print(f"K^B entries retrieved: {len(kb_entries)}\n")

    if model == "tslam":
        from tslam import TSLAMLLM, TSLAMConfig
        llm = TSLAMLLM(TSLAMConfig(
            system_prompt=SYSTEM_PROMPT,
            temperature=0.05, max_new_tokens=2048,
        ))
        raw = llm.complete(prompt)
    else:
        from tele_llm import TeleLLM, TeleLLMConfig
        llm = TeleLLM(TeleLLMConfig(
            system_prompt=SYSTEM_PROMPT,
            temperature=0.05, max_tokens=2048,
        ))
        raw = llm.complete(prompt)

    print("Raw output:")
    print(raw)

    # Structural check
    from orion.llm.llm_backend import extract_json
    try:
        plan = extract_json(raw)
        result = check_plan(plan, slice_request, topology)
        print(f"\nStructural check: {'PASS' if result.is_valid else 'FAIL'}")
        if not result.is_valid:
            print(result.summary())
    except ValueError as e:
        print(f"\nJSON parse failed: {e}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"agent_b_{model.replace('-', '_')}_v2.md"
    out_path.write_text(
        f"## Agent B Placement Plan — {model}\n\n"
        f"```json\n{raw}\n```\n\n"
        f"## Prompt\n\n"
        f"```\n{SYSTEM_PROMPT}\n\n{prompt}\n```\n"
    )
    print(f"\nSaved → {out_path}")
    return {"raw": raw}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["tele-llm", "tslam"], default="tele-llm",
        help="LLM backend to use."
    )
    args = parser.parse_args()
    run(args.model)
