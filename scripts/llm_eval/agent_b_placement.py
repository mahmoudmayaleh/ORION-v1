"""
Agent B: LLM-based abstract placement plan generator.

Reads a static slice request, abstract topology, and few-shot examples,
builds the prompt, and runs it through tele-llm or tslam.

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

DATA_DIR   = Path(__file__).resolve().parents[2] / "data" / "placement_eval"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "llm-outputs"

SLICE_REQUEST_PATH  = DATA_DIR / "slice_request.json"
TOPOLOGY_PATH       = DATA_DIR / "abstract_topology.json"
FEW_SHOT_PATH       = DATA_DIR / "few_shot_examples.json"


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(slice_request: dict, topology: dict, few_shot: list[dict]) -> str:
    few_shot_text = ""
    for i, ex in enumerate(few_shot, 1):
        few_shot_text += f"\n--- Example {i} ---\n"
        few_shot_text += f"Slice Request:\n{json.dumps(ex['slice_request'], indent=2)}\n"
        few_shot_text += f"Placement Plan:\n{json.dumps(ex['placement_plan'], indent=2)}\n"

    return f"""\
You are Agent B in a 6G network slice orchestration system. Your job is to \
produce an abstract placement plan that assigns each VNF in a slice request to \
a network domain.

You receive three inputs:
1. A slice request with SFC chain, per-VNF resource demands, QoS vector, and tier placement rules.
2. An abstract topology with per-domain aggregate residual CPU/RAM and inter-domain link capacities.
3. Few-shot examples of (slice request, high-quality plan) pairs.

PLACEMENT RULES (check all before deciding):
- Assign each VNF to exactly one domain.
- A VNF may only be placed in a domain whose dominant_tiers overlaps with the VNF's permitted_tiers.
- The total cpu_demand of all VNFs assigned to a domain must not exceed that domain's cpu_residual (C4).
- The total ram_demand of all VNFs assigned to a domain must not exceed that domain's ram_residual (C4).
- Minimise the number of inter-domain flow crossings — prefer co-locating VNFs in the same domain when both tier and resource constraints allow.
- Each inter-domain flow must fit within the link's bandwidth_residual_mbps.

OUTPUT FORMAT — respond ONLY with a single JSON object:
{{
  "plan_id": "<request_id>_plan",
  "vnf_assignments": [
    {{
      "vnf_id": "<vnf_id>",
      "domain": "<domain_id>",
      "required_tier": "<the specific tier from permitted_tiers that matches this domain>",
      "cpu_demand": <float>,
      "ram_demand": <float>
    }}
  ],
  "flow_requirements": [
    {{
      "source_vnf": "<vnf_id>",
      "target_vnf": "<vnf_id>",
      "min_bandwidth_mbps": <float>,
      "crosses_domain_boundary": <true|false>
    }}
  ],
  "rationale": "<paragraph: for each domain, show the CPU arithmetic and explain why co-location was chosen or rejected>"
}}

Output only the JSON, no prose.

--- Few-Shot Examples ---
{few_shot_text}
--- Current Task ---

Abstract Topology:
{json.dumps(topology, indent=2)}

Slice Request:
{json.dumps(slice_request, indent=2)}"""


# ── Runner ────────────────────────────────────────────────────────────────────

def run(model: str) -> dict:
    slice_request = json.loads(SLICE_REQUEST_PATH.read_text())
    topology      = json.loads(TOPOLOGY_PATH.read_text())
    few_shot      = json.loads(FEW_SHOT_PATH.read_text())

    prompt = build_prompt(slice_request, topology, few_shot)

    print(f"Running Agent B with model: {model}")
    print(f"Slice: {slice_request['request_id']} ({slice_request['slice_type']}, "
          f"{len(slice_request['vnfs'])} VNFs)\n")

    if model == "tslam":
        from tslam import TSLAMLLM, TSLAMConfig
        llm = TSLAMLLM(TSLAMConfig(temperature=0.05, max_new_tokens=1024))
        raw = llm.complete(prompt)
    else:
        from tele_llm import TeleLLM, TeleLLMConfig
        llm = TeleLLM(TeleLLMConfig(temperature=0.05, max_tokens=1024))
        raw = llm.complete(prompt)

    print("Raw output:")
    print(raw)

    # Save to llm-outputs/
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"agent_b_{model.replace('-', '_')}_v2.md"
    out_path.write_text(
        f"## Agent B Placement Plan — {model}\n\n"
        f"```json\n{raw}\n```\n\n"
        f"## Prompt\n\n"
        f"```\n{prompt}\n```\n"
    )
    print(f"\nSaved → {out_path}")
    return {"raw": raw}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["tele-llm", "tslam"], default="tele-llm",
        help="LLM backend to use."
    )
    args = parser.parse_args()
    run(args.model)
