"""
Translate a natural language slice intent into a SliceRequest JSON.

Usage (interactive):
    python scripts/llm_eval/intent_to_slice.py

Usage (single intent):
    python scripts/llm_eval/intent_to_slice.py "I need ultra-low latency for a V2X corridor"

Output is printed as JSON and optionally saved to ~/llm_models/tele-llm/slice_outputs/.

Note: permitted_nodes is left empty — populate it afterwards using
      _resolve_permitted_nodes() from orion.sim.slice_generator.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "slice_intents"

# ── JSON schema sent to the model ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a 6G network slice specification expert. Your job is to convert a \
natural language service intent into a valid SliceRequest JSON object.

Slice types: eMBB (high throughput), URLLC (ultra-low latency), \
mMTC (massive IoT), V2X (vehicle-to-everything), XR (extended reality).

VNF types per slice:
- eMBB:  Firewall → CDN → vEPC
- URLLC: Firewall → vUPF
- mMTC:  IoTGateway → Aggregator → Analytics
- V2X:   Firewall → V2XController → vEPC
- XR:    Firewall → MediaProc → CDN → vEPC

Tier placement rules (permitted_tiers per VNF type):
- ran_edge / mec: low-latency VNFs (Firewall, vUPF, IoTGateway, V2XController)
- mec / regional_cloud: mid-tier (CDN, Aggregator, MediaProc, V2XController)
- regional_cloud / central_cloud: heavy compute (vEPC, Analytics)

Respond ONLY with a single JSON object matching this schema exactly:
{
  "request_id": "<string>",
  "slice_type": "<eMBB|URLLC|mMTC|V2X|XR>",
  "vnfs": [
    {
      "vnf_id": "<request_id>_f<index>",
      "vnf_type": "<type>",
      "cpu_demand": <float>,
      "ram_demand": <float>,
      "permitted_tiers": ["<tier>", ...],
      "computational_intensity": <float>
    }
  ],
  "flow_edges": [
    {
      "source_vnf": "<vnf_id>",
      "target_vnf": "<vnf_id>",
      "bandwidth_demand": <float in Mbps>
    }
  ],
  "qos": {
    "max_e2e_delay": <float in ms>,
    "min_throughput": <float in Mbps>
  },
  "arrival_time": 0.0,
  "lifetime": 0.0,
  "intent_summary": "<one sentence explaining your interpretation>"
}

Use realistic values. Do not include permitted_nodes (leave it out entirely). \
Output only the JSON, no prose."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Pull the first {...} block out of the model response."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the outermost braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON found in model response:\n{text}")


def _add_request_id(data: dict) -> dict:
    """Ensure a unique request_id."""
    if not data.get("request_id") or data["request_id"] == "<string>":
        data["request_id"] = f"req_{uuid.uuid4().hex[:6]}"
    return data


def _save(data: dict, request_id: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{request_id}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


# ── Core function ─────────────────────────────────────────────────────────────

def intent_to_slice(intent: str, save: bool = True, model: str = "tele-llm") -> dict:
    """
    Translate a natural language intent into a SliceRequest dict.

    Args:
        intent: Free-text description of the slice requirement.
        save:   If True, write the result to OUTPUT_DIR/<request_id>.json.
        model:  "tele-llm" (default, llama-cpp GGUF) or "tslam" (HF Transformers).

    Returns:
        Parsed SliceRequest dict (without permitted_nodes; add via substrate).
    """
    if model == "tslam":
        from tslam import TSLAMLLM, TSLAMConfig  # noqa: PLC0415
        cfg = TSLAMConfig(
            system_prompt=SYSTEM_PROMPT,
            temperature=0.05,
            max_new_tokens=1024,
        )
        llm = TSLAMLLM(cfg)
    else:
        from tele_llm import TeleLLM, TeleLLMConfig  # noqa: PLC0415
        cfg = TeleLLMConfig(
            system_prompt=SYSTEM_PROMPT,
            temperature=0.05,
            max_tokens=1024,
        )
        llm = TeleLLM(cfg)

    print(f"\nIntent: {intent}")
    print("Parsing...", flush=True)

    raw = llm.complete(intent)
    data = _extract_json(raw)
    data = _add_request_id(data)

    if save:
        path = _save(data, data["request_id"])
        print(f"Saved → {path}")

    return data


# ── CLI ───────────────────────────────────────────────────────────────────────

_EXAMPLES = [
    "I need a slice for an autonomous vehicle corridor with sub-5ms latency and 50 Mbps throughput.",
    "Deploy a video streaming service that can handle 800 Mbps with moderate delay tolerance.",
    "Set up a massive sensor network for a smart factory with 10,000 IoT devices and very low bandwidth per device.",
    "Launch an XR telepresence application requiring 4K video processing and 10ms end-to-end delay.",
]

if __name__ == "__main__":
    import argparse, os
    sys.path.insert(0, os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description="Translate intent to SliceRequest JSON.")
    parser.add_argument("intent", nargs="*", help="Natural language intent (optional).")
    parser.add_argument(
        "--model", choices=["tele-llm", "tslam"], default="tele-llm",
        help="LLM backend: 'tele-llm' (llama-cpp GGUF) or 'tslam' (HF Transformers, default: tele-llm).",
    )
    args = parser.parse_args()

    intents = [" ".join(args.intent)] if args.intent else _EXAMPLES

    for intent in intents:
        try:
            result = intent_to_slice(intent, save=True, model=args.model)
            print(json.dumps(result, indent=2))
        except (ValueError, KeyError) as e:
            print(f"[ERROR] {e}")
        print()
