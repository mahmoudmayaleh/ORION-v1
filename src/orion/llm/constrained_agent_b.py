"""Grammar-constrained Agent B: forces valid JSON output for any chain length.

The base AgentB fails ~40% on 3+ VNF chains because the 8B quantized model
drifts over long structured output. This module constrains decoding via
GBNF grammar so the model physically cannot emit malformed structure.

Design:
  - Stripped prompt: emit ONLY the assignments array, not plan_id/rationale/flows
  - GBNF grammar: forces [{vnf_id, domain, tier}, ...] of exactly K elements
  - Single call, no retries (grammar prevents format failure)
  - Flow requirements and rationale are derived post-hoc, not generated
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from orion.llm.llm_backend import LLMConfig

logger = logging.getLogger(__name__)

# ── Stripped system prompt ───────────────────────────────────────────────────

CONSTRAINED_SYSTEM_PROMPT = """\
You are a 6G network slice placement planner. Given a slice request and \
abstract topology, assign each VNF to exactly one domain.

RULES:
- A VNF may only go in a domain whose tiers include one of the VNF's permitted_tiers.
- Total CPU/RAM demand per domain must not exceed its residual.
- Prefer co-locating consecutive VNFs to minimise inter-domain crossings.

Respond with ONLY a JSON array of assignments, one per VNF in SFC order."""


def build_constrained_prompt(
    slice_request: dict,
    abstract_topology: dict,
) -> str:
    """Minimal prompt: topology summary + VNF list. No K^B, no few-shot."""
    # Compact topology
    domains = []
    for d in abstract_topology["domains"]:
        domains.append(
            f"  {d['domain_id']}: tiers={d['dominant_tiers']}, "
            f"cpu_res={d['cpu_residual']}, ram_res={d['ram_residual']}"
        )

    vnfs = []
    for v in slice_request["vnfs"]:
        vnfs.append(
            f"  {v['vnf_id']}: type={v['vnf_type']}, "
            f"cpu={v['cpu_demand']}, ram={v['ram_demand']}, "
            f"tiers={v['permitted_tiers']}"
        )

    return (
        f"Domains:\n" + "\n".join(domains) +
        f"\n\nVNFs (SFC order):\n" + "\n".join(vnfs) +
        f"\n\nAssign each VNF to a domain. Output a JSON array of objects "
        f"with keys: vnf_id, domain (integer), required_tier (string)."
    )


# ── Constrained caller ───────────────────────────────────────────────────────


class ConstrainedAgentB:
    """Agent B with grammar-constrained decoding.

    Uses llama-cpp-python directly (not the OpenAI API) for grammar support.
    Falls back to unconstrained JSON mode if grammar is unavailable.

    Args:
        model_path: Path to the GGUF model file.
        n_threads: CPU threads for inference.
        n_ctx: Context window size.
    """

    def __init__(
        self,
        model_path: str,
        n_threads: int = 32,
        n_ctx: int = 4096,
    ) -> None:
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            chat_format="llama-3",
            verbose=False,
        )
        self._model_path = model_path

    def generate_plan(
        self,
        slice_request: dict,
        abstract_topology: dict,
        max_tokens: int = 512,
    ) -> dict | None:
        """Generate a constrained plan. Returns the parsed plan dict or None."""
        user_msg = build_constrained_prompt(slice_request, abstract_topology)

        # JSON-mode constrained decoding — forces valid JSON output
        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": CONSTRAINED_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.05,
                max_tokens=max_tokens,
            )
            text = response["choices"][0]["message"]["content"]
            parsed = json.loads(text)

            # Extract assignments array from whatever shape the model returned
            if isinstance(parsed, list):
                assignments = parsed
            elif isinstance(parsed, dict):
                # Model may wrap in {"assignments": [...]} or {"vnf_assignments": [...]}
                assignments = (
                    parsed.get("vnf_assignments")
                    or parsed.get("assignments")
                    or parsed.get("result")
                )
                if assignments is None and len(parsed) == 1:
                    # Single key wrapping an array
                    assignments = next(iter(parsed.values()))
                if not isinstance(assignments, list):
                    logger.warning("JSON mode output has no array: %s", list(parsed.keys()))
                    return None
            else:
                return None

            num_vnfs = len(slice_request.get("vnfs", []))
            if len(assignments) != num_vnfs:
                logger.warning(
                    "JSON output has %d assignments, expected %d",
                    len(assignments), num_vnfs,
                )
                return None

            plan = self._build_plan_dict(assignments, slice_request)
            return plan

        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed: %s", e)
            return None
        except Exception as e:
            logger.warning("Constrained call failed: %s", e)
            return None

    def _build_plan_dict(self, assignments: list[dict], slice_request: dict) -> dict:
        """Build the full plan dict from the constrained assignments."""
        return {
            "plan_id": f"{slice_request.get('request_id', 'unknown')}_plan",
            "vnf_assignments": [
                {
                    "vnf_id": a.get("vnf_id", ""),
                    "domain": f"d{a['domain']}" if isinstance(a.get("domain"), int) else str(a.get("domain", "d0")),
                    "required_tier": a.get("required_tier", "edge"),
                    "cpu_demand": next(
                        (v["cpu_demand"] for v in slice_request["vnfs"]
                         if v["vnf_id"] == a.get("vnf_id")), 0.0
                    ),
                    "ram_demand": next(
                        (v["ram_demand"] for v in slice_request["vnfs"]
                         if v["vnf_id"] == a.get("vnf_id")), 0.0
                    ),
                }
                for a in assignments
            ],
            "flow_requirements": self._derive_flows(assignments, slice_request),
            "rationale": "constrained-json-mode",
        }

    def _derive_flows(self, assignments: list[dict], slice_request: dict) -> list[dict]:
        """Derive flow_requirements from assignments + slice request edges."""
        assignment_map = {a.get("vnf_id"): a for a in assignments}
        flows = []
        for edge in slice_request.get("flow_edges", []):
            src = edge["source_vnf"]
            dst = edge["target_vnf"]
            src_dom = assignment_map.get(src, {}).get("domain")
            dst_dom = assignment_map.get(dst, {}).get("domain")
            crosses = src_dom != dst_dom if src_dom and dst_dom else False
            flows.append({
                "source_vnf": src,
                "target_vnf": dst,
                "min_bandwidth_mbps": edge.get("bandwidth_demand", 0.0),
                "crosses_domain_boundary": crosses,
            })
        return flows
