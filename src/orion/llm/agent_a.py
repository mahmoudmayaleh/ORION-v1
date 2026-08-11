"""Agent A -- Intent Translator.

Receives a natural-language service intent and outputs a structured slice
request dict matching the SliceRequest schema. Operates as the first stage
in the CoALA pipeline: intent text -> structured SFC specification.

Inputs:
  1. Natural language intent text
  2. Semantic reference from K^A (top-k entries) -- optional
  3. Episodic few-shot from M^A (top-k entries) -- optional

Output:
  Single JSON dict with request_id, slice_type, vnfs, flow_edges, qos.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from orion.llm.episodic_memory import EpisodicMemory
    from orion.llm.semantic_memory import SemanticMemory

from orion.llm.llm_backend import LLMBackend, extract_json
from orion.types import TIER_ORDER

logger = logging.getLogger("orion.llm.agent_a")

# ── Valid constants ──────────────────────────────────────────────────────────

VALID_SLICE_TYPES = {"eMBB", "URLLC", "mMTC", "V2X", "XR"}

VALID_TIERS = {t.value for t in TIER_ORDER}

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Agent A in a 6G network slice orchestration system. Your job is to \
translate a natural-language service intent into a structured slice request \
that can be processed by downstream placement agents.

You must:
1. Parse the user's natural language intent to determine the service need.
2. Identify the slice type: one of eMBB, URLLC, mMTC, V2X, XR.
3. Select appropriate VNF types and chain them into a service function chain.
4. Set realistic resource demands (cpu_demand, ram_demand) for each VNF.
5. Set correct VCR (Volume Change Ratio) values per VNF type:
   - Firewall: vcr=1.0 (pass-through)
   - CDN: vcr=0.7 (compression/caching)
   - vEPC: vcr=1.0 (pass-through)
   - vUPF: vcr=1.0 (pass-through)
   - IoTGateway: vcr=0.3 (heavy aggregation)
   - Aggregator: vcr=0.5 (moderate aggregation)
   - Analytics: vcr=1.0 (pass-through)
   - V2XController: vcr=1.0 (pass-through)
   - MediaProc: vcr=1.2 (expansion from rendering)
6. Set permitted_tiers for each VNF based on tier placement rules:
   - ran_edge: ultra-low-latency VNFs (Firewall for URLLC, IoTGateway, vUPF)
   - mec: edge processing (Firewall, CDN, vUPF, V2XController, MediaProc)
   - regional_cloud: moderate compute (CDN, vEPC, Aggregator, Analytics, MediaProc)
   - central_cloud: heavy compute (vEPC, Analytics, CDN for XR)
7. Compute VCR-scaled bandwidth for flow edges:
   beta_{k,k+1} = base_bandwidth * product of vcr for VNFs 0..k
8. Set QoS requirements matching the slice type:
   - eMBB: delay 20-100 ms, throughput 50-500 Mbps
   - URLLC: delay 1-10 ms, throughput 10-100 Mbps
   - mMTC: delay 50-500 ms, throughput 1-10 Mbps
   - V2X: delay 5-20 ms, throughput 20-100 Mbps
   - XR: delay 5-30 ms, throughput 100-1000 Mbps

OUTPUT FORMAT -- respond ONLY with a single JSON object:
{
  "request_id": "<generated_id>",
  "slice_type": "eMBB|URLLC|mMTC|V2X|XR",
  "vnfs": [
    {
      "vnf_id": "<request_id>_f<k>",
      "vnf_type": "<type>",
      "cpu_demand": <float>,
      "ram_demand": <float>,
      "permitted_tiers": ["<tier1>", "<tier2>"],
      "computational_intensity": <float>,
      "vcr": <float>
    }
  ],
  "flow_edges": [
    {
      "source_vnf": "<vnf_id>",
      "target_vnf": "<vnf_id>",
      "bandwidth_demand": <float>
    }
  ],
  "qos": {
    "max_e2e_delay": <float>,
    "min_throughput": <float>
  }
}

Output only the JSON, no prose."""


# ── Prompt builder ───────────────────────────────────────────────────────────

def build_user_prompt(
    intent_text: str,
    reference_knowledge: str | None = None,
    few_shot_examples: list[dict] | None = None,
    request_id: str | None = None,
) -> str:
    """Build the user message for Agent A.

    Args:
        intent_text: The natural language service intent.
        reference_knowledge: Optional formatted text from K^A semantic memory.
        few_shot_examples: Optional past valid translations from M^A.
        request_id: Override for the request ID. Auto-generated if not provided.
    """
    if request_id is None:
        request_id = f"intent_{uuid4().hex[:8]}"

    parts: list[str] = []

    # Reference knowledge from K^A
    if reference_knowledge:
        parts.append(reference_knowledge)

    # Few-shot examples from M^A
    if few_shot_examples:
        parts.append("\n--- Past Translations (Few-Shot Examples) ---")
        for i, ex in enumerate(few_shot_examples, 1):
            parts.append(f"\n--- Example {i} ---")
            parts.append(f"Intent: {ex.get('intent', '')}")
            parts.append(f"Slice Request:\n{json.dumps(ex.get('slice_request', {}), indent=2)}")

    # Current task
    parts.append("\n--- Current Task ---")
    parts.append(f"\nRequest ID to use: {request_id}")
    parts.append(f"\nService Intent:\n{intent_text}")

    return "\n".join(parts)


# ── Validation ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of validating a slice spec dict against the expected schema."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)

    def error_text_for_prompt(self) -> str:
        """Format errors as feedback text for a retry prompt."""
        if not self.errors:
            return ""
        lines = ["Your previous output had the following errors:"]
        for err in self.errors:
            lines.append(f"  - {err}")
        return "\n".join(lines)


def validate_slice_spec(spec: dict) -> ValidationResult:
    """Validate a slice spec dict against the SliceRequest schema.

    Checks:
      - Required top-level keys present
      - slice_type is one of the 5 valid types
      - Each VNF has required fields with valid values
      - permitted_tiers contain only valid tier names
      - vcr > 0
      - flow_edges connect consecutive VNFs in the chain
      - qos has max_e2e_delay > 0 and min_throughput > 0

    Returns:
        ValidationResult with is_valid flag and list of error strings.
    """
    errors: list[str] = []

    # Top-level keys
    required_keys = {"request_id", "slice_type", "vnfs", "flow_edges", "qos"}
    missing = required_keys - set(spec.keys())
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return ValidationResult(is_valid=False, errors=errors)

    # slice_type
    if spec["slice_type"] not in VALID_SLICE_TYPES:
        errors.append(
            f"Invalid slice_type '{spec['slice_type']}'. "
            f"Must be one of: {sorted(VALID_SLICE_TYPES)}"
        )

    # VNFs
    vnfs = spec.get("vnfs", [])
    if not isinstance(vnfs, list) or len(vnfs) == 0:
        errors.append("vnfs must be a non-empty list")
    else:
        vnf_ids: list[str] = []
        vnf_required = {"vnf_id", "vnf_type", "cpu_demand", "ram_demand",
                        "permitted_tiers", "computational_intensity", "vcr"}
        for k, vnf in enumerate(vnfs):
            prefix = f"vnfs[{k}]"
            vnf_missing = vnf_required - set(vnf.keys())
            if vnf_missing:
                errors.append(f"{prefix}: missing fields {sorted(vnf_missing)}")
                continue

            vnf_ids.append(vnf["vnf_id"])

            if not isinstance(vnf["cpu_demand"], (int, float)) or vnf["cpu_demand"] <= 0:
                errors.append(f"{prefix}: cpu_demand must be > 0")
            if not isinstance(vnf["ram_demand"], (int, float)) or vnf["ram_demand"] <= 0:
                errors.append(f"{prefix}: ram_demand must be > 0")

            tiers = vnf.get("permitted_tiers", [])
            if not isinstance(tiers, list) or len(tiers) == 0:
                errors.append(f"{prefix}: permitted_tiers must be a non-empty list")
            else:
                invalid_tiers = set(tiers) - VALID_TIERS
                if invalid_tiers:
                    errors.append(
                        f"{prefix}: invalid tier names {sorted(invalid_tiers)}. "
                        f"Valid: {sorted(VALID_TIERS)}"
                    )

            if not isinstance(vnf.get("vcr"), (int, float)) or vnf["vcr"] <= 0:
                errors.append(f"{prefix}: vcr must be > 0")

        # Flow edges: must connect consecutive VNFs
        flow_edges = spec.get("flow_edges", [])
        if not isinstance(flow_edges, list):
            errors.append("flow_edges must be a list")
        elif len(vnf_ids) >= 2:
            expected_count = len(vnf_ids) - 1
            if len(flow_edges) != expected_count:
                errors.append(
                    f"Expected {expected_count} flow_edges for {len(vnf_ids)} VNFs, "
                    f"got {len(flow_edges)}"
                )
            else:
                for k, edge in enumerate(flow_edges):
                    prefix = f"flow_edges[{k}]"
                    if edge.get("source_vnf") != vnf_ids[k]:
                        errors.append(
                            f"{prefix}: source_vnf should be '{vnf_ids[k]}', "
                            f"got '{edge.get('source_vnf')}'"
                        )
                    if edge.get("target_vnf") != vnf_ids[k + 1]:
                        errors.append(
                            f"{prefix}: target_vnf should be '{vnf_ids[k + 1]}', "
                            f"got '{edge.get('target_vnf')}'"
                        )
                    bw = edge.get("bandwidth_demand")
                    if not isinstance(bw, (int, float)) or bw <= 0:
                        errors.append(f"{prefix}: bandwidth_demand must be > 0")

    # QoS
    qos = spec.get("qos", {})
    if not isinstance(qos, dict):
        errors.append("qos must be a dict")
    else:
        delay = qos.get("max_e2e_delay")
        if not isinstance(delay, (int, float)) or delay <= 0:
            errors.append("qos.max_e2e_delay must be > 0")
        throughput = qos.get("min_throughput")
        if not isinstance(throughput, (int, float)) or throughput <= 0:
            errors.append("qos.min_throughput must be > 0")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


# ── Agent A ──────────────────────────────────────────────────────────────────

class AgentA:
    """Intent translator.

    Translates natural-language service intents into structured slice request
    dicts. Supports schema validation with retry and memory-augmented
    translation via K^A and M^A.

    Args:
        llm: LLM backend for chat completion.
        system_prompt: Override the default system prompt if needed.
    """

    def __init__(
        self,
        llm: LLMBackend,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def translate(
        self,
        intent_text: str,
        request_id: str | None = None,
        few_shot_examples: list[dict] | None = None,
        reference_knowledge: str | None = None,
    ) -> dict:
        """Translate a natural-language intent to a slice request dict.

        Single LLM call, no validation or retry.

        Args:
            intent_text: The natural language service intent.
            request_id: Override for the request ID.
            few_shot_examples: Few-shot examples from M^A episodic memory.
            reference_knowledge: Formatted text from K^A semantic memory.

        Returns:
            Parsed slice request dict.

        Raises:
            ValueError: If the LLM output cannot be parsed as JSON.
        """
        user_msg = build_user_prompt(
            intent_text, reference_knowledge,
            few_shot_examples, request_id,
        )
        raw = self.llm.complete(self.system_prompt, user_msg)
        return extract_json(raw)

    def translate_and_validate(
        self,
        intent_text: str,
        request_id: str | None = None,
        few_shot_examples: list[dict] | None = None,
        max_retries: int = 1,
        reference_knowledge: str | None = None,
    ) -> tuple[dict, ValidationResult]:
        """Translate and validate, retrying on schema failure.

        Args:
            intent_text: The natural language service intent.
            request_id: Override for the request ID.
            few_shot_examples: Few-shot examples from M^A episodic memory.
            max_retries: Number of regeneration attempts on schema failure.
            reference_knowledge: Formatted text from K^A semantic memory.

        Returns:
            Tuple of (spec_dict, ValidationResult). The spec may still be
            invalid if all retries are exhausted -- the caller must check
            result.is_valid.
        """
        spec: dict = {}
        result = ValidationResult(is_valid=False, errors=[])
        violation_feedback: str | None = None

        for attempt in range(1 + max_retries):
            try:
                user_msg = build_user_prompt(
                    intent_text, reference_knowledge,
                    few_shot_examples, request_id,
                )
                if violation_feedback:
                    user_msg += f"\n\n--- PREVIOUS ATTEMPT FAILED ---\n{violation_feedback}"

                raw = self.llm.complete(self.system_prompt, user_msg)
                spec = extract_json(raw)
            except ValueError:
                logger.warning(
                    "agent_a_json_parse_failed",
                    extra={"attempt": attempt + 1},
                )
                result = ValidationResult(
                    is_valid=False,
                    errors=[],
                )
                violation_feedback = (
                    "Your previous response was not valid JSON. "
                    "Respond with only a JSON object."
                )
                continue

            result = validate_slice_spec(spec)

            if result.is_valid:
                logger.debug(
                    "agent_a_spec_valid",
                    extra={"attempt": attempt + 1},
                )
                return spec, result

            logger.info(
                "agent_a_validation_failed",
                extra={
                    "attempt": attempt + 1,
                    "errors": len(result.errors),
                },
            )
            violation_feedback = result.error_text_for_prompt()

        return spec, result

    def translate_with_memory(
        self,
        intent_text: str,
        kb: SemanticMemory | None = None,
        mb: EpisodicMemory | None = None,
        request_id: str | None = None,
        max_retries: int = 1,
    ) -> tuple[dict, ValidationResult]:
        """Translate using K^A semantic and M^A episodic memory.

        Retrieves relevant context from both memory systems and feeds it
        into translate_and_validate().

        Args:
            intent_text: The natural language service intent.
            kb: Optional semantic memory (K^A) for reference knowledge.
            mb: Optional episodic memory (M^A) for few-shot examples.
            request_id: Override for the request ID.
            max_retries: Number of regeneration attempts on schema failure.

        Returns:
            Tuple of (spec_dict, ValidationResult).
        """
        reference_knowledge: str | None = None
        if kb is not None:
            kb_entries = kb.retrieve(intent_text, top_k=5)
            formatted = kb.format_for_prompt(kb_entries)
            if formatted:
                reference_knowledge = formatted

        few_shot_examples: list[dict] | None = None
        if mb is not None:
            mb_entries = mb.retrieve(intent_text, top_k=3)
            converted = _convert_episodes_to_few_shot(mb_entries)
            if converted:
                few_shot_examples = converted

        return self.translate_and_validate(
            intent_text,
            request_id=request_id,
            few_shot_examples=few_shot_examples,
            max_retries=max_retries,
            reference_knowledge=reference_knowledge,
        )


# ── Few-shot conversion helper ──────────────────────────────────────────────

def _convert_episodes_to_few_shot(entries: list) -> list[dict]:
    """Convert M^A episodic entries to Agent A few-shot format.

    Parses stored content to extract intent and slice_request dicts.
    Entries that fail to parse are skipped.
    """
    results: list[dict] = []
    for se in entries:
        try:
            intent_text = None
            slice_dict = None
            for line in se.entry.content.splitlines():
                if line.startswith("Slice: "):
                    slice_dict = json.loads(line[len("Slice: "):])
                elif line.startswith("Plan: "):
                    plan = json.loads(line[len("Plan: "):])
                    intent_text = plan.get("intent")
            if intent_text is None or slice_dict is None:
                logger.warning(
                    "agent_a_few_shot_missing_fields",
                    extra={"entry_id": se.entry.entry_id},
                )
                continue
            results.append({
                "intent": intent_text,
                "slice_request": slice_dict,
            })
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning(
                "agent_a_few_shot_parse_failed",
                extra={"entry_id": se.entry.entry_id, "error": str(exc)},
            )
            continue
    return results


# ── M^A write helper ─────────────────────────────────────────────────────────

def record_translation(
    mb: EpisodicMemory,
    intent_text: str,
    spec: dict,
    is_valid: bool,
) -> bool:
    """Record a translation in M^A episodic memory.

    Write trigger: every schema-valid spec is recorded. Invalid specs are
    also recorded as negative examples for future avoidance.

    Args:
        mb: The M^A episodic memory instance.
        intent_text: The original natural language intent.
        spec: The full slice request dict produced by Agent A.
        is_valid: Whether the spec passed schema validation.

    Returns:
        True if the episode was recorded by M^A.
    """
    return mb.record(
        slice_spec=spec,
        plan={"intent": intent_text},
        m_committed=0.0,
        constraints_violated=[] if is_valid else ["schema_invalid"],
        reward=1.0 if is_valid else -1.0,
    )
