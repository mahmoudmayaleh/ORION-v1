"""Agent B — Abstract Plan Proposer.

Emits a single abstract plan per slice arrival. Operates at the service and
tier level only — never sees raw per-node substrate state.

Inputs (v6 Section 5.2):
  1. Technical slice request s
  2. Abstract topology (per-domain aggregates + inter-domain links)
  3. Semantic reference from K^B (top-k entries) — optional, passed as few-shot
  4. Episodic few-shot from M^B (top-k entries) — optional, passed as few-shot
  5. Feedback context (inference retry only) — violation history h_t

Output:
  Single abstract plan with per-VNF domain assignment (suggested partition),
  resource demands, and per-flow bandwidth requirements.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orion.llm.episodic_memory import EpisodicMemory
    from orion.llm.semantic_memory import SemanticMemory

from orion.llm.llm_backend import LLMBackend, extract_json
from orion.llm.structural_checker import CheckResult, check_plan

logger = logging.getLogger("orion.llm.agent_b")

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Agent B in a 6G network slice orchestration system. Your job is to \
produce an abstract placement plan that assigns each VNF in a slice request \
to a network domain.

You receive:
1. A slice request with SFC chain, per-VNF resource demands, QoS vector, \
and tier placement rules.
2. An abstract topology with per-domain aggregate residual CPU/RAM and \
inter-domain link capacities.
3. Reference Knowledge: curated infrastructure guidelines (tier conventions, \
known-good partitioning patterns, anti-patterns). Follow these closely.
4. Few-shot examples of (slice request, plan) pairs.

PLACEMENT RULES (check all before deciding):
- Assign each VNF to exactly one domain.
- A VNF may only be placed in a domain whose dominant_tiers overlaps with \
the VNF's permitted_tiers (C8).
- The total cpu_demand of all VNFs assigned to a domain must not exceed that \
domain's cpu_residual (C4).
- The total ram_demand of all VNFs assigned to a domain must not exceed that \
domain's ram_residual (C4).
- Minimise the number of inter-domain flow crossings — prefer co-locating \
VNFs in the same domain when both tier and resource constraints allow.
- Each inter-domain flow must fit within the link's bandwidth_residual_mbps.

OUTPUT FORMAT — respond ONLY with a single JSON object:
{
  "plan_id": "<request_id>_plan",
  "vnf_assignments": [
    {
      "vnf_id": "<vnf_id>",
      "domain": "<domain_id>",
      "required_tier": "<tier from permitted_tiers matching this domain>",
      "cpu_demand": <float>,
      "ram_demand": <float>
    }
  ],
  "flow_requirements": [
    {
      "source_vnf": "<vnf_id>",
      "target_vnf": "<vnf_id>",
      "min_bandwidth_mbps": <float>,
      "crosses_domain_boundary": <true|false>
    }
  ],
  "rationale": "<paragraph: for each domain, show CPU arithmetic and \
explain why co-location was chosen or rejected>"
}

Output only the JSON, no prose."""


# ── Prompt builder ───────────────────────────────────────────────────────────

def build_user_prompt(
    slice_request: dict,
    abstract_topology: dict,
    few_shot_examples: list[dict] | None = None,
    violation_feedback: str | None = None,
    reference_knowledge: str | None = None,
) -> str:
    """Build the user message for Agent B.

    Args:
        slice_request: The slice request dict.
        abstract_topology: Abstract topology dict from build_abstract_topology().
        few_shot_examples: Optional list of {slice_request, placement_plan} dicts
            from episodic memory M^B.
        violation_feedback: Optional violation text from a failed structural check,
            appended on retry/replan rounds.
        reference_knowledge: Optional formatted text from K^B semantic memory,
            injected as a Reference Knowledge block above the few-shot examples.
    """
    parts: list[str] = []

    # Reference knowledge from K^B (above few-shot, per v6 Section 4.5)
    if reference_knowledge:
        parts.append(reference_knowledge)

    # Few-shot examples from M^B
    if few_shot_examples:
        parts.append("\n--- Past Plans (Few-Shot Examples) ---")
        for i, ex in enumerate(few_shot_examples, 1):
            parts.append(f"\n--- Example {i} ---")
            parts.append(f"Slice Request:\n{json.dumps(ex['slice_request'], indent=2)}")
            parts.append(f"Placement Plan:\n{json.dumps(ex['placement_plan'], indent=2)}")

    # Current task
    parts.append("\n--- Current Task ---")
    parts.append(f"\nAbstract Topology:\n{json.dumps(abstract_topology, indent=2)}")
    parts.append(f"\nSlice Request:\n{json.dumps(slice_request, indent=2)}")

    # Violation feedback for retry
    if violation_feedback:
        parts.append(
            f"\n--- PREVIOUS ATTEMPT FAILED ---\n"
            f"Your previous plan violated the following constraints. "
            f"Fix these issues in your new plan:\n{violation_feedback}"
        )

    return "\n".join(parts)


# ── Agent B ──────────────────────────────────────────────────────────────────

class AgentB:
    """Abstract plan proposer.

    Generates a single plan per invocation. Supports structural-check retry
    (one regeneration attempt with violation feedback).

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

    def generate_plan(
        self,
        slice_request: dict,
        abstract_topology: dict,
        few_shot_examples: list[dict] | None = None,
        violation_feedback: str | None = None,
        reference_knowledge: str | None = None,
    ) -> dict:
        """Generate one abstract plan via LLM call.

        Args:
            slice_request: Slice request dict.
            abstract_topology: Abstract topology dict.
            few_shot_examples: Few-shot examples from M^B episodic memory.
            violation_feedback: Violation text from a prior failed attempt.
            reference_knowledge: Formatted text from K^B semantic memory.

        Returns:
            Parsed plan dict.

        Raises:
            ValueError: If the LLM output cannot be parsed as JSON.
        """
        user_msg = build_user_prompt(
            slice_request, abstract_topology,
            few_shot_examples, violation_feedback, reference_knowledge,
        )
        raw = self.llm.complete(self.system_prompt, user_msg)
        return extract_json(raw)

    def generate_and_check(
        self,
        slice_request: dict,
        abstract_topology: dict,
        few_shot_examples: list[dict] | None = None,
        max_retries: int = 1,
        reference_knowledge: str | None = None,
    ) -> tuple[dict, CheckResult]:
        """Generate a plan and validate it, retrying once on structural failure.

        This implements the structural-checker regeneration logic from v6
        Section 5.3: at most one regeneration during training, with violation
        feedback appended to the prompt.

        Args:
            slice_request: Slice request dict.
            abstract_topology: Abstract topology dict.
            few_shot_examples: Few-shot examples from M^B episodic memory.
            max_retries: Number of regeneration attempts on structural failure.
            reference_knowledge: Formatted text from K^B semantic memory.

        Returns:
            Tuple of (plan_dict, CheckResult). The plan may still be invalid
            if all retries exhausted — the caller must check result.is_valid.
        """
        violation_feedback = None

        for attempt in range(1 + max_retries):
            try:
                plan = self.generate_plan(
                    slice_request, abstract_topology,
                    few_shot_examples, violation_feedback, reference_knowledge,
                )
            except ValueError:
                logger.warning(
                    "agent_b_json_parse_failed",
                    extra={"attempt": attempt + 1},
                )
                # Treat parse failure as a structural failure
                result = CheckResult(
                    is_valid=False,
                    violations=[],
                )
                violation_feedback = "Your previous response was not valid JSON. Respond with only a JSON object."
                continue

            result = check_plan(plan, slice_request, abstract_topology)

            if result.is_valid:
                logger.debug(
                    "agent_b_plan_valid",
                    extra={"attempt": attempt + 1},
                )
                return plan, result

            logger.info(
                "agent_b_structural_check_failed",
                extra={
                    "attempt": attempt + 1,
                    "violations": len(result.violations),
                },
            )
            violation_feedback = result.violation_text_for_prompt()

        return plan, result

    def generate_with_memory(
        self,
        slice_request: dict,
        abstract_topology: dict,
        kb: SemanticMemory | None = None,
        mb: EpisodicMemory | None = None,
        max_retries: int = 1,
    ) -> tuple[dict, CheckResult]:
        """Generate a plan using K^B semantic and M^B episodic memory.

        Retrieves relevant context from both memory systems and feeds it
        into generate_and_check().

        Args:
            slice_request: Slice request dict.
            abstract_topology: Abstract topology dict.
            kb: Optional semantic memory (K^B) for reference knowledge.
            mb: Optional episodic memory (M^B) for few-shot examples.
            max_retries: Number of regeneration attempts on structural failure.

        Returns:
            Tuple of (plan_dict, CheckResult).
        """
        from orion.llm.semantic_memory import build_query_from_slice

        query = build_query_from_slice(slice_request)

        reference_knowledge: str | None = None
        if kb is not None:
            kb_entries = kb.retrieve(query, top_k=5)
            formatted = kb.format_for_prompt(kb_entries)
            if formatted:
                reference_knowledge = formatted

        few_shot_examples: list[dict] | None = None
        if mb is not None:
            mb_entries = mb.retrieve(query, top_k=3)
            converted = mb.to_few_shot(mb_entries)
            if converted:
                few_shot_examples = converted

        return self.generate_and_check(
            slice_request,
            abstract_topology,
            few_shot_examples=few_shot_examples,
            max_retries=max_retries,
            reference_knowledge=reference_knowledge,
        )
