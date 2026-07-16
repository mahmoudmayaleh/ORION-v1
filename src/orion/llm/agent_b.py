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

from pydantic import BaseModel

if TYPE_CHECKING:
    from orion.llm.episodic_memory import EpisodicMemory
    from orion.llm.semantic_memory import SemanticMemory

from orion.llm.llm_backend import LLMBackend, extract_json
from orion.llm.structural_checker import CheckResult, check_plan
from orion.profiling import profiled

logger = logging.getLogger("orion.llm.agent_b")


# ── Output schema (grammar-constrained decoding) ─────────────────────────────
# Passed to the LLM endpoint as a json_object *schema*; llama.cpp converts it to
# a GBNF grammar and constrains decoding to exactly this structure. Format drift
# (wrong/missing keys) becomes impossible by construction, so the structural
# checker only ever sees well-formed JSON and validates *semantics*. This mirrors
# the prompt's OUTPUT FORMAT block field-for-field.

class _AssignmentSchema(BaseModel):
    vnf_id: str
    domain: str
    required_tier: str
    cpu_demand: float
    ram_demand: float


class _FlowSchema(BaseModel):
    source_vnf: str
    target_vnf: str
    min_bandwidth_mbps: float
    crosses_domain_boundary: bool


class AgentBPlanSchema(BaseModel):
    plan_id: str
    vnf_assignments: list[_AssignmentSchema]
    flow_requirements: list[_FlowSchema]


AGENT_B_PLAN_JSON_SCHEMA = AgentBPlanSchema.model_json_schema()


def tier_feasible_domains(vnf: dict, abstract_topology: dict) -> list[str]:
    """D(tau_fk): domains whose dominant_tiers overlap the VNF's permitted_tiers (v6 5.2)."""
    permitted = set(vnf.get("permitted_tiers", []))
    out = []
    for d in abstract_topology.get("domains", []):
        if permitted & set(d.get("dominant_tiers", [])):
            out.append(d["domain_id"])
    return out


def build_pinned_plan_schema(slice_request: dict, abstract_topology: dict) -> dict | None:
    """Per-request JSON schema that PINS the interface contract, not just shape.

    Beyond the shape the static AGENT_B_PLAN_JSON_SCHEMA fixes, this constrains decoding to the
    v6 5.2 interface: each suggested domain m~(f_k) must lie in D(tau_fk), the tier-feasible set,
    identical to the MDO action-space mask (5 3.4). It uses PER-POSITION (tuple) assignment
    schemas (server honors both draft-07 items-list and 2020-12 prefixItems):
      - position i is pinned to VNF i's id via a singleton enum (exact bijection, no omit/dup),
      - domain at position i is an enum of ONLY that VNF's tier-feasible domains D(tau_fk),
      - required_tier is a permissive enum (recomputed deterministically post-generation).
    Grammar-valid output can no longer name a nonexistent VNF, an invalid domain, OR a
    tier-infeasible domain (a contract violation the RL arms structurally cannot make). Only
    genuine C4 (resource) / C5 (inter-domain bandwidth/reachability) infeasibility can remain --
    exactly the prior-quality signal the model owns.

    Returns None if any VNF has NO tier-feasible domain: the slice is genuinely unplaceable, so
    the caller should structural-reject WITHOUT an LLM call (no schema can rescue it).
    """
    vnfs = slice_request.get("vnfs", [])
    vnf_ids = [v["vnf_id"] for v in vnfs]

    def assignment_for(v):
        feas = tier_feasible_domains(v, abstract_topology)
        if not feas:
            return None  # signals genuine tier-infeasibility for this VNF
        tiers = sorted(v.get("permitted_tiers", []))
        return {
            "type": "object",
            "properties": {
                "vnf_id": {"enum": [v["vnf_id"]]},          # pin position -> VNF
                "domain": {"enum": feas},                    # D(tau_fk) only
                "required_tier": ({"enum": tiers} if tiers else {"type": "string"}),
                "cpu_demand": {"type": "number"},
                "ram_demand": {"type": "number"},
            },
            "required": ["vnf_id", "domain", "required_tier", "cpu_demand", "ram_demand"],
            "additionalProperties": False,
        }

    per_position = []
    for v in vnfs:
        a = assignment_for(v)
        if a is None:
            return None  # genuinely infeasible slice
        per_position.append(a)

    flow = {
        "type": "object",
        "properties": {
            "source_vnf": {"enum": vnf_ids},
            "target_vnf": {"enum": vnf_ids},
            "min_bandwidth_mbps": {"type": "number"},
            "crosses_domain_boundary": {"type": "boolean"},
        },
        "required": ["source_vnf", "target_vnf", "min_bandwidth_mbps", "crosses_domain_boundary"],
        "additionalProperties": False,
    }
    k = len(vnf_ids)
    return {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string"},
            # per-position tuple validation (draft-07 items-list): exactly K, position i == VNF i
            "vnf_assignments": {"type": "array", "minItems": k, "maxItems": k, "items": per_position},
            "flow_requirements": {"type": "array", "items": flow},
        },
        "required": ["plan_id", "vnf_assignments", "flow_requirements"],
        "additionalProperties": False,
    }


def recompute_required_tiers(plan: dict, slice_request: dict, abstract_topology: dict) -> None:
    """Set each assignment's required_tier to a tier in (permitted ∩ chosen-domain tiers), in place.

    required_tier must be BOTH in the VNF's permitted set and supported by the assigned domain
    (C8). With domain pinned to a tier-feasible one, that intersection is non-empty; the model's
    stated required_tier can still mismatch the specific chosen domain, so it is recomputed like
    crosses_domain_boundary -- a derived field, not a placement decision.
    """
    permitted = {v["vnf_id"]: set(v.get("permitted_tiers", []))
                 for v in slice_request.get("vnfs", [])}
    dom_tiers = {d["domain_id"]: set(d.get("dominant_tiers", []))
                 for d in abstract_topology.get("domains", [])}
    for a in plan.get("vnf_assignments", []) or []:
        vid, dom = a.get("vnf_id"), a.get("domain")
        inter = permitted.get(vid, set()) & dom_tiers.get(dom, set())
        if inter:
            a["required_tier"] = sorted(inter)[0]


def recompute_flow_boundaries(plan: dict) -> None:
    """Set each flow's crosses_domain_boundary from the plan's OWN assignments, in place.

    crosses_domain_boundary is 100% derived from the VNF->domain assignments, yet the LLM emits
    it as a free field and gets it inconsistent with its own placement on longer chains
    (validity probe 2026-07-10: the dominant post-enum failure — grammar-valid, correct
    partition, rejected only over this computable boolean). Recomputing it does NOT touch the
    LLM's partition decision; it fixes a field the model should never have owned. Genuine
    inter-domain feasibility (C5 reachability/bandwidth, C8 tier) is untouched and still checked.
    """
    dom = {a.get("vnf_id"): a.get("domain") for a in plan.get("vnf_assignments", [])}
    for f in plan.get("flow_requirements", []) or []:
        sd, td = dom.get(f.get("source_vnf")), dom.get(f.get("target_vnf"))
        if sd is not None and td is not None:
            f["crosses_domain_boundary"] = (sd != td)


class PlanTruncationError(ValueError):
    """Raised when the completion hit the context window (finish_reason=length)
    rather than closing naturally. Subclasses ValueError so existing parse-error
    handling still catches it, but lets callers count truncation distinctly from
    a genuine malformed-JSON parse failure (Amendment 3)."""


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
  ]
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
        plan_schema: dict | None = None,
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
        raw = self.llm.complete(
            self.system_prompt, user_msg,
            response_format={"type": "json_object",
                             "schema": plan_schema or AGENT_B_PLAN_JSON_SCHEMA},
        )
        # Distinguish context-window truncation from a genuine parse failure:
        # a truncated grammar-constrained completion is cut-off JSON, but the
        # cause (prompt too large for n_ctx) and the fix are entirely different.
        if getattr(self.llm, "last_finish_reason", None) == "length":
            raise PlanTruncationError(
                "Agent B completion truncated at n_ctx "
                f"(prompt_tokens={getattr(self.llm, 'last_prompt_tokens', None)}, "
                f"n_ctx={getattr(self.llm.config, 'n_ctx', None)})"
            )
        plan = extract_json(raw)
        # Defensive secondary check (v6.5): with grammar-constrained decoding this
        # must already hold. If it ever fires, the constrained path has regressed —
        # log loudly rather than silently repair the model's output.
        try:
            AgentBPlanSchema.model_validate(plan)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "agent_b_schema_defensive_fired",
                extra={"error": str(exc)[:200]},
            )
        return plan

    def generate_and_check(
        self,
        slice_request: dict,
        abstract_topology: dict,
        few_shot_examples: list[dict] | None = None,
        max_retries: int = 1,
        reference_knowledge: str | None = None,
        plan_schema: dict | None = None,
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
        # Fence: grammar-constrained decoding makes a JSON parse failure
        # unreachable, but initialise so the parse-failure branch can never
        # return an unbound `plan` (the former UnboundLocalError at max_retries=0).
        plan: dict = {}

        for attempt in range(1 + max_retries):
            try:
                with profiled("llm.generate", {"attempt": attempt + 1}):
                    plan = self.generate_plan(
                        slice_request, abstract_topology,
                        few_shot_examples, violation_feedback, reference_knowledge,
                        plan_schema=plan_schema,
                    )
            except PlanTruncationError as exc:
                # Context-window exhaustion, NOT malformed output. Counted
                # distinctly so an arm-asymmetric truncation regression is
                # visible in results rather than hiding as a parse failure.
                logger.warning(
                    "agent_b_completion_truncated",
                    extra={"attempt": attempt + 1, "detail": str(exc)[:200]},
                )
                result = CheckResult(is_valid=False, violations=[])
                violation_feedback = "Your previous response was cut off. Respond with a single, complete, minimal JSON object."
                continue
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

            # Fix the derived fields (crosses_domain_boundary flag, required_tier) before
            # validating, so a valid partition is not rejected over values the LLM should not own.
            recompute_flow_boundaries(plan)
            recompute_required_tiers(plan, slice_request, abstract_topology)
            with profiled("struct.check", {"attempt": attempt + 1}):
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
        plan_schema: dict | None = None,
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
            plan_schema=plan_schema,
        )
