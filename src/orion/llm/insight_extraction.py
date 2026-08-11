"""ExpeL insight extraction: experience distilled into rules, not cases.

ORION cites ExpeL (arXiv:2308.10144) for successes-only retrieval and
implemented only that half. ExpeL pairs retrieval with insight extraction, and
its ablation puts the two at near parity. This is the missing half.

Why it is worth building here. The paired copy test (2026-08-05,
`docs/MB_EXEMPLAR_ABSTRACTION_2026-08-05.md`) measured M^B at +0.0 pp with
concrete exemplars and +0.7 pp with abstract ones, both at McNemar p = 1.0, on
149 matched decisions. Cases are consulted only when one is retrievable and only
on plan-cache misses; a rule list rides in every prompt and costs no per-arrival
retrieval.

Faithful to the paper's mechanics:
  * two comparison modes, a failed against a succeeded attempt at the SAME kind
    of slice, and sets of L successes from different slices;
  * four operations, ADD, EDIT, UPVOTE, DOWNVOTE;
  * a new insight starts at importance 2, UPVOTE and EDIT increment, DOWNVOTE
    decrements, and an insight is dropped when importance reaches 0.

One deviation, recorded rather than hidden: rules are forbidden from naming a
domain. The copy test established that this model copies any concrete
assignment it is shown into a network that no longer fits it, so a rule saying
"place URLLC in d0" would reintroduce the defect the abstraction removed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from orion.llm.llm_backend import extract_json

logger = logging.getLogger(__name__)

MAX_INSIGHTS = 8
SUCCESS_BATCH = 4          # L in the paper's "sets of L successes"
NEW_INSIGHT_IMPORTANCE = 2

INSIGHT_SYSTEM_PROMPT = """\
You maintain a short list of reusable rules for placing network slices onto a \
multi-domain substrate.

A rule states a CONDITION and an ACTION. It must generalise beyond the example \
it came from, so never restate a slice: "URLLC, chain of 2, permitted tiers \
[edge]" is a description, not a rule.

Good rules:
  If a chain is short and its tier still has headroom above 0.4, co-locate it.
  When edge headroom drops below 0.25, split the chain and push later VNFs to \
regional tiers rather than forcing them onto edge.
  Prefer fewer domains unless the binding tier is nearly exhausted.

Rules must NEVER name a specific domain. Domains change occupancy constantly, \
so a rule naming one is wrong as soon as the network moves.

Reply with operations on the rule list. Output only JSON."""

# A rule has to read like a rule. The 8B model's first instinct is to echo the
# evidence back as an "insight" (measured: 2 of 2 extracted rules on the first
# run were verbatim slice descriptions), which is the same copying pathology one
# level up. This is a FORM requirement fixed in advance, not a filter tuned
# against the outcome.
_RULE_MARKERS = ("if ", "when ", "unless ", "prefer ", "avoid ", "do not ",
                 "don't ", "never ", "always ", "before ", "after ", "once ")

# The format exemplars in the system prompt fix the rule FORM, and the model
# then proposes them back as if it had learned them (measured: 1 of 3 rules on
# the second live extraction was exemplar 1 verbatim). A rule the prompt handed
# over is not evidence, so exemplars are blocked from the list they demonstrate.
_EXEMPLAR_RULES = (
    "if a chain is short and its tier still has headroom above 0.4, co-locate it.",
    "when edge headroom drops below 0.25, split the chain and push later vnfs to "
    "regional tiers rather than forcing them onto edge.",
    "prefer fewer domains unless the binding tier is nearly exhausted.",
)


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _looks_like_a_rule(text: str) -> bool:
    low = _norm(text)
    if len(low) < 25:
        return False
    if low in _EXEMPLAR_RULES:
        return False
    if "chain of" in low and not any(m in low for m in _RULE_MARKERS):
        return False
    return any(m in low for m in _RULE_MARKERS)

_OPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "op": {"enum": ["ADD", "EDIT", "UPVOTE", "DOWNVOTE"]},
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["op"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["operations"],
    "additionalProperties": False,
}


@dataclass
class Insight:
    """One rule of thumb and its importance count."""

    text: str
    importance: int = NEW_INSIGHT_IMPORTANCE


@dataclass
class Experience:
    """One placement attempt, in the abstract vocabulary the prompt uses."""

    slice_type: str
    chain_length: int
    permitted_tiers: list[str]
    strategy: str | None            # co-locate | split
    n_domains: int
    tiers_used: list[str]
    condition: dict | None
    admitted: bool
    key: tuple = field(init=False)

    def __post_init__(self) -> None:
        # "Same task" for the success/fail pairing, in the paper's sense.
        self.key = (self.slice_type, self.chain_length)

    def render(self) -> str:
        from orion.llm.agent_b import _condition_line

        lines = [
            f"Slice: {self.slice_type}, chain of {self.chain_length}, "
            f"permitted tiers {sorted(self.permitted_tiers)}",
        ]
        if self.strategy:
            lines.append(f"Attempt: {self.strategy} across {self.n_domains} domain(s)"
                         + (f", tiers {self.tiers_used}" if self.tiers_used else ""))
        cond = _condition_line(self.condition)
        if cond:
            lines.append(f"Network at the time: {cond}")
        lines.append(f"Result: {'ADMITTED' if self.admitted else 'REJECTED'}")
        return "\n".join(lines)


def _render_rules(insights: Sequence[Insight]) -> str:
    if not insights:
        return "(no rules yet)"
    return "\n".join(f"{i}. [importance {ins.importance}] {ins.text}"
                     for i, ins in enumerate(insights, 1))


def _apply(insights: list[Insight], ops: Iterable[dict], max_insights: int) -> None:
    """Apply one batch of operations in place, per the paper's counting rules."""
    for op in ops:
        kind = str(op.get("op", "")).upper()
        idx = op.get("index")
        text = (op.get("text") or "").strip()

        if kind == "ADD":
            if not text or len(insights) >= max_insights:
                continue
            if any(_norm(t.text) == _norm(text) for t in insights):
                continue
            if not _looks_like_a_rule(text):
                logger.info("insight_rejected_not_a_rule", extra={"text": text[:120]})
                continue
            insights.append(Insight(text))
            continue

        # The remaining three address an existing rule; a bad index is a no-op
        # rather than an error, since a constrained decode can still miscount.
        if not isinstance(idx, int) or not (1 <= idx <= len(insights)):
            continue
        target = insights[idx - 1]
        if kind == "EDIT":
            if text and _looks_like_a_rule(text):
                target.text = text
            target.importance += 1
        elif kind == "UPVOTE":
            target.importance += 1
        elif kind == "DOWNVOTE":
            target.importance -= 1

    insights[:] = [ins for ins in insights if ins.importance > 0]


def _call(llm, insights: list[Insight], body: str, max_insights: int) -> None:
    user = (
        f"Existing rules:\n{_render_rules(insights)}\n\n"
        f"{body}\n\n"
        f"Propose at most 4 operations. ADD a rule only if it is not already "
        f"covered. UPVOTE a rule this evidence supports, DOWNVOTE one it "
        f"contradicts, EDIT one that is nearly right. Keep the list under "
        f"{max_insights} rules. Never name a domain, and never restate the "
        f"evidence: every rule must say WHEN it applies and WHAT to do."
    )
    try:
        raw = llm.complete(INSIGHT_SYSTEM_PROMPT, user,
                           response_format={"type": "json_object", "schema": _OPS_SCHEMA})
        ops = extract_json(raw).get("operations", [])
    except Exception as exc:  # noqa: BLE001
        # A failed extraction call costs one batch of evidence, not the run.
        logger.warning("insight_extraction_call_failed", extra={"error": str(exc)[:200]})
        return
    _apply(insights, ops, max_insights)


def extract_insights(
    experiences: Sequence[Experience],
    llm,
    max_insights: int = MAX_INSIGHTS,
    success_batch: int = SUCCESS_BATCH,
) -> list[Insight]:
    """Distil experiences into a rule list, ExpeL section 3.2.

    Pass 1 compares a rejected attempt against an admitted one at the same
    (slice_type, chain_length). Pass 2 walks batches of admitted attempts from
    different slices. Both passes edit one shared list, so a rule proposed from
    a failure pair can be upvoted by a later success batch.
    """
    insights: list[Insight] = []
    if not experiences:
        return insights

    by_key: dict[tuple, dict[bool, list[Experience]]] = {}
    for e in experiences:
        by_key.setdefault(e.key, {True: [], False: []})[e.admitted].append(e)

    n_pairs = 0
    for key, groups in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        for fail, ok in zip(groups[False], groups[True]):
            _call(llm, insights,
                  "Two attempts at the same kind of slice, one rejected and one "
                  f"admitted.\n\nREJECTED:\n{fail.render()}\n\n"
                  f"ADMITTED:\n{ok.render()}",
                  max_insights)
            n_pairs += 1

    successes = [e for e in experiences if e.admitted]
    n_batches = 0
    for i in range(0, len(successes), success_batch):
        batch = successes[i:i + success_batch]
        if len(batch) < 2:
            break
        rendered = "\n\n".join(f"[{j}]\n{e.render()}" for j, e in enumerate(batch, 1))
        _call(llm, insights,
              f"{len(batch)} admitted attempts at different slices. Identify what "
              f"they have in common that is worth reusing.\n\n{rendered}",
              max_insights)
        n_batches += 1

    logger.info("insights_extracted", extra={
        "n_experiences": len(experiences), "n_fail_success_pairs": n_pairs,
        "n_success_batches": n_batches, "n_insights": len(insights),
        "llm_calls": n_pairs + n_batches,
    })
    return insights


def format_insights(insights: Sequence[Insight]) -> str:
    """The prompt block. Ordered by importance, the paper's concat(i1,i2,...)."""
    live = [i for i in insights if i.importance > 0]
    if not live:
        return ""
    live = sorted(live, key=lambda i: -i.importance)
    body = "\n".join(f"{n}. {i.text}" for n, i in enumerate(live, 1))
    return ("--- Learned Rules (distilled from past outcomes) ---\n"
            "Apply these to the current network state.\n" + body)
