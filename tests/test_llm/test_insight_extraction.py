"""Tests for ExpeL insight extraction. No live LLM.

The counting rules are the part worth pinning: a miscount silently changes which
rules survive into every Agent B prompt, and no acceptance number would say so.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from orion.llm.insight_extraction import (
    Experience,
    Insight,
    NEW_INSIGHT_IMPORTANCE,
    _apply,
    _EXEMPLAR_RULES,
    extract_insights,
    format_insights,
)


def _mock_llm(responses):
    llm = MagicMock()
    llm.complete = MagicMock(side_effect=[json.dumps(r) for r in responses])
    return llm


def _exp(slice_type="URLLC", k=2, admitted=True, strategy="co-locate", n_dom=1):
    return Experience(
        slice_type=slice_type, chain_length=k, permitted_tiers=["edge"],
        strategy=strategy, n_domains=n_dom, tiers_used=["edge"] * k,
        condition={"cpu_residual_frac": 0.6, "bucket": "moderate",
                   "tier_cpu_residual": {"edge": 0.4}},
        admitted=admitted,
    )


class TestOperationCounting:

    def test_add_starts_at_two(self):
        ins: list[Insight] = []
        _apply(ins, [{"op": "ADD",
                      "text": "If a URLLC chain is short, co-locate it at edge."}], 8)
        assert len(ins) == 1
        assert ins[0].importance == NEW_INSIGHT_IMPORTANCE == 2

    def test_echoed_evidence_is_rejected(self):
        """The 8B model's first instinct is to restate the slice as an insight.

        Measured on the first live extraction: both rules it proposed were
        verbatim slice descriptions. A description in the rule block is worse
        than no rule, since it spends prompt budget teaching nothing.
        """
        ins: list[Insight] = []
        _apply(ins, [{"op": "ADD", "text": "URLLC, chain of 2, permitted tiers ['edge']"}], 8)
        _apply(ins, [{"op": "ADD", "text": "eMBB, chain of 3, co-locate at edge tier"}], 8)
        assert ins == []
        _apply(ins, [{"op": "ADD",
                      "text": "When edge headroom is below 0.25, split the chain."}], 8)
        assert len(ins) == 1

    def test_prompt_exemplars_cannot_be_learned_back(self):
        """A rule the system prompt handed over is not evidence.

        The format exemplars fix the rule form and the model then proposes them
        back as findings: 1 of 3 rules on the second live extraction was
        exemplar 1 verbatim. Without this the rule block reports the prompt to
        itself and reads as if experience produced it.
        """
        ins: list[Insight] = []
        for ex in _EXEMPLAR_RULES:
            _apply(ins, [{"op": "ADD", "text": ex}], 8)
        assert ins == []

    def test_upvote_and_edit_increment(self):
        ins = [Insight("rule a")]
        _apply(ins, [{"op": "UPVOTE", "index": 1}], 8)
        assert ins[0].importance == 3
        _apply(ins, [{"op": "EDIT", "index": 1,
                      "text": "If edge is tight, prefer splitting the chain."}], 8)
        assert ins[0].importance == 4
        assert ins[0].text == "If edge is tight, prefer splitting the chain."

    def test_downvote_to_zero_removes(self):
        ins = [Insight("rule a")]
        _apply(ins, [{"op": "DOWNVOTE", "index": 1}], 8)
        assert ins[0].importance == 1
        _apply(ins, [{"op": "DOWNVOTE", "index": 1}], 8)
        assert ins == []

    def test_cap_respected(self):
        ins = [Insight(f"rule {i}") for i in range(8)]
        _apply(ins, [{"op": "ADD", "text": "one too many"}], 8)
        assert len(ins) == 8

    def test_duplicate_add_ignored(self):
        ins = [Insight("If a chain is short, co-locate it.")]
        _apply(ins, [{"op": "ADD", "text": "IF A CHAIN IS SHORT, CO-LOCATE IT."}], 8)
        assert len(ins) == 1

    @pytest.mark.parametrize("idx", [0, 5, -1, None, "1"])
    def test_bad_index_is_a_noop(self, idx):
        ins = [Insight("rule a")]
        _apply(ins, [{"op": "UPVOTE", "index": idx}], 8)
        assert ins[0].importance == NEW_INSIGHT_IMPORTANCE


class TestExtraction:

    def test_pairs_failures_with_successes_on_same_task(self):
        llm = _mock_llm([{"operations": [{"op": "ADD",
            "text": "If the binding tier is tight, prefer splitting."}]}] * 4)
        exps = [_exp(admitted=False), _exp(admitted=True),
                _exp(slice_type="eMBB", k=3, admitted=True),
                _exp(slice_type="eMBB", k=3, admitted=True)]
        out = extract_insights(exps, llm, max_insights=8, success_batch=4)
        # one fail/success pair on (URLLC,2) plus one batch over the 3 successes
        assert llm.complete.call_count == 2
        assert out and all(i.importance > 0 for i in out)

    def test_no_experiences_makes_no_calls(self):
        llm = _mock_llm([])
        assert extract_insights([], llm) == []
        assert llm.complete.call_count == 0

    def test_llm_failure_costs_a_batch_not_the_run(self):
        llm = MagicMock()
        llm.complete = MagicMock(side_effect=[RuntimeError("server wedged"),
                                              json.dumps({"operations": [{"op": "ADD",
            "text": "If the binding tier is tight, prefer splitting."}]})])
        exps = [_exp(admitted=False), _exp(admitted=True), _exp(admitted=True)]
        out = extract_insights(exps, llm, success_batch=4)
        assert [i.text for i in out] == ["If the binding tier is tight, prefer splitting."]

    def test_prompt_forbids_naming_a_domain(self):
        llm = _mock_llm([{"operations": []}] * 4)
        extract_insights([_exp(admitted=False), _exp(admitted=True)], llm)
        system, user = llm.complete.call_args[0][0], llm.complete.call_args[0][1]
        assert "NEVER name a specific domain" in system
        assert "Never name a domain" in user


class TestFormatting:

    def test_orders_by_importance_and_drops_dead(self):
        ins = [Insight("low", 1), Insight("high", 5), Insight("dead", 0)]
        block = format_insights(ins)
        assert "dead" not in block
        assert block.index("high") < block.index("low")
        assert "Learned Rules" in block

    def test_empty_gives_empty_string(self):
        assert format_insights([]) == ""
        assert format_insights([Insight("x", 0)]) == ""

    def test_block_reaches_the_prompt_above_the_cases(self):
        from orion.llm.agent_b import build_user_prompt
        block = format_insights([Insight("Split long chains when edge is tight.")])
        prompt = build_user_prompt({"vnfs": []}, {"domains": []}, insights=block)
        assert "Learned Rules" in prompt
        assert prompt.index("Learned Rules") < prompt.index("--- Current Task ---")
