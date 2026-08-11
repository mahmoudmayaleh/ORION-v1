"""Tests for Agent A module -- intent translation, validation, and memory integration.

These tests do NOT require a live LLM. They test slice spec validation,
prompt construction, retry logic, and episodic memory recording using
mock responses.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from orion.llm.agent_a import (
    AgentA,
    ValidationResult,
    record_translation,
    validate_slice_spec,
)
from orion.llm.llm_backend import LLMBackend


# -- Valid response templates --------------------------------------------------

VALID_EMBB_SPEC = {
    "request_id": "test_001",
    "slice_type": "eMBB",
    "vnfs": [
        {
            "vnf_id": "test_001_f0",
            "vnf_type": "Firewall",
            "cpu_demand": 2.0,
            "ram_demand": 4.0,
            "permitted_tiers": ["edge", "regional_cloud"],
            "computational_intensity": 0.8,
            "vcr": 1.0,
        },
        {
            "vnf_id": "test_001_f1",
            "vnf_type": "CDN",
            "cpu_demand": 6.0,
            "ram_demand": 12.0,
            "permitted_tiers": ["edge", "regional_cloud"],
            "computational_intensity": 1.2,
            "vcr": 0.7,
        },
    ],
    "flow_edges": [
        {
            "source_vnf": "test_001_f0",
            "target_vnf": "test_001_f1",
            "bandwidth_demand": 100.0,
        }
    ],
    "qos": {"max_e2e_delay": 50.0, "min_throughput": 100.0},
}

VALID_EMBB_RESPONSE = json.dumps(VALID_EMBB_SPEC)

VALID_URLLC_SPEC = {
    "request_id": "test_002",
    "slice_type": "URLLC",
    "vnfs": [
        {
            "vnf_id": "test_002_f0",
            "vnf_type": "Packet_Filter",
            "cpu_demand": 4.0,
            "ram_demand": 8.0,
            "permitted_tiers": ["edge"],
            "computational_intensity": 0.5,
            "vcr": 1.0,
        },
        {
            "vnf_id": "test_002_f1",
            "vnf_type": "Encoder",
            "cpu_demand": 8.0,
            "ram_demand": 16.0,
            "permitted_tiers": ["edge", "regional_cloud"],
            "computational_intensity": 1.5,
            "vcr": 0.9,
        },
    ],
    "flow_edges": [
        {
            "source_vnf": "test_002_f0",
            "target_vnf": "test_002_f1",
            "bandwidth_demand": 200.0,
        }
    ],
    "qos": {"max_e2e_delay": 10.0, "min_throughput": 500.0},
}

VALID_URLLC_RESPONSE = json.dumps(VALID_URLLC_SPEC)

VALID_EMBB_3VNF_SPEC = {
    "request_id": "test_003",
    "slice_type": "eMBB",
    "vnfs": [
        {
            "vnf_id": "test_003_f0",
            "vnf_type": "Firewall",
            "cpu_demand": 2.0,
            "ram_demand": 4.0,
            "permitted_tiers": ["edge"],
            "computational_intensity": 0.8,
            "vcr": 1.0,
        },
        {
            "vnf_id": "test_003_f1",
            "vnf_type": "CDN",
            "cpu_demand": 6.0,
            "ram_demand": 12.0,
            "permitted_tiers": ["edge", "regional_cloud"],
            "computational_intensity": 1.2,
            "vcr": 0.7,
        },
        {
            "vnf_id": "test_003_f2",
            "vnf_type": "Transcoder",
            "cpu_demand": 10.0,
            "ram_demand": 20.0,
            "permitted_tiers": ["regional_cloud"],
            "computational_intensity": 2.0,
            "vcr": 0.5,
        },
    ],
    "flow_edges": [
        {
            "source_vnf": "test_003_f0",
            "target_vnf": "test_003_f1",
            "bandwidth_demand": 100.0,
        },
        {
            "source_vnf": "test_003_f1",
            "target_vnf": "test_003_f2",
            "bandwidth_demand": 80.0,
        },
    ],
    "qos": {"max_e2e_delay": 50.0, "min_throughput": 100.0},
}


# -- Mock helpers --------------------------------------------------------------

def _make_mock_llm(responses: list[str]) -> LLMBackend:
    """Create a mock LLMBackend that returns pre-set responses in order."""
    mock = MagicMock(spec=LLMBackend)
    mock.complete = MagicMock(side_effect=responses)
    return mock


# -- TestValidateSliceSpec -----------------------------------------------------

class TestValidateSliceSpec:

    def test_valid_embb_spec(self):
        """A complete eMBB spec with 3 VNFs passes validation."""
        result = validate_slice_spec(VALID_EMBB_3VNF_SPEC)
        assert result.is_valid
        assert result.errors == []

    def test_valid_urllc_spec(self):
        """A complete URLLC spec with 2 VNFs passes validation."""
        result = validate_slice_spec(VALID_URLLC_SPEC)
        assert result.is_valid
        assert result.errors == []

    def test_missing_required_key(self):
        """Spec without 'vnfs' key fails validation."""
        spec = {
            "request_id": "test_bad",
            "slice_type": "eMBB",
            "flow_edges": [],
            "qos": {"max_e2e_delay": 50.0, "min_throughput": 100.0},
        }
        result = validate_slice_spec(spec)
        assert not result.is_valid
        assert any("vnfs" in e.lower() for e in result.errors)

    def test_invalid_slice_type(self):
        """slice_type='INVALID' fails validation."""
        spec = dict(VALID_EMBB_SPEC)
        spec["slice_type"] = "INVALID"
        result = validate_slice_spec(spec)
        assert not result.is_valid
        assert any("slice_type" in e.lower() or "invalid" in e.lower()
                    for e in result.errors)

    def test_vnf_missing_cpu_demand(self):
        """VNF without cpu_demand fails validation."""
        spec = json.loads(json.dumps(VALID_EMBB_SPEC))
        del spec["vnfs"][0]["cpu_demand"]
        result = validate_slice_spec(spec)
        assert not result.is_valid
        assert any("cpu_demand" in e.lower() for e in result.errors)

    def test_negative_cpu_demand(self):
        """cpu_demand < 0 fails validation."""
        spec = json.loads(json.dumps(VALID_EMBB_SPEC))
        spec["vnfs"][0]["cpu_demand"] = -1.0
        result = validate_slice_spec(spec)
        assert not result.is_valid
        assert any("cpu_demand" in e.lower() or "negative" in e.lower()
                    for e in result.errors)

    def test_invalid_tier_name(self):
        """permitted_tiers=['invalid_tier'] fails validation."""
        spec = json.loads(json.dumps(VALID_EMBB_SPEC))
        spec["vnfs"][0]["permitted_tiers"] = ["invalid_tier"]
        result = validate_slice_spec(spec)
        assert not result.is_valid
        assert any("tier" in e.lower() for e in result.errors)

    def test_flow_edges_must_connect_consecutive_vnfs(self):
        """Non-consecutive flow edge fails validation."""
        spec = json.loads(json.dumps(VALID_EMBB_3VNF_SPEC))
        # Replace the consecutive edges with a non-consecutive one
        spec["flow_edges"] = [
            {
                "source_vnf": "test_003_f0",
                "target_vnf": "test_003_f2",
                "bandwidth_demand": 100.0,
            }
        ]
        result = validate_slice_spec(spec)
        assert not result.is_valid
        assert any("consecutive" in e.lower() or "flow" in e.lower()
                    for e in result.errors)

    def test_qos_delay_must_be_positive(self):
        """max_e2e_delay <= 0 fails validation."""
        spec = json.loads(json.dumps(VALID_EMBB_SPEC))
        spec["qos"]["max_e2e_delay"] = 0.0
        result = validate_slice_spec(spec)
        assert not result.is_valid
        assert any("delay" in e.lower() or "qos" in e.lower()
                    for e in result.errors)

    def test_empty_vnf_list_fails(self):
        """vnfs=[] fails validation."""
        spec = json.loads(json.dumps(VALID_EMBB_SPEC))
        spec["vnfs"] = []
        spec["flow_edges"] = []
        result = validate_slice_spec(spec)
        assert not result.is_valid
        assert any("vnf" in e.lower() or "empty" in e.lower()
                    for e in result.errors)


# -- TestAgentATranslate -------------------------------------------------------

class TestAgentATranslate:

    def test_translate_returns_dict(self):
        """translate() returns a parsed dict from the LLM response."""
        mock_llm = _make_mock_llm([VALID_EMBB_RESPONSE])
        agent = AgentA(llm=mock_llm)

        result = agent.translate("Deploy a high-bandwidth eMBB slice with 2 VNFs.")
        assert isinstance(result, dict)
        assert result["slice_type"] == "eMBB"

    def test_translate_passes_intent_to_prompt(self):
        """The intent text appears in the LLM call."""
        mock_llm = _make_mock_llm([VALID_EMBB_RESPONSE])
        agent = AgentA(llm=mock_llm)

        intent = "Deploy a high-bandwidth eMBB slice for video streaming"
        agent.translate(intent)

        assert mock_llm.complete.call_count == 1
        user_msg = mock_llm.complete.call_args[0][1]
        assert intent in user_msg

    def test_translate_includes_reference_knowledge(self):
        """When provided, reference knowledge appears in prompt."""
        mock_llm = _make_mock_llm([VALID_EMBB_RESPONSE])
        agent = AgentA(llm=mock_llm)

        ref_knowledge = "--- Reference Knowledge ---\neMBB slices typically need CDN and Firewall VNFs."
        agent.translate(
            "Deploy an eMBB slice",
            reference_knowledge=ref_knowledge,
        )

        user_msg = mock_llm.complete.call_args[0][1]
        assert "Reference Knowledge" in user_msg
        assert "CDN and Firewall" in user_msg

    def test_translate_includes_few_shot(self):
        """When provided, few-shot examples appear in prompt."""
        mock_llm = _make_mock_llm([VALID_EMBB_RESPONSE])
        agent = AgentA(llm=mock_llm)

        few_shot = [
            {
                "intent": "Deploy a URLLC factory slice",
                "spec": VALID_URLLC_SPEC,
            }
        ]
        agent.translate(
            "Deploy an eMBB slice",
            few_shot_examples=few_shot,
        )

        user_msg = mock_llm.complete.call_args[0][1]
        assert "URLLC" in user_msg or "Example" in user_msg


# -- TestAgentATranslateAndValidate --------------------------------------------

class TestAgentATranslateAndValidate:

    def test_valid_on_first_attempt(self):
        """Mock returns valid spec, passes on attempt 1."""
        mock_llm = _make_mock_llm([VALID_EMBB_RESPONSE])
        agent = AgentA(llm=mock_llm)

        spec, result = agent.translate_and_validate(
            "Deploy an eMBB slice with 2 VNFs",
        )
        assert result.is_valid
        assert mock_llm.complete.call_count == 1

    def test_invalid_then_valid_retries(self):
        """First response has bad schema, second is valid."""
        bad_spec = json.dumps({"slice_type": "eMBB"})  # missing vnfs, etc.
        mock_llm = _make_mock_llm([bad_spec, VALID_EMBB_RESPONSE])
        agent = AgentA(llm=mock_llm)

        spec, result = agent.translate_and_validate(
            "Deploy an eMBB slice",
            max_retries=1,
        )
        assert result.is_valid
        assert mock_llm.complete.call_count == 2

    def test_all_retries_exhausted(self):
        """Both responses invalid, returns last result with is_valid=False."""
        bad_spec = json.dumps({"slice_type": "eMBB"})
        mock_llm = _make_mock_llm([bad_spec, bad_spec])
        agent = AgentA(llm=mock_llm)

        spec, result = agent.translate_and_validate(
            "Deploy an eMBB slice",
            max_retries=1,
        )
        assert not result.is_valid
        assert mock_llm.complete.call_count == 2

    def test_json_parse_failure_triggers_retry(self):
        """First response is not JSON, retry succeeds."""
        mock_llm = _make_mock_llm(["not json at all", VALID_EMBB_RESPONSE])
        agent = AgentA(llm=mock_llm)

        spec, result = agent.translate_and_validate(
            "Deploy an eMBB slice",
            max_retries=1,
        )
        assert result.is_valid
        assert mock_llm.complete.call_count == 2


# -- TestAgentARecordTranslation -----------------------------------------------

class TestAgentARecordTranslation:

    @staticmethod
    def _make_episodic_memory():
        """Create an EpisodicMemory using keyword-only retrieval (no embeddings)."""
        from orion.retrieval import RetrievalConfig, RetrievalMode
        from orion.llm.episodic_memory import EpisodicMemory

        config = RetrievalConfig(mode=RetrievalMode.KEYWORD_ONLY)
        return EpisodicMemory(config=config)

    def test_valid_translation_recorded(self):
        """is_valid=True records with reward=1.0."""
        mb = self._make_episodic_memory()
        recorded = record_translation(
            mb=mb,
            intent_text="Deploy an eMBB slice",
            spec=VALID_EMBB_SPEC,
            is_valid=True,
        )
        assert recorded is True

        entries = mb.retrieve("eMBB", top_k=5)
        assert len(entries) > 0

    def test_invalid_translation_is_stored_but_not_offered_as_an_exemplar(self):
        """is_valid=False records with violations, and stays out of retrieval.

        This test previously asserted the failure WAS retrievable, which is the
        opposite of the design: `retrieve` defaults to `successes_only`
        (RETRIEVE_SUCCESSES_ONLY, following ExpeL, AAAI 2024) precisely so that
        failures stay in the store as negative evidence without ever being handed
        to the planner as something to imitate. It had been failing since that
        behaviour landed.
        """
        mb = self._make_episodic_memory()
        recorded = record_translation(
            mb=mb,
            intent_text="Deploy an eMBB slice",
            spec={"slice_type": "eMBB"},
            is_valid=False,
        )
        assert recorded is True
        assert len(mb._entries) == 1, "the failure was not stored at all"

        assert mb.retrieve("eMBB", top_k=5) == [], (
            "a schema-invalid translation was offered to the planner as an "
            "exemplar to imitate")
        assert len(mb.retrieve("eMBB", top_k=5, successes_only=False)) > 0, (
            "the failure is unreachable even with successes_only off, so it is "
            "not usable as negative evidence either")


# -- TestAgentAWithMemory ------------------------------------------------------

class TestAgentAWithMemory:

    def test_translate_with_memory_kb_only(self):
        """Passes K^A reference text to translate."""
        mock_llm = _make_mock_llm([VALID_EMBB_RESPONSE])
        agent = AgentA(llm=mock_llm)

        # Create a minimal semantic memory mock with retrieve/format methods
        kb = MagicMock()
        kb.retrieve.return_value = [MagicMock()]
        kb.format_for_prompt.return_value = (
            "--- Reference Knowledge ---\neMBB slices typically use mec tier."
        )

        spec, result = agent.translate_with_memory(
            "Deploy an eMBB slice",
            kb=kb,
            mb=None,
        )

        assert mock_llm.complete.call_count >= 1
        user_msg = mock_llm.complete.call_args[0][1]
        assert "Reference Knowledge" in user_msg

    def test_translate_with_memory_no_memory(self):
        """Works with kb=None, mb=None."""
        mock_llm = _make_mock_llm([VALID_EMBB_RESPONSE])
        agent = AgentA(llm=mock_llm)

        spec, result = agent.translate_with_memory(
            "Deploy an eMBB slice",
            kb=None,
            mb=None,
        )

        assert mock_llm.complete.call_count >= 1
        assert isinstance(spec, dict)
        user_msg = mock_llm.complete.call_args[0][1]
        assert "Reference Knowledge" not in user_msg
