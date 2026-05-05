"""Tests for Agent B module — prompt building, plan parsing, and retry logic.

These tests do NOT require a live LLM. They test the prompt construction,
JSON extraction, and the structural-check retry loop using mock responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orion.llm.agent_b import AgentB, build_user_prompt, SYSTEM_PROMPT
from orion.llm.llm_backend import LLMBackend, LLMConfig, extract_json

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "placement_eval"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def topology() -> dict:
    return json.loads((DATA_DIR / "abstract_topology.json").read_text())


@pytest.fixture
def slice_request() -> dict:
    return json.loads((DATA_DIR / "slice_request.json").read_text())


@pytest.fixture
def few_shot_examples() -> list[dict]:
    return json.loads((DATA_DIR / "few_shot_examples.json").read_text())


def _valid_plan_json() -> str:
    """JSON string of a valid plan for xr_telepresence_005."""
    return json.dumps({
        "plan_id": "xr_telepresence_005_plan",
        "vnf_assignments": [
            {"vnf_id": "xr_telepresence_005_f1", "domain": "d0",
             "required_tier": "mec", "cpu_demand": 4.0, "ram_demand": 8.0},
            {"vnf_id": "xr_telepresence_005_f2", "domain": "d1",
             "required_tier": "mec", "cpu_demand": 14.0, "ram_demand": 30.0},
            {"vnf_id": "xr_telepresence_005_f3", "domain": "d1",
             "required_tier": "mec", "cpu_demand": 10.0, "ram_demand": 25.0},
            {"vnf_id": "xr_telepresence_005_f4", "domain": "d2",
             "required_tier": "regional_cloud", "cpu_demand": 18.0,
             "ram_demand": 45.0},
        ],
        "flow_requirements": [
            {"source_vnf": "xr_telepresence_005_f1",
             "target_vnf": "xr_telepresence_005_f2",
             "min_bandwidth_mbps": 300.0, "crosses_domain_boundary": True},
            {"source_vnf": "xr_telepresence_005_f2",
             "target_vnf": "xr_telepresence_005_f3",
             "min_bandwidth_mbps": 250.0, "crosses_domain_boundary": False},
            {"source_vnf": "xr_telepresence_005_f3",
             "target_vnf": "xr_telepresence_005_f4",
             "min_bandwidth_mbps": 150.0, "crosses_domain_boundary": True},
        ],
        "rationale": "Valid plan.",
    })


def _invalid_plan_json() -> str:
    """JSON string of a plan with CPU overcommit in d1 (28 > 25)."""
    return json.dumps({
        "plan_id": "xr_telepresence_005_plan",
        "vnf_assignments": [
            {"vnf_id": "xr_telepresence_005_f1", "domain": "d1",
             "required_tier": "mec", "cpu_demand": 4.0, "ram_demand": 8.0},
            {"vnf_id": "xr_telepresence_005_f2", "domain": "d1",
             "required_tier": "mec", "cpu_demand": 14.0, "ram_demand": 30.0},
            {"vnf_id": "xr_telepresence_005_f3", "domain": "d1",
             "required_tier": "mec", "cpu_demand": 10.0, "ram_demand": 25.0},
            {"vnf_id": "xr_telepresence_005_f4", "domain": "d2",
             "required_tier": "regional_cloud", "cpu_demand": 18.0,
             "ram_demand": 45.0},
        ],
        "flow_requirements": [
            {"source_vnf": "xr_telepresence_005_f1",
             "target_vnf": "xr_telepresence_005_f2",
             "min_bandwidth_mbps": 300.0, "crosses_domain_boundary": False},
            {"source_vnf": "xr_telepresence_005_f2",
             "target_vnf": "xr_telepresence_005_f3",
             "min_bandwidth_mbps": 250.0, "crosses_domain_boundary": False},
            {"source_vnf": "xr_telepresence_005_f3",
             "target_vnf": "xr_telepresence_005_f4",
             "min_bandwidth_mbps": 150.0, "crosses_domain_boundary": True},
        ],
        "rationale": "Bad plan — CPU overcommit in d1.",
    })


def _make_mock_llm(responses: list[str]) -> LLMBackend:
    """Create a mock LLMBackend that returns pre-set responses in order."""
    mock = MagicMock(spec=LLMBackend)
    mock.complete = MagicMock(side_effect=responses)
    return mock


# ── Tests: prompt construction ───────────────────────────────────────────────

class TestPromptBuilding:

    def test_prompt_contains_topology(self, slice_request, topology):
        prompt = build_user_prompt(slice_request, topology)
        assert "Abstract Topology" in prompt
        assert "d0" in prompt
        assert "cpu_residual" in prompt

    def test_prompt_contains_slice_request(self, slice_request, topology):
        prompt = build_user_prompt(slice_request, topology)
        assert slice_request["request_id"] in prompt
        assert "Firewall" in prompt

    def test_prompt_includes_few_shot(self, slice_request, topology, few_shot_examples):
        prompt = build_user_prompt(slice_request, topology, few_shot_examples)
        assert "Example 1" in prompt
        assert "Example 2" in prompt
        assert "urllc_factory_002" in prompt

    def test_prompt_includes_violation_feedback(self, slice_request, topology):
        prompt = build_user_prompt(
            slice_request, topology,
            violation_feedback="C4: Domain 'd1' CPU overcommit: 28.0 > 25.0",
        )
        assert "PREVIOUS ATTEMPT FAILED" in prompt
        assert "CPU overcommit" in prompt

    def test_prompt_without_optional_fields(self, slice_request, topology):
        prompt = build_user_prompt(slice_request, topology)
        assert "Example" not in prompt
        assert "PREVIOUS ATTEMPT FAILED" not in prompt


# ── Tests: JSON extraction ───────────────────────────────────────────────────

class TestJSONExtraction:

    def test_plain_json(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self):
        text = "Here is the plan:\n```json\n{\"key\": 42}\n```\nDone."
        result = extract_json(text)
        assert result == {"key": 42}

    def test_surrounded_by_prose(self):
        text = 'I think this works: {"a": 1, "b": 2} and that is my answer.'
        result = extract_json(text)
        assert result == {"a": 1, "b": 2}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("This has no JSON at all.")


# ── Tests: Agent B generate_and_check ────────────────────────────────────────

class TestAgentBGenerateAndCheck:

    def test_valid_plan_passes_on_first_attempt(self, slice_request, topology):
        mock_llm = _make_mock_llm([_valid_plan_json()])
        agent = AgentB(llm=mock_llm)

        plan, result = agent.generate_and_check(slice_request, topology)
        assert result.is_valid
        assert mock_llm.complete.call_count == 1

    def test_invalid_then_valid_retries_once(self, slice_request, topology):
        mock_llm = _make_mock_llm([_invalid_plan_json(), _valid_plan_json()])
        agent = AgentB(llm=mock_llm)

        plan, result = agent.generate_and_check(
            slice_request, topology, max_retries=1,
        )
        assert result.is_valid
        assert mock_llm.complete.call_count == 2

        # Second call should include violation feedback
        second_call_args = mock_llm.complete.call_args_list[1]
        user_msg = second_call_args[0][1]  # positional arg: user_message
        assert "PREVIOUS ATTEMPT FAILED" in user_msg
        assert "C4" in user_msg

    def test_all_retries_exhausted(self, slice_request, topology):
        mock_llm = _make_mock_llm([_invalid_plan_json(), _invalid_plan_json()])
        agent = AgentB(llm=mock_llm)

        plan, result = agent.generate_and_check(
            slice_request, topology, max_retries=1,
        )
        assert not result.is_valid
        assert mock_llm.complete.call_count == 2

    def test_json_parse_failure_triggers_retry(self, slice_request, topology):
        mock_llm = _make_mock_llm(["not json at all", _valid_plan_json()])
        agent = AgentB(llm=mock_llm)

        plan, result = agent.generate_and_check(
            slice_request, topology, max_retries=1,
        )
        assert result.is_valid
        assert mock_llm.complete.call_count == 2

    def test_no_retries_returns_first_result(self, slice_request, topology):
        mock_llm = _make_mock_llm([_invalid_plan_json()])
        agent = AgentB(llm=mock_llm)

        plan, result = agent.generate_and_check(
            slice_request, topology, max_retries=0,
        )
        assert not result.is_valid
        assert mock_llm.complete.call_count == 1

    def test_few_shot_passed_to_prompt(self, slice_request, topology, few_shot_examples):
        mock_llm = _make_mock_llm([_valid_plan_json()])
        agent = AgentB(llm=mock_llm)

        agent.generate_and_check(
            slice_request, topology,
            few_shot_examples=few_shot_examples,
        )
        user_msg = mock_llm.complete.call_args[0][1]
        assert "urllc_factory_002" in user_msg


# ── Tests: abstract topology integration ─────────────────────────────────────

class TestAbstractTopologyIntegration:

    def test_topology_from_substrate(self):
        """Build abstract topology from a real SubstrateNetwork."""
        from orion.substrate.topology_generator import generate_multi_domain_topology
        from orion.llm.abstract_topology import build_abstract_topology
        from orion.config import TopologyConfig
        import numpy as np

        rng = np.random.default_rng(42)
        config = TopologyConfig(
            num_domains=3,
            nodes_per_domain=[4, 5, 6],
        )
        substrate = generate_multi_domain_topology(config, rng)
        substrate.reset()

        topo = build_abstract_topology(substrate)

        assert len(topo["domains"]) == 3
        assert all("cpu_residual" in d for d in topo["domains"])
        assert all("dominant_tiers" in d for d in topo["domains"])
        assert len(topo["inter_domain_links"]) > 0
        assert all("bandwidth_residual_mbps" in l for l in topo["inter_domain_links"])

        # Domain IDs should be d0, d1, d2
        domain_ids = {d["domain_id"] for d in topo["domains"]}
        assert domain_ids == {"d0", "d1", "d2"}

        # CPU residual should match sum of node capacities (fresh substrate)
        for dom in topo["domains"]:
            did = int(dom["domain_id"][1:])
            nodes = substrate.nodes_in_domain(did)
            expected_cpu = sum(substrate.graph.nodes[n]["cpu_capacity"] for n in nodes)
            assert abs(dom["cpu_residual"] - expected_cpu) < 0.1
