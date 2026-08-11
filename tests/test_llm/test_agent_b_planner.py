"""Tests for Agent B plan builder integration.

Tests the conversion pipeline (SliceRequest -> dict -> Agent B -> PlanSummary)
and the quality instrumentation, using a mock LLM backend that returns
deterministic plans.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from orion.config import TopologyConfig
from orion.llm.agent_b import AgentB
from orion.llm.agent_b_planner import (
    PlanQualityTracker,
    make_agent_b_plan_builder,
    plan_summary_from_agent_b,
    slice_request_to_dict,
)
from orion.llm.llm_backend import LLMConfig
from orion.sim.slice_generator import generate_slice_request
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier


class MockLLMBackend:
    """Returns a deterministic plan that co-locates all VNFs in domain 0."""

    def __init__(self, substrate: SubstrateNetwork):
        self._substrate = substrate

    def complete(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """Parse the slice request from the prompt and return a co-located plan."""
        # Extract the slice request JSON from the user message
        import re
        sr_match = re.search(r'"request_id":\s*"([^"]+)"', user_message)
        request_id = sr_match.group(1) if sr_match else "unknown"

        # Find all VNF IDs
        vnf_ids = re.findall(r'"vnf_id":\s*"([^"]+)"', user_message)

        # Find flow edges
        flow_matches = re.findall(
            r'"source_vnf":\s*"([^"]+)".*?"target_vnf":\s*"([^"]+)".*?"bandwidth_demand":\s*([\d.]+)',
            user_message, re.DOTALL,
        )

        # Find per-VNF demands
        vnf_blocks = re.findall(
            r'"vnf_id":\s*"([^"]+)".*?"cpu_demand":\s*([\d.]+).*?"ram_demand":\s*([\d.]+).*?"permitted_tiers":\s*\[([^\]]*)\]',
            user_message, re.DOTALL,
        )

        # Build a co-located plan: all VNFs in domain 0
        dom0_tiers = sorted({
            self._substrate.graph.nodes[n]["tier"]
            for n in self._substrate.nodes_in_domain(0)
        })

        assignments = []
        for vnf_id, cpu, ram, tiers_str in vnf_blocks:
            permitted = [t.strip().strip('"') for t in tiers_str.split(",") if t.strip()]
            # Pick first tier that overlaps with domain 0
            tier = next((t for t in permitted if t in dom0_tiers), permitted[0] if permitted else "edge")
            assignments.append({
                "vnf_id": vnf_id,
                "domain": "d0",
                "required_tier": tier,
                "cpu_demand": float(cpu),
                "ram_demand": float(ram),
            })

        flows = []
        for src, dst, bw in flow_matches:
            flows.append({
                "source_vnf": src,
                "target_vnf": dst,
                "min_bandwidth_mbps": float(bw),
                "crosses_domain_boundary": False,
            })

        plan = {
            "plan_id": f"{request_id}_plan",
            "vnf_assignments": assignments,
            "flow_requirements": flows,
            "rationale": "Co-located all VNFs in domain 0 for minimal latency.",
        }
        return json.dumps(plan)


@pytest.fixture
def substrate():
    rng = np.random.default_rng(42)
    return generate_multi_domain_topology(
        TopologyConfig(num_domains=3, nodes_per_domain=[5, 5, 5],
                       intra_link_density=0.6, inter_domain_links=2),
        rng,
    )


@pytest.fixture
def sample_slice(substrate):
    rng = np.random.default_rng(7)
    return generate_slice_request("req_test_ab", substrate, rng)


class TestSliceRequestToDict:
    def test_round_trips_vnf_ids(self, sample_slice, substrate):
        d = slice_request_to_dict(sample_slice, substrate)
        assert [v["vnf_id"] for v in d["vnfs"]] == [v.vnf_id for v in sample_slice.vnfs]

    def test_derives_permitted_tiers(self, sample_slice, substrate):
        d = slice_request_to_dict(sample_slice, substrate)
        for vnf_dict in d["vnfs"]:
            assert "permitted_tiers" in vnf_dict
            assert isinstance(vnf_dict["permitted_tiers"], list)
            assert len(vnf_dict["permitted_tiers"]) > 0


class TestPlanSummaryFromAgentB:
    def test_converts_valid_plan(self, sample_slice):
        plan = {
            "vnf_assignments": [
                {"vnf_id": v.vnf_id, "domain": "d0",
                 "required_tier": "edge", "cpu_demand": v.cpu_demand,
                 "ram_demand": v.ram_demand}
                for v in sample_slice.vnfs
            ],
        }
        ps = plan_summary_from_agent_b(plan, sample_slice)
        assert ps.num_vnfs == len(sample_slice.vnfs)
        assert all(d == 0 for d in ps.suggested_domains)
        assert ps.vnf_ids == [v.vnf_id for v in sample_slice.vnfs]

    def test_missing_vnf_raises(self, sample_slice):
        plan = {"vnf_assignments": []}
        with pytest.raises(ValueError, match="missing assignment"):
            plan_summary_from_agent_b(plan, sample_slice)


class TestAgentBPlanBuilder:
    def test_produces_plan_summary(self, substrate, sample_slice):
        mock_llm = MockLLMBackend(substrate)
        agent_b = AgentB(mock_llm)

        tracker = PlanQualityTracker()
        builder = make_agent_b_plan_builder(agent_b, quality_tracker=tracker)

        plan = builder(sample_slice, substrate)
        assert plan is not None
        assert plan.num_vnfs == len(sample_slice.vnfs)
        assert len(plan.suggested_domains) == plan.num_vnfs

    def test_quality_tracker_records(self, substrate, sample_slice):
        mock_llm = MockLLMBackend(substrate)
        agent_b = AgentB(mock_llm)

        tracker = PlanQualityTracker()
        builder = make_agent_b_plan_builder(agent_b, quality_tracker=tracker)

        builder(sample_slice, substrate)
        assert len(tracker.logs) == 1
        log = tracker.logs[0]
        assert log.structurally_valid
        assert not log.cache_hit

    def test_cache_hit_on_second_call(self, substrate):
        """Same signature -> cache hit, no second LLM call."""
        rng = np.random.default_rng(7)
        req1 = generate_slice_request("req_1", substrate, rng)
        # Same slice type + QoS bucket -> same signature
        rng2 = np.random.default_rng(8)
        req2 = generate_slice_request("req_2", substrate, rng2,
                                       slice_type=req1.slice_type)

        mock_llm = MockLLMBackend(substrate)
        agent_b = AgentB(mock_llm)
        tracker = PlanQualityTracker()
        builder = make_agent_b_plan_builder(agent_b, quality_tracker=tracker)

        plan1 = builder(req1, substrate)
        plan2 = builder(req2, substrate)

        # Both should succeed
        assert plan1 is not None
        assert plan2 is not None

        # Second call should be a cache hit (if same signature)
        from orion.llm.plan_cache import signature
        if signature(req1) == signature(req2):
            assert len(tracker.logs) == 2
            assert not tracker.logs[0].cache_hit
            assert tracker.logs[1].cache_hit

    def test_quality_summary(self, substrate, sample_slice):
        mock_llm = MockLLMBackend(substrate)
        agent_b = AgentB(mock_llm)
        tracker = PlanQualityTracker()
        builder = make_agent_b_plan_builder(agent_b, quality_tracker=tracker)

        builder(sample_slice, substrate)
        summary = tracker.summary()
        assert "total_plans" in summary
        assert summary["total_plans"] == 1
        assert summary["structurally_valid_rate"] == 1.0
