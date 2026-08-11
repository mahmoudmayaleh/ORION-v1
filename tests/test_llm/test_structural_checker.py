"""Tests for the structural constraint checker (C4 + C8)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from orion.llm.structural_checker import CheckResult, check_plan

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


def _valid_plan() -> dict:
    """A hand-verified valid plan for xr_telepresence_005.

    VCR-correct bandwidth: beta_min=300, VCRs=[1.0, 1.2, 0.7, 1.0]
      f1->f2: 300 * 1.0 = 300 Mbps
      f2->f3: 300 * 1.0 * 1.2 = 360 Mbps
      f3->f4: 300 * 1.0 * 1.2 * 0.7 = 252 Mbps

    f1 (Firewall) -> d0: CPU 2/12, RAM 2/24
    f2 (MediaProc) + f3 (CDN) -> d1: CPU 14+4=18/25, RAM 30+8=38/60
    f4 (vEPC) -> d2: CPU 6/100, RAM 12/240
    """
    return {
        "plan_id": "xr_telepresence_005_plan",
        "vnf_assignments": [
            {"vnf_id": "xr_telepresence_005_f1", "domain": "d0",
             "required_tier": "edge", "cpu_demand": 2.0, "ram_demand": 2.0},
            {"vnf_id": "xr_telepresence_005_f2", "domain": "d1",
             "required_tier": "edge", "cpu_demand": 14.0, "ram_demand": 30.0},
            {"vnf_id": "xr_telepresence_005_f3", "domain": "d1",
             "required_tier": "edge", "cpu_demand": 4.0, "ram_demand": 8.0},
            {"vnf_id": "xr_telepresence_005_f4", "domain": "d2",
             "required_tier": "regional_cloud", "cpu_demand": 6.0,
             "ram_demand": 12.0},
        ],
        "flow_requirements": [
            {"source_vnf": "xr_telepresence_005_f1",
             "target_vnf": "xr_telepresence_005_f2",
             "min_bandwidth_mbps": 300.0, "crosses_domain_boundary": True},
            {"source_vnf": "xr_telepresence_005_f2",
             "target_vnf": "xr_telepresence_005_f3",
             "min_bandwidth_mbps": 360.0, "crosses_domain_boundary": False},
            {"source_vnf": "xr_telepresence_005_f3",
             "target_vnf": "xr_telepresence_005_f4",
             "min_bandwidth_mbps": 252.0, "crosses_domain_boundary": True},
        ],
    }


# ── Tests: valid plans ───────────────────────────────────────────────────────

class TestValidPlans:

    def test_valid_plan_passes(self, slice_request, topology):
        plan = _valid_plan()
        result = check_plan(plan, slice_request, topology)
        assert result.is_valid, result.summary()

    def test_few_shot_examples_all_pass(self, few_shot_examples, topology):
        for i, ex in enumerate(few_shot_examples):
            result = check_plan(
                ex["placement_plan"], ex["slice_request"], topology,
            )
            assert result.is_valid, (
                f"Few-shot example {i} ({ex['slice_request']['request_id']}) "
                f"failed:\n{result.summary()}"
            )


# ── Tests: C4 violations ────────────────────────────────────────────────────

class TestC4Violations:

    def test_cpu_overcommit_detected(self, slice_request, topology):
        """Move all VNFs to d0 to exceed its 12 CPU residual."""
        plan = _valid_plan()
        # Move f2 (14 CPU) to d0 — d0 already has f1 (2 CPU), total 16 > 12
        plan["vnf_assignments"][1]["domain"] = "d0"
        plan["vnf_assignments"][1]["required_tier"] = "edge"

        result = check_plan(plan, slice_request, topology)
        assert not result.is_valid
        c4_violations = [v for v in result.violations if v.constraint == "C4"]
        assert len(c4_violations) >= 1
        assert "CPU overcommit" in c4_violations[0].detail
        assert "d0" in c4_violations[0].detail

    def test_ram_overcommit_detected(self, slice_request, topology):
        topo = copy.deepcopy(topology)
        # Shrink d1 RAM so the valid plan fails (needs 30+8=38)
        topo["domains"][1]["ram_residual"] = 30.0

        plan = _valid_plan()
        result = check_plan(plan, slice_request, topo)
        assert not result.is_valid
        c4_violations = [v for v in result.violations if v.constraint == "C4"]
        assert any("RAM overcommit" in v.detail for v in c4_violations)


# ── Tests: C8 violations ────────────────────────────────────────────────────

class TestC8Violations:

    def test_tier_not_in_permitted(self, slice_request, topology):
        """Place Firewall (permitted: edge) with required_tier=central_cloud."""
        plan = _valid_plan()
        plan["vnf_assignments"][0]["required_tier"] = "central_cloud"

        result = check_plan(plan, slice_request, topology)
        assert not result.is_valid
        c8_violations = [v for v in result.violations if v.constraint == "C8"]
        assert any("not in VNF's permitted_tiers" in v.detail for v in c8_violations)

    def test_domain_does_not_support_tier(self, slice_request, topology):
        """Place Firewall in d2 (regional_cloud/central_cloud) with required_tier=mec."""
        plan = _valid_plan()
        plan["vnf_assignments"][0]["domain"] = "d2"
        plan["vnf_assignments"][0]["required_tier"] = "edge"

        result = check_plan(plan, slice_request, topology)
        assert not result.is_valid
        c8_violations = [v for v in result.violations if v.constraint == "C8"]
        assert any("do not include required_tier" in v.detail for v in c8_violations)

    def test_vEPC_in_edge_domain_fails(self, slice_request, topology):
        """vEPC (permitted: regional_cloud, central_cloud) placed in d0 (edge)."""
        plan = _valid_plan()
        plan["vnf_assignments"][3]["domain"] = "d0"
        plan["vnf_assignments"][3]["required_tier"] = "edge"

        result = check_plan(plan, slice_request, topology)
        assert not result.is_valid
        c8_violations = [v for v in result.violations if v.constraint == "C8"]
        assert len(c8_violations) >= 1


# ── Tests: schema violations ────────────────────────────────────────────────

class TestSchemaViolations:

    def test_missing_vnf_detected(self, slice_request, topology):
        plan = _valid_plan()
        plan["vnf_assignments"] = plan["vnf_assignments"][:3]  # Drop f4

        result = check_plan(plan, slice_request, topology)
        assert not result.is_valid
        schema_violations = [v for v in result.violations if v.constraint == "SCHEMA"]
        assert any("missing from plan" in v.detail for v in schema_violations)

    def test_unknown_domain_detected(self, slice_request, topology):
        plan = _valid_plan()
        plan["vnf_assignments"][0]["domain"] = "d99"

        result = check_plan(plan, slice_request, topology)
        assert not result.is_valid
        assert any("unknown domain" in v.detail for v in result.violations)

    def test_boundary_flag_mismatch(self, slice_request, topology):
        """f1 in d0, f2 in d1 but crosses_domain_boundary=false."""
        plan = _valid_plan()
        plan["flow_requirements"][0]["crosses_domain_boundary"] = False

        result = check_plan(plan, slice_request, topology)
        assert not result.is_valid
        assert any("crosses domains" in v.detail for v in result.violations)


# ── Tests: inter-domain bandwidth ────────────────────────────────────────────

class TestBandwidthChecks:

    def test_link_bw_overcommit(self, slice_request, topology):
        """f1->f2 demands 300 Mbps on d0->d1; shrink link to 200 Mbps."""
        topo = copy.deepcopy(topology)
        for link in topo["inter_domain_links"]:
            if link["link_id"] == "l_d0_d1":
                link["bandwidth_residual_mbps"] = 200.0

        plan = _valid_plan()
        result = check_plan(plan, slice_request, topo)
        assert not result.is_valid
        c5_violations = [v for v in result.violations if v.constraint == "C5"]
        assert len(c5_violations) >= 1
        assert "overcommit" in c5_violations[0].detail

    def test_nonexistent_link_detected(self, slice_request, topology):
        """f1(d0)->f2(d1) requires d0-d1 reachability; isolate d0 completely."""
        topo = copy.deepcopy(topology)
        # Remove ALL links involving d0 to make it truly unreachable
        topo["inter_domain_links"] = [
            l for l in topo["inter_domain_links"]
            if l["source_domain"] != "d0" and l["target_domain"] != "d0"
        ]

        plan = _valid_plan()
        result = check_plan(plan, slice_request, topo)
        assert not result.is_valid
        assert any("not reachable" in v.detail for v in result.violations)


# ── Tests: violation feedback for retry ──────────────────────────────────────

class TestViolationFeedback:

    def test_summary_contains_all_violations(self, slice_request, topology):
        plan = _valid_plan()
        # Move f2 to d0 (CPU overcommit: 2+14=16 > 12) AND set wrong tier
        plan["vnf_assignments"][1]["domain"] = "d0"
        plan["vnf_assignments"][1]["required_tier"] = "central_cloud"

        result = check_plan(plan, slice_request, topology)
        assert not result.is_valid
        summary = result.summary()
        assert "C4" in summary
        assert "C8" in summary

    def test_violation_text_for_prompt_nonempty(self, slice_request, topology):
        plan = _valid_plan()
        # Move f2 to d0 (CPU overcommit: 2+14=16 > 12)
        plan["vnf_assignments"][1]["domain"] = "d0"
        plan["vnf_assignments"][1]["required_tier"] = "edge"

        result = check_plan(plan, slice_request, topology)
        text = result.violation_text_for_prompt()
        assert "C4" in text
        assert len(text) > 0

    def test_valid_plan_empty_violation_text(self, slice_request, topology):
        plan = _valid_plan()
        result = check_plan(plan, slice_request, topology)
        assert result.violation_text_for_prompt() == ""
