"""Tests for K^B semantic memory — loading, filtering, retrieval, prompt formatting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion.llm.semantic_memory import (
    KBEntry,
    SemanticMemory,
    build_query_from_slice,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
KB_PATH = DATA_DIR / "memory" / "kb_agent_b.json"
SLICE_REQUEST_PATH = DATA_DIR / "placement_eval" / "slice_request.json"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def kb() -> SemanticMemory:
    return SemanticMemory.from_json(KB_PATH)


@pytest.fixture
def xr_request() -> dict:
    return json.loads(SLICE_REQUEST_PATH.read_text())


# ── Tests: loading ───────────────────────────────────────────────────────────

class TestLoading:

    def test_load_from_json(self, kb):
        assert len(kb.entries) >= 10

    def test_entries_have_required_fields(self, kb):
        for e in kb.entries:
            assert e.topic
            assert e.content
            assert e.slice_type_tag
            assert e.version

    def test_entries_are_kb_entry_type(self, kb):
        for e in kb.entries:
            assert isinstance(e, KBEntry)


# ── Tests: filtering ─────────────────────────────────────────────────────────

class TestFiltering:

    def test_filter_by_slice_type(self, kb):
        results = kb.retrieve("placement", slice_type="URLLC", top_k=100)
        for e in results:
            assert e.slice_type_tag in ("all", "URLLC")

    def test_filter_excludes_wrong_type(self, kb):
        results = kb.retrieve("placement", slice_type="URLLC", top_k=100)
        for e in results:
            assert e.slice_type_tag != "eMBB"

    def test_all_tag_passes_any_filter(self, kb):
        results = kb.retrieve("tier conventions", slice_type="V2X", top_k=100)
        all_entries = [e for e in results if e.slice_type_tag == "all"]
        assert len(all_entries) > 0

    def test_no_filter_returns_all(self, kb):
        results = kb.retrieve("placement", top_k=100)
        assert len(results) == len(kb.entries)


# ── Tests: retrieval ranking ─────────────────────────────────────────────────

class TestRetrieval:

    def test_urllc_query_returns_urllc_entries_first(self, kb):
        results = kb.retrieve(
            "URLLC ultra-low latency Firewall vUPF ran_edge mec",
            slice_type="URLLC",
            top_k=3,
        )
        assert len(results) >= 1
        # At least one result should mention URLLC or low latency
        topics = " ".join(e.topic for e in results)
        assert "URLLC" in topics or "latency" in topics.lower()

    def test_embb_query_finds_co_location_pattern(self, kb):
        results = kb.retrieve(
            "eMBB Firewall CDN vEPC mec regional_cloud high throughput",
            slice_type="eMBB",
            top_k=3,
        )
        topics = " ".join(e.topic + " " + e.content for e in results)
        assert "CDN" in topics or "eMBB" in topics

    def test_top_k_limits_results(self, kb):
        results = kb.retrieve("placement tier", top_k=3)
        assert len(results) <= 3

    def test_empty_query_returns_entries(self, kb):
        results = kb.retrieve("", top_k=5)
        assert len(results) == 5


# ── Tests: query building from slice request ─────────────────────────────────

class TestQueryBuilding:

    def test_xr_request_query(self, xr_request):
        query = build_query_from_slice(xr_request)
        assert "XR" in query
        assert "Firewall" in query
        assert "MediaProc" in query
        assert "mec" in query

    def test_low_delay_triggers_latency_term(self):
        req = {"slice_type": "URLLC", "vnfs": [], "qos": {"max_e2e_delay": 5.0}}
        query = build_query_from_slice(req)
        assert "ultra-low latency" in query

    def test_high_throughput_triggers_throughput_term(self):
        req = {"slice_type": "XR", "vnfs": [], "qos": {"min_throughput": 800.0}}
        query = build_query_from_slice(req)
        assert "high throughput" in query


# ── Tests: prompt formatting ─────────────────────────────────────────────────

class TestPromptFormatting:

    def test_format_nonempty(self, kb):
        entries = kb.retrieve("URLLC latency", top_k=3)
        text = kb.format_for_prompt(entries)
        assert "Reference Knowledge" in text
        assert len(text) > 50

    def test_format_empty_returns_empty(self, kb):
        text = kb.format_for_prompt([])
        assert text == ""

    def test_format_numbers_entries(self, kb):
        entries = kb.retrieve("placement", top_k=3)
        text = kb.format_for_prompt(entries)
        assert "[1]" in text
        assert "[2]" in text


# ── Tests: integration with Agent B prompt ───────────────────────────────────

class TestAgentBIntegration:

    def test_reference_knowledge_in_prompt(self, kb, xr_request):
        from orion.llm.agent_b import build_user_prompt

        topology = json.loads(
            (DATA_DIR / "placement_eval" / "abstract_topology.json").read_text()
        )
        query = build_query_from_slice(xr_request)
        entries = kb.retrieve(query, slice_type=xr_request.get("slice_type"), top_k=5)
        ref_text = kb.format_for_prompt(entries)

        prompt = build_user_prompt(
            xr_request, topology,
            reference_knowledge=ref_text,
        )
        assert "Reference Knowledge" in prompt
        # K^B block should appear before Current Task
        ref_pos = prompt.index("Reference Knowledge")
        task_pos = prompt.index("Current Task")
        assert ref_pos < task_pos

    def test_reference_knowledge_absent_when_none(self, xr_request):
        from orion.llm.agent_b import build_user_prompt

        topology = json.loads(
            (DATA_DIR / "placement_eval" / "abstract_topology.json").read_text()
        )
        prompt = build_user_prompt(xr_request, topology)
        assert "Reference Knowledge" not in prompt
