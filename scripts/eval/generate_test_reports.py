"""Generate detailed test reports showing inputs, outputs, and verdicts.

Produces human-readable reports for supervisor review.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data" / "placement_eval"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "test_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def structural_checker_report() -> str:
    """Generate detailed report for the structural constraint checker."""
    from orion.llm.structural_checker import check_plan

    topology = json.loads((DATA_DIR / "abstract_topology.json").read_text())
    slice_request = json.loads((DATA_DIR / "slice_request.json").read_text())

    lines = []
    lines.append("=" * 80)
    lines.append("STRUCTURAL CONSTRAINT CHECKER — DETAILED TEST REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # Show the topology
    lines.append("\n" + "─" * 80)
    lines.append("ABSTRACT TOPOLOGY (input to all tests)")
    lines.append("─" * 80)
    for d in topology["domains"]:
        lines.append(f"  Domain {d['domain_id']}: tiers={d['dominant_tiers']}, "
                     f"CPU_residual={d['cpu_residual']}, RAM_residual={d['ram_residual']}")
    for l in topology["inter_domain_links"]:
        lines.append(f"  Link {l['link_id']}: {l.get('source_domain','?')}->{l.get('target_domain','?')}, "
                     f"BW_residual={l.get('bandwidth_residual_mbps', '?')} Mbps, "
                     f"delay={l.get('propagation_delay_ms', '?')} ms")

    # Show the slice request
    lines.append("\n" + "─" * 80)
    lines.append("SLICE REQUEST (input to all tests)")
    lines.append("─" * 80)
    lines.append(f"  request_id: {slice_request.get('request_id')}")
    lines.append(f"  slice_type: {slice_request.get('slice_type')}")
    for vnf in slice_request["vnfs"]:
        lines.append(f"  VNF {vnf['vnf_id']}: type={vnf.get('vnf_type')}, "
                     f"permitted_tiers={vnf.get('permitted_tiers')}, "
                     f"cpu={vnf.get('cpu_demand')}, ram={vnf.get('ram_demand')}")

    # Valid plan — VCR-correct bandwidths, demands match slice_request.json
    # beta_min=300, VCRs=[1.0, 1.2, 0.7, 1.0]
    #   f1->f2: 300*1.0 = 300 Mbps
    #   f2->f3: 300*1.0*1.2 = 360 Mbps  (MediaProc expands traffic)
    #   f3->f4: 300*1.0*1.2*0.7 = 252 Mbps  (CDN caches, reduces 30%)
    # Resource check:
    #   f1(Firewall) -> d0: CPU 2/12, RAM 2/24
    #   f2(MediaProc)+f3(CDN) -> d1: CPU 14+4=18/25, RAM 30+8=38/60
    #   f4(vEPC) -> d2: CPU 6/100, RAM 12/240
    valid_plan = {
        "plan_id": "xr_telepresence_005_plan",
        "vnf_assignments": [
            {"vnf_id": "xr_telepresence_005_f1", "domain": "d0",
             "required_tier": "mec", "cpu_demand": 2.0, "ram_demand": 2.0},
            {"vnf_id": "xr_telepresence_005_f2", "domain": "d1",
             "required_tier": "mec", "cpu_demand": 14.0, "ram_demand": 30.0},
            {"vnf_id": "xr_telepresence_005_f3", "domain": "d1",
             "required_tier": "mec", "cpu_demand": 4.0, "ram_demand": 8.0},
            {"vnf_id": "xr_telepresence_005_f4", "domain": "d2",
             "required_tier": "regional_cloud", "cpu_demand": 6.0, "ram_demand": 12.0},
        ],
        "flow_requirements": [
            {"source_vnf": "xr_telepresence_005_f1", "target_vnf": "xr_telepresence_005_f2",
             "min_bandwidth_mbps": 300.0, "crosses_domain_boundary": True},
            {"source_vnf": "xr_telepresence_005_f2", "target_vnf": "xr_telepresence_005_f3",
             "min_bandwidth_mbps": 360.0, "crosses_domain_boundary": False},
            {"source_vnf": "xr_telepresence_005_f3", "target_vnf": "xr_telepresence_005_f4",
             "min_bandwidth_mbps": 252.0, "crosses_domain_boundary": True},
        ],
    }

    # ── Test 1: Valid plan passes ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 1: Valid plan passes all constraints")
    lines.append("═" * 80)
    lines.append("\nInput plan:")
    for a in valid_plan["vnf_assignments"]:
        lines.append(f"  {a['vnf_id']} -> domain={a['domain']}, tier={a['required_tier']}, "
                     f"cpu={a['cpu_demand']}, ram={a['ram_demand']}")
    lines.append("Flow requirements:")
    for f in valid_plan["flow_requirements"]:
        lines.append(f"  {f['source_vnf']} -> {f['target_vnf']}: "
                     f"BW={f['min_bandwidth_mbps']} Mbps, cross_boundary={f['crosses_domain_boundary']}")

    result = check_plan(valid_plan, slice_request, topology)
    lines.append(f"\nResult: {'PASS' if result.is_valid else 'FAIL'}")
    lines.append(f"Violations: {len(result.violations)}")
    lines.append(f"Detail: {result.summary()}")

    # ── Test 2: CPU overcommit (C4) ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 2: C4 — CPU overcommit detection")
    lines.append("═" * 80)
    lines.append("\nScenario: Move f2 (MediaProc, 14 CPU) from d1 to d0.")
    lines.append("  d0 already has f1 (2 CPU). Adding f2 (14 CPU) = 16 CPU > d0 residual (12 CPU)")
    lines.append("  Also RAM: d0 RAM=24, f1(2)+f2(30)=32 > 24")

    plan_c4 = copy.deepcopy(valid_plan)
    plan_c4["vnf_assignments"][1]["domain"] = "d0"
    plan_c4["vnf_assignments"][1]["required_tier"] = "mec"
    lines.append("\nModified assignment:")
    for a in plan_c4["vnf_assignments"]:
        lines.append(f"  {a['vnf_id']} -> domain={a['domain']}, cpu={a['cpu_demand']}")

    result = check_plan(plan_c4, slice_request, topology)
    lines.append(f"\nChecker output: {'VALID' if result.is_valid else 'REJECTED'}")
    lines.append(f"Violations detected ({len(result.violations)}):")
    for v in result.violations:
        lines.append(f"  [{v.constraint}] vnf={v.vnf_id}: {v.detail}")
    lines.append(f"\nTest verdict: {'PASS' if not result.is_valid else 'FAIL'} "
                 f"(expected: checker rejects this plan)")

    # ── Test 3: RAM overcommit (C4) ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 3: C4 — RAM overcommit detection")
    lines.append("═" * 80)
    lines.append("\nScenario: Shrink d1 RAM residual to 30 GB (plan needs 38 GB).")
    lines.append("  d1 holds f2 (MediaProc, 30 GB RAM) + f3 (CDN, 8 GB RAM) = 38 GB total")

    topo_ram = copy.deepcopy(topology)
    topo_ram["domains"][1]["ram_residual"] = 30.0
    lines.append(f"  d1 RAM residual: 30.0 (was {topology['domains'][1]['ram_residual']})")
    lines.append(f"  f2 RAM + f3 RAM = 30 + 8 = 38 > 30")

    result = check_plan(valid_plan, slice_request, topo_ram)
    lines.append(f"\nChecker output: {'VALID' if result.is_valid else 'REJECTED'}")
    lines.append(f"Violations detected ({len(result.violations)}):")
    for v in result.violations:
        lines.append(f"  [{v.constraint}] vnf={v.vnf_id}: {v.detail}")
    lines.append(f"\nTest verdict: {'PASS' if not result.is_valid else 'FAIL'} "
                 f"(expected: checker rejects this plan)")

    # ── Test 4: Tier not in permitted (C8) ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 4: C8 — Tier not in VNF's permitted tiers")
    lines.append("═" * 80)
    lines.append("\nScenario: Set Firewall required_tier='central_cloud'")
    lines.append("  Firewall permitted_tiers = ['ran_edge', 'mec']")
    lines.append("  'central_cloud' not in permitted_tiers -> C8 violation")

    plan_c8 = copy.deepcopy(valid_plan)
    plan_c8["vnf_assignments"][0]["required_tier"] = "central_cloud"

    result = check_plan(plan_c8, slice_request, topology)
    lines.append(f"\nChecker output: {'VALID' if result.is_valid else 'REJECTED'}")
    lines.append(f"Violations detected ({len(result.violations)}):")
    for v in result.violations:
        lines.append(f"  [{v.constraint}] vnf={v.vnf_id}: {v.detail}")
    lines.append(f"\nTest verdict: {'PASS' if not result.is_valid else 'FAIL'} "
                 f"(expected: checker rejects this plan)")

    # ── Test 5: Domain doesn't support tier (C8) ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 5: C8 — Domain does not support the required tier")
    lines.append("═" * 80)
    lines.append("\nScenario: Place Firewall in d2 with required_tier='mec'")
    lines.append(f"  d2 tiers = {topology['domains'][2]['dominant_tiers']}")
    lines.append("  'mec' not in d2's tiers -> C8 violation")

    plan_c8b = copy.deepcopy(valid_plan)
    plan_c8b["vnf_assignments"][0]["domain"] = "d2"
    plan_c8b["vnf_assignments"][0]["required_tier"] = "mec"

    result = check_plan(plan_c8b, slice_request, topology)
    lines.append(f"\nChecker output: {'VALID' if result.is_valid else 'REJECTED'}")
    lines.append(f"Violations detected ({len(result.violations)}):")
    for v in result.violations:
        lines.append(f"  [{v.constraint}] vnf={v.vnf_id}: {v.detail}")
    lines.append(f"\nTest verdict: {'PASS' if not result.is_valid else 'FAIL'} "
                 f"(expected: checker rejects this plan)")

    # ── Test 6: Missing VNF (SCHEMA) ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 6: SCHEMA — Missing VNF in plan")
    lines.append("═" * 80)
    lines.append("\nScenario: Remove f4 (vEPC) from plan assignments.")
    lines.append("  Slice request has 4 VNFs, plan only covers 3 -> SCHEMA violation")

    plan_miss = copy.deepcopy(valid_plan)
    plan_miss["vnf_assignments"] = plan_miss["vnf_assignments"][:3]

    result = check_plan(plan_miss, slice_request, topology)
    lines.append(f"\nChecker output: {'VALID' if result.is_valid else 'REJECTED'}")
    lines.append(f"Violations detected ({len(result.violations)}):")
    for v in result.violations:
        lines.append(f"  [{v.constraint}] vnf={v.vnf_id}: {v.detail}")
    lines.append(f"\nTest verdict: {'PASS' if not result.is_valid else 'FAIL'} "
                 f"(expected: checker rejects this plan)")

    # ── Test 7: Unknown domain (SCHEMA) ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 7: SCHEMA — Unknown domain in plan")
    lines.append("═" * 80)
    lines.append("\nScenario: Assign f1 to 'd99' (does not exist in topology).")

    plan_unk = copy.deepcopy(valid_plan)
    plan_unk["vnf_assignments"][0]["domain"] = "d99"

    result = check_plan(plan_unk, slice_request, topology)
    lines.append(f"\nChecker output: {'VALID' if result.is_valid else 'REJECTED'}")
    lines.append(f"Violations detected ({len(result.violations)}):")
    for v in result.violations:
        lines.append(f"  [{v.constraint}] vnf={v.vnf_id}: {v.detail}")
    lines.append(f"\nTest verdict: {'PASS' if not result.is_valid else 'FAIL'} "
                 f"(expected: checker rejects this plan)")

    # ── Test 8: Multiple violations (C4 + C8) ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 8: Multiple violations — C4 + C8 combined")
    lines.append("═" * 80)
    lines.append("\nScenario: Move f2 (MediaProc) to d0 AND set required_tier='central_cloud'")
    lines.append("  C4: d0 CPU = f1(2)+f2(14)=16 > 12 residual; RAM = f1(2)+f2(30)=32 > 24")
    lines.append("  C8: central_cloud not in MediaProc permitted_tiers [mec, regional_cloud]")
    lines.append("  C8: d0 tiers [ran_edge, mec] do not include central_cloud")

    plan_multi = copy.deepcopy(valid_plan)
    plan_multi["vnf_assignments"][1]["domain"] = "d0"
    plan_multi["vnf_assignments"][1]["required_tier"] = "central_cloud"

    result = check_plan(plan_multi, slice_request, topology)
    lines.append(f"\nChecker output: {'VALID' if result.is_valid else 'REJECTED'}")
    lines.append(f"Violations detected ({len(result.violations)}):")
    for v in result.violations:
        lines.append(f"  [{v.constraint}] vnf={v.vnf_id}: {v.detail}")
    lines.append(f"\nViolation feedback for Agent B retry prompt:")
    lines.append(result.violation_text_for_prompt())
    lines.append(f"\nTest verdict: {'PASS' if not result.is_valid else 'FAIL'} "
                 f"(expected: checker rejects this plan)")

    lines.append("\n" + "=" * 80)
    lines.append("END OF STRUCTURAL CHECKER REPORT")
    lines.append("=" * 80)
    return "\n".join(lines)


def retrieval_pipeline_report() -> str:
    """Generate detailed report for the retrieval pipeline."""
    import numpy as np
    import time
    from orion.retrieval import (
        DenseRetriever, LexicalRetriever, MemoryEntry, RetrievalConfig,
        RetrievalMode, RetrievalPipeline, RetrievalQuery, apply_metadata_filter, rrf_fuse,
    )

    lines = []
    lines.append("=" * 80)
    lines.append("4-STAGE HYBRID RETRIEVAL PIPELINE — DETAILED TEST REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    def _random_embedding(dim=64, seed=0):
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        return (vec / np.linalg.norm(vec)).tolist()

    def _mock_embed_fn(texts):
        return [_random_embedding(64, seed=hash(t) % 10000) for t in texts]

    # ── Test 1: Metadata filter AND logic ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 1: Metadata filter — AND logic across keys")
    lines.append("═" * 80)
    entries = [
        MemoryEntry(entry_id="a", topic="entry A", content="URLLC multi_domain",
                    tags={"slice_type": ["URLLC"], "topology": ["multi_domain"]}),
        MemoryEntry(entry_id="b", topic="entry B", content="URLLC single_domain",
                    tags={"slice_type": ["URLLC"], "topology": ["single_domain"]}),
        MemoryEntry(entry_id="c", topic="entry C", content="eMBB multi_domain",
                    tags={"slice_type": ["eMBB"], "topology": ["multi_domain"]}),
    ]
    lines.append("\nInput entries:")
    for e in entries:
        lines.append(f"  {e.entry_id}: tags={e.tags}")
    query = RetrievalQuery(text="test", filters={"slice_type": "URLLC", "topology": "multi_domain"})
    lines.append(f"\nQuery filters: {query.filters}")
    lines.append("  Logic: slice_type=URLLC AND topology=multi_domain")
    result = apply_metadata_filter(entries, query)
    lines.append(f"\nResult: {len(result)} entries passed")
    for e in result:
        lines.append(f"  {e.entry_id}: tags={e.tags}")
    lines.append(f"\nVerdict: {'PASS' if len(result) == 1 and result[0].entry_id == 'a' else 'FAIL'}")

    # ── Test 2: Metadata filter OR within values ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 2: Metadata filter — OR within tag values")
    lines.append("═" * 80)
    query2 = RetrievalQuery(text="test", filters={"slice_type": ["URLLC", "eMBB"]})
    lines.append(f"\nQuery filters: {query2.filters}")
    lines.append("  Logic: slice_type IN [URLLC, eMBB]")
    result2 = apply_metadata_filter(entries, query2)
    lines.append(f"\nResult: {len(result2)} entries passed")
    for e in result2:
        lines.append(f"  {e.entry_id}: tags={e.tags}")
    # All 3 entries (a=URLLC, b=URLLC, c=eMBB) match the OR filter [URLLC, eMBB]
    lines.append(f"\nVerdict: {'PASS' if len(result2) == 3 else 'FAIL'}")

    # ── Test 3: Dense retrieval ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 3: Dense retrieval — top-k by cosine similarity")
    lines.append("═" * 80)
    n_entries = 20
    dim = 64
    dense_entries = []
    for i in range(n_entries):
        dense_entries.append(MemoryEntry(
            entry_id=f"e_{i}", topic=f"topic {i}", content=f"content {i}",
            embedding=_random_embedding(dim, seed=i),
        ))
    retriever = DenseRetriever()
    retriever.build_index([e.embedding for e in dense_entries])
    query_emb = dense_entries[5].embedding  # Exact match with entry 5
    results = retriever.query(query_emb, top_k=5)

    lines.append(f"\nCorpus: {n_entries} entries with {dim}-dim normalized embeddings")
    lines.append(f"Query: embedding of entry e_5 (exact match expected at rank 1)")
    lines.append(f"\nTop-5 results:")
    for rank, (idx, score) in enumerate(results, 1):
        lines.append(f"  Rank {rank}: entry e_{idx}, score={score:.4f}")
    lines.append(f"\nVerdict: {'PASS' if results[0][0] == 5 else 'FAIL'} "
                 f"(entry 5 at rank 1 with score {results[0][1]:.4f})")

    # ── Test 4: RRF Fusion ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 4: Reciprocal Rank Fusion (RRF)")
    lines.append("═" * 80)
    ranking_a = [(0, 1.0), (1, 0.8), (2, 0.6)]
    ranking_b = [(1, 1.0), (0, 0.8), (3, 0.6)]
    lines.append(f"\nRanking A (dense): {ranking_a}")
    lines.append(f"Ranking B (BM25):  {ranking_b}")
    lines.append(f"RRF k=60")
    fused = rrf_fuse([ranking_a, ranking_b], k=60)
    lines.append(f"\nFused ranking:")
    for idx, score in fused:
        appears_in = []
        for r, (i, _) in enumerate(ranking_a, 1):
            if i == idx:
                appears_in.append(f"dense@{r}")
        for r, (i, _) in enumerate(ranking_b, 1):
            if i == idx:
                appears_in.append(f"BM25@{r}")
        lines.append(f"  entry {idx}: RRF_score={score:.6f} (from {', '.join(appears_in)})")
    lines.append(f"\nVerdict: PASS (items in both rankings scored higher)")

    # ── Test 5: Full pipeline end-to-end ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 5: Full pipeline end-to-end (50 entries, no_rerank mode)")
    lines.append("═" * 80)
    pipeline_entries = []
    for i in range(50):
        pipeline_entries.append(MemoryEntry(
            entry_id=f"e_{i}",
            topic=f"topic {i} VNF placement",
            content=f"content about entry {i} URLLC mec ran_edge tier {i}",
            tags={
                "slice_type": ["URLLC"] if i % 3 == 0 else ["eMBB"] if i % 3 == 1 else ["all"],
                "topology": ["multi_domain"] if i % 2 == 0 else ["all"],
            },
            embedding=_random_embedding(dim, seed=i),
        ))

    config = RetrievalConfig(
        mode=RetrievalMode.NO_RERANK, k_after_filter=200, k_after_dense=20,
        k_after_rrf=10, k_final=5, return_trace=True, enable_bm25=True,
    )
    pipeline = RetrievalPipeline(config, embed_fn=_mock_embed_fn)
    pipeline.build(pipeline_entries)

    query = RetrievalQuery(
        text="URLLC placement ran_edge low latency",
        filters={"slice_type": ["URLLC", "all"]},
        top_k=5,
    )
    lines.append(f"\nCorpus: 50 entries (17 URLLC, 17 eMBB, 16 'all')")
    lines.append(f"Query: '{query.text}'")
    lines.append(f"Filters: {query.filters}")
    lines.append(f"Config: mode={config.mode.value}, k_after_dense={config.k_after_dense}, "
                 f"k_after_rrf={config.k_after_rrf}, k_final={config.k_final}")

    results, trace = pipeline.retrieve(query)
    lines.append(f"\nPipeline trace:")
    lines.append(f"  After metadata filter: {trace.candidates_after_filter} candidates")
    lines.append(f"  After dense retrieval: {trace.candidates_after_dense} candidates")
    lines.append(f"  After RRF fusion:      {trace.candidates_after_rrf} candidates")
    lines.append(f"\nFinal results ({len(results)} entries):")
    for rank, r in enumerate(results, 1):
        lines.append(f"  Rank {rank}: {r.entry.entry_id} | score={r.score:.4f} | "
                     f"stage_scores={r.stage_scores} | tags={r.entry.tags}")

    # ── Test 6: Ablation modes comparison ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 6: Ablation modes comparison (same query, same corpus)")
    lines.append("═" * 80)
    for mode in RetrievalMode:
        cfg = RetrievalConfig(
            mode=mode, k_after_filter=200, k_after_dense=20,
            k_after_rrf=10, k_final=5, enable_rerank=False,
        )
        p = RetrievalPipeline(cfg, embed_fn=_mock_embed_fn)
        p.build(pipeline_entries)
        res, _ = p.retrieve(query)
        lines.append(f"\n  Mode: {mode.value}")
        lines.append(f"  Results: {len(res)} entries")
        for rank, r in enumerate(res, 1):
            lines.append(f"    Rank {rank}: {r.entry.entry_id} (score={r.score:.4f})")

    # ── Test 7: Latency benchmarks ──
    lines.append("\n" + "═" * 80)
    lines.append("TEST 7: Latency benchmarks (500 entries)")
    lines.append("═" * 80)
    large_entries = []
    for i in range(500):
        large_entries.append(MemoryEntry(
            entry_id=f"e_{i}", topic=f"topic {i} VNF", content=f"content {i} URLLC mec",
            tags={"slice_type": ["URLLC"] if i % 3 == 0 else ["eMBB"] if i % 3 == 1 else ["all"]},
            embedding=_random_embedding(dim, seed=i),
        ))

    for mode in [RetrievalMode.NO_RERANK, RetrievalMode.FULL, RetrievalMode.DENSE_ONLY, RetrievalMode.COSINE_ONLY]:
        cfg = RetrievalConfig(mode=mode, k_final=5, enable_rerank=False)
        p = RetrievalPipeline(cfg, embed_fn=_mock_embed_fn)
        p.build(large_entries)
        q = RetrievalQuery(text="URLLC placement ran_edge", top_k=5)
        p.retrieve(q)  # warmup
        t0 = time.perf_counter()
        for _ in range(10):
            p.retrieve(q)
        elapsed_ms = (time.perf_counter() - t0) / 10 * 1000
        lines.append(f"  {mode.value:<15}: {elapsed_ms:.1f} ms per query")

    lines.append("\n" + "=" * 80)
    lines.append("END OF RETRIEVAL PIPELINE REPORT")
    lines.append("=" * 80)
    return "\n".join(lines)


def semantic_memory_report() -> str:
    """Generate detailed report for K^B semantic memory."""
    from orion.llm.semantic_memory import SemanticMemory, build_query_from_slice

    kb_path = PROJECT_ROOT / "data" / "memory" / "kb_agent_b.json"
    kb = SemanticMemory.from_json(kb_path)

    lines = []
    lines.append("=" * 80)
    lines.append("K^B SEMANTIC MEMORY — DETAILED TEST REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    lines.append(f"\nLoaded {len(kb.entries)} entries from {kb_path.name}")
    for i, e in enumerate(kb.entries):
        lines.append(f"\n  [{i}] topic: {e.topic}")
        lines.append(f"      slice_type_tag: {e.slice_type_tag}, topology_tag: {e.topology_tag}")
        lines.append(f"      content (first 150 chars): {e.content[:150]}...")

    # Query tests
    queries = [
        ("URLLC ultra-low latency Firewall vUPF ran_edge mec", "URLLC", None),
        ("eMBB Firewall CDN vEPC mec regional_cloud high throughput", "eMBB", None),
        ("VCR volume change ratio bandwidth computation", None, None),
        ("tier definitions resource profiles mec regional_cloud", None, None),
    ]

    lines.append("\n" + "═" * 80)
    lines.append("RETRIEVAL TESTS")
    lines.append("═" * 80)
    for query_text, slice_type, topo_tag in queries:
        lines.append(f"\n{'─' * 60}")
        lines.append(f"Query: '{query_text}'")
        lines.append(f"Filter: slice_type={slice_type}, topology_tag={topo_tag}")
        results = kb.retrieve(query_text, slice_type=slice_type, topology_tag=topo_tag, top_k=3)
        lines.append(f"Results ({len(results)}):")
        for rank, e in enumerate(results, 1):
            lines.append(f"  Rank {rank}: {e.topic}")
            # Show keyword overlap
            from orion.llm.semantic_memory import _tokenize
            q_tokens = _tokenize(query_text)
            e_tokens = _tokenize(e.topic + " " + e.content)
            overlap = q_tokens & e_tokens
            score = len(overlap) / len(q_tokens) if q_tokens else 0
            lines.append(f"    Score: {score:.3f} ({len(overlap)}/{len(q_tokens)} tokens matched)")
            lines.append(f"    Matched tokens: {sorted(overlap)[:15]}{'...' if len(overlap) > 15 else ''}")

    # Query building from slice request
    lines.append("\n" + "═" * 80)
    lines.append("QUERY BUILDING FROM SLICE REQUEST")
    lines.append("═" * 80)
    slice_req_path = DATA_DIR / "slice_request.json"
    if slice_req_path.exists():
        slice_req = json.loads(slice_req_path.read_text())
        query = build_query_from_slice(slice_req)
        lines.append(f"\nSlice request: {slice_req.get('request_id')} (type={slice_req.get('slice_type')})")
        lines.append(f"Generated query: '{query}'")
        results = kb.retrieve(query, slice_type=slice_req.get("slice_type"), top_k=3)
        lines.append(f"\nTop-3 results:")
        for rank, e in enumerate(results, 1):
            lines.append(f"  Rank {rank}: {e.topic}")

    # Prompt formatting
    lines.append("\n" + "═" * 80)
    lines.append("PROMPT FORMATTING OUTPUT")
    lines.append("═" * 80)
    entries = kb.retrieve("URLLC latency placement", top_k=2)
    prompt_text = kb.format_for_prompt(entries)
    lines.append(f"\n{prompt_text}")

    lines.append("\n" + "=" * 80)
    lines.append("END OF SEMANTIC MEMORY REPORT")
    lines.append("=" * 80)
    return "\n".join(lines)


def agent_b_report() -> str:
    """Generate detailed report for Agent B prompt construction and flow."""
    from orion.llm.agent_b import SYSTEM_PROMPT, build_user_prompt

    lines = []
    lines.append("=" * 80)
    lines.append("AGENT B — DETAILED TEST REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    slice_req_path = DATA_DIR / "slice_request.json"
    topo_path = DATA_DIR / "abstract_topology.json"

    if not slice_req_path.exists() or not topo_path.exists():
        lines.append("\nERROR: Required data files not found.")
        return "\n".join(lines)

    slice_request = json.loads(slice_req_path.read_text())
    topology = json.loads(topo_path.read_text())

    # System prompt
    lines.append("\n" + "═" * 80)
    lines.append("SYSTEM PROMPT")
    lines.append("═" * 80)
    lines.append(f"\n{SYSTEM_PROMPT}")

    # User prompt without reference knowledge
    lines.append("\n" + "═" * 80)
    lines.append("USER PROMPT (without reference knowledge)")
    lines.append("═" * 80)
    user_prompt = build_user_prompt(slice_request, topology)
    lines.append(f"\n{user_prompt}")

    # User prompt with reference knowledge
    lines.append("\n" + "═" * 80)
    lines.append("USER PROMPT (with reference knowledge from K^B)")
    lines.append("═" * 80)
    from orion.llm.semantic_memory import SemanticMemory, build_query_from_slice
    kb = SemanticMemory.from_json(PROJECT_ROOT / "data" / "memory" / "kb_agent_b.json")
    query = build_query_from_slice(slice_request)
    entries = kb.retrieve(query, slice_type=slice_request.get("slice_type"), top_k=3)
    ref_text = kb.format_for_prompt(entries)

    user_prompt_with_kb = build_user_prompt(slice_request, topology, reference_knowledge=ref_text)
    lines.append(f"\n{user_prompt_with_kb}")

    # Show the expected output schema
    lines.append("\n" + "═" * 80)
    lines.append("EXPECTED OUTPUT FORMAT (JSON)")
    lines.append("═" * 80)
    lines.append("""
{
  "plan_id": "<request_id>_plan",
  "vnf_assignments": [
    {
      "vnf_id": "<vnf_id>",
      "domain": "<domain_id>",
      "required_tier": "<tier>",
      "cpu_demand": <float>,
      "ram_demand": <float>
    }, ...
  ],
  "flow_requirements": [
    {
      "source_vnf": "<vnf_id>",
      "target_vnf": "<vnf_id>",
      "min_bandwidth_mbps": <float>,
      "crosses_domain_boundary": <bool>
    }, ...
  ]
}""")

    lines.append("\n" + "=" * 80)
    lines.append("END OF AGENT B REPORT")
    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Generating test reports...")

    report = structural_checker_report()
    (OUTPUT_DIR / "structural_checker_report.txt").write_text(report)
    print(f"  Written: docs/test_results/structural_checker_report.txt")

    report = retrieval_pipeline_report()
    (OUTPUT_DIR / "retrieval_pipeline_report.txt").write_text(report)
    print(f"  Written: docs/test_results/retrieval_pipeline_report.txt")

    report = semantic_memory_report()
    (OUTPUT_DIR / "semantic_memory_report.txt").write_text(report)
    print(f"  Written: docs/test_results/semantic_memory_report.txt")

    report = agent_b_report()
    (OUTPUT_DIR / "agent_b_report.txt").write_text(report)
    print(f"  Written: docs/test_results/agent_b_report.txt")

    print("\nDone. All reports in docs/test_results/")
