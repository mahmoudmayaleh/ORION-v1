"""Offline feasibility evaluation — diagnostic gate for Agent B.

Generates N slice requests, runs Agent B (via LLM backend) on each, applies
the structural checker, and optionally runs the MILP oracle on structurally
valid plans. Reports first-attempt feasibility rate.

Target: >60% structural pass rate before proceeding to RL phases.

Usage:
    # With a live OpenAI-compatible API (vLLM, OpenAI, Anthropic):
    python scripts/llm_eval/offline_feasibility.py \
        --base-url http://localhost:8000/v1 \
        --model gemma-4-26b-moe-it \
        --num-requests 50

    # Dry run (structural check only, skip MILP):
    python scripts/llm_eval/offline_feasibility.py \
        --base-url http://localhost:8000/v1 \
        --model gemma-4-26b-moe-it \
        --num-requests 20 \
        --skip-milp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orion.config import MILPConfig, TopologyConfig
from orion.llm.abstract_topology import build_abstract_topology
from orion.llm.agent_b import AgentB
from orion.llm.llm_backend import LLMBackend, LLMConfig
from orion.llm.semantic_memory import SemanticMemory, build_query_from_slice
from orion.llm.structural_checker import check_plan
from orion.milp.solver import MILPSolver
from orion.sim.slice_generator import generate_slice_request
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import VNF, FlowEdge, QoSRequirements, SliceRequest, SliceType

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
KB_PATH = DATA_DIR / "memory" / "kb_agent_b.json"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "llm-outputs"


def _slice_request_to_dict(req: SliceRequest) -> dict:
    """Convert a SliceRequest dataclass to Agent B's expected dict format."""
    return {
        "request_id": req.request_id,
        "slice_type": req.slice_type.value,
        "vnfs": [
            {
                "vnf_id": v.vnf_id,
                "vnf_type": v.vnf_type,
                "cpu_demand": v.cpu_demand,
                "ram_demand": v.ram_demand,
                "permitted_tiers": _nodes_to_tiers(v.permitted_nodes, substrate_ref[0]),
                "computational_intensity": v.computational_intensity,
                "vcr": v.vcr,
            }
            for v in req.vnfs
        ],
        "flow_edges": [
            {
                "source_vnf": e.source_vnf,
                "target_vnf": e.target_vnf,
                "bandwidth_demand": e.bandwidth_demand,
            }
            for e in req.flow_edges
        ],
        "qos": {
            "max_e2e_delay": req.qos.max_e2e_delay,
            "min_throughput": req.qos.min_throughput,
        },
    }


# Module-level reference to substrate, set in main()
substrate_ref: list = [None]


def _nodes_to_tiers(permitted_nodes: list[str], substrate) -> list[str]:
    """Convert permitted_nodes list back to tier names for Agent B."""
    tiers = set()
    for n in permitted_nodes:
        tiers.add(substrate.graph.nodes[n]["tier"])
    return sorted(tiers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline feasibility evaluation for Agent B."
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="default")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-milp", action="store_true",
                        help="Skip MILP oracle check (structural only).")
    parser.add_argument("--kb-top-k", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Build substrate ──────────────────────────────────────────────────
    config = TopologyConfig(
        num_domains=3,
        nodes_per_domain=[6, 8, 10],
    )
    substrate = generate_multi_domain_topology(config, rng)
    substrate.reset()
    substrate_ref[0] = substrate

    abstract_topo = build_abstract_topology(substrate)

    print(f"Substrate: {substrate.graph.number_of_nodes()} nodes, "
          f"{substrate.graph.number_of_edges()} edges, "
          f"{substrate.num_domains} domains")
    for d in abstract_topo["domains"]:
        print(f"  {d['domain_id']} ({d['label']}): "
              f"CPU={d['cpu_residual']:.0f}, RAM={d['ram_residual']:.0f}, "
              f"tiers={d['dominant_tiers']}")

    # ── Load K^B ─────────────────────────────────────────────────────────
    kb = SemanticMemory.from_json(KB_PATH)
    print(f"\nK^B loaded: {len(kb.entries)} entries")

    # ── Setup LLM backend ────────────────────────────────────────────────
    llm_config = LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )
    llm = LLMBackend(llm_config)
    agent_b = AgentB(llm=llm)

    # ── Setup MILP solver ────────────────────────────────────────────────
    milp_solver = None if args.skip_milp else MILPSolver(MILPConfig(mu=10000.0))

    # ── Generate and evaluate ────────────────────────────────────────────
    print(f"\nGenerating {args.num_requests} slice requests...")
    print(f"Model: {args.model}")
    print(f"Retries: {args.max_retries}")
    print(f"K^B top-k: {args.kb_top_k}")
    print(f"MILP check: {'enabled' if milp_solver else 'skipped'}")
    print("=" * 70)

    results = []
    for i in range(args.num_requests):
        req = generate_slice_request(
            request_id=f"eval_{i:04d}",
            substrate=substrate,
            rng=rng,
        )
        req_dict = _slice_request_to_dict(req)

        # K^B retrieval
        query = build_query_from_slice(req_dict)
        kb_entries = kb.retrieve(
            query,
            slice_type=req_dict.get("slice_type"),
            top_k=args.kb_top_k,
        )
        ref_text = kb.format_for_prompt(kb_entries)

        # Run Agent B
        t0 = time.perf_counter()
        try:
            plan, check_result = agent_b.generate_and_check(
                req_dict, abstract_topo,
                max_retries=args.max_retries,
                reference_knowledge=ref_text,
            )
            elapsed = time.perf_counter() - t0
            structural_pass = check_result.is_valid
        except Exception as e:
            elapsed = time.perf_counter() - t0
            structural_pass = False
            plan = {}
            check_result = None
            print(f"  [{i+1:3d}] {req.request_id} ({req.slice_type:5s}) "
                  f"ERROR: {e}")
            results.append({
                "request_id": req.request_id,
                "slice_type": req.slice_type.value,
                "structural_pass": False,
                "milp_feasible": None,
                "error": str(e),
                "elapsed_s": round(elapsed, 2),
            })
            continue

        # MILP check (if structural pass and MILP enabled)
        milp_feasible = None
        if structural_pass and milp_solver:
            try:
                milp_sol = milp_solver.solve(substrate, [req])
                milp_feasible = milp_sol.admitted.get(req.request_id, False)
            except Exception:
                milp_feasible = None

        status = "PASS" if structural_pass else "FAIL"
        milp_str = ""
        if milp_feasible is not None:
            milp_str = f" | MILP={'admit' if milp_feasible else 'reject'}"

        violations = ""
        if check_result and not check_result.is_valid:
            tags = sorted({v.constraint for v in check_result.violations})
            violations = f" [{','.join(tags)}]"

        print(f"  [{i+1:3d}] {req.request_id} ({req.slice_type:5s}) "
              f"{status}{violations}{milp_str} ({elapsed:.1f}s)")

        results.append({
            "request_id": req.request_id,
            "slice_type": req.slice_type.value,
            "structural_pass": structural_pass,
            "milp_feasible": milp_feasible,
            "elapsed_s": round(elapsed, 2),
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    total = len(results)
    structural_passes = sum(1 for r in results if r["structural_pass"])
    structural_rate = structural_passes / total * 100 if total > 0 else 0

    print(f"Structural pass rate: {structural_passes}/{total} "
          f"({structural_rate:.1f}%)")

    if not args.skip_milp:
        milp_tested = [r for r in results if r["milp_feasible"] is not None]
        milp_passes = sum(1 for r in milp_tested if r["milp_feasible"])
        if milp_tested:
            print(f"MILP feasibility (of structural passes): "
                  f"{milp_passes}/{len(milp_tested)} "
                  f"({milp_passes/len(milp_tested)*100:.1f}%)")

    # Per slice-type breakdown
    from collections import Counter
    type_counts: dict[str, dict[str, int]] = {}
    for r in results:
        st = r["slice_type"]
        if st not in type_counts:
            type_counts[st] = {"total": 0, "structural": 0, "milp": 0}
        type_counts[st]["total"] += 1
        if r["structural_pass"]:
            type_counts[st]["structural"] += 1
        if r.get("milp_feasible"):
            type_counts[st]["milp"] += 1

    print("\nPer slice-type breakdown:")
    for st in sorted(type_counts):
        c = type_counts[st]
        s_rate = c["structural"] / c["total"] * 100
        print(f"  {st:5s}: {c['structural']}/{c['total']} structural "
              f"({s_rate:.0f}%)", end="")
        if not args.skip_milp and c["structural"] > 0:
            m_rate = c["milp"] / c["structural"] * 100 if c["structural"] > 0 else 0
            print(f" | {c['milp']}/{c['structural']} MILP ({m_rate:.0f}%)", end="")
        print()

    avg_time = np.mean([r["elapsed_s"] for r in results])
    print(f"\nAvg latency: {avg_time:.2f}s per request")

    target_met = structural_rate >= 60
    print(f"\nTarget (>60% structural): {'MET' if target_met else 'NOT MET'}")

    # ── Save results ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "offline_feasibility_results.json"
    out_path.write_text(json.dumps({
        "model": args.model,
        "num_requests": total,
        "structural_pass_rate": round(structural_rate, 1),
        "seed": args.seed,
        "results": results,
    }, indent=2))
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
