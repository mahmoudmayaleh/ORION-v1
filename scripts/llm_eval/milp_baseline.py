"""MILP baseline for the Agent B placement evaluation.

Builds a concrete substrate that mirrors abstract_topology.json
(same inter-domain delays: d0-d1=2ms, d1-d2=5ms; per-domain capacities
aligned with abstract residuals), then runs the MILP solver on the XR
telepresence slice and produces a domain-level summary comparable to
Agent B outputs.

Usage:
    python scripts/llm_eval/milp_baseline.py

Output is printed and saved to llm-outputs/agent_b_milp_baseline.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import networkx as nx

from orion.config import MILPConfig
from orion.milp.solver import MILPSolver
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import VNF, FlowEdge, QoSRequirements, SliceRequest, SliceType

DATA_DIR   = Path(__file__).resolve().parents[2] / "data" / "placement_eval"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "llm-outputs"

SLICE_REQUEST_PATH = DATA_DIR / "slice_request.json"


def build_aligned_substrate() -> SubstrateNetwork:
    """Build a 3-domain substrate that mirrors abstract_topology.json.

    Domain layout (matches abstract_topology.json):
      d0 (RAN/Edge)       : 2 nodes — ran_edge + mec     cpu≈12, ram≈24
      d1 (MEC/Regional)   : 2 nodes — mec + regional     cpu≈25, ram≈60
      d2 (Regional/Cloud) : 2 nodes — regional + central cpu≈100, ram≈240

    Inter-domain links exactly match abstract_topology.json:
      d0-d1: 500 Mbps, 2.0 ms propagation
      d1-d2: 1000 Mbps, 5.0 ms propagation
    """
    g = nx.DiGraph()

    # ── Domain 0: RAN/Edge — aggregate ~12 CPU, ~24 RAM ───────────────────────
    # Two nodes; each node must individually hold f1 (cpu=4, ram=8).
    # Using slightly unequal split so either can host f1.
    nodes_d0 = [
        ("d0n0", {"domain_id": 0, "tier": "ran_edge",
                  "cpu_capacity": 6.0,  "ram_capacity": 12.0, "processing_delay": 0.5}),
        ("d0n1", {"domain_id": 0, "tier": "mec",
                  "cpu_capacity": 8.0,  "ram_capacity": 14.0, "processing_delay": 0.8}),
    ]  # total d0: 14 CPU, 26 RAM — close to abstract (12 CPU, 24 RAM)
       # Slightly above abstract aggregate so f1 (cpu=4) always fits

    # ── Domain 1: MEC/Regional — aggregate ~25 CPU, ~60 RAM ──────────────────
    # Two nodes; must individually support f2 (cpu=14, ram=30) and f3 (cpu=10, ram=25).
    # f2+f3 together = 24 CPU < 25; but they cannot share one 15-CPU node.
    nodes_d1 = [
        ("d1n0", {"domain_id": 1, "tier": "mec",
                  "cpu_capacity": 15.0, "ram_capacity": 35.0, "processing_delay": 0.8}),
        ("d1n1", {"domain_id": 1, "tier": "regional_cloud",
                  "cpu_capacity": 12.0, "ram_capacity": 30.0, "processing_delay": 1.0}),
    ]  # total d1: 27 CPU, 65 RAM — f2 must go to d1n0, f3 can go to d1n0 or d1n1
       # (they cannot share d1n0: 14+10=24>15, so f2→d1n0, f3→d1n1)

    # ── Domain 2: Regional/Central — aggregate ~100 CPU, ~240 RAM ────────────
    # Two nodes; must individually support f4 (cpu=18, ram=45).
    nodes_d2 = [
        ("d2n0", {"domain_id": 2, "tier": "regional_cloud",
                  "cpu_capacity": 55.0, "ram_capacity": 125.0, "processing_delay": 1.0}),
        ("d2n1", {"domain_id": 2, "tier": "central_cloud",
                  "cpu_capacity": 50.0, "ram_capacity": 120.0, "processing_delay": 1.5}),
    ]  # total d2: 105 CPU, 245 RAM ✓

    for node_id, attrs in nodes_d0 + nodes_d1 + nodes_d2:
        g.add_node(
            node_id,
            **attrs,
            cpu_residual=attrs["cpu_capacity"],
            ram_residual=attrs["ram_capacity"],
        )

    # ── Intra-domain links (fast, high-BW — not a bottleneck) ─────────────────
    intra = [
        ("d0n0", "d0n1", 2000.0, 0.1),
        ("d1n0", "d1n1", 2000.0, 0.1),
        ("d2n0", "d2n1", 2000.0, 0.1),
    ]
    for u, v, bw, delay in intra:
        for src, dst in [(u, v), (v, u)]:
            g.add_edge(src, dst,
                       link_id=f"l_{src}_{dst}",
                       bandwidth_capacity=bw,
                       bw_residual=bw,
                       propagation_delay=delay,
                       link_type="intra")

    # ── Inter-domain links (matching abstract_topology.json exactly) ──────────
    inter = [
        ("d0n0", "d1n0", 500.0,  2.0),   # l_d0_d1: 500 Mbps, 2 ms
        ("d1n0", "d2n0", 1000.0, 5.0),   # l_d1_d2: 1000 Mbps, 5 ms
    ]
    for u, v, bw, delay in inter:
        for src, dst in [(u, v), (v, u)]:
            g.add_edge(src, dst,
                       link_id=f"l_{src}_{dst}",
                       bandwidth_capacity=bw,
                       bw_residual=bw,
                       propagation_delay=delay,
                       link_type="inter")

    return SubstrateNetwork(graph=g, num_domains=3)


def load_slice_request(substrate: SubstrateNetwork) -> SliceRequest:
    """Load the JSON slice request; map permitted_tiers → permitted_nodes."""
    raw = json.loads(SLICE_REQUEST_PATH.read_text())

    tier_to_nodes: dict[str, list[str]] = {}
    for node_id, attrs in substrate.graph.nodes(data=True):
        tier_to_nodes.setdefault(attrs["tier"], []).append(node_id)

    vnfs = []
    for v in raw["vnfs"]:
        permitted: list[str] = []
        for tier in v["permitted_tiers"]:
            permitted.extend(tier_to_nodes.get(tier, []))
        vnfs.append(VNF(
            vnf_id=v["vnf_id"],
            vnf_type=v["vnf_type"],
            cpu_demand=v["cpu_demand"],
            ram_demand=v["ram_demand"],
            permitted_nodes=list(dict.fromkeys(permitted)),   # deduplicate, keep order
            computational_intensity=v.get("computational_intensity", 1.0),
        ))

    flow_edges = [
        FlowEdge(source_vnf=e["source_vnf"],
                 target_vnf=e["target_vnf"],
                 bandwidth_demand=e["bandwidth_demand"])
        for e in raw["flow_edges"]
    ]

    return SliceRequest(
        request_id=raw["request_id"],
        slice_type=SliceType(raw["slice_type"]),
        vnfs=vnfs,
        flow_edges=flow_edges,
        qos=QoSRequirements(
            max_e2e_delay=raw["qos"]["max_e2e_delay"],
            min_throughput=raw["qos"]["min_throughput"],
        ),
        arrival_time=raw.get("arrival_time", 0.0),
        lifetime=raw.get("lifetime", 0.0),
    )


def domain_of(node_id: str) -> str:
    return node_id.split("n")[0]


def main() -> None:
    print("Building substrate (mirroring abstract_topology.json)...")
    substrate = build_aligned_substrate()
    node_data = dict(substrate.graph.nodes(data=True))

    print("\n=== Substrate Summary ===")
    for d_id in range(3):
        nodes = substrate.nodes_in_domain(d_id)
        total_cpu = sum(node_data[n]["cpu_capacity"] for n in nodes)
        total_ram = sum(node_data[n]["ram_capacity"] for n in nodes)
        tiers = sorted({node_data[n]["tier"] for n in nodes})
        print(f"  d{d_id}: {len(nodes)} nodes | tiers={tiers} | "
              f"total_cpu={total_cpu:.1f} | total_ram={total_ram:.1f}")

    print("\n=== Inter-domain Links ===")
    for u, v, d in substrate.graph.edges(data=True):
        if d["link_type"] == "inter":
            print(f"  {u}->{v}: {d['bandwidth_capacity']:.0f} Mbps, "
                  f"{d['propagation_delay']:.1f} ms")

    print("\nLoading slice request...")
    request = load_slice_request(substrate)

    print(f"\n=== Slice Request: {request.request_id} ===")
    for vnf in request.vnfs:
        domain_set = sorted({domain_of(n) for n in vnf.permitted_nodes})
        print(f"  {vnf.vnf_id} ({vnf.vnf_type}): "
              f"cpu={vnf.cpu_demand}, ram={vnf.ram_demand} | "
              f"permitted_domains={domain_set}")

    print("\nRunning MILP solver...")
    # Use large mu so admission revenue always dominates resource costs,
    # turning the optimisation into feasibility-seeking for a single slice.
    solver = MILPSolver(MILPConfig(mu=10000.0))
    solution = solver.solve(substrate, [request])

    print("\n=== MILP Solution ===")
    print(f"  Status:     {solution.status}")
    print(f"  Objective:  {solution.objective_value:.4f}")
    print(f"  Solve time: {solution.solve_time:.3f}s")
    print(f"  MIP gap:    {solution.gap:.4f}")

    rid = request.request_id
    admitted = solution.admitted.get(rid, False)
    print(f"  Admitted:   {admitted}")

    lines = [
        "## Agent B Placement Plan — MILP Baseline (Optimal)",
        "",
        f"**Slice:** `{rid}` ({request.slice_type}, {len(request.vnfs)} VNFs)",
        f"**Solver:** MILP/CBC  |  **Status:** {solution.status}",
        f"**Admitted:** {admitted}  |  "
        f"**Solve time:** {solution.solve_time:.3f}s  |  "
        f"**Gap:** {solution.gap:.4f}",
        "",
        "**Substrate note:** Concrete substrate built to mirror `abstract_topology.json` "
        "(inter-domain delays d0-d1=2ms, d1-d2=5ms; BW d0-d1=500 Mbps, d1-d2=1000 Mbps).",
        "",
    ]

    if not admitted:
        print("  Slice was REJECTED — check delay/resource constraints.")
        lines.append("### Result: REJECTED (infeasible given substrate constraints)")
        lines.append("")
        lines.append("The MILP could not admit the slice within the given QoS budget "
                     f"(max_e2e_delay={request.qos.max_e2e_delay}ms, "
                     f"min_throughput={request.qos.min_throughput}Mbps).")
    else:
        plan = solution.placements[rid]

        domain_cpu: dict[str, float] = {}
        domain_ram: dict[str, float] = {}
        domain_vnfs: dict[str, list[str]] = {}

        print("\n=== VNF Placements ===")
        for vnf in request.vnfs:
            fid = vnf.vnf_id
            nid = plan.vnf_placements[fid]
            dom = domain_of(nid)
            cpu_alloc = plan.cpu_allocations[fid]
            ram_alloc = plan.ram_allocations[fid]
            tier = node_data[nid]["tier"]
            print(f"  {fid} ({vnf.vnf_type}) -> {nid} [{dom}, {tier}] "
                  f"cpu={cpu_alloc:.1f}, ram={ram_alloc:.1f}")
            domain_cpu[dom] = domain_cpu.get(dom, 0.0) + cpu_alloc
            domain_ram[dom] = domain_ram.get(dom, 0.0) + ram_alloc
            domain_vnfs.setdefault(dom, []).append(f"{fid}({vnf.vnf_type})")

        print("\n=== Domain-Level Summary ===")
        for dom in sorted(domain_vnfs):
            print(f"  {dom}: VNFs={domain_vnfs[dom]} | "
                  f"cpu_used={domain_cpu[dom]:.1f} | ram_used={domain_ram[dom]:.1f}")

        print("\n=== Flow Routes ===")
        for flow, links in plan.flow_routes.items():
            src_dom = domain_of(plan.vnf_placements[flow[0]])
            dst_dom = domain_of(plan.vnf_placements[flow[1]])
            crosses = src_dom != dst_dom
            bw = next((e.bandwidth_demand for e in request.flow_edges
                       if e.source_vnf == flow[0] and e.target_vnf == flow[1]), 0.0)
            print(f"  {flow[0]} -> {flow[1]}: bw={bw:.0f} Mbps | "
                  f"crosses_domain={crosses} | hops={len(links)}")
            if links:
                print(f"    path: {' -> '.join(links)}")

        # ── Markdown output ────────────────────────────────────────────────────
        lines += [
            "### VNF Assignments",
            "",
            "| VNF | Type | Node | Domain | Tier | CPU | RAM |",
            "|-----|------|------|--------|------|-----|-----|",
        ]
        for vnf in request.vnfs:
            fid = vnf.vnf_id
            nid = plan.vnf_placements[fid]
            dom = domain_of(nid)
            tier = node_data[nid]["tier"]
            cpu_alloc = plan.cpu_allocations[fid]
            ram_alloc = plan.ram_allocations[fid]
            lines.append(f"| `{fid}` | {vnf.vnf_type} | `{nid}` | `{dom}` | "
                         f"{tier} | {cpu_alloc:.1f} | {ram_alloc:.1f} |")

        lines += [
            "",
            "### Domain-Level Resource Usage",
            "",
            "| Domain | VNFs | CPU Used | RAM Used |",
            "|--------|------|----------|----------|",
        ]
        for dom in sorted(domain_vnfs):
            vnf_str = ", ".join(domain_vnfs[dom])
            lines.append(f"| `{dom}` | {vnf_str} | "
                         f"{domain_cpu[dom]:.1f} | {domain_ram[dom]:.1f} |")

        lines += [
            "",
            "### Flow Requirements",
            "",
            "| Source VNF | Target VNF | BW (Mbps) | Crosses Domain |",
            "|------------|------------|-----------|----------------|",
        ]
        for flow, links in plan.flow_routes.items():
            src_dom = domain_of(plan.vnf_placements[flow[0]])
            dst_dom = domain_of(plan.vnf_placements[flow[1]])
            crosses = src_dom != dst_dom
            bw = next((e.bandwidth_demand for e in request.flow_edges
                       if e.source_vnf == flow[0] and e.target_vnf == flow[1]), 0.0)
            lines.append(f"| `{flow[0]}` | `{flow[1]}` | {bw:.0f} | {crosses} |")

        # JSON plan in Agent B-compatible format
        json_plan = {
            "plan_id": f"{rid}_plan",
            "vnf_assignments": [
                {
                    "vnf_id": vnf.vnf_id,
                    "domain": domain_of(plan.vnf_placements[vnf.vnf_id]),
                    "node": plan.vnf_placements[vnf.vnf_id],
                    "required_tier": node_data[plan.vnf_placements[vnf.vnf_id]]["tier"],
                    "cpu_demand": plan.cpu_allocations[vnf.vnf_id],
                    "ram_demand": plan.ram_allocations[vnf.vnf_id],
                }
                for vnf in request.vnfs
            ],
            "flow_requirements": [
                {
                    "source_vnf": flow[0],
                    "target_vnf": flow[1],
                    "min_bandwidth_mbps": next(
                        (e.bandwidth_demand for e in request.flow_edges
                         if e.source_vnf == flow[0] and e.target_vnf == flow[1]), 0.0
                    ),
                    "crosses_domain_boundary": (
                        domain_of(plan.vnf_placements[flow[0]]) !=
                        domain_of(plan.vnf_placements[flow[1]])
                    ),
                    "route": list(links),
                }
                for flow, links in plan.flow_routes.items()
            ],
            "solver": "MILP (CBC)",
            "status": solution.status,
            "objective": round(solution.objective_value, 4),
            "solve_time_s": round(solution.solve_time, 3),
        }

        lines += [
            "",
            "### JSON (Agent B-compatible format)",
            "",
            "```json",
            json.dumps(json_plan, indent=2),
            "```",
        ]

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "agent_b_milp_baseline.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
