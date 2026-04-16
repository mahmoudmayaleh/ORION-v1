"""Quick demo of the MILP solver.

Usage:
    python scripts/demo_solver.py
or:
    .venv/bin/python3 scripts/demo_solver.py 
"""
from __future__ import annotations

import numpy as np

from orion.config import MILPConfig, TopologyConfig
from orion.milp.solver import MILPSolver
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import VNF, FlowEdge, QoSRequirements, SliceRequest, SliceType


def make_request(
    request_id: str,
    substrate_nodes: list[str],
) -> SliceRequest:
    """Build a simple 2-VNF eMBB slice request."""
    return SliceRequest(
        request_id=request_id,
        slice_type=SliceType.EMBB,
        vnfs=[
            VNF("f0", "Firewall", cpu_demand=2.0, ram_demand=4.0,
                permitted_nodes=substrate_nodes),
            VNF("f1", "vEPC", cpu_demand=4.0, ram_demand=8.0,
                permitted_nodes=substrate_nodes),
        ],
        flow_edges=[FlowEdge("f0", "f1", bandwidth_demand=50.0)],
        qos=QoSRequirements(max_e2e_delay=80.0, min_throughput=20.0),
        arrival_time=0.0,
        lifetime=0.0,
    )


def main() -> None:
    rng = np.random.default_rng(42)

    # 1. Generate a 3-domain substrate
    substrate = generate_multi_domain_topology(TopologyConfig(), rng=rng)
    all_nodes = list(substrate.graph.nodes())
    print("=== Substrate ===")
    print(f"Domains: {substrate.num_domains}")
    print(f"Nodes:   {len(all_nodes)}")
    print(f"Links:   {substrate.graph.number_of_edges()}")
    print()

    # 2. Build a batch of slice requests
    requests = [make_request(f"req_{i+1:03d}", all_nodes) for i in range(5)]
    print("=== Slice Requests ===")
    for r in requests:
        print(
            f"  {r.request_id}: {len(r.vnfs)} VNFs, "
            f"delay<={r.qos.max_e2e_delay}ms, "
            f"throughput>={r.qos.min_throughput}Mbps"
        )
    print()

    # 3. Solve the full MILP (all C1-C9 constraints)
    solver = MILPSolver(MILPConfig())
    solution = solver.solve(substrate, requests)

    print("=== MILP Solution ===")
    print(f"Status:     {solution.status}")
    print(f"Objective:  {solution.objective_value:.2f}")
    print(f"Admitted:   {sum(solution.admitted.values())}/{len(requests)}")
    print(f"Solve time: {solution.solve_time:.3f}s")
    print()

    # 4. Show placement details
    for sid, admitted in solution.admitted.items():
        if admitted:
            plan = solution.placements[sid]
            print(f"  {sid}: ADMITTED")
            for fid, nid in plan.vnf_placements.items():
                print(f"    {fid} -> {nid}  CPU={plan.cpu_allocations[fid]:.1f}  RAM={plan.ram_allocations[fid]:.1f}")
            for flow, links in plan.flow_routes.items():
                if links:
                    print(f"    route {flow[0]}->{flow[1]}: {' -> '.join(links)}")
        else:
            print(f"  {sid}: REJECTED")


if __name__ == "__main__":
    main()
