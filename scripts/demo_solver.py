"""Quick demo of the MILP solver for the supervisor meeting.

Usage:
    python scripts/demo_solver.py
"""
from __future__ import annotations

import yaml
import numpy as np

from orion.config import MILPConfig, TopologyConfig, SlicingConfig
from orion.milp.solver import MILPSolver
from orion.milp.feasibility import FeasibilityChecker
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.slicing.request_generator import PoissonArrivalGenerator


def main() -> None:
    rng = np.random.default_rng(7)

    # 1. Generate a 3-domain substrate
    substrate = generate_multi_domain_topology(TopologyConfig(), rng=rng)
    print("=== Substrate ===")
    print(f"Domains: {substrate.num_domains}")
    print(f"Nodes:   {substrate.graph.number_of_nodes()}")
    print(f"Links:   {substrate.graph.number_of_edges()}")
    print()

    # 2. Generate slice requests
    with open("configs/slicing/standard_sfc.yaml") as f:
        templates = yaml.safe_load(f)
    gen = PoissonArrivalGenerator(SlicingConfig(), substrate, templates, rng=rng)
    requests = gen.generate_batch(10)

    print("=== Slice Requests ===")
    for r in requests:
        print(
            f"  {r.request_id}: {r.slice_type.value:5s} | "
            f"{len(r.vnfs)} VNFs | "
            f"delay<={r.qos.max_e2e_delay:6.1f}ms  "
            f"throughput>={r.qos.min_throughput:7.1f}Mbps"
        )
    print()

    # 3. Solve the full MILP
    solver = MILPSolver(MILPConfig())
    solution = solver.solve(substrate, requests)

    print("=== MILP Solution ===")
    print(f"Status:     {solution.status}")
    print(f"Objective:  {solution.objective_value:.2f}")
    print(f"Admitted:   {sum(solution.admitted.values())}/{len(requests)}")
    print(f"Solve time: {solution.solve_time:.3f}s")
    print()

    # 4. Show placement details for admitted slices
    for sid, admitted in solution.admitted.items():
        if admitted:
            plan = solution.placements[sid]
            print(f"  {sid}: ADMITTED")
            for fid, nid in plan.vnf_placements.items():
                print(
                    f"    {fid} -> {nid}  "
                    f"CPU={plan.cpu_allocations[fid]:.1f}  "
                    f"RAM={plan.ram_allocations[fid]:.1f}"
                )
            for flow, links in plan.flow_routes.items():
                if links:
                    print(f"    route {flow[0]}->{flow[1]}: {' -> '.join(links)}")
        else:
            print(f"  {sid}: REJECTED")
    print()

    # 5. Feasibility check on first admitted plan
    checker = FeasibilityChecker(substrate)
    for sid, admitted in solution.admitted.items():
        if admitted:
            plan = solution.placements[sid]
            req = [r for r in requests if r.request_id == sid][0]
            res = checker.check(plan, req)
            stc = checker.check_structural(plan, req)
            print(f"=== Feasibility [{sid}] ===")
            tag = "PASS" if res.is_feasible else "FAIL"
            print(f"  Resource (C2,C3,C5,C5b,C7): {tag}")
            if not res.is_feasible:
                print(f"    violations: {res.violated_constraints}")
            tag = "PASS" if stc.is_feasible else "FAIL"
            print(f"  Structural (C1,C4,C6,C8):   {tag}")
            break


if __name__ == "__main__":
    main()
