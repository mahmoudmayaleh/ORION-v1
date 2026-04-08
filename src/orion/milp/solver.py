"""MILPSolver: orchestrates variable creation, constraints, objective, and solving.

Evaluation mode (solve): Full MILP on static batches.
Produces z*_MILP for offline optimality-gap evaluation and
few-shot examples for Agent B.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import Any

import pulp

from orion.config import MILPConfig
from orion.milp.constraints import add_all_constraints
from orion.milp.objective import build_objective
from orion.milp.variables import MILPVariables, create_variables
from orion.slicing.slice_request import slice_request_from_dict, slice_request_to_dict
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import MILPSolution, PlacementPlan, SliceRequest

logger = logging.getLogger("orion.milp")

_STATUS_MAP: dict[int, str] = {
    1: "Optimal",
    0: "Not Solved",
    -1: "Infeasible",
    -2: "Unbounded",
    -3: "Undefined",
}


def _parse_solution(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    solve_time: float,
) -> MILPSolution:
    """Extract a MILPSolution from a solved PuLP problem.

    Binary variables with floating-point deviations are rounded: >= 0.5 is 1.
    """
    status = _STATUS_MAP.get(prob.status, "Unknown")
    obj_val = float(pulp.value(prob.objective) or 0.0)

    admitted: dict[str, bool] = {}
    placements: dict[str, PlacementPlan] = {}

    for s in slices:
        sid = s.request_id
        z_val = pulp.value(vars.z[sid])
        is_admitted = z_val is not None and z_val >= 0.5
        admitted[sid] = is_admitted

        if not is_admitted:
            continue

        vnf_placements: dict[str, str] = {}
        cpu_allocations: dict[str, float] = {}
        ram_allocations: dict[str, float] = {}

        for f in s.vnfs:
            fid = f.vnf_id
            for n_id in f.permitted_nodes:
                x_val = pulp.value(vars.x[sid][fid][n_id])
                if x_val is not None and x_val >= 0.5:
                    vnf_placements[fid] = n_id
                    cpu_allocations[fid] = float(
                        pulp.value(vars.a_c[sid][fid][n_id]) or 0.0
                    )
                    ram_allocations[fid] = float(
                        pulp.value(vars.a_r[sid][fid][n_id]) or 0.0
                    )
                    break

        flow_routes: dict[tuple[str, str], list[str]] = {}
        bw_allocations: dict[tuple[str, str], dict[str, float]] = {}

        for edge in s.flow_edges:
            flow_key = (edge.source_vnf, edge.target_vnf)
            route_links: list[str] = []
            per_link_bw: dict[str, float] = {}

            for l_id, y_var in vars.y[sid][flow_key].items():
                y_val = pulp.value(y_var)
                if y_val is not None and y_val >= 0.5:
                    route_links.append(l_id)
                    per_link_bw[l_id] = float(
                        pulp.value(vars.a_b[sid][flow_key][l_id]) or 0.0
                    )

            flow_routes[flow_key] = route_links
            bw_allocations[flow_key] = per_link_bw

        placements[sid] = PlacementPlan(
            plan_id=f"{sid}_milp",
            vnf_placements=vnf_placements,
            cpu_allocations=cpu_allocations,
            ram_allocations=ram_allocations,
            flow_routes=flow_routes,
            bw_allocations=bw_allocations,
            is_structurally_valid=True,
            source="milp",
        )

    return MILPSolution(
        objective_value=obj_val,
        admitted=admitted,
        placements=placements,
        solve_time=solve_time,
        gap=0.0,
        status=status,
    )


class MILPSolver:
    """Solves the static VNF embedding MILP (evaluation / oracle mode)."""

    def __init__(self, config: MILPConfig) -> None:
        self.config = config

    def solve(
        self,
        substrate: SubstrateNetwork,
        slices: list[SliceRequest],
    ) -> MILPSolution:
        """Solve the full MILP for a single (substrate, slice_batch) instance."""
        prob = pulp.LpProblem("ORION_slice_embedding", pulp.LpMaximize)
        vars = create_variables(prob, slices, substrate)
        add_all_constraints(prob, vars, slices, substrate, self.config)
        build_objective(prob, vars, slices, substrate, self.config)

        solver = pulp.getSolver(
            self.config.solver,
            timeLimit=self.config.time_limit,
            gapRel=self.config.mip_gap,
            msg=0,
        )

        t0 = time.perf_counter()
        prob.solve(solver)
        solve_time = time.perf_counter() - t0

        solution = _parse_solution(prob, vars, slices, solve_time)

        logger.debug(
            "milp_solve_complete",
            extra={
                "status": solution.status,
                "solve_time": round(solve_time, 3),
                "admitted": sum(solution.admitted.values()),
                "total": len(slices),
                "objective": round(solution.objective_value, 2),
            },
        )
        return solution

    def solve_batch(
        self,
        instances: list[tuple[SubstrateNetwork, list[SliceRequest]]],
        n_workers: int = 4,
    ) -> list[MILPSolution]:
        """Solve independent (substrate, slices) instances in parallel."""
        config_dict = self.config.model_dump()
        args = [
            (
                substrate.to_dict(),
                [slice_request_to_dict(r) for r in slices],
                config_dict,
            )
            for substrate, slices in instances
        ]

        with mp.Pool(processes=n_workers) as pool:
            solutions = pool.map(_solve_worker, args)

        return solutions


def _solve_worker(
    args: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
) -> MILPSolution:
    """Worker function for multiprocessing.Pool (must be module-level)."""
    substrate_dict, request_dicts, config_dict = args
    substrate = SubstrateNetwork.from_dict(substrate_dict)
    slices = [slice_request_from_dict(r) for r in request_dicts]
    config = MILPConfig(**config_dict)
    return MILPSolver(config).solve(substrate, slices)
