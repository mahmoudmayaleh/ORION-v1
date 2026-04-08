"""MILP decision variable creation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pulp

from orion.substrate.graph_model import SubstrateNetwork
from orion.types import SliceRequest


def _vname(*parts: str) -> str:
    """
    Args:
        *parts: String components to join with underscores.

    Returns:
        A string safe for use as a PuLP variable name.
    """
    return "_".join(
        p.replace("-", "_").replace(".", "_").replace("(", "").replace(")", "")
        for p in parts
    )


@dataclass
class MILPVariables:
    """Container for all MILP decision variables.

    Attributes:
        x:   x[s_id][f_id][n_id]           — VNF placement binary
        a_c: a_c[s_id][f_id][n_id]         — CPU allocation (continuous)
        a_r: a_r[s_id][f_id][n_id]         — RAM allocation (continuous)
        y:   y[s_id][(f_id, f'_id)][l_id]  — flow routing binary
        a_b: a_b[s_id][(f_id, f'_id)][l_id]— bandwidth allocation (continuous)
        z:   z[s_id]                        — admission decision binary
    """

    x: dict[str, dict[str, dict[str, pulp.LpVariable]]] = field(default_factory=dict)
    a_c: dict[str, dict[str, dict[str, pulp.LpVariable]]] = field(default_factory=dict)
    a_r: dict[str, dict[str, dict[str, pulp.LpVariable]]] = field(default_factory=dict)
    y: dict[str, dict[tuple[str, str], dict[str, pulp.LpVariable]]] = field(default_factory=dict)
    a_b: dict[str, dict[tuple[str, str], dict[str, pulp.LpVariable]]] = field(default_factory=dict)
    z: dict[str, pulp.LpVariable] = field(default_factory=dict)


def create_variables(
    problem: pulp.LpProblem,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
) -> MILPVariables:
    """Create and register all MILP decision variables in the PuLP problem.

    Args:
        problem: The PuLP LpProblem (variables are registered automatically
            when used in constraints added to it).
        slices: List of slice requests in this problem instance.
        substrate: Substrate network providing link and node structure.

    Returns:
        MILPVariables container with all instantiated LpVariable objects.
    """
    link_ids = [d["link_id"] for _, _, d in substrate.graph.edges(data=True)]

    vars = MILPVariables()

    for s in slices:
        sid = s.request_id

        # z[s] -- admission decision
        vars.z[sid] = pulp.LpVariable(_vname("z", sid), cat="Binary")

        vars.x[sid] = {}
        vars.a_c[sid] = {}
        vars.a_r[sid] = {}
        vars.y[sid] = {}
        vars.a_b[sid] = {}

        # Placement and compute variables
        for f in s.vnfs:
            fid = f.vnf_id
            vars.x[sid][fid] = {}
            vars.a_c[sid][fid] = {}
            vars.a_r[sid][fid] = {}
            for n_id in f.permitted_nodes:
                nm = _vname(sid, fid, n_id)
                vars.x[sid][fid][n_id] = pulp.LpVariable(f"x_{nm}", cat="Binary")
                vars.a_c[sid][fid][n_id] = pulp.LpVariable(f"ac_{nm}", lowBound=0.0)
                vars.a_r[sid][fid][n_id] = pulp.LpVariable(f"ar_{nm}", lowBound=0.0)

        # Routing and bandwidth variables — one per (flow, link) pair
        for edge in s.flow_edges:
            flow_key = (edge.source_vnf, edge.target_vnf)
            vars.y[sid][flow_key] = {}
            vars.a_b[sid][flow_key] = {}
            for l_id in link_ids:
                nm = _vname(sid, edge.source_vnf, edge.target_vnf, l_id)
                vars.y[sid][flow_key][l_id] = pulp.LpVariable(f"y_{nm}", cat="Binary")
                vars.a_b[sid][flow_key][l_id] = pulp.LpVariable(f"ab_{nm}", lowBound=0.0)

    return vars
