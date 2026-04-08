"""MILP objective function.

The (a^c + a^r) * x product reduces to (a^c + a^r) because
C2/C3 linearization forces a^c = a^r = 0 whenever x = 0.
"""

from __future__ import annotations

import pulp

from orion.config import MILPConfig
from orion.milp.variables import MILPVariables
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import SliceRequest


def build_objective(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
    config: MILPConfig,
) -> None:
    intra_ids: set[str] = set()
    inter_ids: set[str] = set()
    for _, _, d in substrate.graph.edges(data=True):
        if d["link_type"] == "intra":
            intra_ids.add(d["link_id"])
        else:
            inter_ids.add(d["link_id"])

    # Admission revenue: mu * sum_s z_s
    admission_revenue = config.mu * pulp.lpSum(vars.z.values())

    # Compute resource cost: alpha * sum_{s,f,n} (a_c + a_r)
    resource_cost = config.alpha * pulp.lpSum(
        vars.a_c[s.request_id][f.vnf_id][n_id] + vars.a_r[s.request_id][f.vnf_id][n_id]
        for s in slices
        for f in s.vnfs
        for n_id in f.permitted_nodes
    )

    # Intra-domain bandwidth cost
    bw_cost_intra = config.gamma_intra * pulp.lpSum(
        vars.a_b[s.request_id][(e.source_vnf, e.target_vnf)][l_id]
        for s in slices
        for e in s.flow_edges
        for l_id in intra_ids
        if l_id in vars.a_b[s.request_id].get((e.source_vnf, e.target_vnf), {})
    )

    # Inter-domain bandwidth cost (penalized more heavily, gamma_inter > gamma_intra)
    bw_cost_inter = config.gamma_inter * pulp.lpSum(
        vars.a_b[s.request_id][(e.source_vnf, e.target_vnf)][l_id]
        for s in slices
        for e in s.flow_edges
        for l_id in inter_ids
        if l_id in vars.a_b[s.request_id].get((e.source_vnf, e.target_vnf), {})
    )

    prob += admission_revenue - resource_cost - bw_cost_intra - bw_cost_inter
