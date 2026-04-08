"""MILP constraints C1-C9.

Per Proposed_solution.pdf Section 3.3 / 5.1:
  Structural (checked by Agent B before Module 2): C1, C4, C6, C8
  Resource   (checked by MILP Validator):          C2, C3, C5, C5b, C7, C9
"""

from __future__ import annotations

import pulp

from orion.config import MILPConfig
from orion.milp.variables import MILPVariables
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import SliceRequest


def add_c1_placement_uniqueness(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
) -> None:
    """C1: Each VNF of an admitted slice is placed on exactly one node."""
    for s in slices:
        sid = s.request_id
        for f in s.vnfs:
            prob += (
                pulp.lpSum(vars.x[sid][f.vnf_id][n] for n in f.permitted_nodes) == vars.z[sid],
                f"C1_{sid}_{f.vnf_id}",
            )


def add_c2_cpu_capacity(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
) -> None:
    """C2: Total CPU at a node cannot exceed capacity.
    Linearization: a^c_{s,f,n} <= C_n * x_{s,f,n}.
    """
    for n_id in substrate.graph.nodes():
        c_n = substrate.graph.nodes[n_id]["cpu_capacity"]
        prob += (
            pulp.lpSum(
                vars.a_c[s.request_id][f.vnf_id][n_id]
                for s in slices
                for f in s.vnfs
                if n_id in f.permitted_nodes
            ) <= c_n,
            f"C2_agg_{n_id}",
        )
        for s in slices:
            for f in s.vnfs:
                if n_id in f.permitted_nodes:
                    prob += (
                        vars.a_c[s.request_id][f.vnf_id][n_id]
                        <= c_n * vars.x[s.request_id][f.vnf_id][n_id],
                        f"C2_lin_{s.request_id}_{f.vnf_id}_{n_id}",
                    )


def add_c3_ram_capacity(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
) -> None:
    """C3: Total RAM at a node cannot exceed capacity.
    Linearization: a^r_{s,f,n} <= R_n * x_{s,f,n}.
    """
    for n_id in substrate.graph.nodes():
        r_n = substrate.graph.nodes[n_id]["ram_capacity"]
        prob += (
            pulp.lpSum(
                vars.a_r[s.request_id][f.vnf_id][n_id]
                for s in slices
                for f in s.vnfs
                if n_id in f.permitted_nodes
            ) <= r_n,
            f"C3_agg_{n_id}",
        )
        for s in slices:
            for f in s.vnfs:
                if n_id in f.permitted_nodes:
                    prob += (
                        vars.a_r[s.request_id][f.vnf_id][n_id]
                        <= r_n * vars.x[s.request_id][f.vnf_id][n_id],
                        f"C3_lin_{s.request_id}_{f.vnf_id}_{n_id}",
                    )


def add_c4_resource_sufficiency(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
) -> None:
    """C4: Allocated resources must meet each VNF's minimum demand."""
    for s in slices:
        sid = s.request_id
        for f in s.vnfs:
            for n_id in f.permitted_nodes:
                prob += (
                    vars.a_c[sid][f.vnf_id][n_id] >= f.cpu_demand * vars.x[sid][f.vnf_id][n_id],
                    f"C4_cpu_{sid}_{f.vnf_id}_{n_id}",
                )
                prob += (
                    vars.a_r[sid][f.vnf_id][n_id] >= f.ram_demand * vars.x[sid][f.vnf_id][n_id],
                    f"C4_ram_{sid}_{f.vnf_id}_{n_id}",
                )


def add_c5_bw_capacity(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
) -> None:
    """C5: Total bandwidth on a link cannot exceed capacity.
    Linearization: a^b >= beta * y (demand floor), a^b <= B_l * y (ceiling).
    """
    link_info: dict[str, float] = {
        d["link_id"]: d["bandwidth_capacity"]
        for _, _, d in substrate.graph.edges(data=True)
    }

    for l_id, b_l in link_info.items():
        prob += (
            pulp.lpSum(
                vars.a_b[s.request_id][(e.source_vnf, e.target_vnf)][l_id]
                for s in slices
                for e in s.flow_edges
            ) <= b_l,
            f"C5_agg_{l_id}",
        )
        for s in slices:
            for edge in s.flow_edges:
                flow_key = (edge.source_vnf, edge.target_vnf)
                sid = s.request_id
                y_var = vars.y[sid][flow_key][l_id]
                ab_var = vars.a_b[sid][flow_key][l_id]
                prob += (ab_var <= b_l * y_var, f"C5_lin_{sid}_{flow_key[0]}_{flow_key[1]}_{l_id}")
                prob += (
                    ab_var >= edge.bandwidth_demand * y_var,
                    f"C5_dem_{sid}_{flow_key[0]}_{flow_key[1]}_{l_id}",
                )


def add_c5b_throughput_floor(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
) -> None:
    """C5b: Per-flow throughput must meet the slice QoS minimum.
    Since C5 linearization forces a_b=0 when y=0, the product a_b*y = a_b.
    """
    for s in slices:
        sid = s.request_id
        beta_min = s.qos.min_throughput
        for edge in s.flow_edges:
            flow_key = (edge.source_vnf, edge.target_vnf)
            prob += (
                pulp.lpSum(vars.a_b[sid][flow_key].values()) >= beta_min * vars.z[sid],
                f"C5b_{sid}_{flow_key[0]}_{flow_key[1]}",
            )


def add_c6_flow_conservation(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
) -> None:
    """C6: Routing variables form a valid directed path from placed(f) to placed(f')."""
    for s in slices:
        sid = s.request_id
        vnf_map = {f.vnf_id: f for f in s.vnfs}

        for edge in s.flow_edges:
            f_src_id, f_dst_id = edge.source_vnf, edge.target_vnf
            flow_key = (f_src_id, f_dst_id)
            f_src = vnf_map[f_src_id]
            f_dst = vnf_map[f_dst_id]

            for n_id in substrate.graph.nodes():
                delta_plus, delta_minus = substrate.incident_links(n_id)
                flow_y = vars.y[sid][flow_key]

                outflow = pulp.lpSum(flow_y[lid] for lid in delta_plus if lid in flow_y)
                inflow = pulp.lpSum(flow_y[lid] for lid in delta_minus if lid in flow_y)

                src_term = vars.x[sid][f_src_id][n_id] if n_id in f_src.permitted_nodes else 0
                dst_term = vars.x[sid][f_dst_id][n_id] if n_id in f_dst.permitted_nodes else 0

                prob += (
                    (outflow - inflow) == (src_term - dst_term),
                    f"C6_{sid}_{f_src_id}_{f_dst_id}_{n_id}",
                )


def add_c7_e2e_delay(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
) -> None:
    """C7: Processing + propagation delay must not exceed the slice delay budget."""
    link_delay: dict[str, float] = {
        d["link_id"]: d["propagation_delay"]
        for _, _, d in substrate.graph.edges(data=True)
    }

    for s in slices:
        sid = s.request_id
        proc_delay = pulp.lpSum(
            substrate.graph.nodes[n_id]["processing_delay"] * vars.x[sid][f.vnf_id][n_id]
            for f in s.vnfs
            for n_id in f.permitted_nodes
        )
        prop_delay = pulp.lpSum(
            link_delay[l_id] * vars.y[sid][(e.source_vnf, e.target_vnf)][l_id]
            for e in s.flow_edges
            for l_id in vars.y[sid][(e.source_vnf, e.target_vnf)]
        )
        prob += (
            proc_delay + prop_delay <= s.qos.max_e2e_delay * vars.z[sid],
            f"C7_{sid}",
        )


def add_c9_inter_domain_hops(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
    config: MILPConfig,
) -> None:
    """C9: Routing paths may traverse at most H inter-domain links."""
    inter_link_ids = set(substrate.inter_domain_links())
    h = config.max_inter_domain_hops

    for s in slices:
        sid = s.request_id
        for edge in s.flow_edges:
            flow_key = (edge.source_vnf, edge.target_vnf)
            flow_y = vars.y[sid][flow_key]
            prob += (
                pulp.lpSum(flow_y[lid] for lid in inter_link_ids if lid in flow_y) <= h,
                f"C9_{sid}_{flow_key[0]}_{flow_key[1]}",
            )


def add_all_constraints(
    prob: pulp.LpProblem,
    vars: MILPVariables,
    slices: list[SliceRequest],
    substrate: SubstrateNetwork,
    config: MILPConfig,
) -> None:
    """Add all constraints C1-C9 to the problem."""
    add_c1_placement_uniqueness(prob, vars, slices, substrate)
    add_c2_cpu_capacity(prob, vars, slices, substrate)
    add_c3_ram_capacity(prob, vars, slices, substrate)
    add_c4_resource_sufficiency(prob, vars, slices, substrate)
    add_c5_bw_capacity(prob, vars, slices, substrate)
    add_c5b_throughput_floor(prob, vars, slices, substrate)
    add_c6_flow_conservation(prob, vars, slices, substrate)
    add_c7_e2e_delay(prob, vars, slices, substrate)
    add_c9_inter_domain_hops(prob, vars, slices, substrate, config)
