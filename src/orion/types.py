from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class InfrastructureTier(StrEnum):
    """Node tier in the 6G substrate hierarchy."""

    RAN_EDGE = "ran_edge"
    MEC = "mec"
    REGIONAL_CLOUD = "regional_cloud"
    CENTRAL_CLOUD = "central_cloud"


class LinkType(StrEnum):
    """Whether a link is intra- or inter-domain."""

    INTRA = "intra"
    INTER = "inter"


class SliceType(StrEnum):
    """3GPP/ITU-T 6G network slice service categories."""

    EMBB = "eMBB"
    URLLC = "URLLC"
    MMTC = "mMTC"
    V2X = "V2X"
    XR = "XR"


@dataclass(frozen=True)
class NodeAttributes:
    """Physical/virtual node in the substrate graph.

    Attributes:
        node_id: Unique node identifier, e.g. "d0_n3".
        domain_id: Administrative domain index (0-based).
        tier: Infrastructure tier for placement eligibility (C8).
        cpu_capacity: C_n — total vCPU capacity.
        ram_capacity: R_n — total RAM capacity in GB.
        processing_delay: delta_{f,n} — per-VNF processing time in ms.
    """

    node_id: str
    domain_id: int
    tier: InfrastructureTier
    cpu_capacity: float
    ram_capacity: float
    processing_delay: float


@dataclass(frozen=True)
class LinkAttributes:
    """Directed link in the substrate graph.

    Attributes:
        link_id: Unique link identifier, e.g. "l_d0n2_d0n5".
        source: Source node_id.
        target: Target node_id.
        bandwidth_capacity: B_l — total link bandwidth in Mbps.
        propagation_delay: D_l — one-way propagation delay in ms.
        link_type: Intra- or inter-domain classification.
    """

    link_id: str
    source: str
    target: str
    bandwidth_capacity: float
    propagation_delay: float
    link_type: LinkType


@dataclass
class VNF:
    """A single virtual network function in a service function chain.

    Attributes:
        vnf_id: Unique identifier within the slice request, e.g. "s0_f2".
        vnf_type: Functional role, e.g. "Firewall", "vEPC", "CDN".
        cpu_demand: c̃_f — minimum vCPU allocation required.
        ram_demand: r̃_f — minimum RAM allocation required in GB.
        permitted_nodes: D_f — subset of node_ids satisfying tier/domain rules
            for constraint C8.
        computational_intensity: Multiplier for computing delta_{f,n} =
            node_base_delay * computational_intensity (v4 Remark 1).
        vcr: Volume Change Ratio rho_f. rho < 1 for compression,
            rho ~ 1 for pass-through, rho > 1 for expansion.
            Used to compute per-edge bandwidth via v4 Eq. 3.
    """

    vnf_id: str
    vnf_type: str
    cpu_demand: float
    ram_demand: float
    permitted_nodes: list[str]
    computational_intensity: float = 1.0
    vcr: float = 1.0


@dataclass
class FlowEdge:
    """A directed flow between two consecutive VNFs in a chain.

    Attributes:
        source_vnf: vnf_id of the upstream VNF.
        target_vnf: vnf_id of the downstream VNF.
        bandwidth_demand: beta_{k,k+1} — minimum bandwidth required in Mbps.
    """

    source_vnf: str
    target_vnf: str
    bandwidth_demand: float


@dataclass
class QoSRequirements:
    """QoS vector q_s = (D_max_s, beta_min_s)."""

    max_e2e_delay: float
    min_throughput: float


@dataclass
class SliceRequest:
    """A network slice embedding request.

    Attributes:
        request_id: Unique identifier, e.g. "req_00042".
        slice_type: Service category (eMBB, URLLC, etc.).
        vnfs: F_s — ordered list of VNFs forming the service function chain.
        flow_edges: E_s — directed flows between consecutive VNFs.
        qos: q_s — end-to-end QoS requirements.
        arrival_time: t_arrive — simulation time of arrival (0.0 for static).
        lifetime: Duration drawn from Exp(1/mu); 0.0 for static batch mode.
    """

    request_id: str
    slice_type: SliceType
    vnfs: list[VNF]
    flow_edges: list[FlowEdge]
    qos: QoSRequirements
    arrival_time: float
    lifetime: float


@dataclass
class PlacementPlan:
    """A single candidate placement plan pi_i output by the LLM Generator.

    Attributes:
        plan_id: Unique plan identifier, e.g. "req_00042_c3".
        vnf_placements: x_{f,n} — maps each vnf_id to the assigned node_id.
        cpu_allocations: a^c — maps each vnf_id to allocated vCPUs.
        ram_allocations: a^r — maps each vnf_id to allocated GB RAM.
        flow_routes: y_{l,f,f'} — maps each (src_vnf, dst_vnf) flow to an
            ordered list of link_ids forming the path.
        bw_allocations: a^b_{l,f,f'} — maps each flow to per-link bandwidth
            allocations in Mbps.
        is_structurally_valid: True if the plan passed C1/C4/C8 structural checks.
        source: Originating module — "llm" or "milp".
    """

    plan_id: str
    vnf_placements: dict[str, str]
    cpu_allocations: dict[str, float]
    ram_allocations: dict[str, float]
    flow_routes: dict[tuple[str, str], list[str]]
    bw_allocations: dict[tuple[str, str], dict[str, float]]
    is_structurally_valid: bool = True
    source: str = "llm"


@dataclass
class MILPSolution:
    """Complete solution returned by the MILP oracle.

    Attributes:
        objective_value: Optimal or best-found objective value.
        admitted: z_s — maps each request_id to admission decision.
        placements: Maps each admitted request_id to its PlacementPlan.
        solve_time: Wall-clock seconds consumed by the solver.
        gap: MIP optimality gap at termination.
        status: Solver status string — "Optimal", "Feasible", or "Infeasible".
    """

    objective_value: float
    admitted: dict[str, bool]
    placements: dict[str, PlacementPlan]
    solve_time: float
    gap: float
    status: str


@dataclass
class FeasibilityResult:
    """Output of the MILP Validator checking C2, C3, C5, C5b, C7."""

    is_feasible: bool
    violated_constraints: list[str]
    violation_details: dict[str, float]
