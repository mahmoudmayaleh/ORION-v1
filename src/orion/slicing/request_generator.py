"""Poisson arrival generator for 6G network slice requests.

Generates SliceRequest instances from SFC templates. Two modes:
- Static batch:   S requests with arrival_time = lifetime = 0.0 (for MILP)
- Dynamic stream: Poisson arrivals with exponential lifetimes (for RL)

VNF permitted_nodes are computed at generation time by intersecting the
substrate node set with the VNF type's permitted_tiers, enforcing C8.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from orion.config import SlicingConfig
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import (
    VNF,
    FlowEdge,
    QoSRequirements,
    SliceRequest,
    SliceType,
)

# Maps SliceType enum value -> template key in standard_sfc.yaml
_SLICE_TYPE_TO_TEMPLATE: dict[str, str] = {
    "eMBB": "eMBB",
    "URLLC": "URLLC",
    "mMTC": "mMTC",
    "V2X": "V2X",
    "XR": "XR",
}


class PoissonArrivalGenerator:
    """Generates SliceRequests following a Poisson arrival process.

    Attributes:
        config: SlicingConfig with demand and arrival rate parameters.
        substrate: Substrate graph used to resolve permitted_nodes (C8).
        templates: Parsed content of configs/slicing/standard_sfc.yaml.
        rng: Seeded random generator for all stochastic operations.
    """

    def __init__(
        self,
        config: SlicingConfig,
        substrate: SubstrateNetwork,
        templates: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """Initialize the generator.

        Args:
            config: Slicing configuration (demand ranges, arrival rate, etc.).
            substrate: Substrate network for permitted_nodes computation.
            templates: Dict loaded from configs/slicing/standard_sfc.yaml.
            rng: Seeded NumPy random generator.
        """
        self.config = config
        self.substrate = substrate
        self.templates = templates
        self.rng = rng
        self._request_counter = 0

    def generate_batch(self, num_requests: int) -> list[SliceRequest]:
        """Generate a static batch of slice requests with no timing.

        arrival_time and lifetime are both 0.0. Used as input to the MILP
        oracle for offline dataset generation.

        Args:
            num_requests: Number of SliceRequest instances to generate.

        Returns:
            List of SliceRequest instances.
        """
        return [self._sample_request(arrival_time=0.0, lifetime=0.0) for _ in range(num_requests)]

    def generate_episode(self, duration: float) -> list[SliceRequest]:
        """Generate a dynamic stream of slice requests over a time window.

        Inter-arrival times: t_i ~ Exp(1 / arrival_rate)
        Lifetimes:           l_i ~ Exp(1 / mean_lifetime)

        Args:
            duration: Simulation time window in the same unit as arrival_rate.

        Returns:
            Time-ordered list of SliceRequest instances within [0, duration].
        """
        requests: list[SliceRequest] = []
        current_time = 0.0

        while True:
            inter_arrival = self.rng.exponential(1.0 / self.config.arrival_rate)
            current_time += inter_arrival
            if current_time > duration:
                break
            lifetime = self.rng.exponential(self.config.mean_lifetime)
            requests.append(
                self._sample_request(arrival_time=current_time, lifetime=lifetime)
            )

        return requests

    # ── Internal sampling methods ──────────────────────────────────────────────

    def _sample_request(self, arrival_time: float, lifetime: float) -> SliceRequest:
        """Sample a complete SliceRequest with a unique ID.

        Args:
            arrival_time: Simulation arrival time (0.0 for static batch).
            lifetime: Slice duration (0.0 for static batch).

        Returns:
            A fully populated SliceRequest.
        """
        self._request_counter += 1
        request_id = f"req_{self._request_counter:05d}"
        slice_type = self._sample_slice_type()
        vnfs, flow_edges = self._sample_sfc(slice_type)
        qos = self._sample_qos(slice_type)
        return SliceRequest(
            request_id=request_id,
            slice_type=slice_type,
            vnfs=vnfs,
            flow_edges=flow_edges,
            qos=qos,
            arrival_time=arrival_time,
            lifetime=lifetime,
        )

    def _sample_slice_type(self) -> SliceType:
        """Sample a slice type uniformly from the 5 3GPP categories.

        Returns:
            A SliceType enum value.
        """
        types = list(SliceType)
        idx = int(self.rng.integers(0, len(types)))
        return types[idx]

    def _sample_sfc(self, slice_type: SliceType) -> tuple[list[VNF], list[FlowEdge]]:
        """Sample a service function chain for the given slice type.

        Algorithm:
          1. Look up the template for slice_type.
          2. Sample chain length L from chain_length_range.
          3. Select the first L VNF types from the template chain.
          4. For each VNF, sample cpu_demand ~ Uniform(vnf_cpu_range),
             ram_demand ~ Uniform(vnf_ram_range).
          5. Compute permitted_nodes for each VNF type (C8).
          6. For each consecutive pair (f_k, f_{k+1}), sample
             bandwidth_demand ~ Uniform(flow_bw_range).

        Args:
            slice_type: The slice service category.

        Returns:
            Tuple (vnfs, flow_edges) where vnfs is the ordered VNF chain
            and flow_edges connects consecutive VNFs.

        Raises:
            ValueError: If any VNF type has no permitted nodes on the substrate.
        """
        template_key = _SLICE_TYPE_TO_TEMPLATE[slice_type.value]
        template = self.templates["slice_templates"][template_key]

        lo, hi = template["chain_length_range"]
        chain_len = int(self.rng.integers(lo, hi + 1))
        vnf_types_pool: list[str] = template["chain"]

        # Truncate or wrap to chain_len — always take first chain_len entries
        selected_types = vnf_types_pool[:chain_len]

        vnfs: list[VNF] = []
        for k, vnf_type in enumerate(selected_types):
            cpu = float(self.rng.uniform(*self.config.vnf_cpu_range))
            ram = float(self.rng.uniform(*self.config.vnf_ram_range))
            permitted = self._compute_permitted_nodes(vnf_type)
            vnfs.append(
                VNF(
                    vnf_id=f"f{k}",
                    vnf_type=vnf_type,
                    cpu_demand=round(cpu, 2),
                    ram_demand=round(ram, 2),
                    permitted_nodes=permitted,
                )
            )

        flow_edges: list[FlowEdge] = []
        for k in range(len(vnfs) - 1):
            bw = float(self.rng.uniform(*self.config.flow_bw_range))
            flow_edges.append(
                FlowEdge(
                    source_vnf=f"f{k}",
                    target_vnf=f"f{k + 1}",
                    bandwidth_demand=round(bw, 1),
                )
            )

        return vnfs, flow_edges

    def _compute_permitted_nodes(self, vnf_type: str) -> list[str]:
        """Compute D_f: substrate nodes eligible for a VNF type (constraint C8).

        D_f = {n in N : tier(n) in permitted_tiers(vnf_type)}

        Args:
            vnf_type: VNF type name matching a key in templates["vnf_types"].

        Returns:
            Sorted list of node_ids satisfying the tier constraint.

        Raises:
            ValueError: If the intersection is empty — the substrate has no
                nodes of the required tier(s), making the VNF infeasible.
        """
        permitted_tiers: list[str] = self.templates["vnf_types"][vnf_type]["permitted_tiers"]
        tier_set = set(permitted_tiers)
        nodes = [
            node_id
            for node_id, d in self.substrate.graph.nodes(data=True)
            if d["tier"] in tier_set
        ]
        if not nodes:
            raise ValueError(
                f"VNF type '{vnf_type}' has no eligible nodes on the substrate. "
                f"Required tiers: {permitted_tiers}. "
                "Check that the substrate contains nodes of those tiers."
            )
        return sorted(nodes)

    def _sample_qos(self, slice_type: SliceType) -> QoSRequirements:
        """Sample QoS requirements from the slice type template bounds.

        Args:
            slice_type: The slice service category.

        Returns:
            QoSRequirements with delay and throughput sampled from template bounds.
        """
        template_key = _SLICE_TYPE_TO_TEMPLATE[slice_type.value]
        template = self.templates["slice_templates"][template_key]

        delay = float(self.rng.uniform(*template["delay_budget_ms"]))
        throughput = float(self.rng.uniform(*template["min_throughput_mbps"]))

        return QoSRequirements(
            max_e2e_delay=round(delay, 2),
            min_throughput=round(throughput, 2),
        )
