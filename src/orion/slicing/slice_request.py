"""SliceRequest serialization helpers.

The SliceRequest dataclass is defined in orion.types and has no JSON
dependency. These standalone functions handle dict/JSON round-trips needed
by the DatasetBuilder and the MILP oracle.
"""

from __future__ import annotations

from typing import Any

from orion.types import (
    VNF,
    FlowEdge,
    QoSRequirements,
    SliceRequest,
    SliceType,
)


def slice_request_to_dict(req: SliceRequest) -> dict[str, Any]:
    """Serialize a SliceRequest to a JSON-compatible dictionary.

    Args:
        req: The slice request to serialize.

    Returns:
        A dict with all fields JSON-serializable (no dataclass instances).
    """
    return {
        "request_id": req.request_id,
        "slice_type": req.slice_type.value,
        "vnfs": [
            {
                "vnf_id": v.vnf_id,
                "vnf_type": v.vnf_type,
                "cpu_demand": v.cpu_demand,
                "ram_demand": v.ram_demand,
                "permitted_nodes": v.permitted_nodes,
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
        "arrival_time": req.arrival_time,
        "lifetime": req.lifetime,
    }


def slice_request_from_dict(data: dict[str, Any]) -> SliceRequest:
    """Deserialize a SliceRequest from a dictionary.

    Args:
        data: Dictionary produced by slice_request_to_dict().

    Returns:
        Reconstructed SliceRequest.
    """
    return SliceRequest(
        request_id=data["request_id"],
        slice_type=SliceType(data["slice_type"]),
        vnfs=[
            VNF(
                vnf_id=v["vnf_id"],
                vnf_type=v["vnf_type"],
                cpu_demand=v["cpu_demand"],
                ram_demand=v["ram_demand"],
                permitted_nodes=v["permitted_nodes"],
            )
            for v in data["vnfs"]
        ],
        flow_edges=[
            FlowEdge(
                source_vnf=e["source_vnf"],
                target_vnf=e["target_vnf"],
                bandwidth_demand=e["bandwidth_demand"],
            )
            for e in data["flow_edges"]
        ],
        qos=QoSRequirements(
            max_e2e_delay=data["qos"]["max_e2e_delay"],
            min_throughput=data["qos"]["min_throughput"],
        ),
        arrival_time=data["arrival_time"],
        lifetime=data["lifetime"],
    )
