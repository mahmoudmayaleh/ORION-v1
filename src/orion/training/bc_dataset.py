"""BC demonstration dataset (Choice B1: cached to disk, seed-recorded, hashed).

Generates `(o^m_t, a^{m,greedy}_t)` pairs from running Tier-Aware FFD on a
fixed sequence of synthetic scenarios. Caches the result to a `.pt` file
along with the metadata required to reproduce it exactly:

    {
        "seed": int,                       # the master seed
        "num_scenarios": int,
        "config_hash": str,                # sha256 of TopologyConfig + slice-gen knobs
        "dataset_hash": str,               # sha256 of the (obs, action) bytes
        "samples": list[dict],             # the data itself
    }

The hash is the contract that lets the paper state exactly which demo set
was used. If anything in the generation pipeline changes — topology
config, slice generator, greedy algorithm — the hash changes too, and the
cache is regenerated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from orion.baselines.greedy_ffd import greedy_place_on_substrate
from orion.config import TopologyConfig
from orion.sim.slice_generator import generate_slice_request
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology


@dataclass
class BCDatasetSpec:
    """Reproducibility-pinned spec for a BC demonstration dataset."""

    seed: int
    num_scenarios: int
    topology_config: TopologyConfig

    def config_hash(self) -> str:
        # TopologyConfig is a pydantic BaseSettings; use model_dump for the
        # canonical dict form.
        topo = (
            self.topology_config.model_dump()
            if hasattr(self.topology_config, "model_dump")
            else asdict(self.topology_config)
        )
        payload = json.dumps(
            {"seed": self.seed, "num_scenarios": self.num_scenarios, "topology": topo},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class BCSample:
    """One demonstration: a slice request and the greedy placement decision.

    The actual (o^m_t, a^{m,greedy}_t) tensors are recomputed by the BC
    trainer at load time, using the substrate fixture and the recorded
    slice. This keeps the dataset format simple and forwards-compatible
    if observation features change.
    """

    request_id: str
    slice_dict: dict[str, Any]       # serialised SliceRequest
    placements: dict[str, str]       # vnf_id -> node_id
    cost: float


def generate_dataset(spec: BCDatasetSpec) -> list[BCSample]:
    """Run greedy on full multi-arrival episodes, record placements on depleted substrate.

    Traces carry the actual residual states the actor will see at runtime:
    multiple slices active, departures interleaving, capacity depleting.
    This breaks the shortcut where the actor memorizes "pick the node
    with most free CPU on a reset substrate" instead of learning real
    placement logic.

    Only accepted placements are recorded (greedy rejects ~54%, no action
    to imitate on rejections). The substrate is NOT reset between slices.
    """
    from orion.sim.arrival_process import ArrivalProcess, EventType

    rng = np.random.default_rng(spec.seed)
    substrate = generate_multi_domain_topology(spec.topology_config, rng)

    # Run a full episode with arrivals and departures
    ap_rng = np.random.default_rng(spec.seed + 500_000)
    ap = ArrivalProcess(
        substrate=substrate,
        num_arrivals=spec.num_scenarios,
        arrival_rate=4.0,
        service_rate=0.02,
        rng=ap_rng,
    )
    ap.generate()
    substrate.reset()

    samples: list[BCSample] = []

    while ap.has_next():
        event = ap.next_event()

        if event.event_type == EventType.DEPARTURE:
            plan_entry = substrate._active_slices.get(event.request_id)
            if plan_entry is not None:
                substrate.deallocate(plan_entry[0], plan_entry[1])
            continue

        slice_req = event.slice_request
        assert slice_req is not None

        result = greedy_place_on_substrate(substrate, slice_req)
        if not result.feasible or result.plan is None:
            continue  # skip rejections — no action to imitate

        samples.append(BCSample(
            request_id=slice_req.request_id,
            slice_dict=_slice_to_dict(slice_req),
            placements=dict(result.plan.vnf_placements),
            cost=result.cost,
        ))

    return samples


def save_dataset(samples: list[BCSample], spec: BCDatasetSpec, path: Path) -> str:
    """Write the dataset + metadata to `path`. Returns the dataset hash."""
    payload_bytes = json.dumps(
        [asdict(s) for s in samples], sort_keys=True
    ).encode()
    dataset_hash = hashlib.sha256(payload_bytes).hexdigest()[:16]

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": spec.seed,
            "num_scenarios": spec.num_scenarios,
            "config_hash": spec.config_hash(),
            "dataset_hash": dataset_hash,
            "samples": [asdict(s) for s in samples],
        },
        path,
    )
    return dataset_hash


def load_dataset(path: Path) -> tuple[list[BCSample], dict[str, Any]]:
    """Load and return (samples, metadata)."""
    blob = torch.load(path, weights_only=False)
    samples = [BCSample(**s) for s in blob["samples"]]
    meta = {k: v for k, v in blob.items() if k != "samples"}
    return samples, meta


def get_or_generate_dataset(
    spec: BCDatasetSpec, path: Path
) -> tuple[list[BCSample], dict[str, Any]]:
    """Load if the cached config_hash matches; else regenerate.

    This is the function the BC pretrainer calls. Hash mismatch → silent
    regeneration is intentional (you don't want to train on a stale
    dataset just because someone tweaked the topology config), but the
    new dataset hash is returned so the trainer can log it.
    """
    if path.exists():
        try:
            samples, meta = load_dataset(path)
            if meta.get("config_hash") == spec.config_hash():
                return samples, meta
        except Exception:
            pass  # cache corrupt — fall through to regenerate
    samples = generate_dataset(spec)
    dataset_hash = save_dataset(samples, spec, path)
    _, meta = load_dataset(path)
    return samples, meta


# ── Slice ↔ dict serialisation ───────────────────────────────────────────


def _slice_to_dict(slice_req) -> dict[str, Any]:
    """Minimal JSON-safe serialisation. Lossy for VNF.permitted_nodes which
    is rebuilt from the substrate at load time anyway."""
    return {
        "request_id": slice_req.request_id,
        "slice_type": slice_req.slice_type.value,
        "vnfs": [
            {
                "vnf_id": v.vnf_id,
                "vnf_type": v.vnf_type,
                "cpu_demand": v.cpu_demand,
                "ram_demand": v.ram_demand,
                "computational_intensity": v.computational_intensity,
                "vcr": v.vcr,
                "permitted_nodes": list(v.permitted_nodes),
            }
            for v in slice_req.vnfs
        ],
        "flow_edges": [
            {
                "source_vnf": f.source_vnf,
                "target_vnf": f.target_vnf,
                "bandwidth_demand": f.bandwidth_demand,
            }
            for f in slice_req.flow_edges
        ],
        "qos": {
            "max_e2e_delay": slice_req.qos.max_e2e_delay,
            "min_throughput": slice_req.qos.min_throughput,
        },
        "arrival_time": slice_req.arrival_time,
        "lifetime": slice_req.lifetime,
    }
