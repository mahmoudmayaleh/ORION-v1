"""BC pretraining for domain actors from greedy FFD demonstrations.

Each domain actor is warm-started by minimising

    L_BC^m = - E[ log pi^m_theta(a* | o^m) ]  -  lambda_ent * H[pi^m_theta(. | o^m)]

over per-VNF (observation, action) pairs extracted from greedy traces.

The projection replays greedy placements in the ACTOR's order (SFC within
domain), deducting resources at each step so the observation matches what
the actor would see at runtime if it followed greedy's decisions. This is
the critical alignment: if the observation doesn't match runtime, BC loss
goes down but cold-start admission stays near zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from orion.actors.action_mask import compute_action_mask
from orion.actors.domain_actor import DomainActor
from orion.actors.domain_observation import build_domain_observation
from orion.actors.policy import DomainPolicy, VNF_CONTEXT_DIM
from orion.actors.types import VNFAssignment
from orion.config import TopologyConfig
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.training.bc_dataset import (
    BCDatasetSpec,
    BCSample,
    get_or_generate_dataset,
)
from orion.training.config import MAPPOConfig
from orion.types import (
    FlowEdge,
    InfrastructureTier,
    QoSRequirements,
    SliceRequest,
    SliceType,
    TIER_ORDER,
    VNF,
)

logger = logging.getLogger(__name__)


@dataclass
class BCEpochResult:
    epoch: int
    imitation_loss: float
    entropy_bonus: float
    num_samples: int


@dataclass
class BCStep:
    """One (observation, target_action) pair for BC training."""
    graph_data: Any        # PyG Data
    vnf_context: torch.Tensor
    action_mask: torch.Tensor
    target_node_idx: int   # index into the domain's node list
    domain_id: int


# ── Projection: BCSample -> per-VNF BCSteps ──────────────────────────────


def _reconstruct_slice(sample: BCSample, substrate: SubstrateNetwork) -> SliceRequest:
    """Rebuild a SliceRequest from a BCSample's stored dict."""
    sd = sample.slice_dict
    vnfs = [
        VNF(
            vnf_id=v["vnf_id"],
            vnf_type=v["vnf_type"],
            cpu_demand=v["cpu_demand"],
            ram_demand=v["ram_demand"],
            permitted_nodes=v["permitted_nodes"],
            computational_intensity=v.get("computational_intensity", 1.0),
            vcr=v.get("vcr", 1.0),
        )
        for v in sd["vnfs"]
    ]
    flow_edges = [
        FlowEdge(
            source_vnf=f["source_vnf"],
            target_vnf=f["target_vnf"],
            bandwidth_demand=f["bandwidth_demand"],
        )
        for f in sd["flow_edges"]
    ]
    return SliceRequest(
        request_id=sd["request_id"],
        slice_type=SliceType(sd["slice_type"]),
        vnfs=vnfs,
        flow_edges=flow_edges,
        qos=QoSRequirements(
            max_e2e_delay=sd["qos"]["max_e2e_delay"],
            min_throughput=sd["qos"]["min_throughput"],
        ),
        arrival_time=sd.get("arrival_time", 0.0),
        lifetime=sd.get("lifetime", 0.0),
    )


def _node_id_to_domain(node_id: str) -> int:
    if not node_id.startswith("d"):
        return -1
    try:
        return int(node_id.split("n")[0][1:])
    except (ValueError, IndexError):
        return -1


# Canonical ordering lives in orion.types (one definition, see TIER_ORDER).
_TIER_ORDER = list(TIER_ORDER)
_TIER_TO_IDX = {t: i for i, t in enumerate(_TIER_ORDER)}


def project_sample(
    sample: BCSample,
    substrate: SubstrateNetwork,
) -> list[BCStep]:
    """Project one BCSample into per-VNF BCSteps.

    Replays greedy placements in SFC order per domain, deducting resources
    at each step so the observation matches what the actor would see at
    runtime. Uses the SAME build_domain_observation, compute_action_mask,
    and _build_vnf_context functions the live actor uses.
    """
    slice_req = _reconstruct_slice(sample, substrate)
    g = substrate.graph

    # Group VNFs by domain (from greedy's placements), preserving SFC order
    vnf_by_id = {v.vnf_id: v for v in slice_req.vnfs}
    domain_vnfs: dict[int, list[VNF]] = {}
    for vnf in slice_req.vnfs:
        node_id = sample.placements.get(vnf.vnf_id)
        if node_id is None:
            continue
        domain = _node_id_to_domain(node_id)
        if domain < 0:
            continue
        domain_vnfs.setdefault(domain, []).append(vnf)

    steps: list[BCStep] = []
    placed_node_ids: set[str] = set()

    for domain_id, vnfs in domain_vnfs.items():
        node_ids = sorted(substrate.nodes_in_domain(domain_id))
        if not node_ids:
            continue
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Normalization constants (same as DomainActor.act)
        max_cpu = max(g.nodes[n]["cpu_capacity"] for n in node_ids)
        max_ram = max(g.nodes[n]["ram_capacity"] for n in node_ids)
        max_bw = 1.0
        node_set = set(node_ids)
        for u, v, d in g.edges(data=True):
            if u in node_set and v in node_set:
                max_bw = max(max_bw, d["bandwidth_capacity"])

        resource_overrides: dict[str, tuple[float, float]] = {}

        for vnf in vnfs:
            target_node = sample.placements[vnf.vnf_id]
            if target_node not in node_id_to_idx:
                continue
            target_idx = node_id_to_idx[target_node]

            # Build observation using the LIVE functions
            obs_data, obs_node_ids = build_domain_observation(
                substrate, domain_id,
                target_domain_ids=set(),
                placed_node_ids=placed_node_ids,
            )

            # Build VNF assignment for action mask
            tier = InfrastructureTier(g.nodes[target_node]["tier"])
            vnf_assignment = VNFAssignment(
                vnf_id=vnf.vnf_id,
                vnf_type=vnf.vnf_type,
                cpu_demand=vnf.cpu_demand,
                ram_demand=vnf.ram_demand,
                required_tier=tier,
                computational_intensity=vnf.computational_intensity,
                vcr=vnf.vcr,
                bandwidth_in=0.0,
                permitted_nodes=vnf.permitted_nodes,
                position_in_sfc=0,
                sfc_length=len(slice_req.vnfs),
            )

            # Action mask using the LIVE function
            mask = compute_action_mask(
                substrate, obs_node_ids, vnf_assignment,
                resource_overrides=resource_overrides,
            )

            # VNF context using the SAME logic as DomainActor._build_vnf_context
            vnf_ctx = torch.zeros(VNF_CONTEXT_DIM)
            vnf_ctx[0] = vnf.cpu_demand / max_cpu if max_cpu > 0 else 0.0
            vnf_ctx[1] = vnf.ram_demand / max_ram if max_ram > 0 else 0.0
            tier_idx = _TIER_TO_IDX.get(tier, 0)
            vnf_ctx[2 + tier_idx] = 1.0
            vnf_ctx[6] = vnf.vcr
            vnf_ctx[7] = 0.0  # bandwidth_in not critical for placement
            vnf_ctx[8] = 0.0  # position_in_sfc normalized

            steps.append(BCStep(
                graph_data=obs_data,
                vnf_context=vnf_ctx,
                action_mask=mask,
                target_node_idx=target_idx,
                domain_id=domain_id,
            ))

            # Deduct resources in-place (matching actor's autoregressive update)
            # so build_domain_observation sees correct residuals for next VNF.
            if target_node in resource_overrides:
                old_cpu, old_ram = resource_overrides[target_node]
            else:
                old_cpu = substrate.get_residual_cpu(target_node)
                old_ram = substrate.get_residual_ram(target_node)
            new_cpu = old_cpu - vnf.cpu_demand
            new_ram = old_ram - vnf.ram_demand
            resource_overrides[target_node] = (new_cpu, new_ram)
            g.nodes[target_node]["cpu_residual"] = new_cpu
            g.nodes[target_node]["ram_residual"] = new_ram
            placed_node_ids.add(target_node)

    return steps


# ── BC training ──────────────────────────────────────────────────────────


def bc_pretrain(
    domain_actors: dict[int, DomainActor],
    spec: BCDatasetSpec,
    dataset_path: Path,
    config: MAPPOConfig,
) -> tuple[dict[int, list[BCEpochResult]], dict[str, str]]:
    """Run BC pretraining on greedy demonstrations.

    Projects each BCSample into per-VNF (observation, action) pairs using
    the live observation builder, then trains each domain actor's policy
    via cross-entropy against greedy's placements.
    """
    samples, meta = get_or_generate_dataset(spec, dataset_path)
    logger.info(
        "BC dataset: %d samples, hash=%s", len(samples), meta.get("dataset_hash"),
    )

    if not samples:
        logger.warning("BC dataset empty — skipping pretraining.")
        return {d: [] for d in domain_actors}, meta

    # Replay the full episode to reconstruct the depleted substrate state
    # at each sample's acceptance event. The dataset was generated from a
    # multi-arrival episode with departures, so we must replay arrivals and
    # departures in order to match the residual state at each placement.
    from orion.sim.arrival_process import ArrivalProcess, EventType
    from orion.baselines.greedy_ffd import greedy_place_on_substrate

    rng = np.random.default_rng(spec.seed)
    substrate = generate_multi_domain_topology(spec.topology_config, rng)

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

    # Index samples by request_id for lookup during replay
    sample_by_id = {s.request_id: s for s in samples}

    logger.info("Projecting %d samples via episode replay...", len(samples))
    all_steps: list[BCStep] = []

    while ap.has_next():
        event = ap.next_event()

        if event.event_type == EventType.DEPARTURE:
            plan_entry = substrate._active_slices.get(event.request_id)
            if plan_entry is not None:
                substrate.deallocate(plan_entry[0], plan_entry[1])
            continue

        slice_req = event.slice_request
        assert slice_req is not None

        if slice_req.request_id not in sample_by_id:
            # This arrival was rejected by greedy — replay greedy to advance
            # substrate state correctly (greedy doesn't allocate on rejection)
            greedy_place_on_substrate(substrate, slice_req)
            continue

        sample = sample_by_id[slice_req.request_id]

        # Snapshot residuals before projection (project_sample mutates
        # substrate in-place for autoregressive observation accuracy)
        node_snapshot = {
            n: (d["cpu_residual"], d["ram_residual"])
            for n, d in substrate.graph.nodes(data=True)
        }

        # Project at the CURRENT depleted substrate state
        steps = project_sample(sample, substrate)
        all_steps.extend(steps)

        # Restore residuals so greedy_place_on_substrate sees the pre-projection state
        for n, (cpu, ram) in node_snapshot.items():
            substrate.graph.nodes[n]["cpu_residual"] = cpu
            substrate.graph.nodes[n]["ram_residual"] = ram

        # Advance substrate state: apply greedy's allocation
        result = greedy_place_on_substrate(substrate, slice_req)

    # Partition steps by domain
    per_domain: dict[int, list[BCStep]] = {d: [] for d in domain_actors}
    for step in all_steps:
        if step.domain_id in per_domain:
            per_domain[step.domain_id].append(step)

    logger.info(
        "BC steps per domain: %s",
        {d: len(ss) for d, ss in per_domain.items()},
    )

    per_domain_logs: dict[int, list[BCEpochResult]] = {d: [] for d in domain_actors}

    for epoch in range(config.bc_epochs):
        for domain_id, actor in domain_actors.items():
            steps = per_domain.get(domain_id, [])
            if not steps:
                continue
            result = _run_one_epoch(actor, steps, config.bc_lr, config.bc_entropy_coef, epoch)
            per_domain_logs[domain_id].append(result)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            for d, logs in per_domain_logs.items():
                if logs:
                    last = logs[-1]
                    logger.info(
                        "  BC epoch %d domain %d: loss=%.4f ent=%.4f n=%d",
                        last.epoch, d, last.imitation_loss, last.entropy_bonus, last.num_samples,
                    )

    return per_domain_logs, {
        "dataset_hash": meta.get("dataset_hash", ""),
        "config_hash": meta.get("config_hash", ""),
    }


def _run_one_epoch(
    actor: DomainActor,
    steps: list[BCStep],
    lr: float,
    entropy_coef: float,
    epoch: int,
) -> BCEpochResult:
    """One BC epoch: cross-entropy loss against greedy targets."""
    policy = actor.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    total_loss = 0.0
    total_entropy = 0.0
    n = 0

    # Shuffle steps
    indices = list(range(len(steps)))
    np.random.shuffle(indices)

    optimizer.zero_grad()
    batch_loss = torch.zeros(1)

    for i, idx in enumerate(indices):
        step = steps[idx]

        # Forward: get logits over [N+1] (nodes + NULL)
        logits = policy._encode_and_score(
            step.graph_data, step.vnf_context, step.action_mask,
        )

        # Target is the node index (not NULL)
        target = torch.tensor(step.target_node_idx, dtype=torch.long)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

        # Entropy bonus
        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        step_loss = ce_loss - entropy_coef * entropy
        batch_loss = batch_loss + step_loss

        total_loss += ce_loss.item()
        total_entropy += entropy.item()
        n += 1

        # Mini-batch update every 32 steps
        if (i + 1) % 32 == 0 or i == len(indices) - 1:
            (batch_loss / min(32, n)).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            batch_loss = torch.zeros(1)

    return BCEpochResult(
        epoch=epoch,
        imitation_loss=total_loss / max(n, 1),
        entropy_bonus=entropy_coef * total_entropy / max(n, 1),
        num_samples=n,
    )
