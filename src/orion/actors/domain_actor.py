"""Domain actor: orchestrates observation, policy, routing, and response.

Implements the interleaved place-then-route paradigm for a single domain:
  For each VNF k in SFC order within the fragment:
    1. Build observation (full GNN re-encode)
    2. Compute action mask
    3. Run policy → sample node placement
    4. Update node residuals
    5. Route flow from VNF k-1 to VNF k (if intra-domain), update edge BW
    6. If routing fails → abort, return infeasible

This interleaving ensures the GNN sees routing-induced BW changes before
placing the next VNF, eliminating the "placed fine but unroutable" failure
class (review point 4).

Each domain actor maintains its own parameters (architecture sharing, no
weight sharing per HARL, Zhong et al. JMLR 2024).

Centralized critic V_phi(s_t) is NOT part of the domain actor — it lives
in Phase 5 with the CTDE training loop. The actor outputs log_probs and
entropy only.
"""

from __future__ import annotations

import math

import torch

from orion.actors.action_mask import compute_action_mask
from orion.actors.domain_observation import build_domain_observation
from orion.actors.policy import DomainPolicy, VNF_CONTEXT_DIM  # noqa: F401 (VNF_CONTEXT_DIM re-exported)
from orion.actors.routing import (
    RoutingSelector,
    allocate_route_bw,
    deallocate_route_bw,
    route_flow,
)
from orion.config import MDO_DELAY_REF
from orion.actors.types import ActorStepRecord, DomainResponse, PlanFragment, VNFAssignment
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import InfrastructureTier, TIER_INDEX, TIER_ORDER

# Canonical ordering lives in orion.types (one definition, see TIER_ORDER).
_TIER_ORDER = list(TIER_ORDER)
_TIER_TO_IDX = {t: i for i, t in enumerate(_TIER_ORDER)}


class DomainActor:
    """Orchestrates placement and routing for one domain.

    Args:
        domain_id: The domain this actor manages.
        policy: The DomainPolicy network (owns its own weights).
        routing_mode: Path selection strategy ("min_cost", "min_delay", "min_hops").
        k_paths: Number of shortest paths to consider in Yen's algorithm.
        cost_alpha: Weight for compute resources in cost calculation.
        cost_gamma_intra: Weight for intra-domain bandwidth in cost calculation.
    """

    def __init__(
        self,
        domain_id: int,
        policy: DomainPolicy | None = None,
        routing_mode: str = "min_cost",
        k_paths: int = 3,
        cost_alpha: float = 1.0,
        cost_gamma_intra: float = 0.1,
    ) -> None:
        self.domain_id = domain_id
        self.policy = policy or DomainPolicy()
        self.routing_selector = RoutingSelector(routing_mode)
        self.k_paths = k_paths
        self.cost_alpha = cost_alpha
        self.cost_gamma_intra = cost_gamma_intra

    # Rollout/eval mode switch (2026-08-20). The coordinator calls
    # `act(substrate, fragment)` with no decode argument, so which decode a
    # learned actor uses has to live on the actor. Training rollouts need
    # sampling (PPO is on-policy); every eval must be argmax, matching how the
    # frozen greedy actor is deterministic everywhere. `train_approach` flips
    # this around its eval calls.
    stochastic: bool = True

    def act(
        self,
        substrate: SubstrateNetwork,
        fragment: PlanFragment,
        deterministic: bool | None = None,
    ) -> DomainResponse:
        """Execute interleaved placement and routing for a plan fragment.

        Args:
            substrate: Current substrate state (will be modified in-place
                for temporary BW allocations during routing; rolled back on
                failure).
            fragment: The VNFs and flows assigned to this domain.
            deterministic: If True, use argmax instead of sampling.

        Returns:
            DomainResponse with feasibility, placements, routes, and RL signals.
        """
        if fragment.is_empty:
            return DomainResponse.empty(self.domain_id)

        node_ids = sorted(substrate.nodes_in_domain(self.domain_id))
        if not node_ids:
            return DomainResponse(domain_id=self.domain_id, feasible=False)

        # Normalization constants for VNF context
        g = substrate.graph
        max_cpu = max(g.nodes[n]["cpu_capacity"] for n in node_ids)
        max_ram = max(g.nodes[n]["ram_capacity"] for n in node_ids)
        max_bw = 1.0
        for u, v, d in g.edges(data=True):
            if u in set(node_ids) and v in set(node_ids):
                max_bw = max(max_bw, d["bandwidth_capacity"])

        placements: dict[str, str] = {}
        routes: dict[tuple[str, str], list[str]] = {}
        bw_allocated: dict[tuple[str, str], float] = {}
        actions: list[int] = []
        log_probs: list[torch.Tensor] = []
        total_entropy = 0.0
        total_proc_delay = 0.0
        total_route_delay = 0.0
        total_resource_cost = 0.0
        total_bw_cost = 0.0

        # Snapshot original residuals for rollback on failure
        node_snapshots: dict[str, tuple[float, float]] = {}  # node_id -> (cpu, ram)

        # Track resource overrides for autoregressive mask updates
        resource_overrides: dict[str, tuple[float, float]] = {}
        placed_node_ids: set[str] = set()
        allocated_bw: list[tuple[list[str], float]] = []
        step_records: list[ActorStepRecord] = []

        # Build flow lookup: target_vnf -> FlowEdge
        intra_flow_by_target: dict[str, tuple[str, float]] = {}
        for fe in fragment.intra_flows:
            intra_flow_by_target[fe.target_vnf] = (fe.source_vnf, fe.bandwidth_demand)

        if deterministic is None:
            deterministic = not self.stochastic

        vnf_list = fragment.vnf_assignments
        delay_remaining = fragment.delay_budget_ms

        for step_idx, vnf in enumerate(vnf_list):
            # 1. Build observation with current state
            obs_data, obs_node_ids = build_domain_observation(
                substrate, self.domain_id,
                target_domain_ids=fragment.target_domain_ids,
                placed_node_ids=placed_node_ids,
            )

            # 2. Compute action mask
            mask = compute_action_mask(
                substrate, obs_node_ids, vnf,
                resource_overrides=resource_overrides,
            )

            # 3. Build VNF context vector
            # `delay_remaining` is threaded in so the policy can see the budget
            # it is spending, not just the price of each node (§9).
            vnf_ctx = self._build_vnf_context(
                vnf, max_cpu, max_ram, max_bw,
                delay_remaining=delay_remaining,
                delay_budget=fragment.delay_budget_ms,
            )

            # 4. Run policy (NULL slot is always available even if mask is all-False)
            action_idx, log_prob, entropy = self.policy(
                obs_data, vnf_ctx, mask, deterministic=deterministic,
            )

            # Record step inputs for PPO re-evaluation (CTDE).
            # Detach graph_data and tensors — these are replay inputs, not
            # part of the gradient graph. The PPO update re-evaluates through
            # the current policy to get new_log_prob.
            step_records.append(ActorStepRecord(
                graph_data=obs_data.clone() if hasattr(obs_data, 'clone') else obs_data,
                vnf_context=vnf_ctx.detach().clone(),
                action_mask=mask.detach().clone(),
                action_idx=action_idx,
                log_prob=float(log_prob.detach().item()),
                entropy=entropy,
            ))

            # NULL/refuse action: actor declines placement for this VNF
            if action_idx == DomainPolicy.NULL_ACTION:
                log_probs.append(log_prob)
                total_entropy += entropy
                self._rollback(substrate, allocated_bw, node_snapshots)
                return DomainResponse(
                    domain_id=self.domain_id,
                    feasible=False,
                    actions=actions,
                    log_probs=torch.stack(log_probs) if log_probs else torch.tensor([]),
                    entropy=total_entropy / (step_idx + 1),
                    step_records=step_records,
                )

            chosen_node = obs_node_ids[action_idx]
            placements[vnf.vnf_id] = chosen_node
            actions.append(action_idx)
            log_probs.append(log_prob)
            total_entropy += entropy

            # 5. Update node residuals (autoregressive)
            # Snapshot original residual before first modification
            if chosen_node not in node_snapshots:
                node_snapshots[chosen_node] = (
                    substrate.get_residual_cpu(chosen_node),
                    substrate.get_residual_ram(chosen_node),
                )

            if chosen_node in resource_overrides:
                old_cpu, old_ram = resource_overrides[chosen_node]
            else:
                old_cpu = substrate.get_residual_cpu(chosen_node)
                old_ram = substrate.get_residual_ram(chosen_node)

            new_cpu = old_cpu - vnf.cpu_demand
            new_ram = old_ram - vnf.ram_demand
            resource_overrides[chosen_node] = (new_cpu, new_ram)

            # Update substrate in-place for observation accuracy
            g.nodes[chosen_node]["cpu_residual"] = new_cpu
            g.nodes[chosen_node]["ram_residual"] = new_ram
            placed_node_ids.add(chosen_node)

            # Processing delay
            proc_delay = (
                g.nodes[chosen_node]["processing_delay"] * vnf.computational_intensity
            )
            total_proc_delay += proc_delay
            # Same defect as GreedyDomainActor had: the budget tracker was
            # decremented only by routing propagation delay and never by the
            # processing delay it was itself accumulating, so `delay_remaining`
            # overstated what was left for every VNF after the first.
            delay_remaining -= proc_delay

            # Resource cost
            total_resource_cost += self.cost_alpha * (vnf.cpu_demand + vnf.ram_demand)

            # 6. Route flow from previous VNF (if intra-domain)
            if vnf.vnf_id in intra_flow_by_target:
                src_vnf_id, bw_demand = intra_flow_by_target[vnf.vnf_id]
                src_node = placements[src_vnf_id]
                dst_node = chosen_node

                result = route_flow(
                    substrate, src_node, dst_node,
                    bw_demand=bw_demand,
                    delay_budget=delay_remaining,
                    domain_node_ids=obs_node_ids,
                    k=self.k_paths,
                    selector=self.routing_selector,
                )

                if not result.feasible:
                    self._rollback(substrate, allocated_bw, node_snapshots)
                    return DomainResponse(
                        domain_id=self.domain_id, feasible=False,
                        step_records=step_records,
                    )

                flow_key = (src_vnf_id, vnf.vnf_id)
                routes[flow_key] = result.path_links
                bw_allocated[flow_key] = bw_demand
                total_route_delay += result.propagation_delay
                delay_remaining -= result.propagation_delay
                total_bw_cost += self.cost_gamma_intra * bw_demand * len(result.path_links)

                # Update edge BW residuals for subsequent VNF placements
                allocate_route_bw(substrate, result.path_links, bw_demand)
                allocated_bw.append((result.path_links, bw_demand))

        # Aggregate log probs
        if log_probs:
            stacked = torch.stack(log_probs)
        else:
            stacked = torch.tensor([])

        avg_entropy = total_entropy / len(vnf_list) if vnf_list else 0.0

        return DomainResponse(
            domain_id=self.domain_id,
            feasible=True,
            placements=placements,
            routes=routes,
            bw_allocated=bw_allocated,
            intra_delay=total_proc_delay + total_route_delay,
            resource_cost=total_resource_cost + total_bw_cost,
            actions=actions,
            log_probs=stacked,
            entropy=avg_entropy,
            step_records=step_records,
        )

    def _build_vnf_context(
        self,
        vnf: VNFAssignment,
        max_cpu: float,
        max_ram: float,
        max_bw: float,
        delay_remaining: float | None = None,
        delay_budget: float | None = None,
    ) -> torch.Tensor:
        """Build the VNF context vector [VNF_CONTEXT_DIM=12].

        Slots 9-11 are the delay budget (2026-08-20). They default to zeros when
        no budget is supplied, which keeps the pre-2026-08-20 call signature
        working, but a real episode always passes one.
        """
        ctx = torch.zeros(VNF_CONTEXT_DIM)
        ctx[0] = vnf.cpu_demand / max_cpu if max_cpu > 0 else 0.0
        ctx[1] = vnf.ram_demand / max_ram if max_ram > 0 else 0.0
        tier_idx = _TIER_TO_IDX.get(vnf.required_tier, 0)
        ctx[2 + tier_idx] = 1.0
        ctx[6] = vnf.vcr
        ctx[7] = vnf.bandwidth_in / max_bw if max_bw > 0 else 0.0
        ctx[8] = vnf.position_in_sfc / vnf.sfc_length if vnf.sfc_length > 0 else 0.0
        if delay_remaining is not None:
            rem = max(0.0, float(delay_remaining))
            ctx[9] = min(1.0, rem / MDO_DELAY_REF)
            ctx[10] = math.log1p(rem) / math.log1p(MDO_DELAY_REF)
            ctx[11] = (rem / delay_budget) if delay_budget else 0.0
        return ctx

    def _rollback(
        self,
        substrate: SubstrateNetwork,
        allocated_bw: list[tuple[list[str], float]],
        node_snapshots: dict[str, tuple[float, float]],
    ) -> None:
        """Roll back all temporary allocations on failure.

        Restores node CPU/RAM from snapshots and link BW via deallocation.
        """
        # Rollback BW allocations (reverse order)
        for path_links, bw in reversed(allocated_bw):
            deallocate_route_bw(substrate, path_links, bw)

        # Rollback node resources from pre-fragment snapshots
        g = substrate.graph
        for node_id, (orig_cpu, orig_ram) in node_snapshots.items():
            g.nodes[node_id]["cpu_residual"] = orig_cpu
            g.nodes[node_id]["ram_residual"] = orig_ram
