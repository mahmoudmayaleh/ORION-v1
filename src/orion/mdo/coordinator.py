

"""MDO Coordinator: partition retry loop, dispatch to domain actors, reward.

Implements the per-arrival control flow:
  1. Build MDO observation from substrate + plan + retry history
  2. Partition retry loop (j = 1..N_part):
     a. Sample partition from MDO policy (or deterministic mode)
     b. Split plan into PlanFragments per domain
     c. Dispatch to domain actors
     d. Collect DomainResponses, run pre-commit checks
     e. COMMIT or check rejection triggers
  3. Compute reward components (SMDP: terminal reward shared across trials)

The coordinator does NOT own training — PPO updates, the centralised critic,
and the training loop are Phase 5.  Phase 4 delivers this coordinator running
end-to-end with random or deterministic ("follow m̃") MDO policies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.distributions import Categorical

if TYPE_CHECKING:
    from orion.actors.domain_actor import DomainActor

from orion.actors.routing import (
    allocate_route_bw,
    deallocate_route_bw,
    route_cross_domain_flow,
)
from orion.actors.types import DomainResponse, PlanFragment, VNFAssignment
from orion.mdo.observation import (
    build_mdo_observation,
    build_tier_masks,
    observation_to_tensor,
)
from orion.mdo.policy import MDOPolicy
from orion.mdo.rejection import check_rejection_triggers
from orion.mdo.types import (
    MDOAction,
    MDOResult,
    PartitionAttempt,
    PlanSummary,
    RejectReason,
    RetryHistory,
    RewardComponents,
    ViolationInfo,
)
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import FlowEdge, QoSRequirements, SliceRequest

logger = logging.getLogger(__name__)


@dataclass
class MDOConfig:
    """Configuration for the MDO coordinator."""

    n_part: int = 3
    n_retry: int = 1
    max_inter_domain_hops: int = 3
    gamma_inter: float = 1.0
    # Reward weights (v6.2 Eq. 9)
    mu: float = 100.0
    alpha: float = 1.0
    lambda_viol: float = 10.0
    eta: float = 1.0
    xi: float = 0.5
    # Rejection trigger (iii) threshold — disabled by default until trained
    tau_v: float = -float("inf")
    stability_k: int = 2


class MDOCoordinator:
    """Orchestrates the partition retry loop for one slice arrival.

    Args:
        policy: The MDO policy network (or None for random/deterministic modes).
        domain_actors: Dict mapping domain_id -> DomainActor.
        config: MDO configuration.
    """

    def __init__(
        self,
        policy: MDOPolicy | None,
        domain_actors: dict[int, DomainActor],
        config: MDOConfig | None = None,
    ) -> None:
        self.policy = policy
        self.domain_actors = domain_actors
        self.config = config or MDOConfig()

    def resolve_arrival(
        self,
        substrate: SubstrateNetwork,
        slice_req: SliceRequest,
        plan: PlanSummary,
        inter_domain_delays: dict[tuple[int, int], float],
        mode: str = "sample",
        cost_greedy: float | None = None,
    ) -> MDOResult:
        """Run the full partition retry loop for one slice arrival.

        Args:
            substrate: Current substrate state.
            slice_req: The arriving slice request.
            plan: Flattened abstract plan from Agent B.
            inter_domain_delays: Inter-domain link propagation delays.
            mode: "sample" (policy sampling), "deterministic" (policy argmax),
                  "follow_prior" (always use Agent B's suggestion m̃),
                  "random" (uniform random over feasible domains).
            cost_greedy: Cost of greedy FFD baseline for LocalScore.
                         If None, quality shaping is skipped.

        Returns:
            MDOResult with final outcome, reward, and retry history.
        """
        history = RetryHistory()
        obs_struct = build_mdo_observation(substrate, plan)
        tier_mask = build_tier_masks(plan, obs_struct.domain_summaries)
        obs_tensor = observation_to_tensor(obs_struct)
        bw_demands = plan.bw_demands
        cfg = self.config

        # Map canonical indices (policy output) to actual domain IDs.
        # Domain summaries are sorted by (tier_type, domain_id), so
        # canonical index i corresponds to domain_summaries[i].domain_id.
        canonical_to_domain = [
            s.domain_id for s in obs_struct.domain_summaries
        ]

        committed_partition: list[int] | None = None
        committed_responses: dict[int, DomainResponse] = {}
        committed_cross_routes: dict[tuple[str, str], list[str]] = {}
        committed_cross_bw: dict[tuple[str, str], float] = {}
        committed_e2e = 0.0
        committed_cost = 0.0
        final_action = MDOAction.REJECT
        reject_reason: RejectReason | None = None
        all_log_probs: list[torch.Tensor] = []
        all_entropy = 0.0
        last_value = 0.0

        for j in range(cfg.n_part):
            # --- Sample partition (canonical indices) ---
            partition_canonical, log_probs, _logits, entropy, value = self._sample_partition(
                obs_tensor, tier_mask, plan, mode,
                canonical_to_domain=canonical_to_domain,
            )
            all_log_probs.append(log_probs)
            all_entropy += entropy
            last_value = value

            # Map canonical indices to actual domain IDs
            partition = [canonical_to_domain[c] for c in partition_canonical]

            # --- Snapshot substrate CPU/RAM before actor dispatch ---
            node_snapshot = self._snapshot_node_residuals(substrate, partition)

            # --- Split into PlanFragments and dispatch ---
            fragments, cross_domain_flows = self._build_fragments(
                plan, partition, slice_req,
            )
            responses = self._dispatch_to_actors(substrate, fragments)

            # --- Route cross-domain flows on full graph ---
            actor_infeasible = any(
                not r.feasible for r in responses.values()
            )
            cross_feasible = True
            cross_routes: dict[tuple[str, str], list[str]] = {}
            cross_bw: dict[tuple[str, str], float] = {}
            cross_delay = 0.0
            cross_hops = 0

            if not actor_infeasible and cross_domain_flows:
                intra_delay = sum(
                    r.intra_delay for r in responses.values()
                )
                remaining_delay = slice_req.qos.max_e2e_delay - intra_delay
                (
                    cross_feasible, cross_routes, cross_bw,
                    cross_delay, cross_hops,
                ) = self._route_cross_domain_flows(
                    substrate, cross_domain_flows, responses,
                    delay_budget=remaining_delay,
                )

            # --- Build violation info from actual routing ---
            intra_delay = sum(r.intra_delay for r in responses.values())
            e2e = intra_delay + cross_delay
            inter_bw = sum(cross_bw.values())
            # Intra-domain routes use subgraph(domain_set) — no inter-domain
            # edges. All inter-domain hops come from cross-domain routing.
            total_inter_hops = cross_hops

            violation = ViolationInfo(
                c7_violated=e2e > slice_req.qos.max_e2e_delay,
                c9_violated=total_inter_hops > cfg.max_inter_domain_hops,
                actor_infeasible=actor_infeasible,
                cross_domain_infeasible=not cross_feasible,
                e2e_delay=e2e,
                e2e_budget=slice_req.qos.max_e2e_delay,
                total_bw=inter_bw,
                min_bw=slice_req.qos.min_throughput,
                inter_domain_hops=total_inter_hops,
                max_inter_domain_hops=cfg.max_inter_domain_hops,
            )

            passes = not violation.has_violation

            # Compute cost
            intra_cost = sum(r.resource_cost for r in responses.values())
            cost = intra_cost + cfg.gamma_inter * inter_bw

            # --- Record attempt ---
            # Store canonical partition for PPO re-evaluation (policy space).
            attempt = PartitionAttempt(
                trial_index=j,
                partition=partition_canonical,
                domain_responses=responses,
                violation=violation if not passes else None,
                e2e_delay=e2e,
                total_cost=cost,
                value_estimate=value,
                log_probs=log_probs.detach(),
                entropy=entropy,
            )
            history.attempts.append(attempt)

            if passes:
                # Frame-consistency asserts: certify canonical↔domain-ID
                # round-trip and tier feasibility at the dispatch site.
                domain_to_canonical = {d: i for i, d in enumerate(canonical_to_domain)}
                roundtrip = [domain_to_canonical[d] for d in partition]
                assert roundtrip == partition_canonical, (
                    f"Canonical round-trip failed: {partition_canonical} -> "
                    f"{partition} -> {roundtrip}"
                )
                domain_tiers = {
                    s.domain_id: s.supported_tiers
                    for s in obs_struct.domain_summaries
                }
                for k, domain_id in enumerate(partition):
                    required = plan.required_tiers[k]
                    assert required in domain_tiers.get(domain_id, []), (
                        f"VNF {k} requires {required} but domain {domain_id} "
                        f"supports {domain_tiers.get(domain_id, [])}"
                    )

                # COMMIT — restore substrate to pre-dispatch state.
                # The episode runner's allocate() applies the definitive
                # allocation from the PlacementPlan (which now includes
                # cross-domain routes). Undo all provisional mutations.
                self._restore_node_residuals(substrate, node_snapshot)
                for resp in responses.values():
                    if resp.feasible:
                        self._rollback_domain(substrate, resp)
                self._rollback_cross_domain(substrate, cross_routes, cross_bw)

                final_action = MDOAction.COMMIT
                committed_partition = partition
                committed_responses = responses
                committed_cross_routes = cross_routes
                committed_cross_bw = cross_bw
                committed_e2e = e2e
                committed_cost = cost
                break

            # --- Rollback substrate state ---
            self._restore_node_residuals(substrate, node_snapshot)
            for resp in responses.values():
                if resp.feasible:
                    self._rollback_domain(substrate, resp)
            self._rollback_cross_domain(substrate, cross_routes, cross_bw)

            # --- Check rejection triggers ---
            reject_reason = check_rejection_triggers(
                history=history,
                n_part=cfg.n_part,
                tau_v=cfg.tau_v,
                stability_k=cfg.stability_k,
            )
            if reject_reason is not None:
                final_action = MDOAction.REJECT
                break

        # If loop ended without COMMIT or explicit REJECT
        if final_action not in (MDOAction.COMMIT, MDOAction.REJECT):
            final_action = MDOAction.REJECT
            reject_reason = RejectReason.BUDGET_EXHAUSTED

        # --- Compute reward (SMDP: terminal reward for the whole arrival) ---
        admitted = final_action == MDOAction.COMMIT
        reward = self._compute_reward(
            admitted=admitted,
            cost=committed_cost if admitted else 0.0,
            e2e=committed_e2e if admitted else 0.0,
            qos=slice_req.qos,
            num_trials=history.num_attempts,
            cost_greedy=cost_greedy,
        )

        # Stack log probs from all trials
        if all_log_probs:
            stacked = torch.cat(all_log_probs)
        else:
            stacked = torch.tensor([])

        return MDOResult(
            request_id=slice_req.request_id,
            action=final_action,
            admitted=admitted,
            partition=committed_partition,
            domain_responses=committed_responses,
            cross_domain_routes=committed_cross_routes,
            cross_domain_bw=committed_cross_bw,
            e2e_delay=committed_e2e,
            total_cost=committed_cost,
            reward=reward,
            retry_history=history,
            reject_reason=reject_reason,
            log_probs=stacked,
            entropy=all_entropy / max(history.num_attempts, 1),
            value_estimate=last_value,
            obs_tensor=obs_tensor.detach(),
            tier_mask=tier_mask.detach(),
            num_vnfs=plan.num_vnfs,
        )

    def _sample_partition(
        self,
        obs_tensor: torch.Tensor,
        tier_mask: torch.Tensor,
        plan: PlanSummary,
        mode: str,
        canonical_to_domain: list[int] | None = None,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, float, float]:
        """Sample a partition using the specified mode.

        Returns: (partition_canonical, log_probs, logits, entropy, value_estimate)
        All partition outputs use canonical indices (policy space). The caller
        maps to actual domain IDs via canonical_to_domain.
        """
        K = plan.num_vnfs
        M = tier_mask.shape[1]

        if mode == "follow_prior":
            # Convert domain IDs to canonical indices
            domain_to_canonical = {}
            if canonical_to_domain:
                domain_to_canonical = {d: i for i, d in enumerate(canonical_to_domain)}
            partition = [
                domain_to_canonical.get(d, d) for d in plan.suggested_domains
            ]
            log_probs = torch.zeros(K)
            logits = torch.zeros(K, M)
            return partition, log_probs, logits, 0.0, 0.0

        if mode == "random":
            partition = []
            for k in range(K):
                feasible = tier_mask[k].nonzero(as_tuple=True)[0]
                if len(feasible) == 0:
                    partition.append(0)
                else:
                    idx = torch.randint(len(feasible), (1,)).item()
                    partition.append(feasible[idx].item())
            log_probs = torch.zeros(K)
            logits = torch.zeros(K, M)
            return partition, log_probs, logits, 0.0, 0.0

        if self.policy is None:
            raise ValueError(f"Policy required for mode '{mode}' but is None")

        if mode == "sequential_argmax":
            return self._sequential_argmax(obs_tensor, tier_mask, K)

        deterministic = mode == "deterministic"
        with torch.no_grad() if deterministic else torch.enable_grad():
            partition, log_probs, logits, entropy = self.policy(
                obs_tensor, tier_mask, K, deterministic=deterministic,
            )
            value = self.policy.get_value(obs_tensor).item()

        return partition, log_probs, logits, entropy, value

    def _sequential_argmax(
        self,
        obs_tensor: torch.Tensor,
        tier_mask: torch.Tensor,
        num_vnfs: int,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, float, float]:
        """Capacity-aware sequential argmax: decode VNFs in order, penalizing
        domains already loaded by prior VNFs of this slice.

        Uses the trained policy's logits but decodes autoregressively at
        inference time. No retraining. Discriminates "policy preferences are
        useful but independent argmax destroys them" from "policy has no
        useful preferences."
        """
        assert self.policy is not None
        with torch.no_grad():
            if obs_tensor.dim() == 1:
                obs_tensor_b = obs_tensor.unsqueeze(0)
            else:
                obs_tensor_b = obs_tensor
            h = self.policy.encoder(obs_tensor_b)
            raw = self.policy.actor_head(h).view(
                -1, self.policy.max_vnfs, self.policy.num_domains,
            ).squeeze(0)
            logits = raw[:num_vnfs]

            neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype)
            masked = torch.where(tier_mask[:num_vnfs], logits, neg_inf)

            M = self.policy.num_domains
            domain_count = [0] * M
            partition = []
            log_prob_list = []
            entropy_sum = 0.0

            for k in range(num_vnfs):
                adjusted = masked[k].clone()
                for m in range(M):
                    if adjusted[m] != float("-inf"):
                        adjusted[m] -= 2.0 * domain_count[m]
                dist = Categorical(logits=adjusted)
                action = adjusted.argmax()
                partition.append(action.item())
                log_prob_list.append(dist.log_prob(action))
                entropy_sum += dist.entropy().item()
                domain_count[action.item()] += 1

            log_probs = torch.stack(log_prob_list)
            entropy = entropy_sum / num_vnfs if num_vnfs > 0 else 0.0
            value = self.policy.get_value(obs_tensor).item()

        return partition, log_probs, logits, entropy, value

    def _build_fragments(
        self,
        plan: PlanSummary,
        partition: list[int],
        slice_req: SliceRequest,
    ) -> tuple[dict[int, PlanFragment], list[FlowEdge]]:
        """Split the abstract plan into per-domain PlanFragments.

        Returns:
            (fragments, cross_domain_flows) — fragments for domain actors,
            plus cross-domain flows that the coordinator routes on the full
            graph after actor dispatch.
        """
        fragments: dict[int, list[VNFAssignment]] = {}
        domain_flows: dict[int, list[FlowEdge]] = {}
        cross_domain_flows: list[FlowEdge] = []

        for k, vnf in enumerate(slice_req.vnfs):
            domain_id = partition[k]
            if domain_id not in fragments:
                fragments[domain_id] = []
                domain_flows[domain_id] = []

            adj_domains: set[int] = set()
            if k > 0 and partition[k - 1] != domain_id:
                adj_domains.add(partition[k - 1])
            if k < len(partition) - 1 and partition[k + 1] != domain_id:
                adj_domains.add(partition[k + 1])

            bw_in = plan.bw_demands[k] if k < len(plan.bw_demands) else 0.0

            assignment = VNFAssignment(
                vnf_id=vnf.vnf_id,
                vnf_type=vnf.vnf_type,
                cpu_demand=vnf.cpu_demand,
                ram_demand=vnf.ram_demand,
                required_tier=plan.required_tiers[k],
                computational_intensity=vnf.computational_intensity,
                vcr=vnf.vcr,
                bandwidth_in=bw_in,
                permitted_nodes=vnf.permitted_nodes,
                position_in_sfc=k,
                sfc_length=len(slice_req.vnfs),
                adjacent_domain_ids=adj_domains,
            )
            fragments[domain_id].append(assignment)

        for fe in slice_req.flow_edges:
            src_idx = next(
                i for i, v in enumerate(slice_req.vnfs) if v.vnf_id == fe.source_vnf
            )
            dst_idx = next(
                i for i, v in enumerate(slice_req.vnfs) if v.vnf_id == fe.target_vnf
            )
            if partition[src_idx] == partition[dst_idx]:
                domain_id = partition[src_idx]
                if domain_id not in domain_flows:
                    domain_flows[domain_id] = []
                domain_flows[domain_id].append(fe)
            else:
                cross_domain_flows.append(fe)

        result: dict[int, PlanFragment] = {}
        all_domains = set(partition)
        for domain_id in all_domains:
            vnf_list = fragments.get(domain_id, [])
            flows = domain_flows.get(domain_id, [])
            target_doms = {
                partition[k]
                for k in range(len(partition))
                if partition[k] != domain_id
            }
            result[domain_id] = PlanFragment(
                domain_id=domain_id,
                vnf_assignments=vnf_list,
                intra_flows=flows,
                delay_budget_ms=slice_req.qos.max_e2e_delay,
                target_domain_ids=target_doms,
            )

        return result, cross_domain_flows

    def _dispatch_to_actors(
        self,
        substrate: SubstrateNetwork,
        fragments: dict[int, PlanFragment],
    ) -> dict[int, DomainResponse]:
        """Dispatch fragments to domain actors and collect responses."""
        responses: dict[int, DomainResponse] = {}

        for domain_id, fragment in fragments.items():
            if domain_id in self.domain_actors:
                actor = self.domain_actors[domain_id]
                responses[domain_id] = actor.act(substrate, fragment)
            else:
                # No actor for this domain — return infeasible
                responses[domain_id] = DomainResponse(
                    domain_id=domain_id, feasible=False,
                )

        return responses

    def _snapshot_node_residuals(
        self,
        substrate: SubstrateNetwork,
        partition: list[int],
    ) -> dict[str, tuple[float, float]]:
        """Snapshot CPU/RAM residuals for all nodes in domains touched by the partition."""
        domains = set(partition)
        snapshot: dict[str, tuple[float, float]] = {}
        g = substrate.graph
        for d in domains:
            for nid in substrate.nodes_in_domain(d):
                snapshot[nid] = (g.nodes[nid]["cpu_residual"], g.nodes[nid]["ram_residual"])
        return snapshot

    def _restore_node_residuals(
        self,
        substrate: SubstrateNetwork,
        snapshot: dict[str, tuple[float, float]],
    ) -> None:
        """Restore CPU/RAM residuals from a snapshot."""
        g = substrate.graph
        for nid, (cpu, ram) in snapshot.items():
            g.nodes[nid]["cpu_residual"] = cpu
            g.nodes[nid]["ram_residual"] = ram

    def _route_cross_domain_flows(
        self,
        substrate: SubstrateNetwork,
        cross_domain_flows: list[FlowEdge],
        responses: dict[int, DomainResponse],
        delay_budget: float,
    ) -> tuple[
        bool,
        dict[tuple[str, str], list[str]],
        dict[tuple[str, str], float],
        float,
        int,
    ]:
        """Route cross-domain flows on the full substrate graph.

        Called after domain actors have placed VNFs and routed intra-domain
        flows. Uses the actual VNF placements from domain responses to
        determine src/dst nodes, then routes each cross-domain flow on the
        full directed graph (multi-hop through transit domains).

        Provisionally debits BW on routed edges so subsequent flows in the
        same partition see updated residuals. On failure, the caller must
        rollback all provisional debits.

        Args:
            substrate: Current substrate state (actors already mutated it).
            cross_domain_flows: Flows between VNFs in different domains.
            responses: Domain actor responses with VNF placements.
            delay_budget: Remaining delay budget after intra-domain routing.

        Returns:
            (feasible, routes, bw_allocated, total_delay, inter_hops)
        """
        all_placements: dict[str, str] = {}
        for resp in responses.values():
            all_placements.update(resp.placements)

        routes: dict[tuple[str, str], list[str]] = {}
        bw_allocated: dict[tuple[str, str], float] = {}
        total_delay = 0.0
        inter_hops = 0
        remaining_delay = delay_budget

        for fe in cross_domain_flows:
            src_node = all_placements.get(fe.source_vnf)
            dst_node = all_placements.get(fe.target_vnf)
            if src_node is None or dst_node is None:
                return False, {}, {}, 0.0, 0

            result = route_cross_domain_flow(
                substrate, src_node, dst_node,
                bw_demand=fe.bandwidth_demand,
                delay_budget=remaining_delay,
            )

            if not result.feasible:
                # Rollback all previously allocated cross-domain BW
                for prev_key, prev_bw in bw_allocated.items():
                    prev_links = routes[prev_key]
                    deallocate_route_bw(substrate, prev_links, prev_bw)
                return False, {}, {}, 0.0, 0

            flow_key = (fe.source_vnf, fe.target_vnf)
            routes[flow_key] = result.path_links
            bw_allocated[flow_key] = fe.bandwidth_demand
            total_delay += result.propagation_delay
            remaining_delay -= result.propagation_delay

            # Count inter-domain hops on this path
            g = substrate.graph
            for link_id in result.path_links:
                for u, v, d in g.edges(data=True):
                    if d["link_id"] == link_id:
                        if g.nodes[u]["domain_id"] != g.nodes[v]["domain_id"]:
                            inter_hops += 1
                        break

            # Provisional BW debit for subsequent flows in same partition
            allocate_route_bw(substrate, result.path_links, fe.bandwidth_demand)

        return True, routes, bw_allocated, total_delay, inter_hops

    def _rollback_cross_domain(
        self,
        substrate: SubstrateNetwork,
        routes: dict[tuple[str, str], list[str]],
        bw_allocated: dict[tuple[str, str], float],
    ) -> None:
        """Rollback provisional BW debits from cross-domain routing."""
        for flow_key, bw in bw_allocated.items():
            links = routes.get(flow_key, [])
            if links:
                deallocate_route_bw(substrate, links, bw)

    def _rollback_domain(
        self,
        substrate: SubstrateNetwork,
        response: DomainResponse,
    ) -> None:
        """Roll back BW allocations from a domain actor's intra-domain routing."""
        for flow_key, bw in response.bw_allocated.items():
            route_links = response.routes.get(flow_key, [])
            if route_links:
                deallocate_route_bw(substrate, route_links, bw)
        # SCAFFOLD: no-op until Part B.

    def _compute_reward(
        self,
        admitted: bool,
        cost: float,
        e2e: float,  # noqa: ARG002 — used by hard penalty in Phase 5
        qos: QoSRequirements,  # noqa: ARG002 — used by hard penalty in Phase 5
        num_trials: int,
        cost_greedy: float | None = None,
    ) -> RewardComponents:
        """Compute decomposed reward (v6.2 Eq. 9).

        SMDP credit assignment: this terminal reward is shared across all
        partition trials within this arrival. No bootstrapping between retries.

        Efficiency uses normalized cost ratio (cost/cost_greedy) so the
        penalty stays bounded relative to the admission bonus mu.
        """
        cfg = self.config

        admission = cfg.mu if admitted else 0.0

        efficiency = 0.0
        if admitted and cost_greedy is not None and cost_greedy > 0:
            efficiency = -cfg.alpha * (cost / cost_greedy)

        hard_penalty = 0.0

        quality_shaping = 0.0
        if admitted and cost_greedy is not None and cost_greedy > 0:
            local_score = max(0.0, (cost_greedy - cost) / cost_greedy)
            quality_shaping = cfg.eta * local_score

        trial_penalty = -cfg.xi * max(0, num_trials - 1)

        return RewardComponents(
            admission=admission,
            efficiency=efficiency,
            hard_penalty=hard_penalty,
            quality_shaping=quality_shaping,
            trial_penalty=trial_penalty,
        )
