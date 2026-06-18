

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

if TYPE_CHECKING:
    from orion.actors.domain_actor import DomainActor

from orion.actors.types import DomainResponse, PlanFragment, VNFAssignment
from orion.mdo.observation import (
    build_mdo_observation,
    build_tier_masks,
    observation_to_tensor,
)
from orion.mdo.policy import MDOPolicy
from orion.mdo.precommit_check import (
    inter_domain_residual_by_pair,
    precommit_check,
)
from orion.mdo.rejection import check_rejection_triggers
from orion.mdo.types import (
    MDOAction,
    MDOResult,
    PartitionAttempt,
    PlanSummary,
    RejectReason,
    RetryHistory,
    RewardComponents,
)
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import QoSRequirements, SliceRequest

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

        # Inter-domain C5b residual source (aggregate per domain-pair).
        # Built once per arrival: inter-domain reservation happens at COMMIT
        # (slice departure releases it — Phase 5), so the residual is constant
        # across this arrival's retry loop.
        # SCAFFOLD (Part B): replace this substrate-derived aggregate with the
        # live per-pair residual counter that the reserve/release path
        # maintains and the observation also reads. See _reserve_inter_domain_bw
        # and docs/c5b_inter_domain_changeset.md.
        inter_domain_residuals = inter_domain_residual_by_pair(
            obs_struct.inter_domain_links,
        )

        committed_partition: list[int] | None = None
        committed_responses: dict[int, DomainResponse] = {}
        committed_e2e = 0.0
        committed_cost = 0.0
        final_action = MDOAction.REJECT
        reject_reason: RejectReason | None = None
        all_log_probs: list[torch.Tensor] = []
        all_entropy = 0.0
        last_value = 0.0

        for j in range(cfg.n_part):
            # --- Sample partition ---
            partition, log_probs, _logits, entropy, value = self._sample_partition(
                obs_tensor, tier_mask, plan, mode,
            )
            all_log_probs.append(log_probs)
            all_entropy += entropy
            last_value = value

            # --- Split into PlanFragments and dispatch ---
            fragments = self._build_fragments(plan, partition, slice_req)
            responses = self._dispatch_to_actors(substrate, fragments)

            # --- Pre-commit check ---
            passes, violation, e2e, cost = precommit_check(
                partition=partition,
                domain_responses=responses,
                inter_domain_delays=inter_domain_delays,
                qos=slice_req.qos,
                bw_demands=bw_demands,
                max_inter_domain_hops=cfg.max_inter_domain_hops,
                gamma_inter=cfg.gamma_inter,
                inter_domain_residuals=inter_domain_residuals,
            )

            # --- Record attempt ---
            attempt = PartitionAttempt(
                trial_index=j,
                partition=partition,
                domain_responses=responses,
                violation=violation if not passes else None,
                e2e_delay=e2e,
                total_cost=cost,
                value_estimate=value,
            )
            history.attempts.append(attempt)

            if passes:
                # COMMIT
                final_action = MDOAction.COMMIT
                committed_partition = partition
                committed_responses = responses
                committed_e2e = e2e
                committed_cost = cost
                # Reserve inter-domain bandwidth for the committed partition.
                # SCAFFOLD (Part B) — currently a no-op; see method docstring.
                self._reserve_inter_domain_bw(substrate, partition, bw_demands)
                break

            # --- Rollback substrate state ---
            # Domain actors handle their own rollback on infeasible results,
            # but committed BW/CPU needs rollback for feasible-but-failed-precommit
            for resp in responses.values():
                if resp.feasible:
                    self._rollback_domain(substrate, resp)

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
            e2e_delay=committed_e2e,
            total_cost=committed_cost,
            reward=reward,
            retry_history=history,
            reject_reason=reject_reason,
            log_probs=stacked,
            entropy=all_entropy / max(history.num_attempts, 1),
            value_estimate=last_value,
        )

    def _sample_partition(
        self,
        obs_tensor: torch.Tensor,
        tier_mask: torch.Tensor,
        plan: PlanSummary,
        mode: str,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, float, float]:
        """Sample a partition using the specified mode.

        Returns: (partition, log_probs, logits, entropy, value_estimate)
        """
        K = plan.num_vnfs
        M = tier_mask.shape[1]

        if mode == "follow_prior":
            partition = list(plan.suggested_domains)
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

        deterministic = mode == "deterministic"
        with torch.no_grad() if deterministic else torch.enable_grad():
            partition, log_probs, logits, entropy = self.policy(
                obs_tensor, tier_mask, K, deterministic=deterministic,
            )
            value = self.policy.get_value(obs_tensor).item()

        return partition, log_probs, logits, entropy, value

    def _build_fragments(
        self,
        plan: PlanSummary,
        partition: list[int],
        slice_req: SliceRequest,
    ) -> dict[int, PlanFragment]:
        """Split the abstract plan into per-domain PlanFragments."""
        fragments: dict[int, list[VNFAssignment]] = {}
        domain_flows: dict[int, list[FlowEdge]] = {}

        for k, vnf in enumerate(slice_req.vnfs):
            domain_id = partition[k]
            if domain_id not in fragments:
                fragments[domain_id] = []
                domain_flows[domain_id] = []

            # Determine adjacent domains for border-node awareness
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

        # Build intra-domain flows
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

        # Assemble PlanFragments
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

        return result

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

    def _rollback_domain(
        self,
        substrate: SubstrateNetwork,
        response: DomainResponse,
    ) -> None:
        """Roll back substrate state for a domain that passed actor placement
        but failed the MDO pre-commit check.

        Restores link BW from the response's allocations. Node CPU/RAM
        rollback requires the substrate snapshot/restore mechanism — domain
        actors handle their own rollback on infeasible results, but
        feasible-but-precommit-failed paths need explicit BW deallocation.
        Full node rollback is wired in Phase 5 via SubstrateNetwork.restore().
        """
        from orion.actors.routing import deallocate_route_bw

        for flow_key, bw in response.bw_allocated.items():
            route_links = response.routes.get(flow_key, [])
            if route_links:
                deallocate_route_bw(substrate, route_links, bw)

    def _reserve_inter_domain_bw(
        self,
        substrate: SubstrateNetwork,  # noqa: ARG002 — used by Part B
        partition: list[int],  # noqa: ARG002 — used by Part B
        bw_demands: list[float],  # noqa: ARG002 — used by Part B
    ) -> None:
        """Reserve inter-domain bandwidth for a committed partition.

        SCAFFOLD (Part B, Phase 5) — currently a no-op. This is the second
        half of the C5b work item; Part A (the check in precommit_check) is
        live but reads capacity-not-residual until this lands.

        When implemented, this decrements an aggregate per-domain-pair
        residual counter by the cross-domain demand
        (`inter_domain_demand_by_pair(partition, bw_demands)`), and that SAME
        counter is what feeds both precommit_check's `inter_domain_residuals`
        and the MDO observation's inter-domain link features. Keeping one
        counter is load-bearing: split sources reintroduce the stale-residual
        bug (the policy trained to believe inter-domain links never deplete).

        Reservation happens at COMMIT; the matching release happens at slice
        DEPARTURE in the simulator lifecycle (Phase 5), not on a failed retry —
        failed trials never reserved inter-domain BW under commit-time
        reservation, so there is no per-trial inter-domain rollback. This must
        land together with the substrate snapshot/restore so the counter is
        restored cleanly across episodes.

        Design pin: the MDO reserves against the AGGREGATE per-pair counter
        (summary granularity). Concrete per-edge charging — choosing which
        physical inter-domain link to debit — is the simulator's ground-truth
        job, not the MDO's; the MDO has no per-edge information to make that
        choice well. See docs/c5b_inter_domain_changeset.md.
        """
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
        """
        cfg = self.config

        # Admission
        admission = cfg.mu if admitted else 0.0

        # Efficiency
        efficiency = -cfg.alpha * cost if admitted else 0.0

        # Hard penalty (fires when pre-commit check was wrong under actual load)
        # At Phase 4, this is always 0 — the simulator's ground-truth check
        # is wired in Phase 5.
        hard_penalty = 0.0

        # Quality shaping via LocalScore
        quality_shaping = 0.0
        if admitted and cost_greedy is not None and cost_greedy > 0:
            local_score = max(0.0, (cost_greedy - cost) / cost_greedy)
            quality_shaping = cfg.eta * local_score

        # Trial penalty
        trial_penalty = -cfg.xi * max(0, num_trials - 1)

        return RewardComponents(
            admission=admission,
            efficiency=efficiency,
            hard_penalty=hard_penalty,
            quality_shaping=quality_shaping,
            trial_penalty=trial_penalty,
        )
