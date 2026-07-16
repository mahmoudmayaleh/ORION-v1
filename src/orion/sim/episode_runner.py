"""Episode-level scenario driver.

This is the imperative interface the Phase 5 MAPPO training loop will use.
It is intentionally NOT a Gymnasium env — Gym semantics fit poorly when one
"environment step" internally runs an N_part retry loop and N_struct LLM
regenerations. `sim/env.py` provides a thin Gym wrapper around this for
monitoring tools that expect that API.

Per-arrival flow (matches v6.2 §6.3):
    1. Pop next event from `ArrivalProcess`.
    2. If DEPARTURE: deallocate, record bookkeeping, return.
    3. If ARRIVAL:
         a. Compute Cost_greedy on a clone of the current substrate (Choice B1).
         b. Coordinator.resolve_arrival(...) runs the MDO partition retry loop,
            calling the domain actors per trial. Returns an MDOResult.
         c. If COMMIT: build a PlacementPlan from the domain responses, allocate
            on the substrate, run the post-commit verifier under load-dependent
            M/M/1 (Choice C1 + E1), and assemble the final reward (override
            hard_penalty from the verdict).
         d. Append per-agent rollout transitions.

Choices wired in here:
    - A3: imperative interface for training; Gym wrapper sits on top.
    - B1: per-arrival substrate clone for Cost_greedy.
    - C1: load-dependent delay via the verifier.
    - D2: structurally-rejected slices update the KPI counter but do NOT
          generate MDO rollout transitions.
    - E1: post-commit ground-truth verification overrides hard_penalty.
    - F1: fixed N arrivals per episode (Coordinator-agnostic; the runner
          terminates when the ArrivalProcess is drained).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from orion.baselines.greedy_ffd import compute_cost_greedy
from orion.mdo.coordinator import MDOCoordinator
from orion.mdo.observation import build_domain_summaries
from orion.mdo.types import MDOAction, MDOResult, PlanSummary, RewardComponents
from orion.profiling import profiled
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.reward import RewardWeights, finalize_reward
from orion.sim.rollout_buffer import (
    DomainActorTransition,
    MDOTransition,
    MultiAgentRollout,
)
from orion.sim.verifier import GroundTruthVerdict, verify_committed_plan
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import PlacementPlan, SliceRequest

logger = logging.getLogger(__name__)


# ── Public per-episode metrics ───────────────────────────────────────────────


@dataclass
class EpisodeStats:
    """KPIs reported per episode. Counters decoupled from the RL buffer (D2).

    `total_arrivals` includes structurally-rejected slices; the RL buffer
    only contains transitions for arrivals the MDO actually acted on.
    """

    total_arrivals: int = 0
    admitted: int = 0
    rejected_by_mdo: int = 0
    rejected_structural: int = 0
    departures: int = 0
    hard_penalty_fires: int = 0
    cumulative_reward: float = 0.0
    per_slice_type_admitted: dict[str, int] = field(default_factory=dict)
    per_slice_type_total: dict[str, int] = field(default_factory=dict)

    @property
    def admission_rate(self) -> float:
        if self.total_arrivals == 0:
            return 0.0
        return self.admitted / self.total_arrivals


@dataclass
class EpisodeResult:
    """Full episode output for downstream training / evaluation."""

    stats: EpisodeStats
    rollout: MultiAgentRollout
    mdo_results: list[MDOResult]
    # Per-arrival behavioral trace (PREREG §N.2) — permanent gate-runner
    # instrumentation so criterion (b) / the k-analysis are computable from any
    # run, not a side probe. One dict per MDO-reaching arrival:
    #   {index, rid, k, partition (domain list), admit, hm (per-domain h^m snapshot)}
    arrival_trace: list[dict] = field(default_factory=list)


# ── Episode runner ───────────────────────────────────────────────────────────


class EpisodeRunner:
    """Drives one episode end-to-end.

    Args:
        substrate: The substrate (its residuals are mutated; caller should
            call substrate.reset() before reusing it for another episode).
        arrival_process: Pre-generated event stream.
        coordinator: MDO coordinator (holds policy + domain actors).
        inter_domain_delays: Static propagation-delay table; passed to the
            coordinator for its light-load pre-commit. Ground-truth delay
            uses load-dependent M/M/1 via the verifier.
        reward_weights: Weights for the final reward assembly.
        plan_builder: Callable that maps a slice_req + plan_summary to the
            PlanSummary the coordinator consumes. Lets eval bypass the real
            Agent B and inject canned plans.
        max_inter_domain_hops: C9 limit propagated to the verifier.
    """

    def __init__(
        self,
        substrate: SubstrateNetwork,
        arrival_process: ArrivalProcess,
        coordinator: MDOCoordinator,
        inter_domain_delays: dict[tuple[int, int], float],
        reward_weights: RewardWeights | None = None,
        plan_builder=None,
        max_inter_domain_hops: int = 3,
    ) -> None:
        self.substrate = substrate
        self.arrival_process = arrival_process
        self.coordinator = coordinator
        self.inter_domain_delays = inter_domain_delays
        self.reward_weights = reward_weights or RewardWeights()
        self.plan_builder = plan_builder or _default_plan_builder
        self.max_inter_domain_hops = max_inter_domain_hops

        self._active_plans: dict[str, PlacementPlan] = {}
        self._active_requests: dict[str, SliceRequest] = {}

    def reset(self) -> None:
        """Reset substrate residuals and arrival index for a new episode."""
        self.substrate.reset()
        self.arrival_process.reset()
        self._active_plans.clear()
        self._active_requests.clear()

    def run_episode(self, mdo_mode: str = "sample") -> EpisodeResult:
        """Drive the full event stream to completion.

        Args:
            mdo_mode: Forwarded to coordinator.resolve_arrival; one of
                "sample", "deterministic", "follow_prior", "random".
        """
        stats = EpisodeStats()
        rollout = MultiAgentRollout()
        mdo_results: list[MDOResult] = []
        arrival_trace: list[dict] = []

        while self.arrival_process.has_next():
            event = self.arrival_process.next_event()

            if event.event_type == EventType.DEPARTURE:
                self._handle_departure(event.request_id, stats)
                continue

            assert event.slice_request is not None
            self._handle_arrival(
                event.slice_request, mdo_mode, rollout, mdo_results, stats,
                arrival_trace,
            )

        return EpisodeResult(
            stats=stats, rollout=rollout, mdo_results=mdo_results,
            arrival_trace=arrival_trace,
        )

    # ── Event handlers ──────────────────────────────────────────────────────

    def _handle_arrival(
        self,
        slice_req: SliceRequest,
        mdo_mode: str,
        rollout: MultiAgentRollout,
        mdo_results: list[MDOResult],
        stats: EpisodeStats,
        arrival_trace: list[dict] | None = None,
    ) -> None:
        stats.total_arrivals += 1
        arrival_index = stats.total_arrivals - 1  # global stream position (§N.2)
        stats.per_slice_type_total[slice_req.slice_type.value] = (
            stats.per_slice_type_total.get(slice_req.slice_type.value, 0) + 1
        )

        with profiled("plan_build", {"slice_type": slice_req.slice_type.value,
                                     "k": len(slice_req.vnfs)}):
            plan_summary = self.plan_builder(slice_req, self.substrate)

        # Structurally infeasible slices skip the MDO entirely (Choice D2).
        # They count as system-level rejections in the KPI but contribute no
        # RL transitions.
        if plan_summary is None:
            stats.rejected_structural += 1
            return

        cost_greedy = compute_cost_greedy(self.substrate, slice_req)

        # h^m snapshot (§N.2) — captured pre-decision / pre-allocation, so it is the
        # same per-domain headroom the policy's observation was built from.
        hm_snapshot = {
            ds.domain_id: ds.max_node_headroom
            for ds in build_domain_summaries(self.substrate)
        } if arrival_trace is not None else None

        with profiled("mdo.decision", {"mode": mdo_mode, "k": len(slice_req.vnfs)}):
            mdo_result = self.coordinator.resolve_arrival(
                substrate=self.substrate,
                slice_req=slice_req,
                plan=plan_summary,
                inter_domain_delays=self.inter_domain_delays,
                mode=mdo_mode,
                cost_greedy=cost_greedy if cost_greedy != float("inf") else None,
            )
        mdo_results.append(mdo_result)

        verdict: GroundTruthVerdict | None = None
        if mdo_result.admitted and mdo_result.partition is not None:
            placement_plan = self._build_placement_plan(slice_req, mdo_result)
            if placement_plan is not None:
                self.substrate.allocate(placement_plan, slice_req)
                with profiled("verify"):
                    verdict = verify_committed_plan(
                        self.substrate,
                        placement_plan,
                        slice_req,
                        max_inter_domain_hops=self.max_inter_domain_hops,
                    )
                if verdict.hard_penalty_fired:
                    stats.hard_penalty_fires += 1
                    self.substrate.deallocate(placement_plan, slice_req)
                    mdo_result.admitted = False
                else:
                    self._active_plans[slice_req.request_id] = placement_plan
                    self._active_requests[slice_req.request_id] = slice_req

        final_reward, final_components = finalize_reward(
            mdo_components=mdo_result.reward,
            admitted=mdo_result.admitted,
            verdict=verdict,
            weights=self.reward_weights,
        )
        mdo_result.reward = final_components

        self._update_stats(slice_req, mdo_result, final_reward, stats)
        self._append_rollout(slice_req, mdo_result, final_reward, rollout, plan_summary)

        # Per-arrival behavioral trace (§N.2), recorded AFTER verify so `admit`
        # reflects the true-load outcome (hard-penalty deallocation flips it False).
        # `partition` is the policy's SELECTED partition (final attempt) — recorded
        # even on reject, so criterion (b) / the k-analysis see WHERE it placed, not
        # only what committed. `committed` separates selection from admission.
        if arrival_trace is not None:
            _attempts = (mdo_result.retry_history.attempts
                         if mdo_result.retry_history is not None else [])
            selected = (list(_attempts[-1].partition) if _attempts
                        else (list(mdo_result.partition)
                              if mdo_result.partition is not None else None))
            arrival_trace.append({
                "index": arrival_index,
                "rid": slice_req.request_id,
                "k": len(slice_req.vnfs),
                "partition": selected,
                "committed": mdo_result.partition is not None,
                "admit": bool(mdo_result.admitted),
                "hm": hm_snapshot,
            })

    def _handle_departure(self, request_id: str, stats: EpisodeStats) -> None:
        plan = self._active_plans.pop(request_id, None)
        request = self._active_requests.pop(request_id, None)
        if plan is None or request is None:
            return  # slice was never admitted; departure is a no-op
        self.substrate.deallocate(plan, request)
        stats.departures += 1

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_placement_plan(
        self,
        slice_req: SliceRequest,
        mdo_result: MDOResult,
    ) -> PlacementPlan | None:
        """Synthesise a substrate-level PlacementPlan from domain responses.

        Combines intra-domain routes (from domain actors) with cross-domain
        routes (from the coordinator's full-graph routing). Both land in
        flow_routes and bw_allocations so substrate.allocate() charges all
        edges and the verifier covers all flows.
        """
        vnf_placements: dict[str, str] = {}
        cpu_allocations: dict[str, float] = {}
        ram_allocations: dict[str, float] = {}
        flow_routes: dict[tuple[str, str], list[str]] = {}
        bw_allocations: dict[tuple[str, str], dict[str, float]] = {}

        vnf_by_id = {v.vnf_id: v for v in slice_req.vnfs}

        for _domain_id, response in mdo_result.domain_responses.items():
            for vnf_id, node_id in response.placements.items():
                vnf_placements[vnf_id] = node_id
                vnf = vnf_by_id[vnf_id]
                cpu_allocations[vnf_id] = vnf.cpu_demand
                ram_allocations[vnf_id] = vnf.ram_demand

            for flow_key, route_link_ids in response.routes.items():
                flow_routes[flow_key] = route_link_ids
                per_link_bw_value = response.bw_allocated.get(flow_key, 0.0)
                bw_allocations[flow_key] = {
                    link_id: per_link_bw_value for link_id in route_link_ids
                }

        # Add cross-domain routes from coordinator's full-graph routing
        for flow_key, route_link_ids in mdo_result.cross_domain_routes.items():
            flow_routes[flow_key] = route_link_ids
            bw = mdo_result.cross_domain_bw.get(flow_key, 0.0)
            bw_allocations[flow_key] = {
                link_id: bw for link_id in route_link_ids
            }

        if any(v.vnf_id not in vnf_placements for v in slice_req.vnfs):
            logger.warning(
                "Incomplete VNF placements for committed slice %s",
                slice_req.request_id,
            )
            return None

        return PlacementPlan(
            plan_id=f"{slice_req.request_id}_mdo",
            vnf_placements=vnf_placements,
            cpu_allocations=cpu_allocations,
            ram_allocations=ram_allocations,
            flow_routes=flow_routes,
            bw_allocations=bw_allocations,
            is_structurally_valid=True,
            source="mdo",
        )

    def _update_stats(
        self,
        slice_req: SliceRequest,
        mdo_result: MDOResult,
        final_reward: float,
        stats: EpisodeStats,
    ) -> None:
        stats.cumulative_reward += final_reward
        if mdo_result.admitted:
            stats.admitted += 1
            stats.per_slice_type_admitted[slice_req.slice_type.value] = (
                stats.per_slice_type_admitted.get(slice_req.slice_type.value, 0) + 1
            )
        else:
            stats.rejected_by_mdo += 1

    def _append_rollout(
        self,
        slice_req: SliceRequest,
        mdo_result: MDOResult,
        final_reward: float,
        rollout: MultiAgentRollout,
        plan_summary: PlanSummary | None = None,
    ) -> None:
        """One MDO transition per trial; one domain-actor transition per
        domain per trial. Terminal reward broadcast to all (SMDP)."""
        history = mdo_result.retry_history
        suggested = plan_summary.suggested_domains if plan_summary is not None else []
        for attempt in history.attempts:
            rollout.append_mdo(
                MDOTransition(
                    request_id=slice_req.request_id,
                    trial_index=attempt.trial_index,
                    obs=mdo_result.obs_tensor,
                    action=attempt.partition,
                    log_probs=attempt.log_probs,
                    entropy=attempt.entropy,
                    value_estimate=attempt.value_estimate,
                    terminal_reward=final_reward,
                    committed=attempt.trial_index == history.num_attempts - 1,
                    tier_mask=mdo_result.tier_mask,
                    num_vnfs=mdo_result.num_vnfs,
                    info={
                        "e2e_delay": attempt.e2e_delay,
                        "cost": attempt.total_cost,
                        "suggested_domains": suggested,
                    },
                )
            )
            for domain_id, response in attempt.domain_responses.items():
                rollout.append_domain_actor(
                    DomainActorTransition(
                        request_id=slice_req.request_id,
                        trial_index=attempt.trial_index,
                        domain_id=domain_id,
                        log_probs=response.log_probs,
                        entropy=getattr(response, "entropy", 0.0),
                        accepted=response.feasible,
                        intra_delay=response.intra_delay,
                        resource_cost=response.resource_cost,
                        terminal_reward=final_reward,
                        steps=getattr(response, "step_records", []),
                    )
                )


# ── Default plan builder (eval / random-policy mode) ─────────────────────────


def _default_plan_builder(
    slice_req: SliceRequest, substrate: SubstrateNetwork
) -> PlanSummary | None:
    """Build a PlanSummary directly from the slice request, no LLM.

    Used by the Phase 1 milestone (random/deterministic MDO mode) and by
    evaluation when Agent B is bypassed. The "abstract plan" here is just the
    structural skeleton — required tiers come from each VNF's permitted_nodes,
    suggested partition is a tier-greedy default the MDO is free to override.

    Returns None if no domain supports a required tier (structural
    infeasibility — D2 path).
    """
    g = substrate.graph

    vnf_ids = [v.vnf_id for v in slice_req.vnfs]
    cpu_demands = [v.cpu_demand for v in slice_req.vnfs]
    ram_demands = [v.ram_demand for v in slice_req.vnfs]
    vcrs = [v.vcr for v in slice_req.vnfs]
    bw_demands = [f.bandwidth_demand for f in slice_req.flow_edges]

    required_tiers = []
    suggested_domains = []
    for vnf in slice_req.vnfs:
        tier = _infer_required_tier(vnf, substrate)
        if tier is None:
            return None
        required_tiers.append(tier)

        # Suggested domain: first domain with a node of the required tier.
        chosen = None
        for n in vnf.permitted_nodes:
            if n in g.nodes and g.nodes[n]["tier"] == tier.value:
                chosen = int(g.nodes[n]["domain_id"])
                break
        if chosen is None:
            return None
        suggested_domains.append(chosen)

    return PlanSummary(
        vnf_ids=vnf_ids,
        required_tiers=required_tiers,
        suggested_domains=suggested_domains,
        cpu_demands=cpu_demands,
        ram_demands=ram_demands,
        vcrs=vcrs,
        bw_demands=bw_demands,
    )


def _infer_required_tier(vnf, substrate):
    """Return the InfrastructureTier required by a VNF (modal tier of permitted_nodes)."""
    from orion.types import InfrastructureTier

    if not vnf.permitted_nodes:
        return None
    tiers = {
        substrate.graph.nodes[n]["tier"]
        for n in vnf.permitted_nodes
        if n in substrate.graph.nodes
    }
    if not tiers:
        return None
    # If unambiguous, use it; otherwise pick the most "edge-ward" tier.
    tier_order = [
        InfrastructureTier.RAN_EDGE,
        InfrastructureTier.MEC,
        InfrastructureTier.REGIONAL_CLOUD,
        InfrastructureTier.CENTRAL_CLOUD,
    ]
    for t in tier_order:
        if t.value in tiers:
            return t
    return None
