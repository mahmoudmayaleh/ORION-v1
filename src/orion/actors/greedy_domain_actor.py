"""Deterministic best-fit domain actor for memory experiments.

Implements the same .act(substrate, fragment) interface as DomainActor but
uses best-fit placement (tightest CPU fit) and shortest-path routing. No
learnable parameters. Identical across all experiment approaches — removes actor
stochasticity from the memory comparison.

Node selection: lowest processing delay among feasible nodes (2026-08-20; it
was best-fit on CPU and delay-blind, which made `post_commit_c7_delay` the
largest rejection bin in the system). Tier match is a hard constraint.
Deterministic tiebreak by node_id.
"""

from __future__ import annotations

import torch

from orion.actors.routing import (
    allocate_route_bw,
    deallocate_route_bw,
    route_flow,
    RoutingSelector,
)
from orion.actors.types import DomainResponse, PlanFragment, VNFAssignment
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import InfrastructureTier, TIER_INDEX, TIER_ORDER


# Canonical ordering lives in orion.types (one definition, see TIER_ORDER).
_TIER_ORDER = TIER_INDEX


class GreedyDomainActor:
    """Deterministic best-fit placer within a single domain.

    Sorts VNFs by decreasing CPU then RAM, picks the lowest-delay feasible node
    (tier match first, then least processing delay), routes intra-domain flows
    on the shortest delay-feasible path. No RL signals produced.
    Identical across all experiment approaches.
    """

    def __init__(self, domain_id: int, k_paths: int = 3):
        self.domain_id = domain_id
        self.k_paths = k_paths
        self.routing_selector = RoutingSelector("min_delay")

    def act(
        self,
        substrate: SubstrateNetwork,
        fragment: PlanFragment,
        deterministic: bool = True,
    ) -> DomainResponse:
        if fragment.is_empty:
            return DomainResponse.empty(self.domain_id)

        g = substrate.graph
        domain_nodes = sorted(substrate.nodes_in_domain(self.domain_id))
        if not domain_nodes:
            return DomainResponse(domain_id=self.domain_id, feasible=False)

        vnf_list = sorted(
            fragment.vnf_assignments,
            key=lambda v: (-v.cpu_demand, -v.ram_demand, v.vnf_id),
        )

        placements: dict[str, str] = {}
        node_snapshots: dict[str, tuple[float, float]] = {}
        allocated_bw: list[tuple[list[str], float]] = []
        routes: dict[tuple[str, str], list[str]] = {}
        bw_allocated: dict[tuple[str, str], float] = {}
        total_proc_delay = 0.0
        total_route_delay = 0.0
        total_resource_cost = 0.0

        delay_remaining = fragment.delay_budget_ms

        for placed_i, vnf in enumerate(vnf_list):
            # Budget the CHAIN, not this hop. `delay_remaining` is the whole
            # residual e2e budget, so testing one VNF against it passes almost
            # always and the filter never binds -- every VNF fits, the sum does
            # not. Each VNF gets an equal share of what is left, which is what
            # makes the constraint bite before the chain is already over.
            n_left = max(1, len(vnf_list) - placed_i)
            node_id = self._select_node(
                substrate, vnf, domain_nodes,
                delay_remaining=delay_remaining, vnfs_remaining=n_left)
            if node_id is None:
                self._rollback(substrate, allocated_bw, node_snapshots)
                return DomainResponse(
                    domain_id=self.domain_id, feasible=False,
                )

            placements[vnf.vnf_id] = node_id

            if node_id not in node_snapshots:
                node_snapshots[node_id] = (
                    g.nodes[node_id]["cpu_residual"],
                    g.nodes[node_id]["ram_residual"],
                )

            g.nodes[node_id]["cpu_residual"] -= vnf.cpu_demand
            g.nodes[node_id]["ram_residual"] -= vnf.ram_demand

            proc_delay = (
                g.nodes[node_id]["processing_delay"] * vnf.computational_intensity
            )
            total_proc_delay += proc_delay
            # The budget tracker used to be decremented ONLY by routing
            # propagation delay, never by processing delay -- so it ignored the
            # term it was itself accumulating into `intra_delay`, and every
            # `route_flow` after the first VNF was handed a budget that had
            # already been partly spent.
            delay_remaining -= proc_delay
            total_resource_cost += vnf.cpu_demand + vnf.ram_demand

        # ── routing: a SECOND pass, over every intra-domain flow ────────────────
        #
        # This used to run inside the placement loop, keyed on the flow's TARGET,
        # and skipped the flow outright when its source was not placed yet
        # (`if src_vnf_id not in placements: continue`). Placement order is by
        # decreasing CPU, not chain order, so the source is placed after the target
        # on 77.7% of chain edges (measured over 2000 generated slices: K=2 100%,
        # K=3 82.5%, K=4 33.3%). Those edges were never routed, never charged
        # bandwidth and never charged propagation delay, and the verifier could not
        # see it: `_check_c5b` reads a missing `bw_allocations` entry as "same-node
        # placement, contracts met" and `_compute_ground_truth_e2e` adds nothing for
        # an empty path.
        #
        # Cross-domain flows never had the skip (`_route_cross_domain_flows` runs
        # after all actors, off the assembled placements), so intra-domain chains
        # were charged for almost nothing while split chains were charged in full.
        # That is a thumb on the scale for colocation, in a system whose whole
        # question is when to split.
        #
        # Placing first and routing second is what removes the ordering dependence:
        # every endpoint exists before any route is asked for. The delay budget is
        # unchanged in total, since it was already debited by every placement's
        # processing delay before the first route.
        for fe in fragment.intra_flows:
            src_node = placements.get(fe.source_vnf)
            dst_node = placements.get(fe.target_vnf)
            if src_node is None or dst_node is None:
                continue  # the edge leaves this fragment; the coordinator routes it
            flow_key = (fe.source_vnf, fe.target_vnf)
            if src_node == dst_node:
                # Co-located on one node: no link, no bandwidth, no propagation.
                # Recorded as an empty route so the flow is visibly accounted for.
                routes[flow_key] = []
                continue

            result = route_flow(
                substrate, src_node, dst_node,
                bw_demand=fe.bandwidth_demand,
                delay_budget=delay_remaining,
                domain_node_ids=domain_nodes,
                k=self.k_paths,
                selector=self.routing_selector,
            )

            if not result.feasible:
                self._rollback(substrate, allocated_bw, node_snapshots)
                return DomainResponse(
                    domain_id=self.domain_id, feasible=False,
                )

            routes[flow_key] = result.path_links
            bw_allocated[flow_key] = fe.bandwidth_demand
            total_route_delay += result.propagation_delay
            delay_remaining -= result.propagation_delay

            allocate_route_bw(substrate, result.path_links, fe.bandwidth_demand)
            allocated_bw.append((result.path_links, fe.bandwidth_demand))

        return DomainResponse(
            domain_id=self.domain_id,
            feasible=True,
            placements=placements,
            routes=routes,
            bw_allocated=bw_allocated,
            intra_delay=total_proc_delay + total_route_delay,
            resource_cost=total_resource_cost,
            actions=[],
            log_probs=torch.tensor([]),
            entropy=0.0,
            step_records=[],
        )

    def _select_node(
        self,
        substrate: SubstrateNetwork,
        vnf: VNFAssignment,
        domain_nodes: list[str],
        delay_remaining: float | None = None,
        vnfs_remaining: int = 1,
    ) -> str | None:
        """Lowest-delay feasible node. Tier match is a hard constraint.

        This used to be best-fit on CPU alone and was completely DELAY-BLIND,
        while `proc_delay = node.processing_delay * computational_intensity` is
        the dominant term in `intra_delay` and the one the ground-truth verifier
        then tests. Measured consequence (2026-08-20,
        docs/RL_DIAGNOSIS_2026-08-20.md §8): `post_commit_c7_delay` ran 396-464
        of 2000 arrivals for EVERY MDO approach at L3, invariant across
        inter-domain hops from 0.0 to 1.34 and across `actor_infeasible` from 1
        to 587 -- untouched by anything the orchestrator decides, because it was
        decided here. It was the largest rejection bin in the system, and the
        only placer in the tree that never looked at delay was the one choosing
        the nodes whose delay it is made of.

        Four rules were measured over 3 seeds x L2/L3/L4, `MDO-partial` /
        follow_prior, changing nothing but this ranking key:

            rule              L2      L3      L4
            best-fit (old)   base    base    base
            min-delay       +10.5    +4.9    +3.0     <- shipped
            delay-then-fit   +9.7    +5.7    +1.8
            fit-then-delay   +0.3    -1.0    -1.1

        Min-delay wins two of the three levels and is positive in all nine
        cells. `delay-then-fit` (min delay, best-fit as the tie-break) is within
        0.4 pp of it on average and wins at L3, so the choice between those two
        is NOT load-bearing and should not be read as tuned; what matters is
        that delay enters the key at all.

        A filter form was also tried -- keep only nodes whose delay fits the
        remaining budget, then best-fit among them -- in both a whole-budget and
        an equal-share variant. Both are inert (+0.05 pp at L3): the residual
        e2e budget is large relative to any SINGLE node's delay, so every
        candidate passes and the rule collapses to the old best-fit. Each VNF
        fits; the chain total does not. Greedy minimisation is what reduces the
        accumulating sum, which is why it is a ranking key and not a filter.

        SCOPE, and it matters for how results are read: this actor is the SHARED
        executor. Every approach places through it, and `MDO-partial` /
        `MDO-fullobs` are ORION's own ablations rather than external baselines,
        so this change lifts EVERY row at once -- baselines included. It is
        applied uniformly by construction and must be reported as its own
        contribution, never folded into a claim about the planner or the policy.

        Cost, stated: this abandons the anti-fragmentation objective, and
        `actor_infeasible` rises with it (L4, seed 42: 136 -> 179). The delay
        gain dominates at every level measured, but that trade is the reason
        this is a ranking change and not a strict improvement.

        `delay_remaining` and `vnfs_remaining` are accepted but unused by this
        rule; they are the interface a LEARNED actor needs, and the signature is
        fixed here so swapping one in does not change the call site.
        """
        g = substrate.graph
        permitted = set(vnf.permitted_nodes) if vnf.permitted_nodes else None
        required_tier = vnf.required_tier
        candidates: list[tuple[int, float, str]] = []

        for node_id in domain_nodes:
            if permitted is not None and node_id not in permitted:
                continue
            d = g.nodes[node_id]
            cpu_avail = float(d["cpu_residual"])
            ram_avail = float(d["ram_residual"])
            if cpu_avail < vnf.cpu_demand or ram_avail < vnf.ram_demand:
                continue

            tier_match = 1 if d["tier"] == required_tier.value else 0
            proc_delay = (
                float(d["processing_delay"]) * float(vnf.computational_intensity)
            )
            # Ascending sort; node_id is the deterministic tiebreak, as before.
            candidates.append((-tier_match, proc_delay, node_id))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][2]

    def _rollback(
        self,
        substrate: SubstrateNetwork,
        allocated_bw: list[tuple[list[str], float]],
        node_snapshots: dict[str, tuple[float, float]],
    ) -> None:
        for path_links, bw in reversed(allocated_bw):
            deallocate_route_bw(substrate, path_links, bw)
        g = substrate.graph
        for node_id, (orig_cpu, orig_ram) in node_snapshots.items():
            g.nodes[node_id]["cpu_residual"] = orig_cpu
            g.nodes[node_id]["ram_residual"] = orig_ram
