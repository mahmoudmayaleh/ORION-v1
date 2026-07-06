"""Plan cache: signature-keyed Agent B plans with stale-flag invalidation.

Per the plan step 2a:
  - Cache hit, entry not marked stale: structural revalidation; if it passes
    the cached plan is reused with no LLM call.
  - Cache hit, entry marked stale: Agent B is invoked once to refresh.
  - Cache miss: Agent B is invoked.

Two signature granularities:
  - `signature` (coarse): (slice_type, qos_bucket) — 15 keys, no eviction.
  - `plan_signature` (fine): (slice_type, qos_bucket, sfc_template) — many keys,
    bounded LRU eviction required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Hashable, TypeVar

from orion.mdo.types import PlanSummary
from orion.types import InfrastructureTier, QoSRequirements, SliceRequest, SliceType

if TYPE_CHECKING:
    from orion.substrate.graph_model import SubstrateNetwork


# ── Signature ────────────────────────────────────────────────────────────


def qos_bucket(qos: QoSRequirements) -> str:
    """Coarse QoS tier for cache keying.

    Three buckets matching the slice-class natural cut points:
        "tight"  : max_e2e_delay < 10 ms   (URLLC, V2X-low)
        "medium" : 10-50 ms                 (V2X-high, XR, eMBB-low)
        "loose"  : > 50 ms                  (eMBB-high, mMTC)
    """
    if qos.max_e2e_delay < 10.0:
        return "tight"
    if qos.max_e2e_delay < 50.0:
        return "medium"
    return "loose"


def signature(slice_req: SliceRequest) -> tuple[str, str]:
    """Coarse substrate-independent signature — (slice_type, qos_bucket).

    5 x 3 = 15 possible keys. Superseded by plan_signature for abstract plan
    caching (the coarse key cannot distinguish chains of different shape).
    """
    return (slice_req.slice_type.value, qos_bucket(slice_req.qos))


def sfc_template(slice_req: SliceRequest) -> tuple[str, ...]:
    """Ordered VNF-type tuple — the structural identity of the chain.

    Two requests with the same template and QoS class share an abstract plan:
    the suggested-domain-per-position vector aligns, and the bandwidth profile
    is implied (per-hop BW multipliers are a function of vnf_type).
    """
    return tuple(v.vnf_type for v in slice_req.vnfs)


def plan_signature(slice_req: SliceRequest) -> tuple[str, str, tuple[str, ...]]:
    """Fine intent-level signature: (slice_type, qos_bucket, sfc_template).

    The canonical key for abstract plan caching and the strategy monitor's
    per-plan drift stream. Exact matching — the ordered VNF-type tuple is the
    keyword set with order; request-specific scalars (cpu/ram/bw demands,
    vnf_ids, beta_in) are instantiated fresh per request, never keyed on.
    """
    return (
        slice_req.slice_type.value,
        qos_bucket(slice_req.qos),
        sfc_template(slice_req),
    )


# ── Abstract plan (the cached, request-invariant core) ───────────────────


@dataclass(frozen=True)
class AbstractPlan:
    """The request-INVARIANT core of an Agent B plan, keyed by plan_signature.

    Only stores the per-chain-position tier + domain assignment. Request-specific
    scalars (vnf_ids, cpu/ram, vcrs, bw) are grafted on per arrival via
    instantiate_plan.
    """

    sfc_template: tuple[str, ...]
    required_tiers: list[InfrastructureTier]
    suggested_domains: list[int]


def instantiate_plan(abstract: AbstractPlan, slice_req: SliceRequest) -> PlanSummary:
    """Graft a concrete request's scalars onto a cached AbstractPlan."""
    types = tuple(v.vnf_type for v in slice_req.vnfs)
    if types != abstract.sfc_template:
        raise ValueError(
            f"instantiate_plan template drift: request {types} != cached "
            f"{abstract.sfc_template}"
        )
    return PlanSummary(
        vnf_ids=[v.vnf_id for v in slice_req.vnfs],
        required_tiers=list(abstract.required_tiers),
        suggested_domains=list(abstract.suggested_domains),
        cpu_demands=[v.cpu_demand for v in slice_req.vnfs],
        ram_demands=[v.ram_demand for v in slice_req.vnfs],
        vcrs=[v.vcr for v in slice_req.vnfs],
        bw_demands=[f.bandwidth_demand for f in slice_req.flow_edges],
    )


def revalidate_plan(plan: PlanSummary, substrate: "SubstrateNetwork") -> bool:
    """Cheap structural revalidation of a reused plan vs the current substrate.

    Checks that every required tier is still supported by its suggested domain.
    Residual-sensitive checks (C4, C8) are enforced downstream by the
    coordinator and verifier.
    """
    present = {
        (int(d.get("domain_id", -1)), d.get("tier"))
        for _, d in substrate.graph.nodes(data=True)
    }
    return all(
        (dom, tier.value) in present
        for tier, dom in zip(plan.required_tiers, plan.suggested_domains)
    )


# ── Cache entry ──────────────────────────────────────────────────────────


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """One cached plan with bookkeeping fields."""

    plan: T
    stale: bool = False
    hit_count: int = 0
    miss_after_stale_count: int = 0


# ── Cache ────────────────────────────────────────────────────────────────


@dataclass
class PlanCache(Generic[T]):
    """Signature-keyed plan cache with two size disciplines.

    - capacity=None (default): unbounded, for the coarse signature (<=15 keys).
    - capacity set: bounded LRU, for the finer plan_signature (many keys).
    """

    _MAX_EXPECTED_SIZE: int = 16

    capacity: int | None = None
    entries: dict[Hashable, CacheEntry[T]] = field(default_factory=dict)

    def get(self, key: Hashable) -> CacheEntry[T] | None:
        entry = self.entries.get(key)
        if entry is None or entry.stale:
            return None
        entry.hit_count += 1
        self._touch(key)
        return entry

    def put(self, key: Hashable, plan: T) -> None:
        self.entries[key] = CacheEntry(plan=plan)
        self._enforce_size_invariant()

    def refresh(self, key: Hashable, plan: T) -> None:
        entry = self.entries.get(key)
        if entry is None:
            self.put(key, plan)
            return
        entry.plan = plan
        entry.stale = False
        entry.miss_after_stale_count += 1
        self._touch(key)

    def mark_stale(self, key: Hashable) -> None:
        entry = self.entries.get(key)
        if entry is not None:
            entry.stale = True

    def __len__(self) -> int:
        return len(self.entries)

    def _touch(self, key: Hashable) -> None:
        if self.capacity is not None and key in self.entries:
            self.entries[key] = self.entries.pop(key)

    def _enforce_size_invariant(self) -> None:
        if self.capacity is None:
            if len(self.entries) > self._MAX_EXPECTED_SIZE:
                raise RuntimeError(
                    f"PlanCache size {len(self.entries)} exceeds expected bound "
                    f"{self._MAX_EXPECTED_SIZE}. Set capacity= for bounded LRU."
                )
            return
        while len(self.entries) > self.capacity:
            lru_key = next(iter(self.entries))
            del self.entries[lru_key]
