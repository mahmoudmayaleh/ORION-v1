"""Plan cache tests.

The single most important test here is `test_cache_size_stays_small` —
it pins the contract that the cache signature is substrate-independent
so the unbounded design assumption holds. If this ever fails, the design
choice E1 (no eviction) is broken and an eviction policy must be added.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from orion.llm.plan_cache import PlanCache, qos_bucket, signature
from orion.types import QoSRequirements, SliceRequest, SliceType


@dataclass
class _FakePlan:
    payload: str


def _slice(slice_type: SliceType, delay: float) -> SliceRequest:
    return SliceRequest(
        request_id="r",
        slice_type=slice_type,
        vnfs=[],
        flow_edges=[],
        qos=QoSRequirements(max_e2e_delay=delay, min_throughput=10.0),
        arrival_time=0.0,
        lifetime=0.0,
    )


class TestSignature:
    def test_qos_bucket_partitions(self) -> None:
        assert qos_bucket(QoSRequirements(max_e2e_delay=5.0, min_throughput=1.0)) == "tight"
        assert qos_bucket(QoSRequirements(max_e2e_delay=20.0, min_throughput=1.0)) == "medium"
        assert qos_bucket(QoSRequirements(max_e2e_delay=200.0, min_throughput=1.0)) == "loose"

    def test_signature_is_substrate_independent(self) -> None:
        # Same slice type + delay → same signature, regardless of anything else.
        a = signature(_slice(SliceType.EMBB, 30.0))
        b = signature(_slice(SliceType.EMBB, 30.0))
        assert a == b


class TestCache:
    def test_put_then_get_returns_entry(self) -> None:
        cache: PlanCache[_FakePlan] = PlanCache()
        cache.put(("eMBB", "medium"), _FakePlan("plan-1"))
        entry = cache.get(("eMBB", "medium"))
        assert entry is not None
        assert entry.plan.payload == "plan-1"
        assert entry.hit_count == 1

    def test_stale_entry_returns_none_via_get_but_persists(self) -> None:
        cache: PlanCache[_FakePlan] = PlanCache()
        key = ("URLLC", "tight")
        cache.put(key, _FakePlan("v1"))
        cache.mark_stale(key)
        assert cache.get(key) is None  # get treats stale as missing
        # The entry is still there for refresh().
        assert len(cache) == 1

    def test_refresh_clears_stale_and_replaces(self) -> None:
        cache: PlanCache[_FakePlan] = PlanCache()
        key = ("V2X", "tight")
        cache.put(key, _FakePlan("v1"))
        cache.mark_stale(key)
        cache.refresh(key, _FakePlan("v2"))
        entry = cache.get(key)
        assert entry is not None
        assert entry.plan.payload == "v2"
        assert entry.miss_after_stale_count == 1


class TestSizeContract:
    def test_cache_size_stays_small_over_many_slice_types(self) -> None:
        """The single most important guard: with the coarse signature, the
        cache must never grow beyond ~15 entries, regardless of how many
        slices are processed.
        """
        cache: PlanCache[_FakePlan] = PlanCache()
        import random

        rng = random.Random(0)
        for i in range(10_000):
            stype = rng.choice(list(SliceType))
            delay = rng.uniform(1.0, 500.0)
            slc = _slice(stype, delay)
            key = signature(slc)
            if cache.get(key) is None:
                cache.put(key, _FakePlan(f"p-{i}"))

        # 5 slice types × 3 buckets = 15 keys maximum.
        assert len(cache) <= 15

    def test_oversized_signature_trips_guard(self) -> None:
        """If something extends the signature to include substrate state,
        the size guard must fire. Simulate by inserting many distinct keys.
        """
        cache: PlanCache[_FakePlan] = PlanCache()
        # 16 fits (= _MAX_EXPECTED_SIZE), 17 must trip.
        for i in range(16):
            cache.put((f"st{i}", "x"), _FakePlan("p"))
        with pytest.raises(RuntimeError, match="exceeds expected bound"):
            cache.put(("st17", "x"), _FakePlan("p"))
