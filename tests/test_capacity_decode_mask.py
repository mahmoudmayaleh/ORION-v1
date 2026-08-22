"""§AC — capacity belongs in the decode mask, not the prompt.

Agent B is already told to colocate: the system prompt carries an ordered decision
procedure ("assign EVERY VNF to the single domain with the largest margin, and
stop... Only if that list is empty may you split") and the topology already carries
`largest_free_node_by_tier`. Measured on 40 real L3 arrivals (seed 42, M^B off,
cache bypassed, temperature 0): `partial_obs_builder` colocates 100% of chains and
Agent B splits 85%, on arrivals where a whole-chain host always exists.

Restating the derived set as prose is known to backfire (see `_build_user_prompt`:
78.4% -> 80.2% and 77.2% -> 96.4% split rate on paired L3 probes), so the fix goes
into the grammar next to the tier contract.
"""
from __future__ import annotations

import importlib
import os

import pytest


def _topo():
    return {"domains": [
        {"domain_id": "d0", "dominant_tiers": ["edge", "regional_cloud"],
         "cpu_residual": 100.0, "ram_residual": 200.0,
         "largest_free_node_by_tier": {"edge": {"cpu": 20, "ram": 40},
                                       "regional_cloud": {"cpu": 30, "ram": 60}}},
        # too small in aggregate AND holds no regional tier
        {"domain_id": "d1", "dominant_tiers": ["edge"],
         "cpu_residual": 5.0, "ram_residual": 5.0,
         "largest_free_node_by_tier": {"edge": {"cpu": 2, "ram": 2}}},
        {"domain_id": "d2", "dominant_tiers": ["edge", "regional_cloud"],
         "cpu_residual": 500.0, "ram_residual": 900.0,
         "largest_free_node_by_tier": {"edge": {"cpu": 50, "ram": 90},
                                       "regional_cloud": {"cpu": 50, "ram": 90}}},
    ]}


def _sr():
    return {"vnfs": [
        {"vnf_id": "v1", "cpu_demand": 10, "ram_demand": 20,
         "permitted_tiers": ["edge"]},
        {"vnf_id": "v2", "cpu_demand": 10, "ram_demand": 20,
         "permitted_tiers": ["regional_cloud"]},
    ]}


@pytest.fixture
def A_on(monkeypatch):
    monkeypatch.setenv("ORION_CAPACITY_MASK", "1")
    import orion.llm.agent_b as A
    return importlib.reload(A)


@pytest.fixture
def A_off(monkeypatch):
    monkeypatch.delenv("ORION_CAPACITY_MASK", raising=False)
    import orion.llm.agent_b as A
    return importlib.reload(A)


def test_default_is_off_so_banked_cells_reproduce(A_off):
    assert A_off.CAPACITY_MASK is False


def test_hosts_ranked_by_slack_and_exclude_the_infeasible(A_on):
    assert A_on.whole_chain_hosts(_sr(), _topo()) == ["d2", "d0"], (
        "d1 must be excluded (no regional tier, and too small), and the larger "
        "slack must come first")


def test_h_guard_excludes_a_domain_with_ample_aggregate_but_no_big_enough_node(A_on):
    """The case the aggregate cannot see: residual is real but spread thin."""
    topo = _topo()
    topo["domains"][0]["largest_free_node_by_tier"]["edge"] = {"cpu": 1, "ram": 1}
    assert A_on.whole_chain_hosts(_sr(), topo) == ["d2"]


def test_empty_when_no_domain_takes_the_whole_chain(A_on):
    """That is exactly when splitting is legitimate, so the mask must not bind."""
    topo = _topo()
    for d in topo["domains"]:
        d["cpu_residual"] = 1.0
    assert A_on.whole_chain_hosts(_sr(), topo) == []


def test_mask_narrows_every_position_to_whole_chain_hosts(A_on):
    sch = A_on.build_pinned_plan_schema(_sr(), _topo())
    enums = [i["properties"]["domain"]["enum"]
             for i in sch["properties"]["vnf_assignments"]["items"]]
    assert enums == [["d2", "d0"], ["d2", "d0"]]


def test_mask_off_keeps_the_tier_contract_only(A_off):
    sch = A_off.build_pinned_plan_schema(_sr(), _topo())
    enums = [i["properties"]["domain"]["enum"]
             for i in sch["properties"]["vnf_assignments"]["items"]]
    assert enums == [["d0", "d1", "d2"], ["d0", "d2"]], (
        "with the mask off the enum is D(tau) exactly as every banked cell had it")


def test_falls_back_to_tier_contract_when_no_host_exists(A_on):
    topo = _topo()
    for d in topo["domains"]:
        d["cpu_residual"] = 1.0
    sch = A_on.build_pinned_plan_schema(_sr(), topo)
    enums = [i["properties"]["domain"]["enum"]
             for i in sch["properties"]["vnf_assignments"]["items"]]
    assert enums == [["d0", "d1", "d2"], ["d0", "d2"]]


def test_still_returns_none_on_genuine_tier_infeasibility(A_on):
    sr = _sr()
    sr["vnfs"][0]["permitted_tiers"] = ["central_cloud"]
    assert A_on.build_pinned_plan_schema(sr, _topo()) is None


def test_adds_no_information_beyond_its_two_arguments(A_on):
    """The mask is derived from the dicts already handed to the schema builder, so
    it cannot smuggle in substrate state the planner was not shown."""
    import inspect
    src = inspect.getsource(A_on.whole_chain_hosts)
    for leak in ("substrate", "graph", "nodes_in_domain", "cpu_residual_frac"):
        assert leak not in src, f"whole_chain_hosts reads {leak!r}, which is not an argument"


# ── §AD: the colocation contract ──────────────────────────────────────────────

@pytest.fixture
def A_contract(monkeypatch):
    monkeypatch.setenv("ORION_CAPACITY_MASK", "1")
    monkeypatch.setenv("ORION_COLOCATION_CONTRACT", "1")
    import orion.llm.agent_b as A
    return importlib.reload(A)


def test_contract_default_is_off(A_off):
    assert A_off.COLOCATION_CONTRACT is False


def test_contract_emits_one_host_domain_from_the_qualifying_hosts(A_contract):
    sch = A_contract.build_pinned_plan_schema(_sr(), _topo())
    assert sch == {
        "type": "object",
        "properties": {"host_domain": {"enum": ["d2", "d0"]}},
        "required": ["host_domain"],
        "additionalProperties": False,
    }, "a split must be UNREPRESENTABLE, not merely discouraged"


def test_contract_falls_back_to_per_vnf_when_no_host_exists(A_contract):
    """Exactly when a split is the right answer, the model gets to author it."""
    topo = _topo()
    for d in topo["domains"]:
        d["cpu_residual"] = 1.0
    sch = A_contract.build_pinned_plan_schema(_sr(), topo)
    assert "host_domain" not in sch["properties"]
    assert "vnf_assignments" in sch["properties"]


def test_expand_produces_the_shape_every_downstream_stage_expects(A_contract):
    plan = {"host_domain": "d2"}
    assert A_contract.expand_host_domain(plan, _sr()) is True
    assert plan == {"vnf_assignments": [{"vnf_id": "v1", "domain": "d2"},
                                        {"vnf_id": "v2", "domain": "d2"}]}
    assert "host_domain" not in plan, "the raw field must not survive into the plan"


def test_expand_is_a_noop_on_a_per_vnf_plan(A_contract):
    plan = {"vnf_assignments": [{"vnf_id": "v1", "domain": "d0"}]}
    before = dict(plan)
    assert A_contract.expand_host_domain(plan, _sr()) is False
    assert plan == before


def test_expanded_plan_has_exactly_one_domain(A_contract):
    plan = {"host_domain": "d0"}
    A_contract.expand_host_domain(plan, _sr())
    assert len({a["domain"] for a in plan["vnf_assignments"]}) == 1


def test_expand_runs_before_derived_fields_are_filled(A_contract):
    """`fill_derived_fields` and the defensive validator both read
    `vnf_assignments`, so expanding after either would make a contract plan look
    malformed and fire the regression warning on every arrival."""
    import inspect
    src = inspect.getsource(A_contract.AgentB.generate_plan)
    assert src.index("expand_host_domain") < src.index("fill_derived_fields")


def test_contract_alone_implies_the_mask(monkeypatch):
    """ORION_COLOCATION_CONTRACT=1 without ORION_CAPACITY_MASK must still bind.

    The contract is DEFINED in terms of the whole-chain hosts, so gating the host
    computation on CAPACITY_MASK alone would make the contract silently inert -- the
    worst kind of failure, since the run completes and banks a cell.
    """
    monkeypatch.delenv("ORION_CAPACITY_MASK", raising=False)
    monkeypatch.setenv("ORION_COLOCATION_CONTRACT", "1")
    import orion.llm.agent_b as A
    A = importlib.reload(A)
    assert A.CAPACITY_MASK is False and A.COLOCATION_CONTRACT is True
    sch = A.build_pinned_plan_schema(_sr(), _topo())
    assert "host_domain" in sch["properties"]


# ── §AD.1: the competitiveness band ───────────────────────────────────────────

def test_band_default_keeps_every_qualifying_host(A_contract):
    assert A_contract.HOST_SLACK_BAND == 0.0
    assert A_contract.whole_chain_hosts(_sr(), _topo()) == ["d2", "d0"]


def test_band_drops_hosts_far_below_the_best(monkeypatch):
    """d0's slack is far under d2's, so a 0.5 band must leave only d2.

    Without this, Agent B's fixed preference for one domain lets it ride that domain
    down to its margin -- where `fits_a_node` stops being sufficient -- while a much
    emptier host sits unused in the enum.
    """
    monkeypatch.setenv("ORION_COLOCATION_CONTRACT", "1")
    monkeypatch.setenv("ORION_HOST_SLACK_BAND", "0.5")
    import orion.llm.agent_b as A
    A = importlib.reload(A)
    assert A.HOST_SLACK_BAND == 0.5
    assert A.whole_chain_hosts(_sr(), _topo()) == ["d2"]


def test_band_keeps_hosts_that_are_genuinely_close(monkeypatch):
    """The band must not collapse the choice when the hosts really are comparable,
    or the model has no authorship left and ORION is `partial_obs_builder`."""
    monkeypatch.setenv("ORION_COLOCATION_CONTRACT", "1")
    monkeypatch.setenv("ORION_HOST_SLACK_BAND", "0.5")
    import orion.llm.agent_b as A
    A = importlib.reload(A)
    topo = _topo()
    topo["domains"][0]["cpu_residual"] = 480.0   # d0 now close to d2
    topo["domains"][0]["ram_residual"] = 880.0
    assert A.whole_chain_hosts(_sr(), topo) == ["d2", "d0"]


def test_band_never_empties_a_non_empty_host_list(monkeypatch):
    """The best host always survives its own cutoff, at any band."""
    monkeypatch.setenv("ORION_COLOCATION_CONTRACT", "1")
    monkeypatch.setenv("ORION_HOST_SLACK_BAND", "0.99")
    import orion.llm.agent_b as A
    A = importlib.reload(A)
    assert A.whole_chain_hosts(_sr(), _topo()) == ["d2"]
