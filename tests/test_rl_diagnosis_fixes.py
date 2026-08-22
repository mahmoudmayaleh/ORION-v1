"""Guards for the four 2026-08-20 RL fixes (docs/RL_DIAGNOSIS_2026-08-20.md).

Every one of these pins a property that, when it broke, produced a run that
COMPLETED and reported a plausible acceptance number. That is the failure mode
this project keeps rediscovering, so each fix gets a test that fails loudly
rather than a comment saying it was done.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from orion.config import MDO_DELAY_REF
from orion.mdo.observation import (
    OBS_VERSION,
    SLICE_FEAT_DIM,
    VNF_FEAT_DIM,
    build_domain_summaries,
    build_inter_domain_links,
    build_mdo_observation,
    observation_to_tensor,
)
from orion.mdo.policy import AutoregMDOPolicy
from orion.mdo.types import PlanSummary
from orion.substrate.hierarchical_topology import generate_hierarchical_topology
from orion.sim.slice_generator import generate_slice_request
from orion.training.global_state import (
    REQUEST_FEAT_DIM,
    GlobalStateStats,
    encode_global_state,
    probe_global_state_dim,
)
from orion.types import InfrastructureTier, QoSRequirements


@pytest.fixture(scope="module")
def sub():
    return generate_hierarchical_topology(0)


@pytest.fixture(scope="module")
def req(sub):
    rng = np.random.default_rng(0)
    return generate_slice_request("r0", sub, rng, arrival_time=0.0, lifetime=20.0)


def _plan(sub, sr):
    first_tier = lambda v: InfrastructureTier(
        sub.graph.nodes[sorted(v.permitted_nodes)[0]]["tier"])
    return PlanSummary(
        vnf_ids=[v.vnf_id for v in sr.vnfs],
        required_tiers=[first_tier(v) for v in sr.vnfs],
        suggested_domains=[0] * len(sr.vnfs),
        cpu_demands=[v.cpu_demand for v in sr.vnfs],
        ram_demands=[v.ram_demand for v in sr.vnfs],
        vcrs=[v.vcr for v in sr.vnfs],
        bw_demands=[f.bandwidth_demand for f in sr.flow_edges])


# ── Fix 1: the observation ────────────────────────────────────────────────


def test_vnf_demand_and_h_m_share_a_normaliser(sub, req):
    """The fit test `d_k <= h^m` must be a FUNCTION OF THE TENSOR.

    Per-VNF demand used to divide by max(plan.cpu_demands), a per-arrival scalar
    that appeared nowhere in the observation -- 137 distinct values over 2.00 to
    16.00 CPU on one measured L3 stream -- so the largest VNF of every chain
    encoded as exactly 1.0 whether it needed 2 CPU or 16, and no setting of the
    weights could answer the question the domain actor then failed on.
    """
    plan = _plan(sub, req)
    t = observation_to_tensor(
        build_mdo_observation(sub, plan, slice_req=req), max_vnfs=10)

    domains = build_domain_summaries(sub)
    n_dom = len(domains)
    n_links = len(build_inter_domain_links(sub))
    max_cpu_cap = max(s.cpu_capacity for s in domains)

    dom_block = 17 * n_dom
    plan_off = dom_block + 3 * n_links
    stride = VNF_FEAT_DIM + n_dom

    for k in range(plan.num_vnfs):
        emitted = float(t[plan_off + k * stride])
        assert emitted == pytest.approx(plan.cpu_demands[k] / max_cpu_cap), (
            f"VNF {k} cpu slot is not on the substrate normaliser")

    # The regression this replaces: the chain's largest VNF encoding as 1.0.
    largest = max(float(t[plan_off + k * stride]) for k in range(plan.num_vnfs))
    assert largest < 0.5, (
        "the largest VNF still encodes near 1.0 -- demand is back on a "
        "per-arrival normaliser and the fit test is unexpressible again")

    # h^m rides on the same constant, which is what makes them comparable.
    hm = float(t[5 + 2])  # domain 0, first tier, tier_max_node_cpu
    assert hm == pytest.approx(
        domains[0].tier_max_node_cpu.get(list(domains[0].tier_max_node_cpu)[0], 0.0)
        / max_cpu_cap)


def test_delay_budget_reaches_the_tensor(sub, req):
    """`post_commit_c7_delay` is the largest rejection bin; the budget it tests
    against was absent from the observation entirely until 2026-08-20."""
    plan = _plan(sub, req)

    def emit(delay):
        obs = build_mdo_observation(sub, plan)
        obs.qos = QoSRequirements(max_e2e_delay=delay, min_throughput=10.0)
        return observation_to_tensor(obs, max_vnfs=10)

    tight, loose = emit(5.0), emit(300.0)
    assert not torch.equal(tight, loose), (
        "two slices 60x apart in delay budget produce an identical observation")

    slice_block = slice(-SLICE_FEAT_DIM, None)
    assert float(tight[slice_block][0]) == pytest.approx(5.0 / MDO_DELAY_REF)
    # The linear slot saturates on the loose tail; the log slot must still order it.
    assert float(loose[slice_block][0]) == pytest.approx(1.0)
    assert float(loose[slice_block][1]) > float(tight[slice_block][1])
    assert float(loose[slice_block][1]) == pytest.approx(
        math.log1p(300.0) / math.log1p(MDO_DELAY_REF))


def test_missing_qos_zeros_the_block_without_changing_the_width(sub, req):
    """A width probe may omit the request; it must not change the shape."""
    plan = _plan(sub, req)
    with_q = observation_to_tensor(
        build_mdo_observation(sub, plan, slice_req=req), max_vnfs=10)
    without = observation_to_tensor(
        build_mdo_observation(sub, plan), max_vnfs=10)
    assert with_q.shape == without.shape
    assert torch.equal(without[-SLICE_FEAT_DIM:], torch.zeros(SLICE_FEAT_DIM))


# ── Fix 2: the critic's s_t ───────────────────────────────────────────────


def test_global_state_carries_the_arriving_request(sub, req):
    """V_phi(s_t) cannot value an arrival it cannot see.

    Measured before this landed: a linear read of the whole 86-d s_t explained
    R^2 = 0.038 of the RL reward, against 0.215 for the slice features it
    excluded. After: 0.344.
    """
    stats = GlobalStateStats(total_arrivals=10, max_arrivals=2000)
    bare = encode_global_state(sub, stats)
    withreq = encode_global_state(sub, stats, slice_req=req)

    assert bare.numel() == withreq.numel() == probe_global_state_dim(sub)
    assert not torch.equal(bare, withreq), (
        "s_t is identical with and without the arriving request")
    assert torch.equal(bare[-REQUEST_FEAT_DIM:], torch.zeros(REQUEST_FEAT_DIM))

    # Two different arrivals at the SAME substrate state must differ in s_t --
    # that is the whole point, and it is what makes the advantage action-relevant.
    rng = np.random.default_rng(7)
    other = generate_slice_request("r1", sub, rng, arrival_time=0.0, lifetime=20.0)
    assert not torch.equal(
        encode_global_state(sub, stats, slice_req=req),
        encode_global_state(sub, stats, slice_req=other))


# ── Fix 3: entropy on the raw policy ──────────────────────────────────────


def test_entropy_is_the_raw_policys_not_the_advised_one():
    """Maximising the ADVISED entropy has its fixed point at raw[m~] = -w.

    So the entropy bonus used to be a systematic force cancelling the advisory
    channel, at up to ENT_COEF_MAX = 0.5 * 0.456 per step, competing with a PPO
    term whose whitened advantage has mean zero.
    """
    torch.manual_seed(0)
    pol = AutoregMDOPolicy(obs_dim=239, num_domains=5, max_vnfs=10)
    obs = torch.randn(239) * 0.3
    tm = torch.ones(3, 5, dtype=torch.bool)
    pri = torch.zeros(3, 5)
    pri[:, 0] = 1.0
    act = torch.tensor([0, 0, 0])

    _, ent_advised, _ = pol.evaluate_actions(
        obs, tm, act, 3, prior_logits=pri, prior_weight=2.0)
    _, ent_raw, _ = pol.evaluate_actions(obs, tm, act, 3)
    assert float(ent_advised) == pytest.approx(float(ent_raw), abs=1e-6), (
        "the entropy term still follows the advisory bias, so the bonus is "
        "again paying the policy to disagree with its advisor")

    # An unadvised call must be untouched by the change.
    _, ent_none, _ = pol.evaluate_actions(
        obs, tm, act, 3, prior_logits=None, prior_weight=0.0)
    assert float(ent_none) == pytest.approx(float(ent_raw), abs=1e-9)

    # It must still be a live exploration signal, not a constant.
    pol.zero_grad()
    _, e, _ = pol.evaluate_actions(
        obs, tm, act, 3, prior_logits=pri, prior_weight=2.0)
    e.backward()
    grads = [p.grad for p in pol.decoder.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads), (
        "the entropy term no longer has a gradient -- the floor is inert")


def test_advised_decode_still_follows_the_prior_at_init():
    """The invariant the whole diagnosis rests on.

    An untrained policy in advised mode IS follow_prior, because orthogonal gain
    0.01 puts the raw logits far below ADVISE_WEIGHT. If an initialisation or
    decode change ever breaks this, `.6650` stops being the number the RL has to
    beat and every comparison in the diagnosis is void.
    """
    torch.manual_seed(0)
    pol = AutoregMDOPolicy(obs_dim=239, num_domains=5, max_vnfs=10)
    tm = torch.ones(4, 5, dtype=torch.bool)
    for trial in range(25):
        obs = torch.randn(239) * 0.5
        want = int(torch.randint(5, (1,)).item())
        pri = torch.zeros(4, 5)
        pri[:, want] = 1.0
        part, _, _, _ = pol(obs, tm, 4, deterministic=True,
                            prior_logits=pri, prior_weight=2.0)
        assert part == [want] * 4, (
            f"untrained advised decode deviated from m~ on trial {trial}")


# ── Fix 4 + the version guard ─────────────────────────────────────────────


def _wp7_source() -> str:
    """Read scripts/wp7_runner.py as text.

    Deliberately not an import: `scripts/` is not on the test path and the module
    pulls the whole LLM stack in at import time. These two assertions are about
    what the file SAYS, so reading it is both sufficient and cheaper.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return (root / "scripts" / "wp7_runner.py").read_text(encoding="utf-8")


def test_ppo_minibatch_is_wired_not_declared():
    """`MAPPOConfig.minibatch_size = 64` was read by nothing in the tree.

    With it dead the optimizer stepped once per epoch, four times per round,
    800 times for the whole 200-round curriculum, and PPO's clip could never
    bind because epoch 1 has ratio == 1.0 exactly.
    """
    import re

    from orion.training.config import MAPPOConfig

    text = _wp7_source()
    m = re.search(r'PPO_MINIBATCH = int\(os\.environ\.get\(\s*"ORION_PPO_MINIBATCH",\s*"(\d+)"',
                  text)
    assert m, "PPO_MINIBATCH is no longer defined from the environment"
    default = int(m.group(1))
    assert default > 0, (
        "minibatching is off by default again; the update is back to four "
        "vanilla policy-gradient steps per round")
    assert default == MAPPOConfig().minibatch_size, (
        "the wired default has drifted from the declared MAPPOConfig value")

    # The optimizer step must sit inside the minibatch loop, not the epoch loop.
    assert "for _mb in _batches:" in text
    step_at = text.index("opt_mdo.step()")
    mb_at = text.index("for _mb in _batches:")
    assert mb_at < step_at, "the optimizer step escaped the minibatch loop"
    assert '"ppo_clip_frac"' in text, "clip fraction is not on the round curve"

    # The step count the curriculum actually gets, at the as-run shape.
    n, epochs = 500, MAPPOConfig().update_epochs
    per_round = epochs * math.ceil(n / default)
    assert per_round >= 8 * epochs, "fewer steps per round than 8x the old rate"


def test_obs_version_is_stamped_and_checked():
    """A width change fails loudly in load_state_dict; a same-width change of
    MEANING would not, and checkpoint filenames encode only
    (scenario, config, seed, segment)."""
    from pathlib import Path

    assert OBS_VERSION >= 2
    text = _wp7_source()
    assert '"obs_version": OBS_VERSION' in text, "checkpoints do not stamp it"
    assert "refusing to warm-start" in text, "no guard on a mismatched warm start"

    grid = (Path(__file__).resolve().parents[1]
            / "scripts" / "grid_runner.py").read_text(encoding="utf-8")
    assert "obs_version" in grid, "eval-only path does not check the stamp"


# ── Fix 5: the domain actor's node choice ─────────────────────────────────


def test_node_selection_is_delay_aware(sub):
    """`GreedyDomainActor._select_node` must rank on processing delay.

    It ranked on tightest CPU fit and never looked at delay, while
    `proc_delay = node.processing_delay * computational_intensity` is the
    dominant term in `intra_delay` and the one the verifier tests.
    `post_commit_c7_delay` was 396-464 of 2000 arrivals for EVERY MDO approach
    at L3 -- invariant to the partition, because it was decided here.

    Measured over 3 seeds x L2/L3/L4 on `MDO-partial`: +10.5 / +4.9 / +3.0 pp,
    positive in all nine cells.

    SCOPE: this actor is the SHARED executor, so the change lifts every row
    including the baselines, and `MDO-partial` / `MDO-fullobs` are ORION's own
    ablations. It is applied uniformly by construction and must be reported as
    its own contribution, never folded into a planner or policy claim.
    """
    from orion.actors.greedy_domain_actor import GreedyDomainActor
    from orion.actors.types import VNFAssignment

    g = sub.graph
    same_tier = None
    for dom in range(sub.num_domains):
        by_tier: dict[str, list[str]] = {}
        for n in sorted(sub.nodes_in_domain(dom)):
            by_tier.setdefault(g.nodes[n]["tier"], []).append(n)
        for t, ns in by_tier.items():
            if len(ns) >= 2:
                same_tier, tier = ns[:2], t
                break
        if same_tier:
            break
    assert same_tier, "no domain holds two same-tier nodes"
    tight_slow, roomy_fast = same_tier

    # The tight node is the better BIN-PACK and the worse DELAY choice; the old
    # rule took it every time.
    g.nodes[tight_slow]["cpu_residual"] = 10.0
    g.nodes[tight_slow]["ram_residual"] = 20.0
    g.nodes[tight_slow]["processing_delay"] = 5.0
    g.nodes[roomy_fast]["cpu_residual"] = 100.0
    g.nodes[roomy_fast]["ram_residual"] = 200.0
    g.nodes[roomy_fast]["processing_delay"] = 0.1

    vnf = VNFAssignment(
        vnf_id="v0", vnf_type="Firewall", cpu_demand=8.0, ram_demand=16.0,
        vcr=1.0, bandwidth_in=10.0, position_in_sfc=0, sfc_length=1,
        required_tier=InfrastructureTier(tier),
        permitted_nodes=[tight_slow, roomy_fast],
        computational_intensity=1.0)

    picked = GreedyDomainActor(0)._select_node(sub, vnf, [tight_slow, roomy_fast])
    assert picked == roomy_fast, (
        "node selection is back on tightest-fit and ignoring delay; "
        "post_commit_c7_delay is the largest rejection bin when it does")


def test_actor_delay_budget_counts_processing_delay():
    """`delay_remaining` was decremented only by routing propagation delay.

    So the actor ignored the term it was itself accumulating into `intra_delay`,
    and every `route_flow` after the first VNF got a budget already partly spent.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "src" / "orion" / "actors" / "greedy_domain_actor.py"
           ).read_text(encoding="utf-8")
    assert "delay_remaining -= proc_delay" in src, (
        "the actor's budget tracker no longer charges processing delay")
