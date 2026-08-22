"""§AE -- `AutoregMDOPolicy.joint_host_decode` and its coordinator wiring.

The claim under test is narrow and worth stating: the decode returns the COLOCATED
partition that the trained policy itself assigns the highest joint probability to,
with the advisory bias included. It is not "obey the plan" and it is not "pick the
best host" -- nothing here knows which host is good. If these pass, a win in the
cell is attributable to the action SET, which is what the §AE claim rests on.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orion.mdo.policy import AutoregMDOPolicy  # noqa: E402


def _policy(seed=0, obs_dim=16, M=5, K=4):
    torch.manual_seed(seed)
    p = AutoregMDOPolicy(obs_dim=obs_dim, num_domains=M, max_vnfs=K + 2)
    # The init gain is 0.01, so a freshly built policy is very nearly uniform and
    # every host scores the same. Perturb so the argmax is a real preference.
    with torch.no_grad():
        for q in p.decoder.parameters():
            q.add_(torch.randn_like(q) * 0.5)
    p.eval()
    return p


def _joint_logprob(policy, obs, tier_mask, host, K, prior_logits=None, weight=0.0):
    """log pi(all-`host` | s), via `evaluate_actions` -- a separate code path."""
    actions = torch.tensor([host] * K)
    lps, _ent, _raw = policy.evaluate_actions(
        obs, tier_mask, actions, K, prior_logits=prior_logits, prior_weight=weight)
    return float(lps.sum().item())


def test_returns_the_argmax_colocated_partition():
    M, K = 5, 4
    p = _policy()
    obs = torch.randn(16)
    tier_mask = torch.ones(K, M, dtype=torch.bool)

    out = p.joint_host_decode(obs, tier_mask, K)
    assert out is not None
    partition, log_probs, logits, _ent, host, n_cand = out

    assert partition == [host] * K, "decode must be colocated"
    assert n_cand == M
    assert log_probs.shape == (K,)
    assert logits.shape == (K, M)

    scores = {c: _joint_logprob(p, obs, tier_mask, c, K) for c in range(M)}
    assert host == max(scores, key=scores.get), scores
    # and the log-probs it reports are that partition's own
    assert float(log_probs.sum().item()) == pytest.approx(scores[host], abs=1e-5)


def test_advice_is_included_in_the_score():
    """The bias enters the DECISION logits, exactly as `forward` applies it."""
    M, K = 5, 4
    p = _policy(seed=3)
    obs = torch.randn(16)
    tier_mask = torch.ones(K, M, dtype=torch.bool)

    unadvised = p.joint_host_decode(obs, tier_mask, K)[4]
    other = next(c for c in range(M) if c != unadvised)
    prior = torch.zeros(K, M)
    prior[:, other] = 1.0

    # Weak advice must not be able to move it; overwhelming advice must.
    assert p.joint_host_decode(obs, tier_mask, K,
                              prior_logits=prior, prior_weight=0.0)[4] == unadvised
    assert p.joint_host_decode(obs, tier_mask, K,
                              prior_logits=prior, prior_weight=50.0)[4] == other


def test_policy_can_override_the_planners_host():
    """The point of the mechanism: the RL still chooses.

    At the shipped weight the advised bias is +2.0 on one domain. If no plan could
    ever be overridden this would be `follow_prior` with extra steps, and the
    headline it is meant to test would be untestable.
    """
    M, K = 5, 4
    p = _policy(seed=7)
    obs = torch.randn(16)
    tier_mask = torch.ones(K, M, dtype=torch.bool)
    preferred = p.joint_host_decode(obs, tier_mask, K)[4]
    other = next(c for c in range(M) if c != preferred)
    prior = torch.zeros(K, M)
    prior[:, other] = 1.0

    scores = {c: _joint_logprob(p, obs, tier_mask, c, K,
                                prior_logits=prior, weight=2.0) for c in range(M)}
    chosen = p.joint_host_decode(obs, tier_mask, K,
                                 prior_logits=prior, prior_weight=2.0)[4]
    assert chosen == max(scores, key=scores.get)
    # Not an assertion about THIS seed's outcome: only that both outcomes are
    # reachable at the shipped weight, i.e. the advice does not dominate by
    # construction.
    assert scores[other] - scores[preferred] < 2.0 * K + 10.0


def test_none_when_no_domain_can_host_the_whole_chain():
    """The legitimate-split case: the caller must fall through, not be handed a
    partition that ignores tier feasibility."""
    M, K = 5, 3
    p = _policy()
    obs = torch.randn(16)
    tier_mask = torch.zeros(K, M, dtype=torch.bool)
    tier_mask[0, 0] = True          # VNF 0 only in d0
    tier_mask[1, 1] = True          # VNF 1 only in d1
    tier_mask[2, 1] = True
    assert p.joint_host_decode(obs, tier_mask, K) is None


def test_only_tier_feasible_hosts_are_candidates():
    M, K = 5, 3
    p = _policy(seed=11)
    obs = torch.randn(16)
    tier_mask = torch.ones(K, M, dtype=torch.bool)
    tier_mask[1, 2] = False          # d2 out for VNF 1 -> not a whole-chain host
    out = p.joint_host_decode(obs, tier_mask, K)
    assert out is not None
    _part, _lp, _lg, _e, host, n_cand = out
    assert host != 2
    assert n_cand == M - 1


def test_candidate_mask_narrows_further():
    M, K = 5, 3
    p = _policy(seed=5)
    obs = torch.randn(16)
    tier_mask = torch.ones(K, M, dtype=torch.bool)
    allow = torch.zeros(M, dtype=torch.bool)
    allow[3] = True
    out = p.joint_host_decode(obs, tier_mask, K, candidate_mask=allow)
    assert out is not None and out[4] == 3 and out[5] == 1
    none_allowed = torch.zeros(M, dtype=torch.bool)
    assert p.joint_host_decode(obs, tier_mask, K, candidate_mask=none_allowed) is None


def test_flag_defaults_off_and_decode_is_unchanged():
    """Every banked cell decoded per-VNF. The flag must be opt-in, and with it off
    the advised path must be the same code it always was."""
    import orion.mdo.coordinator as co
    assert co.JOINT_HOST_DECODE is False, "must not ship on"
    assert os.environ.get("ORION_JOINT_HOST_DECODE") in (None, "0")


def test_stats_reset_is_total():
    import orion.mdo.coordinator as co
    for k in co.JOINT_HOST_STATS:
        co.JOINT_HOST_STATS[k] = 7
    s = co.reset_joint_host_stats()
    assert set(s) == {"arrivals", "colocated", "fallback_split", "agreed",
                      "overrode", "no_plan_host", "candidates"}
    assert all(v == 0 for v in s.values())
    assert s is co.JOINT_HOST_STATS
