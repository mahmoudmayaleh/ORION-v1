"""PPO update for the per-domain placement actors (2026-08-20, RL_DIAGNOSIS §9).

Until now no experiment ever trained a domain actor: `build_stack` created no
actor optimizer and every run placed with the frozen `GreedyDomainActor`. The
2026-08-20 diagnosis showed the actor's node choice -- not the MDO partition --
owns the largest rejection bin (`post_commit_c7_delay`, 396-464/2000 at L3 for
every approach), so this module is where the RL's headroom actually is.

Design, and where each choice comes from:

* **CTDE with one centralised critic** (MAPPO; Yu et al., NeurIPS 2022,
  arXiv:2103.01955). The critic is the existing `CentralisedCritic` over s_t --
  which since 2026-08-20 sees the arriving request -- and the actors reuse the
  SAME arrival-level advantages the MDO update uses, mapped by request_id. No
  per-agent critics.

* **Sequential update with inter-agent correction** (HAPPO; Zhong, Kuba et al.,
  "Heterogeneous-Agent Reinforcement Learning", JMLR 25(32), 2024). Agents
  update one at a time in a fresh random permutation each round; after agent i
  updates, its post-update ratio on each arrival multiplies into a per-arrival
  correction factor M(rid), and every later agent optimises `M(rid) * A(rid)`.
  This is the multi-agent advantage decomposition made tractable, and it is
  exact here because domain fragments of one arrival are DISJOINT, so the joint
  policy factorises across domains. The codebase already cites HARL for its
  no-weight-sharing actors; this makes the training side match. On mostly
  colocated traffic the correction rarely fires (one domain per arrival), which
  makes it cheap, not pointless: split arrivals are exactly the ones where one
  actor's update changes what the other should have done.

* **One ratio per fragment, log-probs summed over its VNFs** (the multi-discrete
  rule from "The 37 Implementation Details of PPO", Huang et al., ICLR Blog
  Track 2022). The fragment's placement is one composite action under a chain
  rule; per-step ratios with a shared fragment advantage would misweight long
  chains.

* **Replay-based re-evaluation.** `ActorStepRecord` stores the exact per-step
  observation (the graph snapshot INCLUDING `has_placed_vnf` and the running
  action mask), so re-evaluating a recorded step under the current policy is
  correct without re-simulating the autoregressive rollout: the conditioning is
  baked into the recorded inputs.

* The rest of the 37-details checklist as it applies: per-MINIBATCH advantage
  normalisation, Adam eps 1e-5 (set where the optimizer is built), global grad
  clip 0.5, approx-KL early stop per agent (the sequential scheme's cheap trust
  region), entropy bonus on the masked distribution (the mask is genuine
  feasibility, not advice -- unlike the MDO's advisory bias, there is no
  channel here for the entropy term to fight). Value-loss clipping is NOT used:
  both Andrychowicz et al. and the blog find it neutral-to-harmful, and the
  critic path here is the already-shipped Huber-on-normalised-targets (§O.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ActorPPOConfig:
    update_epochs: int = 4
    minibatch_size: int = 32          # fragments per minibatch
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    # Approx-KL early stop, computed as mean((ratio-1) - log ratio) (the k3
    # estimator, unbiased and always positive). 0.02 is the conventional
    # PPO2 stop; the sequential scheme makes this the per-agent trust region.
    target_kl: float = 0.02
    # Per-minibatch advantage re-normalisation (37-details). The advantages
    # arrive already whitened at arrival level; this recentres each minibatch.
    normalize_adv_per_minibatch: bool = True


@dataclass
class ActorUpdateStats:
    """Telemetry for one round's actor update, aggregated over agents."""
    steps: int = 0                    # optimizer steps taken
    clip_frac: float = 0.0            # fraction of fragment ratios clipped
    approx_kl: float = 0.0            # mean over agents' final epochs
    entropy: float = 0.0              # mean fragment entropy (nats/VNF step)
    null_rate: float = 0.0            # fraction of steps that chose NULL
    fragments: int = 0                # fragments seen across all agents
    early_stops: int = 0              # agents that hit target_kl
    per_agent_kl: dict = field(default_factory=dict)
    # Fragments per domain. A domain with 0 gets no gradient and keeps its
    # RANDOM initialisation -- correct when it is never dispatched to (on the
    # conventional mix the central-cloud-only domain receives zero fragments),
    # and a live hazard if the mix ever changes, because a random pointer actor
    # would then be placing real slices. Published so that is visible rather
    # than inferred.
    fragments_by_domain: dict = field(default_factory=dict)
    untrained_domains: list = field(default_factory=list)


def _fragment_new_logprob(policy, steps):
    """Sum of current-policy log-probs over a fragment's recorded steps.

    Also returns the summed entropy (gradient-carrying) and the NULL count.
    """
    lp_sum = None
    ent_sum = None
    nulls = 0
    for rec in steps:
        lp, ent = policy.evaluate_step(
            rec.graph_data, rec.vnf_context, rec.action_mask, rec.action_idx)
        lp_sum = lp if lp_sum is None else lp_sum + lp
        ent_sum = ent if ent_sum is None else ent_sum + ent
        if rec.action_idx == policy.NULL_ACTION:
            nulls += 1
    return lp_sum, ent_sum, nulls


def update_domain_actors(
    domain_rollout: dict[int, list],
    rid_to_adv: dict[str, float],
    actors: dict[int, object],
    optimizers: dict[int, torch.optim.Optimizer],
    cfg: ActorPPOConfig | None = None,
    generator: torch.Generator | None = None,
) -> ActorUpdateStats:
    """One HAPPO round over every trainable domain actor.

    Args:
        domain_rollout: `MultiAgentRollout.domain_actor` -- per-domain lists of
            `DomainActorTransition`, each carrying its `ActorStepRecord`s.
        rid_to_adv: request_id -> arrival-level advantage (the SAME values the
            MDO update consumed; whitened at arrival level).
        actors: domain_id -> actor. Only actors with a `.policy` attribute and
            an optimizer entry are updated; frozen greedy actors pass through.
        optimizers: domain_id -> Adam over that actor's policy parameters.
        generator: torch.Generator for the permutation/shuffles (seeded by the
            caller so the round is reproducible).

    Returns:
        ActorUpdateStats.
    """
    cfg = cfg or ActorPPOConfig()
    stats = ActorUpdateStats()

    trainable = [d for d, a in actors.items()
                 if d in optimizers and getattr(a, "policy", None) is not None]
    if not trainable:
        return stats
    stats.fragments_by_domain = {
        d: sum(1 for tr in domain_rollout.get(d, []) if tr.steps)
        for d in trainable}
    stats.untrained_domains = sorted(
        d for d, n in stats.fragments_by_domain.items() if n == 0)

    # Fragments per agent: (old_logprob_sum, steps, request_id). A fragment
    # with no recorded steps (empty fragment / greedy response) is skipped.
    per_agent: dict[int, list] = {}
    for d in trainable:
        frags = []
        for tr in domain_rollout.get(d, []):
            if not tr.steps or tr.request_id not in rid_to_adv:
                continue
            old_lp = float(sum(r.log_prob for r in tr.steps))
            frags.append((old_lp, tr.steps, tr.request_id))
        if frags:
            per_agent[d] = frags
    if not per_agent:
        return stats

    # HAPPO: fresh random agent order each call, and a per-arrival correction
    # factor that accumulates the already-updated agents' post-update ratios.
    order = [list(per_agent)[i] for i in
             torch.randperm(len(per_agent), generator=generator).tolist()]
    correction: dict[str, float] = {}

    clip_hits = 0
    ratio_count = 0
    ent_total = 0.0
    ent_steps = 0
    null_total = 0
    kl_by_agent: dict[int, float] = {}

    for d in order:
        policy = actors[d].policy
        opt = optimizers[d]
        frags = per_agent[d]
        stats.fragments += len(frags)
        n = len(frags)
        mb = cfg.minibatch_size if cfg.minibatch_size > 0 else n

        stopped = False
        agent_kl = 0.0
        for _ in range(cfg.update_epochs):
            if stopped:
                break
            perm = torch.randperm(n, generator=generator).tolist()
            for lo in range(0, n, mb):
                batch = [frags[i] for i in perm[lo:lo + mb]]

                # Per-minibatch advantage (re)normalisation, with the HAPPO
                # correction applied BEFORE normalising so the correction
                # changes direction, not just scale.
                advs = torch.tensor(
                    [correction.get(rid, 1.0) * rid_to_adv[rid]
                     for _, _, rid in batch], dtype=torch.float32)
                if cfg.normalize_adv_per_minibatch and len(advs) > 1 \
                        and float(advs.std()) > 1e-8:
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                loss = torch.tensor(0.0)
                kl_accum = 0.0
                for j, (old_lp, steps, _rid) in enumerate(batch):
                    new_lp, ent, nulls = _fragment_new_logprob(policy, steps)
                    logratio = new_lp - old_lp
                    ratio = torch.exp(logratio)
                    adv_j = float(advs[j])
                    surr = -torch.min(
                        ratio * adv_j,
                        torch.clamp(ratio, 1 - cfg.clip_eps,
                                    1 + cfg.clip_eps) * adv_j)
                    loss = loss + surr - cfg.entropy_coef * (ent / max(1, len(steps)))
                    with torch.no_grad():
                        r = float(ratio)
                        kl_accum += (r - 1.0) - float(logratio)  # k3 estimator
                        clip_hits += int(abs(r - 1.0) > cfg.clip_eps)
                        ratio_count += 1
                        ent_total += float(ent)
                        ent_steps += len(steps)
                        null_total += nulls

                opt.zero_grad()
                (loss / max(1, len(batch))).backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), cfg.max_grad_norm)
                opt.step()
                stats.steps += 1

                agent_kl = kl_accum / max(1, len(batch))
                if cfg.target_kl is not None and agent_kl > cfg.target_kl:
                    stopped = True
                    stats.early_stops += 1
                    break

        kl_by_agent[d] = agent_kl

        # HAPPO correction: this agent's POST-update ratio on each of its
        # arrivals scales what every later agent in the permutation optimises.
        with torch.no_grad():
            for old_lp, steps, rid in frags:
                new_lp, _, _ = _fragment_new_logprob(policy, steps)
                r = float(torch.exp(new_lp - old_lp))
                # Clamp: a wild ratio from one agent must not blow up the
                # next agent's objective (HAPPO uses the clipped surrogate for
                # the same reason).
                correction[rid] = correction.get(rid, 1.0) * max(
                    1 - cfg.clip_eps, min(1 + cfg.clip_eps, r))

    stats.clip_frac = clip_hits / max(1, ratio_count)
    stats.approx_kl = sum(kl_by_agent.values()) / max(1, len(kl_by_agent))
    stats.entropy = ent_total / max(1, ent_steps)
    stats.null_rate = null_total / max(1, ent_steps)
    stats.per_agent_kl = kl_by_agent
    return stats
