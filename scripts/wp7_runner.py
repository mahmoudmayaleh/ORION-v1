#!/usr/bin/env python3
"""WP7 headline runner: LLM+RL vs RL-alone, MDO in the loop.

Three claim arms, IDENTICAL MDO architecture + IDENTICAL frozen greedy domain
actors (no actor gradient updates — the RL-alone vs LLM+RL gap is attributable
to the LLM, not to actor learning). Only the MDO policy pi^MDO_phi learns.

  RL-alone       : beta=0, NO KL prior, NO Agent B. m~ = deterministic greedy
                   structural plan (LLM-free), used only for the tier mask /
                   obs; NOT as a KL target (beta=0).
  LLM+RL memoff  : KL prior toward Agent B's suggested partition m~, beta
                   linearly decayed. Agent B active + K^B. M^B OFF.
  LLM+RL full    : same as memoff plus M^B episodic ON.

Per-arrival flow (EpisodeRunner): spec -> plan_builder (m~) -> MDO partition
(pi^MDO_phi, mode="sample", n_part=3) -> frozen GreedyDomainActor place -> verify
(E2E/C5b/C7/C9) -> commit/reject -> reward Rt -> PPO update (MDO + critic only).

Metrics (all three): paired eval FoC vs exhaustive ceiling per round (learning
curve), samples-to-threshold, final FoC. num_domains is read from the substrate.

Reduced-scale on-claim usage (1 family, seed 42):
  python scripts/wp7_runner.py --family C-_T-_B- --seed 42 \
      --arrivals 60 --rounds 30 --arms RL-alone LLM+RL-memoff --port 8000
Fast plumbing check (no LLM; LLM arms fall back to greedy m~):
  python scripts/wp7_runner.py --family C-_T-_B- --rounds 2 --arrivals 40 --mock
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import five_arm_runner as R  # domain-parse + AgentB/K^B/M^B setup + ceiling
from orion.actors.greedy_domain_actor import GreedyDomainActor
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.kl_prior import analytical_kl, build_prior_logits, beta_schedule
from orion.mdo.observation import (
    build_domain_summaries,
    build_mdo_observation,
    observation_to_tensor,
)
from orion.mdo.policy import MDOPolicy
from orion.mdo.types import PlanSummary
from orion.profiling import get_collector, profiled
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.training.buffer import PPORolloutBuffer
from orion.training.config import MAPPOConfig
from orion.training.gae import compute_gae
from orion.training.critic import CentralisedCritic
from orion.training.global_state import (
    GlobalStateStats,
    encode_global_state,
    probe_global_state_dim,
)
from orion.training.value_norm import ValueNormalizer
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wp7")

ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
MAX_VNFS = 10
FAM = {f.short_name: f for f in R.ALL_FAMILIES}

LLM_ARMS = {"LLM+RL-memoff", "LLM+RL-full"}
ALL_ARMS = ["RL-alone", "LLM+RL-memoff", "LLM+RL-full"]


def build_delays(substrate):
    delays = {}
    g = substrate.graph
    for u, v, d in g.edges(data=True):
        sd, dd = g.nodes[u]["domain_id"], g.nodes[v]["domain_id"]
        if sd != dd:
            key = (min(sd, dd), max(sd, dd))
            delays[key] = min(delays.get(key, float("inf")), d["propagation_delay"])
    return delays


# ── §R routing-critical injection hooks (None = original family behavior) ────
# Set by scripts/rc_train_runner.py before calling train_arm, so the conformant
# §O trainer runs on the RC-v2 family + cut-sensitive workload with cache-ON,
# WITHOUT perturbing the validated family path. All default None → gate behavior.
RC_SUBSTRATE_FN = None    # callable(seed) -> SubstrateNetwork
RC_SLICE_FACTORY = None   # passed to every training/eval ArrivalProcess
RC_USE_PLAN_CACHE = False  # wrap the LLM plan builder in a signature cache (cache-ON)
# §R Δ2-R (2026-07-15): cache-OFF Full-ORION. The signature cache keys on
# (slice_type, qos_bucket, sfc_template), so distinct chains on one template share a
# cached cut. Cache-OFF was registered to remove that as a variable. NOTE: the
# stronger claim that the cache is *pathological* on this family was WITHDRAWN by
# §R Δ3-R -- the clean re-run supports no family-wide cache claim in either
# direction (it behaves as a lottery over a handful of frozen plans). Cache-OFF
# stands as the registered configuration, not as a remedy for a proven fault.
# To get cache-OFF per-arrival plan quality at feasible cost, FIX the training
# stream across rounds (byte-identical
# every round, matching R.2's single stream exactly) and memoize the LLM plan by
# request_id — a temp-0 LLM on a fixed stream yields a constant plan per arrival, so
# this is ~arrivals calls TOTAL, not arrivals*rounds (25k/seed -> ~100/seed). The
# R.4-vs-R.2 comparison becomes exact same-stream selector isolation. Ratified
# plan_cache/plan_signature untouched (the tier-class key fix stays a deferred Δ).
RC_FIXED_TRAIN_STREAM = False


def _make_sub(fam_name, seed):
    """Training/eval substrate — RC instance if the hook is set, else the family."""
    if RC_SUBSTRATE_FN is not None:
        return RC_SUBSTRATE_FN(seed)
    return R.generate_family_instance(FAM[fam_name], seed=0)


def _make_ap(sub, n, rng):
    return ArrivalProcess(sub, n, ARRIVAL_RATE, SERVICE_RATE, rng,
                          slice_factory=RC_SLICE_FACTORY)


def _cached_plan_builder(inner, plan_cache):
    """Cache-ON wrapper (§6.3): reuse the abstract plan per signature; the LLM is
    called only on a cache miss (~6 signatures on RC → ~6 calls, then served)."""
    from orion.llm.plan_cache import (
        plan_signature, sfc_template, AbstractPlan, instantiate_plan, revalidate_plan)

    def _builder(sr, substrate):
        key = plan_signature(sr)
        entry = plan_cache.get(key)
        if entry is not None and revalidate_plan(entry.plan, substrate):
            try:
                return instantiate_plan(entry.plan, sr)
            except ValueError:
                pass
        ps = inner(sr, substrate)
        if ps is not None:
            ab = AbstractPlan(sfc_template=sfc_template(sr),
                              required_tiers=list(ps.required_tiers),
                              suggested_domains=list(ps.suggested_domains))
            if key in plan_cache.entries:
                plan_cache.refresh(key, ab)
            else:
                plan_cache.put(key, ab)
        return ps
    return _builder


def _slice_memo_key(sr):
    """Content signature of a slice request — everything a deterministic plan
    depends on (substrate is fixed per seed). EXCLUDES request_id (the field that
    collides across the train/eval streams) and INCLUDES permitted_nodes so forced
    and flexible chains never share a key (the Anomaly-A collision, avoided)."""
    vnf_part = tuple(
        (v.vnf_type, round(float(v.cpu_demand), 3), round(float(v.ram_demand), 3),
         round(float(v.vcr), 4), tuple(sorted(v.permitted_nodes)))
        for v in sr.vnfs)
    bw_part = tuple(round(float(f.bandwidth_demand), 3) for f in sr.flow_edges)
    qos = (round(float(sr.qos.max_e2e_delay), 3), round(float(sr.qos.min_throughput), 3))
    return (len(sr.vnfs), vnf_part, bw_part, qos)


def greedy_plan_builder(slice_req, substrate):
    """LLM-free structural plan (m~): greedy FFD -> PlanSummary via graph domain_id."""
    result = _run_greedy_ffd(substrate, slice_req, GreedyConfig())
    if not result.feasible or result.plan is None:
        return None
    return R.plan_to_summary(result, slice_req, substrate)


def make_llm_plan_builder(agent_b, kb, mb_getter):
    """m~ from Agent B (K^B grounded; M^B optional via mb_getter()).

    Returns a PlanSummary whose suggested_domains is Agent B's partition m~
    (the KL-prior target). Returns None on structural-check failure (-> the
    arrival is a structural reject, no MDO transition), matching EpisodeRunner.
    """
    def _builder(slice_req, substrate):
        from orion.llm.agent_b import build_pinned_plan_schema
        abstract_topo = R.build_abstract_topology(substrate)
        sr_dict = R._slice_request_to_dict(slice_req, substrate)
        mb = mb_getter()
        # Per-request enum+length schema (2026-07-10 content fix): pins vnf_id/domain
        # to the real request/topology so grammar-valid m~ can't name a nonexistent
        # VNF or domain on 3+ VNF chains (validity probe: content 40% -> ~0%).
        # Per-request schema pins the v6 5.2 interface: exact VNF bijection + domain in the
        # tier-feasible set D(tau_fk) (identical to the MDO mask). None => some VNF has no
        # tier-feasible domain => genuinely unplaceable slice => structural reject, no LLM call.
        with profiled("plan.schema_pin"):
            plan_schema = build_pinned_plan_schema(sr_dict, abstract_topo)
        if plan_schema is None:
            return None
        # N_struct=2 (v6 5.3 default; admissible {1,2}) => max_retries=1 (one regeneration). With
        # tier pinning the residual is only C4/C5, so retry pressure is second-order.
        plan_dict, check = agent_b.generate_with_memory(
            sr_dict, abstract_topo, kb=kb, mb=mb, max_retries=1, plan_schema=plan_schema)
        if not getattr(check, "is_valid", False):
            return None
        assignments = plan_dict.get("vnf_assignments", [])
        dom_of = {}
        for a in assignments:
            d = R._parse_abstract_domain(a.get("domain"))
            if a.get("vnf_id") is None or d is None:
                return None
            dom_of[a["vnf_id"]] = d
        suggested, tiers = [], []
        g = substrate.graph
        valid_domains = {g.nodes[n]["domain_id"] for n in g.nodes}
        for v in slice_req.vnfs:
            d = dom_of.get(v.vnf_id)
            if d is None or d not in valid_domains:
                return None
            suggested.append(d)
            # required tier: prefer the tier the LLM/vnf implies; fall back to a
            # permitted tier so the tier mask stays feasible.
            perm = sorted({g.nodes[n]["tier"] for n in v.permitted_nodes if n in g.nodes})
            tiers.append(InfrastructureTier(perm[0]) if perm else InfrastructureTier.MEC)
        return PlanSummary(
            vnf_ids=[v.vnf_id for v in slice_req.vnfs],
            required_tiers=tiers, suggested_domains=suggested,
            cpu_demands=[v.cpu_demand for v in slice_req.vnfs],
            ram_demands=[v.ram_demand for v in slice_req.vnfs],
            vcrs=[v.vcr for v in slice_req.vnfs],
            bw_demands=[f.bandwidth_demand for f in slice_req.flow_edges],
        )
    return _builder


def build_stack(substrate, seed, lr, actors=None, mdo_cfg=None):
    """MDO policy + FROZEN domain actors + coordinator + critic + optimizers.
    num_domains is read from the substrate (families have 5 domains).
    `actors` (dict domain_id->actor) lets a caller inject frozen BC-warm-started
    DomainActors instead of the default deterministic GreedyDomainActor. Either
    way NO actor optimizer is created -> actors are frozen.
    `mdo_cfg` overrides the reward/n_part weights (default = shipped 1,.1,.1,.1)."""
    num_domains = substrate.num_domains
    dummy = PlanSummary(
        vnf_ids=["v0", "v1"], required_tiers=[InfrastructureTier.MEC] * 2,
        suggested_domains=[0, 1], cpu_demands=[1.0] * 2, ram_demands=[1.0] * 2,
        vcrs=[1.0] * 2, bw_demands=[10.0])
    obs_dim = observation_to_tensor(
        build_mdo_observation(substrate, dummy), max_vnfs=MAX_VNFS).shape[0]

    policy = MDOPolicy(obs_dim=obs_dim, num_domains=num_domains,
                       max_vnfs=MAX_VNFS, hidden_dim=128, num_layers=2)
    if actors is None:
        actors = {d: GreedyDomainActor(d) for d in range(num_domains)}  # FROZEN (no params)
    coord = MDOCoordinator(policy, actors,
                           mdo_cfg or MDOConfig(n_part=3, mu=1.0, alpha=0.1, xi=0.1, eta=0.1))
    # §O.3 (Choice A1, conformance): the centralised critic consumes the designed
    # global state s_t, NOT the local o^MDO.
    critic = CentralisedCritic(input_dim=probe_global_state_dim(substrate),
                               hidden_dim=128, num_layers=2)
    opt_mdo = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)
    return policy, coord, critic, opt_mdo, opt_critic, obs_dim, num_domains


def eval_foc(coord, fam_name, seed, arrivals, ceiling, delays, plan_builder=None,
             mode="deterministic"):
    """Eval episode on a fixed held-out stream (seed+777) -> FoC vs ceiling.
    `plan_builder` defaults to the LLM-free greedy m~; pass the arm's own builder
    (e.g. Agent B) so the policy is evaluated on the SAME m~ it trained on."""
    sub = _make_sub(fam_name, seed)
    rng = np.random.default_rng(seed + 777)
    ap = _make_ap(sub, arrivals, rng)
    ap.generate()
    runner = EpisodeRunner(sub, ap, coord, delays,
                           plan_builder=plan_builder or greedy_plan_builder)
    runner.reset()
    ep = runner.run_episode(mdo_mode=mode)
    adm = ep.stats.admitted
    # m~-agreement (fifth-degenerate check, PREREG 5I): fraction of committed per-VNF domain
    # choices that equal the prior m~. Near-1.0 with collapsed entropy => arm 2 is just
    # following the prior (behaviourally == Prior-only), so (2)~=(3) would be a reward artifact.
    # §O.5: t.action is in CANONICAL (sorted) index space while suggested_domains are raw
    # domain IDs — map actions through canonical_to_domain before comparing. Historical
    # mtilde_agreement values (pre-§O) are void.
    num = den = 0
    try:
        canonical_to_domain = [s.domain_id for s in build_domain_summaries(sub)]
        for t in ep.rollout.mdo:
            sug = list(t.info.get("suggested_domains", [])) if getattr(t, "info", None) else []
            act = list(t.action) if getattr(t, "action", None) is not None else []
            if sug and act and len(sug) == len(act):
                for a_j, s_j in zip(act, sug):
                    if not (0 <= int(a_j) < len(canonical_to_domain)):
                        continue
                    den += 1
                    num += int(int(canonical_to_domain[int(a_j)]) == int(s_j))
    except Exception:  # noqa: BLE001  -- instrumentation must never break eval
        num = den = 0
    agreement = (num / den) if den else None
    return (adm / ceiling if ceiling > 0 else 0.0), adm, ep.stats.total_arrivals, agreement


def instrumented_eval(coord, fam_name, seed, arrivals, delays, plan_builder=None,
                      mode="deterministic"):
    """One eval episode returning its per-arrival behavioral trace (PREREG §N.2).

    Uses the SAME fixed held-out stream as `eval_foc` (seed+777), so a learned
    (`mode="deterministic"`) pass and a `mode="random"` reference pass are on
    byte-identical arrivals — exactly the contrast criterion (b) / the k-analysis
    need. This is the permanent replacement for the old side probe.
    """
    sub = _make_sub(fam_name, seed)
    rng = np.random.default_rng(seed + 777)
    ap = _make_ap(sub, arrivals, rng)
    ap.generate()
    runner = EpisodeRunner(sub, ap, coord, delays,
                           plan_builder=plan_builder or greedy_plan_builder)
    runner.reset()
    ep = runner.run_episode(mdo_mode=mode)
    return {"mode": mode, "admit": ep.stats.admitted,
            "total": ep.stats.total_arrivals, "pairs": ep.arrival_trace}


# §O.3 — per-arrival s_t capture for the centralised critic. Built as a dynamic
# subclass of the CURRENT module-level EpisodeRunner so canary/test harnesses
# that patch `wp7_runner.EpisodeRunner` compose transparently.
_STATE_CAPTURE_CACHE: dict = {}


def _with_state_capture(base_cls):
    cls = _STATE_CAPTURE_CACHE.get(base_cls)
    if cls is not None:
        return cls

    class _StateCaptureRunner(base_cls):
        """s_t = encode_global_state at decision time (pre-decision, post-(t-1)
        allocations, departures applied) — Choice A1 conformance (§O.3)."""

        def run_episode(self, mdo_mode: str = "sample"):
            self.s_t_by_request = {}
            return super().run_episode(mdo_mode)

        def _handle_arrival(self, slice_req, mdo_mode, rollout, mdo_results, stats,
                            arrival_trace=None):
            gs = GlobalStateStats(
                total_arrivals=stats.total_arrivals,
                admitted=stats.admitted,
                rejected_by_mdo=stats.rejected_by_mdo,
                max_arrivals=getattr(self, "_cap_max_arrivals", 100),
            )
            self.s_t_by_request[slice_req.request_id] = encode_global_state(
                self.substrate, gs).detach()
            super()._handle_arrival(slice_req, mdo_mode, rollout, mdo_results, stats,
                                    arrival_trace)

    _STATE_CAPTURE_CACHE[base_cls] = _StateCaptureRunner
    return _StateCaptureRunner


def gae_over_arrivals(buffer, trial_values, gamma, lam):
    """Conformant credit assignment (PREREG §N.1) — GAE-λ over the ARRIVAL STREAM.

    The MDO buffer holds one entry per retry TRIAL. IMPLEMENTATION_PLAN §4.8 keeps
    within-arrival credit a contextual bandit (terminal reward shared across an
    arrival's trials, no bootstrap between retries), so GAE runs at ARRIVAL
    granularity and each arrival's advantage/return is broadcast back to its trials.
    `buffer.dones[i] == 1.0` marks each arrival's last (committed) trial; the whole
    arrival stream is one episode, so `done` for GAE is set only at the final arrival.

    Returns per-trial (advantages, returns) tensors aligned to buffer order.
    Advantages are normalized to match `mappo_trainer._compute_advantages` (ruling 2).
    """
    dones = buffer.dones  # list[float], 1.0 at each arrival's last trial
    T = len(dones)
    # Map each trial -> its arrival ordinal (contiguous blocks ending at done==1).
    arrival_of = [0] * T
    a = 0
    for i in range(T):
        arrival_of[i] = a
        if dones[i] >= 0.5:
            a += 1
    term_idx = [i for i in range(T) if dones[i] >= 0.5]

    # ORDERING GUARD (ruling 4): the last transition must close an arrival, and
    # every trial must map into an existing arrival. A buffer that is not in
    # arrival order (e.g. shuffled) breaks the backward GAE recursion silently, so
    # this is a stop-and-report — never a silent sort.
    if not term_idx or dones[-1] < 0.5 or arrival_of[-1] != len(term_idx) - 1:
        raise RuntimeError(
            "MDO buffer not in arrival order (dones do not close the stream) — "
            "GAE-over-stream requires time-ordered transitions; refusing to train. "
            "See PREREG §N.1 ruling 4."
        )

    # Arrival-level reward + value: trials of an arrival share obs & terminal
    # reward, so the arrival-terminal trial carries the arrival's (reward, V).
    ar_rewards = torch.tensor([buffer.rewards[i] for i in term_idx], dtype=torch.float32)
    ar_values = torch.tensor(
        [float(trial_values[i]) for i in term_idx] + [0.0],  # bootstrap tail = 0
        dtype=torch.float32,
    )
    ar_dones = torch.zeros(len(term_idx), dtype=torch.float32)
    ar_dones[-1] = 1.0  # episode = whole stream: terminal only at the final arrival

    ar_adv, ar_ret = compute_gae(ar_rewards, ar_values, ar_dones, gamma=gamma, lam=lam)

    # Normalize advantages — MANDATORY (ruling 2), identical to the conformant path.
    if ar_adv.numel() > 1:
        ar_adv = (ar_adv - ar_adv.mean()) / (ar_adv.std() + 1e-8)

    # Broadcast arrival-level adv/return back to per-trial buffer order (§4.8).
    advantages = torch.tensor(
        [float(ar_adv[arrival_of[i]]) for i in range(T)], dtype=torch.float32
    )
    returns = torch.tensor(
        [float(ar_ret[arrival_of[i]]) for i in range(T)], dtype=torch.float32
    )
    return advantages, returns


def train_arm(arm, fam_name, seed, rounds, arrivals, lr,
              beta_start, beta_end, agent_b, kb, mock, actors=None,
              mdo_cfg=None, eval_with_train_builder=False, return_coord=False,
              entropy_schedule=None, train_trace_path=None, ckpt_path=None,
              use_mb=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sub0 = _make_sub(fam_name, seed)
    delays = build_delays(sub0)

    policy, coord, critic, opt_mdo, opt_critic, obs_dim, num_domains = build_stack(
        sub0, seed, lr, actors=actors, mdo_cfg=mdo_cfg)

    # §O.1 — value normalization: running stats updated ONCE PER ROUND from that
    # round's return batch, BEFORE the critic epochs, frozen within the update
    # loop (pinned cadence). State rides in the §O.7 checkpoint.
    value_norm = ValueNormalizer()

    # §O.4 — the KL prior target must live in the same CANONICAL (sorted) frame
    # as the policy logits and tier mask. suggested_domains are raw domain IDs;
    # convert exactly as coordinator._sample_partition does for follow_prior.
    # Canonical order is static per substrate (sort key uses static tiers) and
    # the family instance is deterministic (seed=0), so compute once.
    canonical_to_domain = [s.domain_id for s in build_domain_summaries(sub0)]
    domain_to_canonical = {d: i for i, d in enumerate(canonical_to_domain)}

    # M^B only for the full arm -- and only when `use_mb`. §T Δ2-T D.3 runs Track D
    # with mb=None: §M's M.1 measured cache-OFF + M^B-live as the WORST known config
    # on RC (-10..-13 admits/100 vs mb=None), and it is the one R45's Full-ORION ran.
    # Default True keeps R45's as-run behavior reproducible.
    mb = None
    if arm == "LLM+RL-full" and use_mb:
        from orion.llm.episodic_memory import EpisodicMemory
        from orion.retrieval import RetrievalConfig, RetrievalMode
        mb = EpisodicMemory(
            config=RetrievalConfig(mode=RetrievalMode.NO_RERANK, apply_recency=True, k_final=3),
            max_entries=R.MEMORY_CAPACITY_K)

    if arm in LLM_ARMS and not mock and agent_b is not None:
        plan_builder = make_llm_plan_builder(agent_b, kb, lambda: mb)
        # §R cache-ON: warm ~6 signatures on round 1, serve cached after → the
        # LLM leaves the per-arrival hot loop, which is what makes full training
        # feasible (the gate's 22h was per-arrival Agent B with no cache).
        if RC_USE_PLAN_CACHE:
            from orion.llm.plan_cache import PlanCache
            plan_builder = _cached_plan_builder(plan_builder, PlanCache(capacity=64))
        elif RC_FIXED_TRAIN_STREAM:
            # §R Δ2-R cache-OFF: memoize the raw LLM plan by SLICE CONTENT (not
            # request_id). A temp-0 LLM is a deterministic function of slice content,
            # so identical content → identical plan (safe reuse); this collapses the
            # fixed train stream to ~1 call/arrival across rounds. Keying on content
            # (not request_id) is required because train (seed) and eval (seed+777)
            # share the req_0000.. namespace but hold DIFFERENT slices — a request_id
            # key would serve a train plan for an eval slice of a different length
            # (partition/vnf mismatch → coordinator crash).
            _content_cache = {}
            _raw_pb = plan_builder
            def plan_builder(sr, substrate, _c=_content_cache, _pb=_raw_pb):
                key = _slice_memo_key(sr)
                if key in _c:
                    return _c[key]
                r = _pb(sr, substrate)
                _c[key] = r
                return r
    else:
        plan_builder = greedy_plan_builder  # RL-alone, or mock stand-in
    # Eval on the SAME m~ the arm trained on (avoids a train/eval prior mismatch
    # for the LLM arm); default keeps the LLM-free greedy m~ for back-compat.
    # The eval stream (seed+777) is byte-identical every round and Agent B's m~
    # depends only on (slice_req, substrate) not the policy -> memoize by
    # request_id so an LLM eval costs ~arrivals calls TOTAL, not arrivals*rounds.
    if eval_with_train_builder:
        _eval_cache = {}
        _train_pb = plan_builder
        def eval_pb(slice_req, substrate):
            rid = getattr(slice_req, "request_id", None)
            if rid is not None and rid in _eval_cache:
                return _eval_cache[rid]
            r = _train_pb(slice_req, substrate)
            if rid is not None:
                _eval_cache[rid] = r
            return r
    else:
        eval_pb = None

    cfg = MAPPOConfig(kl_beta_initial=beta_start, kl_beta_final=beta_end,
                      update_epochs=4, clip_eps=0.2, entropy_coef=0.01,
                      gamma=0.99, gae_lambda=0.95)

    # Fixed eval ceiling (once). §O.6: ceiling counted over the SAME stream
    # length the eval episode runs, not ARRIVALS_PER_INSTANCE.
    eval_sub = _make_sub(fam_name, seed)
    _, eval_ceiling = R.compute_ceiling(eval_sub, seed + 777, num_arrivals=arrivals,
                                        slice_factory=RC_SLICE_FACTORY)

    init_params = {n: p.clone() for n, p in policy.named_parameters()}
    curve = []
    cumulative_arrivals = 0

    logger.info("=" * 66)
    logger.info("ARM %s  family=%s seed=%d rounds=%d arrivals=%d beta=%.2f->%.2f "
                "num_domains=%d eval_ceiling=%d",
                arm, fam_name, seed, rounds, arrivals, beta_start, beta_end,
                num_domains, eval_ceiling)
    logger.info("=" * 66)

    for rnd in range(rounds):
        t0 = time.time()
        # KL beta: linear decay over ROUNDS (high early, ~0 late), constant
        # within a round. This is the annealing axis; total-step-based decay
        # collapses to 0 inside round 1 because PPO takes many sub-steps.
        beta_t = beta_schedule(rnd, max(1, rounds - 1), beta_start, beta_end)
        # Entropy-floor schedule (PREREG 2026-07-11 §M.4-Δ5): coefficient decays from c0 to a
        # floor over rounds, never below. Protects against premature lock-in while h^m is
        # learned, without forcing uniformity at convergence. None -> constant cfg.entropy_coef.
        if entropy_schedule is not None:
            _c0, _cfloor = entropy_schedule
            ent_coef_t = max(_cfloor, _c0 * (1.0 - rnd / max(1, rounds - 1)))
        else:
            ent_coef_t = cfg.entropy_coef
        sub = _make_sub(fam_name, seed)
        # §R Δ2-R: fixed train stream (byte-identical every round) enables the
        # request_id plan memo above and makes R.4 train on R.2's exact stream.
        # Default: fresh per-round stream (rnd in the seed) for coverage.
        # Fixed: default_rng(seed) == R.2's exact arrival stream (run_q_cell uses
        # default_rng(arrival_seed=seed)); eval stays held-out at seed+777.
        stream_seed = seed if RC_FIXED_TRAIN_STREAM else (seed + rnd * 1_000_000 + 1)
        rng = np.random.default_rng(stream_seed)
        ap = _make_ap(sub, arrivals, rng)
        ap.generate()
        # §O.3: train episodes run under the s_t-capture subclass of whatever
        # EpisodeRunner currently is (composes with test-harness patches).
        runner = _with_state_capture(EpisodeRunner)(
            sub, ap, coord, delays, plan_builder=plan_builder)
        runner._cap_max_arrivals = arrivals
        runner.reset()
        ep = runner.run_episode(mdo_mode="sample")
        cumulative_arrivals += ep.stats.total_arrivals

        # Training-time behavioral trace (§N.2): append this round's per-arrival
        # trace as one JSONL line, flushed immediately so it is visible/parseable
        # after round 1 (the pre-fifteen-rounds "recorder is recording" spot-check).
        if train_trace_path is not None:
            with open(train_trace_path, "a") as _tf:
                _tf.write(json.dumps({"round": rnd + 1, "pairs": ep.arrival_trace}) + "\n")

        # Build buffer + advantages. Conformance (PREREG §N.1): advantages come from
        # GAE-λ over the arrival stream (gae_over_arrivals), NOT the old per-arrival
        # bandit `reward - V`. done is set per arrival's last trial in _append_rollout;
        # gae_over_arrivals treats the whole stream as one episode.
        buffer = PPORolloutBuffer()
        suggested_list = []
        with torch.no_grad():
            for t in ep.rollout.mdo:
                # §O.3: critic input is the designed s_t captured at decision
                # time, not the local o^MDO. §O.1: values denormalized.
                st = runner.s_t_by_request.get(t.request_id)
                if st is None:  # defensive; every MDO-reaching arrival is captured
                    st = torch.zeros(probe_global_state_dim(sub0))
                cv = float(value_norm.denormalize(critic(st.unsqueeze(0))).item())
                buffer.append_mdo(
                    mdo_obs=t.obs, action=t.action, log_prob=t.log_probs,
                    entropy=t.entropy, aux_value=t.value_estimate,
                    global_state=st, critic_value=cv,
                    reward=t.terminal_reward, done=t.committed,
                    tier_mask=t.tier_mask, num_vnfs=t.num_vnfs)
                suggested_list.append(t.info.get("suggested_domains", []))

        kl_sum = 0.0
        motion = 0.0
        kl_frame_skips = 0
        ev = corr_adv_pos = neg_adv_admitted = float("nan")
        if len(buffer) > 0:
            with profiled("train.gae"):
                with torch.no_grad():
                    vals = value_norm.denormalize(
                        critic(torch.stack(buffer.global_states)).reshape(-1)
                    ).to(torch.float32)  # reshape (not squeeze): [1,1]/[1]->[1], never 0-dim
                advantages, returns = gae_over_arrivals(
                    buffer, vals, gamma=cfg.gamma, lam=cfg.gae_lambda)
                buffer.set_gae(advantages, returns)

            # §O.8 — permanent EV / advantage-sanity telemetry (arrival granularity).
            # A run whose EV sits near zero is invalid by inspection.
            _term = [i for i, d in enumerate(buffer.dones) if d >= 0.5]
            if len(_term) > 2:
                _ret = np.array([float(returns[i]) for i in _term])
                _val = np.array([float(vals[i]) for i in _term])
                _adv = np.array([float(advantages[i]) for i in _term])
                _rew = np.array([buffer.rewards[i] for i in _term])
                _mu = coord.config.mu
                _var = float(np.var(_ret))
                ev = 1.0 - float(np.var(_ret - _val)) / _var if _var > 1e-12 else float("nan")
                _ord = np.arange(len(_term), dtype=float)
                corr_adv_pos = (float(np.corrcoef(_adv, _ord)[0, 1])
                                if np.std(_adv) > 1e-12 else float("nan"))
                _adm = _rew > _mu / 2.0
                neg_adv_admitted = (float(np.mean(_adv[_adm] < 0.0))
                                    if _adm.any() else float("nan"))

            # §O.1 — update normalizer ONCE from this round's return batch,
            # BEFORE the critic epochs; frozen through all epochs (pinned).
            value_norm.update(returns)

            # Critic update — §O.2: Huber on normalized targets; the
            # max(raw, clip) value clipping is removed (zero-gradient region
            # demonstrated in the 2026-07-13 fault report).
            with profiled("train.critic_update"):
                norm_targets = value_norm.normalize(returns).detach()
                for _ in range(cfg.update_epochs):
                    gs = torch.stack(buffer.global_states)
                    new_v = critic(gs).squeeze(-1)
                    v_loss = torch.nn.functional.smooth_l1_loss(new_v, norm_targets)
                    opt_critic.zero_grad(); (0.5 * v_loss).backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt_critic.step()

            # MDO PPO update with KL prior toward m~
            steps = 0
            with profiled("train.ppo_update"):
                for _ in range(cfg.update_epochs):
                    epoch_loss = torch.tensor(0.0); cnt = 0
                    for i in range(len(buffer.mdo_obs)):
                        obs_i = buffer.mdo_obs[i]
                        if obs_i.numel() == 0:
                            continue
                        k = buffer.mdo_num_vnfs[i]
                        if k == 0 or i >= len(buffer.mdo_tier_masks):
                            continue
                        action_i = torch.tensor(buffer.mdo_actions[i], dtype=torch.long)
                        tm_i = buffer.mdo_tier_masks[i]
                        adv_i = float(advantages[i]) if i < len(advantages) else 0.0
                        new_lp, new_ent, new_logits = policy.evaluate_actions(
                            obs_i, tm_i, action_i, k)
                        ratio = torch.exp(new_lp.sum() - buffer.mdo_log_probs[i].sum())
                        step_loss = -torch.min(
                            ratio * adv_i,
                            torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_i)
                        kl_term = torch.tensor(0.0)
                        if beta_t > 0 and i < len(suggested_list):
                            sug = suggested_list[i]
                            if sug and len(sug) == k:
                                # §O.4 — canonicalize m~ (raw domain IDs) into the
                                # policy/mask frame, matching coordinator follow_prior.
                                sug_c = [domain_to_canonical.get(d) for d in sug]
                                if any(c is None for c in sug_c):
                                    kl_frame_skips += 1
                                else:
                                    prior = build_prior_logits(sug_c, num_domains, tm_i)
                                    kl_term = analytical_kl(
                                        new_logits[:k], prior[:k],
                                        tm_i[:k] if tm_i.dim() == 2 else None)
                        epoch_loss = epoch_loss + step_loss - ent_coef_t * new_ent + beta_t * kl_term
                        cnt += 1; steps += 1
                        kl_sum += float(kl_term)
                    if cnt > 0:
                        opt_mdo.zero_grad(); (epoch_loss / cnt).backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt_mdo.step()
            motion = sum((p - init_params[n]).abs().sum().item()
                         for n, p in policy.named_parameters())

        # MDO policy entropy (stable-vs-trained: is the policy still exploring?)
        ent_vals = [float(t.entropy) for t in ep.rollout.mdo]
        mdo_entropy = float(np.mean(ent_vals)) if ent_vals else 0.0

        foc, eadm, etot, magree = eval_foc(coord, fam_name, seed, arrivals, eval_ceiling, delays,
                                           plan_builder=eval_pb)
        curve.append({
            "round": rnd + 1, "cumulative_arrivals": cumulative_arrivals,
            "train_admit": ep.stats.admitted, "train_total": ep.stats.total_arrivals,
            "eval_foc": foc, "eval_admit": eadm, "eval_total": etot,
            "eval_ceiling": eval_ceiling, "beta": beta_t,
            "kl_mean": kl_sum / max(1, len(buffer)), "param_motion": motion,
            "mdo_entropy": mdo_entropy, "mtilde_agreement": magree,
            "ent_coef": ent_coef_t,
            # §O.8 telemetry — near-zero EV means the run is invalid by inspection.
            "ev": ev, "corr_adv_pos": corr_adv_pos,
            "neg_adv_admitted_frac": neg_adv_admitted,
            "kl_frame_skips": kl_frame_skips,
            "value_norm_mean": value_norm.mean, "value_norm_std": value_norm.std,
        })
        # §O.9 — peak RSS/VRAM sampled at round boundaries.
        _coll = get_collector()
        if _coll is not None and hasattr(_coll, "sample_memory"):
            _coll.sample_memory()
        logger.info("[%s] R%d: train=%d/%d  eval_FoC=%.1f%% (%d/%d)  beta=%.3f "
                    "kl=%.4f motion=%.3f ent=%.3f  EV=%.3f corr_pos=%.2f negAdv@adm=%.2f  %.1fs",
                    arm, rnd + 1, ep.stats.admitted, ep.stats.total_arrivals,
                    100 * foc, eadm, eval_ceiling, beta_t,
                    kl_sum / max(1, len(buffer)), motion, mdo_entropy,
                    ev, corr_adv_pos, neg_adv_admitted, time.time() - t0)

    # §R Δ2-R — same-stream selector isolation. Eval the SAME trained coord on the
    # SAME held-out stream + plans (eval_pb) in follow_prior mode: the trained
    # selector (per-round eval_foc, deterministic) vs plan-following, only the
    # selector differs. Δ = the RL selector's contribution beyond following the plan.
    if curve:
        foc_follow, fadm, ftot, _ = eval_foc(
            coord, fam_name, seed, arrivals, eval_ceiling, delays,
            plan_builder=eval_pb, mode="follow_prior")
        curve[-1]["eval_foc_follow_prior"] = foc_follow
        curve[-1]["eval_admit_follow_prior"] = fadm
        curve[-1]["eval_foc_trained"] = curve[-1]["eval_foc"]
        logger.info("[%s] SELECTOR ISOLATION (held-out seed+777): trained=%.1f%% "
                    "follow_prior=%.1f%%  selector_delta=%+.1fpp",
                    arm, 100 * curve[-1]["eval_foc"], 100 * foc_follow,
                    100 * (curve[-1]["eval_foc"] - foc_follow))

    # §O.7 — permanent checkpointing (policy + critic + value-norm state).
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with profiled("ckpt.write"):
            torch.save({
                "arm": arm, "family": fam_name, "seed": seed,
                "rounds": rounds, "arrivals": arrivals,
                "policy_state": policy.state_dict(),
                "critic_state": critic.state_dict(),
                "value_norm_state": value_norm.state_dict(),
                "obs_dim": obs_dim, "num_domains": num_domains,
                "canonical_to_domain": canonical_to_domain,
                "final_curve": curve[-1] if curve else None,
            }, ckpt_path)
        logger.info("[%s] checkpoint -> %s", arm, ckpt_path)
    return (curve, coord) if return_coord else curve


def samples_to_threshold(curve, frac=0.9):
    """Cumulative training arrivals until eval FoC first reaches frac*final_FoC."""
    if not curve:
        return None
    final = curve[-1]["eval_foc"]
    if final <= 0:
        return None
    target = frac * final
    for c in curve:
        if c["eval_foc"] >= target:
            return c["cumulative_arrivals"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="C-_T-_B-")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--arrivals", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--beta-start", type=float, default=1.0)
    ap.add_argument("--beta-end", type=float, default=0.0)
    ap.add_argument("--arms", nargs="+", default=ALL_ARMS)
    ap.add_argument("--port", type=int, default=8000, help="llama.cpp server port for Agent B")
    ap.add_argument("--mock", action="store_true", help="no LLM; LLM arms use greedy m~")
    ap.add_argument("--out", default="results/wp7")
    args = ap.parse_args()

    agent_b, kb = None, None
    need_llm = (not args.mock) and any(a in LLM_ARMS for a in args.arms)
    if need_llm:
        from orion.llm.llm_backend import LLMBackend, LLMConfig
        from orion.llm.agent_b import AgentB
        from orion.llm.semantic_memory import SemanticMemory
        cfg = LLMConfig(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY",
                        model="default", temperature=0.05, max_tokens=2048)
        agent_b = AgentB(LLMBackend(cfg))
        kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
        if kb_path.exists():
            kb = SemanticMemory.from_json(kb_path)
            logger.info("K^B loaded: %d entries", len(kb.entries))
        else:
            logger.warning("K^B not found at %s (LLM arms run without K^B grounding)", kb_path)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.family}_seed{args.seed}{'_mock' if args.mock else ''}"
    results = {}
    for arm in args.arms:
        # Per-arm checkpoint: skip an arm already completed in a prior run so a
        # kill on this shared box loses at most the in-progress arm's work.
        arm_ckpt = out_dir / f"arm_{arm}_{tag}.json"
        if arm_ckpt.exists():
            with open(arm_ckpt) as f:
                results[arm] = json.load(f)
            logger.info("[%s] already complete -> loaded checkpoint %s", arm, arm_ckpt)
            continue
        bs, be = (0.0, 0.0) if arm == "RL-alone" else (args.beta_start, args.beta_end)
        curve = train_arm(arm, args.family, args.seed, args.rounds, args.arrivals,
                          args.lr, bs, be, agent_b, kb, args.mock)
        results[arm] = {
            "curve": curve,
            "final_foc": curve[-1]["eval_foc"] if curve else 0.0,
            "samples_to_90pct": samples_to_threshold(curve, 0.9),
        }
        tmp = arm_ckpt.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(results[arm], f, indent=2)
        tmp.replace(arm_ckpt)  # atomic
        logger.info("[%s] checkpoint written -> %s", arm, arm_ckpt)

    summary = {
        "family": args.family, "seed": args.seed, "rounds": args.rounds,
        "arrivals": args.arrivals, "beta": [args.beta_start, args.beta_end],
        "mock": args.mock, "arms": {a: {k: v for k, v in results[a].items() if k != "curve"}
                                    for a in results},
    }
    out_path = out_dir / f"wp7_{args.family}_seed{args.seed}{'_mock' if args.mock else ''}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    logger.info("\n" + "=" * 66)
    logger.info("WP7 SUMMARY  family=%s seed=%d%s", args.family, args.seed,
                " [MOCK]" if args.mock else "")
    logger.info("=" * 66)
    for arm in args.arms:
        r = results[arm]
        logger.info("  %-16s final_FoC=%.1f%%  samples_to_90%%=%s",
                    arm, 100 * r["final_foc"], r["samples_to_90pct"])
    if "RL-alone" in results:
        base = results["RL-alone"]["final_foc"]
        for arm in args.arms:
            if arm in LLM_ARMS:
                logger.info("  headline: %s vs RL-alone = %+.1f pp (final FoC)",
                            arm, 100 * (results[arm]["final_foc"] - base))
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
