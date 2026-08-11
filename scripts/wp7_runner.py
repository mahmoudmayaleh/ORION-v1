#!/usr/bin/env python3
"""WP7 headline runner: LLM+RL vs RL-alone, MDO in the loop.

Three claim approaches, IDENTICAL MDO architecture + IDENTICAL frozen greedy domain
actors (no actor gradient updates — the RL-alone vs LLM+RL gap is attributable
to the LLM, not to actor learning). Only the MDO policy pi^MDO_phi learns.

  RL-alone       : beta=0, NO KL prior, NO Agent B. m~ = deterministic greedy
                   structural plan (LLM-free), used only for the tier mask /
                   obs; NOT as a KL target (beta=0).
  LLM+RL memoff  : KL prior toward Agent B's suggested partition m~, beta
                   linearly decayed. Agent B active + K^B. M^B OFF.
  LLM+RL full    : same as memoff plus M^B episodic ON.

Per-arrival flow (EpisodeRunner): spec -> plan_builder (m~) -> MDO partition
(pi^MDO_phi, mode="sample") -> frozen GreedyDomainActor place -> verify
(E2E/C5b/C7/C9) -> commit/reject -> reward Rt -> PPO update (MDO + critic only).

Metrics (all three): paired eval FoC vs exhaustive ceiling per round (learning
curve), samples-to-threshold, final FoC. num_domains is read from the substrate.

Reduced-scale on-claim usage (1 family, seed 42):
  python scripts/wp7_runner.py --family C-_T-_B- --seed 42 \
      --arrivals 60 --rounds 30 --approaches RL-alone LLM+RL-memoff --port 8000
Fast plumbing check (no LLM; LLM approaches fall back to greedy m~):
  python scripts/wp7_runner.py --family C-_T-_B- --rounds 2 --arrivals 40 --mock
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import approach_runner as R  # domain-parse + AgentB/K^B/M^B setup + ceiling
from orion.actors.greedy_domain_actor import GreedyDomainActor
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.llm.condition_signature import compute_condition_signature
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.kl_prior import analytical_kl, build_prior_logits, beta_schedule
from orion.mdo.observation import (
    build_domain_summaries,
    build_mdo_observation,
    observation_to_tensor,
)
from orion.mdo.policy import MDOPolicy, DirectJointPolicy, AutoregMDOPolicy
from orion.mdo.types import PlanSummary
from orion.profiling import get_collector, profiled
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.episode_runner import EpisodeRunner
from orion.sim.reward import RewardWeights
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
from orion.sim.load_levels import NUM_ARRIVALS as Y_NUM_ARRIVALS
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wp7")

# PRE-§Y constants. Retained only so banked pre-§Y cells replay byte-identically.
# They are the ones §Y.2 diagnosed: mean lifetime 1/0.02 = 50 time units against a
# ~25 t.u. episode, so essentially nothing departed and difficulty could not come
# from load. §Y sets RC_LOAD_LEVEL instead; see _make_ap.
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
MAX_VNFS = 10

LLM_APPROACHES = {"LLM+RL-memoff", "LLM+RL-full"}
ALL_APPROACHES = ["RL-alone", "LLM+RL-memoff", "LLM+RL-full"]


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
# Set by scripts/grid_runner.py::_wire before calling train_approach, so the conformant
# §O trainer runs on the RC-v2 family + cut-sensitive workload with cache-ON,
# WITHOUT perturbing the validated family path. All default None → gate behavior.
RC_SUBSTRATE_FN = None    # callable(seed) -> SubstrateNetwork
RC_SLICE_FACTORY = None   # passed to every training/eval ArrivalProcess

# §Y.2 load-level seam. When set to a `load_levels.LoadLevel`, every training and
# eval ArrivalProcess is built from the CALIBRATED lambda/mu and the §Y episode
# length instead of the pre-§Y module constants. Left None, the pre-§Y path is
# byte-identical, so banked pre-§Y cells stay reproducible.
RC_LOAD_LEVEL = None      # load_levels.LoadLevel, or None for the pre-§Y constants
RC_NUM_ARRIVALS = None    # override episode length (defaults to the level's N)
# The plan cache is part of the pipeline, not an ablation. Without it an LLM cell
# is 2000 calls at ~6s = 3.36 h against a 4 h cell timeout, i.e. 538 h for the
# grid's 160 LLM cells -- the run is not merely slow, it does not complete. It is
# also the mechanism the design claims (consult the model per distinct planning
# situation, not per arrival), so running without it would measure a system
# nobody proposes. grid_runner refuses to turn it off.
RC_USE_PLAN_CACHE = True
# Per-arrival post-outcome callback handed to the EVAL EpisodeRunner. Same seam
# idiom as EVAL_TOPO_SIG / CUSTOM_PLAN_BUILDER: None leaves the eval path
# byte-identical. grid_runner sets it when M^B accumulates during evaluation.
# Deliberately NOT wired into the training loop at `train_approach` -- writing to
# the store during training is a separate change with its own protocol question.
EVAL_ON_DECISION = None
PLAN_CACHE_CAPACITY = 256  # peak live keys in any 200-arrival window is 114 (measured)
PLAN_CACHE_STATS: dict = {}
# §R Δ2-R (2026-07-15): cache-OFF Full-ORION. The signature cache is known-pathological
# on RC (forced/flex chains collide on one sfc_template → wrong cut reused; see
# results/r_local_R12_RESULT.md). To get cache-OFF per-arrival plan quality (no
# collision) at feasible cost, FIX the training stream across rounds (byte-identical
# every round, matching R.2's single stream exactly) and memoize the LLM plan by
# request_id — a temp-0 LLM on a fixed stream yields a constant plan per arrival, so
# this is ~arrivals calls TOTAL, not arrivals*rounds (25k/seed -> ~100/seed). The
# R.4-vs-R.2 comparison becomes exact same-stream selector isolation. Ratified
# plan_cache/plan_signature untouched (the tier-class key fix stays a deferred Δ).
RC_FIXED_TRAIN_STREAM = False

# ── §U.1 MDO representation fix (2026-07-18) ─────────────────────────────────
# The factored MDOPolicy cannot express colocation (the only reliably feasible
# partition) without one-hot collapse → born-rejecting 2-5% admit. DirectJointPolicy
# is a single Categorical over enumerated feasible joints. Default "factored" keeps
# every pre-§U path byte-identical; the RC/Phase-1 driver sets "joint".
MDO_POLICY_KIND = "factored"   # "factored" (legacy) | "joint" (DirectJointPolicy, small K) | "autoreg" (§U.1e, RC)
MAX_CHAIN_LEN_RC = 5           # DirectJointPolicy enumeration cap (small-K diagnostic only)
JOINT_PRIOR_TEMP = 0.2         # §U.1b committed: peak logit 5.0 on the m̃ atom (joint)
# §U.1e (2026-07-18): DirectJointPolicy enumerates M^K joints — infeasible at RC's K=6
# (5^6=15625, beyond its "few hundred" scope; born-rejecting risk + slow). AutoregMDOPolicy
# decodes VNF domains sequentially, each conditioned on prior placements → captures
# colocation AND scales O(K·M), no chain cap. Uses the factored [K,M] KL path with a
# per-step prior toward m̃[k] at this temperature.
AUTOREG_PRIOR_TEMP = 0.3       # §U.1e committed: peak logit 3.33 on m̃[k] per step
# §U.1h: optional custom plan builder (sr, substrate)->PlanSummary|None, overrides the
# LLM/greedy m̃ source. Used to inject a GOOD prior (Plain's partition). None = off.
CUSTOM_PLAN_BUILDER = None

# §W.1 (2026-07-25): advantage construction for the MDO update. §V.4 (BCD1)
# confirmed the stream-GAE update destroys a known-good policy: with the whole
# 100-arrival stream one gamma=0.99 episode, an arrival's return is dominated by
# how many admissions remain after it (position), not by its own action
# (corr_adv_pos reached -0.85 during the destruction). Per-arrival modes score
# the action on what it can actually influence:
#   "stream_gae"  PREREG §N.1 legacy (control approach; the confirmed-faulty path)
#   "td0"         A_a = r_a + gamma*V(s_{a+1}) - V(s_a); critic target = TD(0).
#                 Keeps the inter-arrival externality through the learned V.
#   "bandit"      A_a = r_a - V(s_a); critic target = r_a (immediate reward).
ADV_MODE = "stream_gae"
# §X.4 (2026-07-26): prior-coupling loss for autoreg policies.
#   "sampled_kl"  legacy: analytical KL of the policy's conditionals ALONG THE
#                 SAMPLED prefix vs per-step m~ targets, summed over slots. For
#                 a colocation m~ this is a contradictory objective (after
#                 sampling domain X at step 0, it pulls step 1 toward m~[1]
#                 conditioned on the wrong prefix, against the decoder's
#                 count-conditioning) — measured: cannot align at any beta.
#   "distill"     teacher-forced sequence distillation: -sum_k log pi(m~_k |
#                 m~_<k), the exact gradient form §V.4 BC validated on this
#                 network. Skips (counted in kl_frame_skips) if any m~ step is
#                 mask-infeasible.
#
# §Z.1 (2026-08-06) REDUCTION: both terms now reduce over the K slots by SUM.
# "distill" previously used mean() while "sampled_kl" reaches analytical_kl,
# which ends kl_per_slot.sum(). K runs 2-6, so switching PRIOR_LOSS silently
# rescaled the effective beta by ~K, and KLS4/KLS5 therefore compared a distill
# arm at beta=25 against a legacy arm carrying 4-6x that dose. Sum is the
# sequence-level convention Shah et al. (arXiv:2512.21852) analyse; averaging is
# outside what they verify. Any beta read from a pre-§Z run is in the old units.
PRIOR_LOSS = "sampled_kl"
# §W.2: single reward-config source. RewardWeights.lambda_viol defaulted to 10
# against the +1.0 admission bonus (MDOConfig.lambda_viol is dead code); §V.3
# showed 10:1 wrecks the critic/advantage telemetry and 1:1 fixes it with no
# FoC cost. Mainline default is now 1.0; the ordering clean-admit > reject >=
# violating-admit is preserved.
REWARD_LAMBDA_VIOL = 1.0

# §Y.13 (2026-08-02) — adaptive entropy floor, see the controller in
# train_approach. ENT_TARGET_FRAC sets the target policy entropy as a fraction of
# log(num_domains), the uniform-policy entropy: 0.25 * log(5) = 0.402 nats, which
# is where the two §Y.10 seeds that generalized cleanly (44, 45) settled on their
# own, and roughly 20x above where the seed that overfit hardest (42) ended up.
# Expressing it as a fraction of log|A| rather than as a bare constant keeps it
# meaningful if the domain count changes.
ENT_TARGET_FRAC = 0.25
ENT_DUAL_LR = 0.5      # dual step on log-coefficient, per round
ENT_COEF_MAX = 0.5     # ceiling; the controller must not drown the policy loss
ENT_ADAPTIVE = True    # False restores the pure §M.4-Δ5 schedule byte-for-byte

# §U.2b (2026-07-18): the LLM occasionally suggests a tier-infeasible domain (VNF
# needs MEC, assigned cloud-only). In sample mode the tier mask blocks it; the
# follow_prior comparator raw-COPIES m̃ and hits the coordinator COMMIT frame
# assert mid-run. Enabling this filters such plans to None (== a plan-reject, what
# the mask does), so training AND the follow_prior comparator operate on tier-valid
# plans (masked-copy). Default False keeps every legacy path byte-identical; the
# joint/RC smoke sets True. The proper coordinator-side follow_prior fix is deferred.
TIER_FILTER_LLM_PLANS = False
# Topology signature for M^B topology-keyed retrieval in make_llm_plan_builder's
# eval path; set by the grid runner before eval_foc, reset to None after.
EVAL_TOPO_SIG = None
# §Y.6: operating-point label ("L1".."L4") stamped into the M^B condition
# signature. The condition itself is NOT a global — it is read off the substrate
# at each arrival inside the plan builder, because congestion is exactly the
# thing that changes between arrivals. Only the load label is ambient.
EVAL_LOAD_LEVEL = None


def tier_filtered(inner):
    """Wrap a plan builder: return None for any plan whose suggested domain cannot
    host a VNF (§U.1d node-based: the domain holds NO permitted node for the VNF —
    a genuine infeasibility, matching the node-based mask). Under the fixed mask
    this is a near-no-op guard. Exposes `.stats` = {built, none_inner, tier_filtered}."""
    _cache: dict = {}
    stats = {"built": 0, "none_inner": 0, "tier_filtered": 0}

    def _domain_nodes(substrate):
        key = id(substrate)
        if key not in _cache:
            _cache[key] = {d: set(substrate.nodes_in_domain(d))
                           for d in range(substrate.num_domains)}
        return _cache[key]

    def _builder(sr, substrate):
        r = inner(sr, substrate)
        if r is None:
            stats["none_inner"] += 1
            return None
        dn = _domain_nodes(substrate)
        feasible = all(
            set(sr.vnfs[k].permitted_nodes) & dn.get(d, set())
            for k, d in enumerate(r.suggested_domains)
            if k < len(sr.vnfs)
        )
        if not feasible:
            stats["tier_filtered"] += 1
            return None
        stats["built"] += 1
        return r
    _builder.stats = stats
    return _builder


def _make_sub(fam_name, seed):
    """Training/eval substrate for the currently wired §Y instance.

    This used to fall back to a PRE-§Y family substrate when the hook was unset.
    That is the silent-wrong-substrate failure mode: the run completes, every cell
    carries a plausible number, and nothing says the episode happened on a
    different network than the one being reported. It refuses now, for the same
    reason `get_level` refuses an uncalibrated lambda.
    """
    if RC_SUBSTRATE_FN is None:
        raise RuntimeError(
            "no substrate is wired: call grid_runner._wire(scenario, level, instance) "
            "before building an episode. There is deliberately no default substrate.")
    return RC_SUBSTRATE_FN(seed)


def _make_ap(sub, n, rng):
    """Arrival process for one episode.

    Under §Y the rate comes from the calibrated level, not from the module
    constants: ARRIVAL_RATE=4.0 / SERVICE_RATE=0.02 gave a mean lifetime of 50
    time units against a ~25 t.u. episode, so nothing departed and there was no
    load axis at all. `n` still wins if a caller passes one, so short smoke
    episodes stay possible.
    """
    if RC_LOAD_LEVEL is not None:
        n = n or RC_NUM_ARRIVALS or Y_NUM_ARRIVALS
        return ArrivalProcess(sub, n, RC_LOAD_LEVEL.arrival_rate,
                              RC_LOAD_LEVEL.service_rate, rng,
                              slice_factory=RC_SLICE_FACTORY)
    return ArrivalProcess(sub, n, ARRIVAL_RATE, SERVICE_RATE, rng,
                          slice_factory=RC_SLICE_FACTORY)


def _cached_plan_builder(inner, plan_cache, stats=None):
    """Cache-ON wrapper: reuse the abstract plan per PLAN PROFILE, so the LLM is
    called once per distinct planning situation rather than per arrival.

    The key is `plan_profile` = intent + quantised network condition, NOT the
    substrate-independent `plan_signature`. Keying on intent alone served a plan
    produced on an empty network to a saturated one (24 distinct plans per
    2000-arrival episode); the profile gives 174 at a 91.3% hit rate, measured.
    That also keeps the cache and M^B agreeing on what "the same situation"
    means, instead of the cache silently answering for a memory it bypassed.

    `stats` (optional dict) collects hits/misses so the cache hit rate is
    reported rather than assumed.
    """
    from orion.llm.plan_cache import (
        plan_profile, sfc_template, AbstractPlan, instantiate_plan, revalidate_plan)
    from orion.llm.condition_signature import compute_condition_signature

    def _bump(field):
        if stats is not None:
            stats[field] = stats.get(field, 0) + 1

    def _builder(sr, substrate):
        key = plan_profile(sr, compute_condition_signature(substrate, EVAL_LOAD_LEVEL))
        _bump("lookups")
        entry = plan_cache.get(key)
        if entry is not None and revalidate_plan(entry.plan, substrate):
            try:
                served = instantiate_plan(entry.plan, sr)
                # Counted only where it MATTERS: a hit that revalidated and
                # instantiated is an LLM call not made. A hit that failed either
                # check still costs a call, so counting it would report a hit
                # rate the run never enjoyed.
                _bump("hits")
                return served
            except ValueError:
                pass
        _bump("misses")
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
        # §Y.6 — condition read at DECISION time (pre-allocation state), so the
        # retrieved episodes are the ones recorded under comparable congestion.
        cond_sig = (compute_condition_signature(substrate, EVAL_LOAD_LEVEL)
                    if mb is not None else None)
        plan_dict, check = agent_b.generate_with_memory(
            sr_dict, abstract_topo, kb=kb, mb=mb, max_retries=1, plan_schema=plan_schema,
            topology_signature=EVAL_TOPO_SIG, condition_signature=cond_sig)
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
            tiers.append(InfrastructureTier(perm[0]) if perm else InfrastructureTier.EDGE)
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
    `mdo_cfg` overrides the reward weights (default = shipped 1,.1,.1)."""
    num_domains = substrate.num_domains
    dummy = PlanSummary(
        vnf_ids=["v0", "v1"], required_tiers=[InfrastructureTier.EDGE] * 2,
        suggested_domains=[0, 1], cpu_demands=[1.0] * 2, ram_demands=[1.0] * 2,
        vcrs=[1.0] * 2, bw_demands=[10.0])
    obs_dim = observation_to_tensor(
        build_mdo_observation(substrate, dummy), max_vnfs=MAX_VNFS).shape[0]

    if MDO_POLICY_KIND == "autoreg":
        # §U.1e — autoregressive sequential decode (scales to RC's K=6, no cap).
        policy = AutoregMDOPolicy(obs_dim=obs_dim, num_domains=num_domains,
                                  max_vnfs=MAX_VNFS, hidden_dim=128, num_layers=2)
    elif MDO_POLICY_KIND == "joint":
        # §U.1a — single Categorical over enumerated feasible joint partitions (small K only).
        policy = DirectJointPolicy(obs_dim=obs_dim, num_domains=num_domains,
                                   max_chain_length=MAX_CHAIN_LEN_RC,
                                   hidden_dim=128, num_layers=2)
    else:
        policy = MDOPolicy(obs_dim=obs_dim, num_domains=num_domains,
                           max_vnfs=MAX_VNFS, hidden_dim=128, num_layers=2)
    if actors is None:
        actors = {d: GreedyDomainActor(d) for d in range(num_domains)}  # FROZEN (no params)
    coord = MDOCoordinator(policy, actors,
                           mdo_cfg or MDOConfig(mu=1.0, alpha=0.1, eta=0.1))
    # §O.3 (Choice A1, conformance): the centralised critic consumes the designed
    # global state s_t, NOT the local o^MDO.
    critic = CentralisedCritic(input_dim=probe_global_state_dim(substrate),
                               hidden_dim=128, num_layers=2)
    opt_mdo = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)
    return policy, coord, critic, opt_mdo, opt_critic, obs_dim, num_domains


def _agreement_support(pairs, exact_arrivals, n_arrivals):
    """The support a pooled m~-agreement ratio is actually computed on.

    KLS7 read 175 of 200 rounds below 0.05 and 24 above 0.95 with a single
    value in between, which is a switch and not a ratio. Both sides of the
    comparison collapse: partial_obs_builder colocates a chain onto one
    domain, and the deterministic decode also lands on ~one domain, so the
    pooled ratio asks whether two collapsed choices coincided at an effective
    n of one rather than averaging over the slots it appears to count.

    `target_modal_frac` is the score a constant policy playing the modal m~
    domain would get, so agreement below it is worse than a constant;
    `off_modal_agreement` removes those slots and is the informative part;
    `exact_arrival_frac` counts whole arrivals, which is the real n.
    """
    out = {"n_slots": len(pairs), "n_arrivals": n_arrivals,
           "target_modal_frac": None, "policy_modal_frac": None,
           "off_modal_agreement": None, "off_modal_slots": 0,
           "exact_arrival_frac": None}
    if not pairs:
        return out
    tgt, pol = {}, {}
    for s_j, d_j in pairs:
        tgt[s_j] = tgt.get(s_j, 0) + 1
        pol[d_j] = pol.get(d_j, 0) + 1
    total = len(pairs)
    modal_t = max(tgt, key=lambda kk: tgt[kk])
    out["target_modal_frac"] = tgt[modal_t] / total
    out["policy_modal_frac"] = max(pol.values()) / total
    off = [(a, b) for a, b in pairs if a != modal_t]
    out["off_modal_slots"] = len(off)
    if off:
        out["off_modal_agreement"] = sum(1 for a, b in off if a == b) / len(off)
    if n_arrivals:
        out["exact_arrival_frac"] = exact_arrivals / n_arrivals
    return out


def eval_acceptance(coord, fam_name, seed, arrivals, delays, plan_builder=None,
                    mode="deterministic", cost_out=None, report_out=None,
                    agree_out=None):
    """§Y.5 eval: acceptance ratio on the held-out stream, no feasibility oracle.

    Same episode as `eval_foc`, different denominator: admitted / offered rather
    than admitted / ceiling. `report_out`, if given, is filled with the §Y.5
    rejection breakdown (`AcceptanceReport.to_dict()`).

    Returns (acceptance, admitted, offered, mtilde_agreement) so it is a drop-in
    for `eval_foc`'s tuple shape at every call site.
    """
    from orion.sim.acceptance import build_report

    acc, adm, tot, agree, ep = _eval_episode(
        coord, fam_name, seed, arrivals, delays, plan_builder, mode, cost_out,
        agree_out)
    if report_out is not None:
        report_out.update(build_report(ep).to_dict())
    return acc, adm, tot, agree


def eval_foc(coord, fam_name, seed, arrivals, ceiling, delays, plan_builder=None,
             mode="deterministic", cost_out=None):
    """PRE-§Y eval episode on a fixed held-out stream (seed+777) -> FoC vs ceiling.

    §Y.5 deletes fraction-of-ceiling: the feasibility oracle is contention-blind
    and unbounded at §Y substrate sizes. Retained only to replay pre-§Y banked
    cells. New code calls `eval_acceptance`.
    """
    _acc, adm, tot, agree, _ep = _eval_episode(
        coord, fam_name, seed, arrivals, delays, plan_builder, mode, cost_out)
    return (adm / ceiling if ceiling > 0 else 0.0), adm, tot, agree


def _eval_episode(coord, fam_name, seed, arrivals, delays, plan_builder=None,
                  mode="deterministic", cost_out=None, agree_out=None):
    """One eval episode on the fixed held-out stream (seed+777).

    Shared implementation behind `eval_acceptance` (§Y) and `eval_foc` (pre-§Y),
    so the two metrics can never diverge in how the episode itself is run.
    `plan_builder` defaults to the LLM-free greedy m~; pass the approach's own
    builder (e.g. Agent B) so the policy is evaluated on the SAME m~ it trained
    on. `cost_out`: optional dict; filled with the per-admission secondary cost
    summary (cost_metrics.CostAccumulator) over this episode's admissions.

    Returns (acceptance, admitted, offered, mtilde_agreement, episode).
    """
    sub = _make_sub(fam_name, seed)
    rng = np.random.default_rng(seed + 777)
    ap = _make_ap(sub, arrivals, rng)
    ap.generate()
    runner = EpisodeRunner(sub, ap, coord, delays,
                           plan_builder=plan_builder or greedy_plan_builder)
    runner.on_decision = EVAL_ON_DECISION
    runner.reset()
    ep = runner.run_episode(mdo_mode=mode)
    adm = ep.stats.admitted
    if cost_out is not None:
        try:
            from cost_metrics import CostAccumulator
            sr_by_rid = {ev.slice_request.request_id: ev.slice_request
                         for ev in ap.events
                         if ev.event_type == EventType.ARRIVAL
                         and ev.slice_request is not None}
            acc = CostAccumulator(sub)
            for res in ep.mdo_results:
                sr = sr_by_rid.get(res.request_id)
                if res.admitted and sr is not None:
                    acc.add_mdo(sr, res)
            cost_out.update(acc.summary())
        except Exception:  # noqa: BLE001 -- secondary instrumentation must never break eval
            cost_out.setdefault("n_admitted", None)
    # m~-agreement (fifth-degenerate check, PREREG 5I): fraction of committed per-VNF domain
    # choices that equal the prior m~. Near-1.0 with collapsed entropy => approach 2 is just
    # following the prior (behaviourally == Prior-only), so (2)~=(3) would be a reward artifact.
    # §O.5: t.action is in CANONICAL (sorted) index space while suggested_domains are raw
    # domain IDs — map actions through canonical_to_domain before comparing. Historical
    # mtilde_agreement values (pre-§O) are void.
    num = den = 0
    # §Z.6 — the ratio is unchanged; what is added is the support it stands on.
    pairs = []
    exact = arrivals_scored = 0
    try:
        canonical_to_domain = [s.domain_id for s in build_domain_summaries(sub)]
        for t in ep.rollout.mdo:
            sug = list(t.info.get("suggested_domains", [])) if getattr(t, "info", None) else []
            act = list(t.action) if getattr(t, "action", None) is not None else []
            if sug and act and len(sug) == len(act):
                hits = slots = 0
                for a_j, s_j in zip(act, sug):
                    if not (0 <= int(a_j) < len(canonical_to_domain)):
                        continue
                    den += 1
                    d_j = int(canonical_to_domain[int(a_j)])
                    hit = int(d_j == int(s_j))
                    num += hit
                    pairs.append((int(s_j), d_j))
                    hits += hit
                    slots += 1
                if slots:
                    arrivals_scored += 1
                    exact += int(hits == slots)
    except Exception:  # noqa: BLE001  -- instrumentation must never break eval
        num = den = 0
        pairs = []
        exact = arrivals_scored = 0
    agreement = (num / den) if den else None
    if agree_out is not None:
        agree_out.update(_agreement_support(pairs, exact, arrivals_scored))
    offered = ep.stats.total_arrivals
    return (adm / offered if offered > 0 else 0.0), adm, offered, agreement, ep


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
    runner.on_decision = EVAL_ON_DECISION
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

    GAE runs at ARRIVAL granularity; each arrival's advantage/return maps onto
    its buffer entry. `buffer.dones[i] == 1.0` marks each arrival's terminal
    entry; the whole arrival stream is one episode, so `done` for GAE is set
    only at the final arrival.

    Returns (advantages, returns) tensors aligned to buffer order.
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


def per_arrival_advantages(buffer, trial_values, gamma, mode):
    """§W.1 — per-arrival credit assignment ("td0" | "bandit").

    Same arrival mapping, ordering guard, whitening (ruling 2), and per-trial
    broadcast as gae_over_arrivals; only the arrival-level (advantage, return)
    pair differs:
      td0     A_a = r_a + gamma*V_{a+1} - V_a,  return_a = r_a + gamma*V_{a+1}
      bandit  A_a = r_a - V_a,                  return_a = r_a
    """
    dones = buffer.dones
    T = len(dones)
    arrival_of = [0] * T
    a = 0
    for i in range(T):
        arrival_of[i] = a
        if dones[i] >= 0.5:
            a += 1
    term_idx = [i for i in range(T) if dones[i] >= 0.5]
    if not term_idx or dones[-1] < 0.5 or arrival_of[-1] != len(term_idx) - 1:
        raise RuntimeError(
            "MDO buffer not in arrival order (dones do not close the stream) — "
            "refusing to train. See PREREG §N.1 ruling 4."
        )

    ar_rewards = torch.tensor([buffer.rewards[i] for i in term_idx], dtype=torch.float32)
    ar_values = torch.tensor([float(trial_values[i]) for i in term_idx],
                             dtype=torch.float32)
    if mode == "td0":
        next_v = torch.cat([ar_values[1:], torch.zeros(1)])  # bootstrap tail = 0
        ar_ret = ar_rewards + gamma * next_v
    elif mode == "bandit":
        ar_ret = ar_rewards
    else:
        raise ValueError(f"per_arrival_advantages: unknown mode {mode!r}")
    ar_adv = ar_ret - ar_values

    if ar_adv.numel() > 1:
        ar_adv = (ar_adv - ar_adv.mean()) / (ar_adv.std() + 1e-8)

    advantages = torch.tensor(
        [float(ar_adv[arrival_of[i]]) for i in range(T)], dtype=torch.float32)
    returns = torch.tensor(
        [float(ar_ret[arrival_of[i]]) for i in range(T)], dtype=torch.float32)
    return advantages, returns


def train_approach(approach, fam_name, seed, rounds, arrivals, lr,
              beta_start, beta_end, agent_b, kb, mock, actors=None,
              mdo_cfg=None, eval_with_train_builder=False, return_coord=False,
              entropy_schedule=None, train_trace_path=None, ckpt_path=None,
              use_mb=True, init_from=None, target_entropy=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sub0 = _make_sub(fam_name, seed)
    delays = build_delays(sub0)

    policy, coord, critic, opt_mdo, opt_critic, obs_dim, num_domains = build_stack(
        sub0, seed, lr, actors=actors, mdo_cfg=mdo_cfg)

    # Curriculum warm-start (transfer grid): load a prior segment's policy+critic
    # weights so one stack can accumulate experience across the train families.
    # obs_dim (121 on the §Y substrate) and the global-state dim are constant across
    # every segment, because the topology size and inter-domain adjacency are fixed.
    if init_from is not None:
        _sd = torch.load(init_from, map_location="cpu")
        policy.load_state_dict(_sd["policy_state"])
        critic.load_state_dict(_sd["critic_state"])
        logger.info("curriculum warm-start: loaded policy+critic from %s", init_from)

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

    # M^B only for the full approach -- and only when `use_mb`. §T Δ2-T D.3 runs Track D
    # with mb=None: §M's M.1 measured cache-OFF + M^B-live as the WORST known config
    # on RC (-10..-13 admits/100 vs mb=None), and it is the one R45's Full-ORION ran.
    # Default True keeps R45's as-run behavior reproducible.
    mb = None
    if approach == "LLM+RL-full" and use_mb:
        from orion.llm.episodic_memory import EpisodicMemory
        from orion.retrieval import RetrievalConfig, RetrievalMode
        mb = EpisodicMemory(
            config=RetrievalConfig(mode=RetrievalMode.NO_RERANK, apply_recency=True, k_final=3),
            max_entries=R.MEMORY_CAPACITY_K)

    _llm_tier_stats = None
    if CUSTOM_PLAN_BUILDER is not None:
        # §U.1h — inject a custom m̃ source (e.g. Plain's partition as a GOOD, LLM-free
        # KL prior) to test whether the RL breaks the floor with good guidance.
        plan_builder = CUSTOM_PLAN_BUILDER
    elif approach in LLM_APPROACHES and not mock and agent_b is not None:
        plan_builder = make_llm_plan_builder(agent_b, kb, lambda: mb)
        # §U.2b masked-copy: filter tier-infeasible LLM plans (== sample-mode mask)
        # so the follow_prior comparator's raw-copy never hits the COMMIT frame
        # assert. Applied before the cache/memo so filtered results are cached.
        if TIER_FILTER_LLM_PLANS:
            plan_builder = tier_filtered(plan_builder)
            _llm_tier_stats = plan_builder.stats
        # §R cache-ON: warm ~6 signatures on round 1, serve cached after → the
        # LLM leaves the per-arrival hot loop, which is what makes full training
        # feasible (the gate's 22h was per-arrival Agent B with no cache).
        if RC_USE_PLAN_CACHE:
            from orion.llm.plan_cache import PlanCache
            PLAN_CACHE_STATS.clear()
            plan_builder = _cached_plan_builder(
                plan_builder, PlanCache(capacity=PLAN_CACHE_CAPACITY),
                stats=PLAN_CACHE_STATS)
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
    # Eval on the SAME m~ the approach trained on (avoids a train/eval prior mismatch
    # for the LLM approach); default keeps the LLM-free greedy m~ for back-compat.
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

    # §Y.5: no feasibility ceiling. The per-round eval reports acceptance
    # (admitted / offered), so there is no denominator to precompute -- and the
    # enumerator refuses §Y-scale substrates anyway, which is what surfaced this
    # call still being here.

    init_params = {n: p.clone() for n, p in policy.named_parameters()}
    curve = []
    cumulative_arrivals = 0

    # §Y.13 adaptive entropy floor. `target_entropy=None` resolves to a fraction of
    # the uniform-policy entropy log(num_domains); pass 0.0 (or set ENT_ADAPTIVE
    # False) to disable the controller and keep the legacy schedule exactly.
    if not ENT_ADAPTIVE or entropy_schedule is None:
        target_entropy = None
    elif target_entropy is None:
        target_entropy = ENT_TARGET_FRAC * math.log(max(2, num_domains))
    elif target_entropy <= 0.0:
        target_entropy = None
    # Lagged by one round: the controller reacts to the entropy the previous
    # round's rollout actually measured, so it stays None until round 1.
    _measured_entropy = None
    _ent_coef_dual = entropy_schedule[0] if entropy_schedule else cfg.entropy_coef
    ent_floor_hits = 0

    logger.info("=" * 66)
    logger.info("APPROACH %s  scenario=%s seed=%d rounds=%d arrivals=%d "
                "beta=%.2f->%.2f num_domains=%d load=%s",
                approach, fam_name, seed, rounds, arrivals, beta_start, beta_end,
                num_domains, RC_LOAD_LEVEL.name if RC_LOAD_LEVEL else "pre-Y")
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
        # §Y.13 (2026-08-02) — the schedule above floors the COEFFICIENT, not the
        # policy. Across a chained curriculum that is not enough: the coefficient
        # sits at its floor while the policy keeps sharpening, and seeds that ran
        # down to a measured entropy of 0.02-0.11 nats overfit their training
        # instances (seed 42: 67-73% on its own draw, 51.8% held out) while seeds
        # that stayed near 0.4 generalized. So the floor moves onto the measured
        # quantity: a dual controller raises the coefficient whenever policy
        # entropy sits below target and relaxes back to the schedule when it does
        # not. This is SAC's automatic temperature adjustment (Haarnoja et al.,
        # "Soft Actor-Critic Algorithms and Applications", arXiv:1812.05905 §5) in
        # its discrete-action form (Christodoulou, arXiv:1910.07207), where the
        # target is set as a fraction of log|A| rather than -dim(A).
        #
        # The controller can only ever push the coefficient ABOVE the schedule, so
        # a run whose entropy never falls below target trains exactly as it did
        # before this change.
        _ent_coef_sched = ent_coef_t
        if target_entropy is not None and _measured_entropy is not None:
            if _measured_entropy >= target_entropy:
                # Healthy: release immediately and sit on the schedule. Letting the
                # dual decay multiplicatively instead would keep the coefficient a
                # little above the schedule for a few rounds after recovery, which
                # is harmless but makes "engaged" mean two different things.
                _ent_coef_dual = ent_coef_t
            else:
                ent_coef_t = float(np.clip(
                    _ent_coef_dual * math.exp(ENT_DUAL_LR * (target_entropy - _measured_entropy)),
                    ent_coef_t, ENT_COEF_MAX))
                _ent_coef_dual = ent_coef_t
                ent_floor_hits += 1
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
            sub, ap, coord, delays, plan_builder=plan_builder,
            reward_weights=RewardWeights(lambda_viol=REWARD_LAMBDA_VIOL))
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
        prior_fire_rate = train_agreement = None  # §Y.9 train-side prior telemetry
        train_agreement_fr = None                 # §Z.5 free-running counterpart
        train_support_tf = train_support_fr = None  # §Z.6 support of both
        ev = corr_adv_pos = neg_adv_admitted = float("nan")
        if len(buffer) > 0:
            with profiled("train.gae"):
                with torch.no_grad():
                    vals = value_norm.denormalize(
                        critic(torch.stack(buffer.global_states)).reshape(-1)
                    ).to(torch.float32)  # reshape (not squeeze): [1,1]/[1]->[1], never 0-dim
                # §W.1 — advantage construction is a pre-registered approach axis.
                if ADV_MODE == "stream_gae":
                    advantages, returns = gae_over_arrivals(
                        buffer, vals, gamma=cfg.gamma, lam=cfg.gae_lambda)
                else:
                    advantages, returns = per_arrival_advantages(
                        buffer, vals, gamma=cfg.gamma, mode=ADV_MODE)
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
                                elif isinstance(policy, AutoregMDOPolicy):
                                    if PRIOR_LOSS == "distill":
                                        # §X.4 — teacher-forced distillation on
                                        # m̃'s own prefix (the §V.4-validated
                                        # gradient form).
                                        if all(bool(tm_i[j][sug_c[j]]) for j in range(k)):
                                            sug_t = torch.tensor(sug_c, dtype=torch.long)
                                            lp_sug, _, _ = policy.evaluate_actions(
                                                obs_i, tm_i, sug_t, k)
                                            # §Z.1: sum, not mean — same reduction
                                            # analytical_kl uses, so beta is one unit.
                                            kl_term = -lp_sug.sum()
                                        else:
                                            kl_frame_skips += 1
                                    else:
                                        # §U.1e legacy — per-step KL along the
                                        # sampled prefix (temp committed).
                                        prior = build_prior_logits(
                                            sug_c, num_domains, tm_i,
                                            temperature=AUTOREG_PRIOR_TEMP)
                                        kl_term = analytical_kl(
                                            new_logits[:k], prior[:k],
                                            tm_i[:k] if tm_i.dim() == 2 else None)
                                elif isinstance(policy, DirectJointPolicy):
                                    # §U.1b — joint KL toward the single m̃ atom.
                                    jp = policy.joint_prior_logits(
                                        tm_i, k, sug_c, JOINT_PRIOR_TEMP)
                                    if jp is None:
                                        kl_frame_skips += 1  # m̃ not a feasible atom
                                    else:
                                        prior_logits, jmask = jp
                                        kl_term = analytical_kl(
                                            new_logits, prior_logits, jmask)
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

            # §Y.9 — TRAIN-side prior telemetry. §X.4 declared the prior channel
            # inert on the strength of EVAL m~-agreement alone, which conflates
            # three different failures: the term never firing, the term firing
            # and not aligning, and the term aligning on the training states but
            # not generalizing to the eval stream. scripts/probe_prior_gradient.py
            # rules out the middle one — the identical objective (beta=25,
            # lr 1e-3, clip 0.5) drives argmax-agreement to 1.0 on a fixed batch
            # in ~40 steps. So the discriminating measurement is agreement HERE,
            # on the states the update actually saw, plus how often the term
            # fired at all. Both are cheap and read-only.
            fired = eligible = 0
            tr_num = tr_den = 0
            fr_num = fr_den = 0
            tf_pairs, fr_pairs, fr_exact = [], [], 0   # §Z.6
            with torch.no_grad():
                for i in range(len(buffer.mdo_obs)):
                    obs_i = buffer.mdo_obs[i]
                    k = buffer.mdo_num_vnfs[i]
                    if obs_i.numel() == 0 or k == 0 or i >= len(buffer.mdo_tier_masks):
                        continue
                    if i >= len(suggested_list):
                        continue
                    sug = suggested_list[i]
                    if not sug or len(sug) != k:
                        continue
                    eligible += 1
                    sug_c = [domain_to_canonical.get(d) for d in sug]
                    if any(c is None for c in sug_c):
                        continue
                    tm_i = buffer.mdo_tier_masks[i]
                    if not all(bool(tm_i[j][sug_c[j]]) for j in range(k)):
                        continue
                    fired += 1
                    sug_t = torch.tensor(sug_c, dtype=torch.long)
                    _lp, _h, lg = policy.evaluate_actions(obs_i, tm_i, sug_t, k)
                    masked = lg[:k].masked_fill(~tm_i[:k], float("-inf"))
                    tf_pick = masked.argmax(dim=-1).tolist()
                    tr_num += sum(int(a) == int(b)
                                  for a, b in zip(tf_pick, sug_c))
                    tr_den += k
                    tf_pairs.extend((int(b), int(a))
                                    for a, b in zip(tf_pick, sug_c))
                    # §Z.5 — the same comparison under a FREE-RUNNING decode.
                    # evaluate_actions advances its running counts on the actions
                    # passed to it, so the line above is teacher-forced on m~ and
                    # measures agreement at prefixes the decoder never has to
                    # produce. The eval-side mtilde_agreement is free-running, so
                    # the two were never the same measurement and their difference
                    # was read as transfer failure. Free-running HERE isolates
                    # exposure bias (this minus the teacher-forced number, same
                    # states) from transfer (this minus the eval number).
                    part_fr, _lpf, _lgf, _entf = policy(
                        obs_i, tm_i, k, deterministic=True)
                    fr_hits = sum(int(a) == int(b)
                                  for a, b in zip(part_fr, sug_c))
                    fr_num += int(fr_hits)
                    fr_den += k
                    fr_pairs.extend((int(b), int(a))
                                    for a, b in zip(part_fr, sug_c))
                    fr_exact += int(fr_hits == k)
            prior_fire_rate = (fired / eligible) if eligible else None
            train_agreement = (tr_num / tr_den) if tr_den else None
            train_agreement_fr = (fr_num / fr_den) if fr_den else None
            train_support_tf = _agreement_support(tf_pairs, 0, 0)
            train_support_fr = _agreement_support(fr_pairs, fr_exact, fired)

        # MDO policy entropy (stable-vs-trained: is the policy still exploring?)
        ent_vals = [float(t.entropy) for t in ep.rollout.mdo]
        mdo_entropy = float(np.mean(ent_vals)) if ent_vals else 0.0
        # §Y.13: close the dual loop. An empty rollout leaves the controller on its
        # last reading rather than driving it with a fake 0.0.
        if ent_vals:
            _measured_entropy = mdo_entropy

        eval_support = {}
        foc, eadm, etot, magree = eval_acceptance(coord, fam_name, seed, arrivals, delays,
                                           plan_builder=eval_pb,
                                           agree_out=eval_support)
        curve.append({
            "round": rnd + 1, "cumulative_arrivals": cumulative_arrivals,
            "train_admit": ep.stats.admitted, "train_total": ep.stats.total_arrivals,
            "eval_acceptance": foc, "eval_admit": eadm, "eval_total": etot,
            "beta": beta_t,
            "kl_mean": kl_sum / max(1, len(buffer)), "param_motion": motion,
            "mdo_entropy": mdo_entropy, "mtilde_agreement": magree,
            "ent_coef": ent_coef_t,
            "ent_coef_sched": _ent_coef_sched, "ent_target": target_entropy,
            # §O.8 telemetry — near-zero EV means the run is invalid by inspection.
            "ev": ev, "corr_adv_pos": corr_adv_pos,
            "neg_adv_admitted_frac": neg_adv_admitted,
            "kl_frame_skips": kl_frame_skips,
            # §Y.9 — separates "the prior term never fired" from "it fired and
            # did not align" from "it aligned on train and did not transfer".
            # mtilde_agreement above is the EVAL-stream number; this is the
            # train-stream one, on the states the update actually saw.
            "prior_fire_rate": prior_fire_rate,
            "train_mtilde_agreement": train_agreement,
            # §Z.5 — free-running on the SAME train states. Kept alongside the
            # teacher-forced key rather than replacing it: the gap between them IS
            # the exposure-bias measurement, and the old key's meaning must not
            # change under banked readers.
            "train_mtilde_agreement_fr": train_agreement_fr,
            # §Z.6 — a pooled agreement ratio is uninterpretable without the
            # modal baseline it has to beat and the off-modal slots it is
            # really made of. Published for both train decodes and for eval.
            "train_support_tf": train_support_tf,
            "train_support_fr": train_support_fr,
            "eval_support": eval_support or None,
            "adv_mode": ADV_MODE, "lambda_viol": REWARD_LAMBDA_VIOL,
            "value_norm_mean": value_norm.mean, "value_norm_std": value_norm.std,
        })
        # §O.9 — peak RSS/VRAM sampled at round boundaries.
        _coll = get_collector()
        if _coll is not None and hasattr(_coll, "sample_memory"):
            _coll.sample_memory()
        logger.info("[%s] R%d: train=%d/%d  eval_acceptance=%.1f%% (%d/%d)  beta=%.3f "
                    "kl=%.4f motion=%.3f ent=%.3f/%s coef=%.4f  "
                    "EV=%.3f corr_pos=%.2f negAdv@adm=%.2f  %.1fs",
                    approach, rnd + 1, ep.stats.admitted, ep.stats.total_arrivals,
                    100 * foc, eadm, etot, beta_t,
                    kl_sum / max(1, len(buffer)), motion, mdo_entropy,
                    "--" if target_entropy is None else "%.3f" % target_entropy, ent_coef_t,
                    ev, corr_adv_pos, neg_adv_admitted, time.time() - t0)

    if target_entropy is not None:
        logger.info("[%s] entropy floor: target=%.3f nats, controller engaged in "
                    "%d/%d rounds, final ent=%.3f", approach, target_entropy,
                    ent_floor_hits, rounds, curve[-1]["mdo_entropy"] if curve else float("nan"))

    # §R Δ2-R — same-stream selector isolation. Eval the SAME trained coord on the
    # SAME held-out stream + plans (eval_pb) in follow_prior mode: the trained
    # selector (per-round eval_foc, deterministic) vs plan-following, only the
    # selector differs. Δ = the RL selector's contribution beyond following the plan.
    if curve:
        foc_follow, fadm, ftot, _ = eval_acceptance(
            coord, fam_name, seed, arrivals, delays,
            plan_builder=eval_pb, mode="follow_prior")
        curve[-1]["eval_acceptance_follow_prior"] = foc_follow
        curve[-1]["eval_admit_follow_prior"] = fadm
        curve[-1]["eval_acceptance_trained"] = curve[-1]["eval_acceptance"]
        logger.info("[%s] SELECTOR ISOLATION (held-out seed+777): trained=%.1f%% "
                    "follow_prior=%.1f%%  selector_delta=%+.1fpp",
                    approach, 100 * curve[-1]["eval_acceptance"], 100 * foc_follow,
                    100 * (curve[-1]["eval_acceptance"] - foc_follow))

    # §U.2b — LLM tier-error rate is a measured property of the planner (telemetry).
    if curve and _llm_tier_stats is not None:
        st = dict(_llm_tier_stats)
        curve[-1]["llm_tier_stats"] = st
        _tot = st["built"] + st["tier_filtered"]
        logger.info("[%s] LLM tier-error: %d tier-filtered / %d tier-valid "
                    "(%.1f%% of built plans), %d inner-invalid",
                    approach, st["tier_filtered"], st["built"],
                    100.0 * st["tier_filtered"] / _tot if _tot else 0.0,
                    st["none_inner"])

    # §O.7 — permanent checkpointing (policy + critic + value-norm state).
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with profiled("ckpt.write"):
            torch.save({
                "approach": approach, "family": fam_name, "seed": seed,
                "rounds": rounds, "arrivals": arrivals,
                "policy_state": policy.state_dict(),
                "critic_state": critic.state_dict(),
                "value_norm_state": value_norm.state_dict(),
                "obs_dim": obs_dim, "num_domains": num_domains,
                "canonical_to_domain": canonical_to_domain,
                "final_curve": curve[-1] if curve else None,
            }, ckpt_path)
        logger.info("[%s] checkpoint -> %s", approach, ckpt_path)
    return (curve, coord) if return_coord else curve


def samples_to_threshold(curve, frac=0.9):
    """Cumulative training arrivals until eval FoC first reaches frac*final_FoC."""
    if not curve:
        return None
    final = curve[-1]["eval_acceptance"]
    if final <= 0:
        return None
    target = frac * final
    for c in curve:
        if c["eval_acceptance"] >= target:
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
    ap.add_argument("--approaches", nargs="+", default=ALL_APPROACHES)
    ap.add_argument("--port", type=int, default=8000, help="llama.cpp server port for Agent B")
    ap.add_argument("--mock", action="store_true", help="no LLM; LLM approaches use greedy m~")
    ap.add_argument("--out", default="results/wp7")
    args = ap.parse_args()

    agent_b, kb = None, None
    need_llm = (not args.mock) and any(a in LLM_APPROACHES for a in args.approaches)
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
            logger.warning("K^B not found at %s (LLM approaches run without K^B grounding)", kb_path)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.family}_seed{args.seed}{'_mock' if args.mock else ''}"
    results = {}
    for approach in args.approaches:
        # Per-approach checkpoint: skip an approach already completed in a prior run so a
        # kill on this shared box loses at most the in-progress approach's work.
        approach_ckpt = out_dir / f"approach_{approach}_{tag}.json"
        if approach_ckpt.exists():
            with open(approach_ckpt) as f:
                results[approach] = json.load(f)
            logger.info("[%s] already complete -> loaded checkpoint %s", approach, approach_ckpt)
            continue
        bs, be = (0.0, 0.0) if approach == "RL-alone" else (args.beta_start, args.beta_end)
        curve = train_approach(approach, args.family, args.seed, args.rounds, args.arrivals,
                          args.lr, bs, be, agent_b, kb, args.mock)
        results[approach] = {
            "curve": curve,
            "final_acceptance": curve[-1]["eval_acceptance"] if curve else 0.0,
            "samples_to_90pct": samples_to_threshold(curve, 0.9),
        }
        tmp = approach_ckpt.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(results[approach], f, indent=2)
        tmp.replace(approach_ckpt)  # atomic
        logger.info("[%s] checkpoint written -> %s", approach, approach_ckpt)

    summary = {
        "family": args.family, "seed": args.seed, "rounds": args.rounds,
        "arrivals": args.arrivals, "beta": [args.beta_start, args.beta_end],
        "mock": args.mock, "approaches": {a: {k: v for k, v in results[a].items() if k != "curve"}
                                    for a in results},
    }
    out_path = out_dir / f"wp7_{args.family}_seed{args.seed}{'_mock' if args.mock else ''}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    logger.info("\n" + "=" * 66)
    logger.info("WP7 SUMMARY  family=%s seed=%d%s", args.family, args.seed,
                " [MOCK]" if args.mock else "")
    logger.info("=" * 66)
    for approach in args.approaches:
        r = results[approach]
        logger.info("  %-16s final_FoC=%.1f%%  samples_to_90%%=%s",
                    approach, 100 * r["final_acceptance"], r["samples_to_90pct"])
    if "RL-alone" in results:
        base = results["RL-alone"]["final_acceptance"]
        for approach in args.approaches:
            if approach in LLM_APPROACHES:
                logger.info("  headline: %s vs RL-alone = %+.1f pp (final FoC)",
                            approach, 100 * (results[approach]["final_acceptance"] - base))
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
