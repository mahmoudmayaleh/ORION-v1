#!/usr/bin/env python
"""§exp transfer grid — THE final paper run (Table~\\ref{tab:exp-main} + memory ablation).

§Y: approaches x 2 scenario classes x 4 calibrated LOAD LEVELS x >=5 seeds on ONE
fixed hierarchical substrate. Stacks are curriculum-trained at TRAINING_LEVEL over
the TRAINING INSTANCES (warm-started rotation; obs_dim=121 is constant because the
topology size is fixed) and evaluated at every level on HELD-OUT instances, so the
generalization claim is over congestion regime and network instance, not over
topology family.

Approaches (monotone ablation ladder; all trained approaches share ONE curriculum stack per
(scenario, seed) except RL-alone which is trained with beta=0):
  Plain       heuristic colocation-first + FF fallback, full substrate visibility,
              allocated directly (no coordinator). no LLM, no RL.
  Plain-fullobs  §Y.8: Plain's OWN colocation-first placer (full substrate) executed
              through the coordinator + greedy actors, so it differs from Plain ONLY
              in pipeline and from Plain-partial ONLY in observability. Exists
              because Plain vs Plain-partial confounds the two.
  Plain-ffd   §Y.8 control: plain FFD (no colocation preference) through the same
              pipeline. Measures what scattering a chain across domains costs; NOT
              part of the observability ablation.
  Plain-partial  §V.1: the SAME heuristic idea at the MDO's observability —
              colocation-first partition from DomainSummary aggregates + the
              node-based K x M mask only, executed through the coordinator +
              greedy actors (follow_prior). no LLM, no RL, no training.
  RL-alone    trained selector (beta=0), greedy m~. no LLM.
  RL-poprior  §V.2: trained selector, KL prior = partial-obs heuristic partition
              (LLM-free), beta 1->BETA_FLOOR during training; eval deterministic
              (argmax, no advice) with the SAME partial-obs m~ in the obs.
  Memory-off  LLM plan as eval prior, M^B disabled.
  Full        LLM plan + M^B (outcome-driven: selective write, violation-tagged).

Memory approaches warm M^B on the TRAIN-family streams, then keep writing on the held-out
stream (adaptation "written during operation"). Non-memory approaches go through wp7's
eval_foc; memory approaches use grid_memory_instance (reuses approach_runner.write_to_mb /
run_llm_approach / _extract_plan_shape) so outcomes are recorded per arrival.

RESUMABLE: every cell writes data/grid_cells/<scenario>_<approach>_<seed>_<fam>.json the
moment it finishes and is skipped if present. Trained stacks cache to
results/wp7/ckpt_grid/. Kill/relaunch is always safe.

PARTS: --part {smoke,1,full}. smoke = 1 test family / 1 seed / few rounds, all approaches.
part 1 = one scenario+seed, all approaches+families (bug gate). full = everything.

Requires PYTHONHASHSEED=0 (BC/hash determinism) — guarded below.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# ── PYTHONHASHSEED guard (docs/PREREG_AMENDMENT_2026-07-18_U.md §U.2b) ─────────
# Only when run as a script. The guard used to fire on IMPORT, so ANY process that
# imported this module was silently replaced via os.execv with a re-exec'd copy of
# itself: under pytest that killed the whole session with no traceback and no exit
# code, and under any other importer it would relaunch that importer's argv. The
# determinism pin is a property of the RUN, so it belongs behind __main__.
# Importers that need the pin must set PYTHONHASHSEED=0 in their own environment;
# `main()` verifies it below rather than trusting the re-exec.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)  # re-exec once, pinned

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import wp7_runner as W  # noqa: E402
import approach_runner as F  # noqa: E402
import r_local_runner as R  # noqa: E402
import orion.mdo.observation as OBS  # noqa: E402
from orion.substrate.hierarchical_topology import (  # noqa: E402
    HELDOUT_INSTANCES, TRAIN_INSTANCES, generate_hierarchical_topology,
)
from orion.sim.load_levels import (  # noqa: E402
    NUM_ARRIVALS, TRAINING_LEVEL, WARMUP_ARRIVALS, get_level,
)
from orion.sim.scenario_slices import make_scenario_slice_factory  # noqa: E402
from orion.sim.arrival_process import ArrivalProcess, EventType  # noqa: E402
from orion.baselines.colocation_ffd import colocation_ffd  # noqa: E402
from orion.sim.acceptance import REJECT_BINS, AcceptanceReport  # noqa: E402
from orion.sim.qos_gate import plan_qos_reason  # noqa: E402
from orion.baselines.greedy_ffd import GreedyConfig  # noqa: E402
from orion.llm.condition_signature import compute_condition_signature  # noqa: E402
from orion.llm.episodic_memory import EpisodicMemory  # noqa: E402
from orion.sim.episode_runner import build_placement_plan  # noqa: E402
from orion.sim.verifier import verify_committed_plan  # noqa: E402
from orion.profiling import profiled  # noqa: E402
from orion.retrieval import RetrievalConfig, RetrievalMode  # noqa: E402
from orion.provenance import git_provenance, serving_provenance  # noqa: E402
from partial_obs_prior import partial_obs_builder  # noqa: E402
from cost_metrics import CostAccumulator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("grid")

PREREG = "docs/PREREG_AMENDMENT_2026-07-27_Y_DRAFT.md"
# Banked-cell directory. Overridable for the same reason as ORION_CKPT_DIR below:
# a diagnostic or a re-derive that wants to compare against the results of record
# must be able to write somewhere other than on top of them. The default stays the
# real one, so no production invocation changes behaviour.
CELLS = Path(os.environ.get("ORION_CELL_DIR", "data/grid_cells"))
# Checkpoint directory. `curriculum_train` writes segment checkpoints here by a
# name derived only from (scenario, config, seed, segment), so ANY caller that
# reuses those coordinates overwrites the results of record in place. A probe
# calling curriculum_train to inspect the curriculum did exactly that on
# 2026-08-02 and clobbered three of seed 42's segment checkpoints. Diagnostics
# must set ORION_CKPT_DIR to a scratch path; the default stays the real one so
# no production invocation changes behaviour.
CKPTS = Path(os.environ.get("ORION_CKPT_DIR", "results/wp7/ckpt_grid"))
APPROACHES = ["Plain", "MDO-fullobs", "MDO-partial", "MDO-partial-noh", "MDO-partial-modal", "MDO-partial-obsenc", "MDO-partial-fact", "MDO-partial-noh-fact", "MDO-ffd", "RL-alone", "RL-advised", "RL-poprior", "Memory-off", "Memory-off-rpg", "Memory-off-rp", "Full", "Full-rp"]
# M^B capacity. Defined HERE, not inherited from r_local_runner, because this is
# the runner that decides how much the store is asked to hold.
#
# Was R.MEMORY_CAPACITY_K = 50, set on 2026-07-02 (d21dcd8) to force 4-8x
# oversubscription so a FIFO-vs-importance eviction ablation had something to
# measure. That ablation is no longer in APPROACHES, so the 50 was a leftover
# that every §Y.15 cell ran under: warm-up offers 4 x --warm-arrivals candidate
# writes, and all 34 banked memory cells came back at exactly mb_entries=50,
# i.e. saturated, with most of the warm-up evicted before eval began.
#
# 500 was tried on 2026-08-11 and STILL bound: the warm-up offers 4 x
# --warm-arrivals candidate writes, so at --warm-arrivals 200 that is 800 into
# 500, and Probe A's Full cell tripped the saturation guard at 500/500.
#
# A capacity constant is the wrong shape for this. M^B is an accumulating record
# of past episodes, and the claim it exists to support is that experience helps,
# so a cap that bites limits the mechanism rather than describing it. None =
# unbounded, which is the default; an int is still honoured so a capacity
# ablation stays possible, and `_mb_capacity` says when one would bind.
MEMORY_CAPACITY_K = None
# "frozen" is the §P transfer protocol: warm the store, freeze it, evaluate. That
# measures a memory with accumulation off. "accumulate" lets it grow during the
# evaluation, which is what the mechanism claims to do. Recorded per cell as
# mb_mode, so a frozen cell is never confused with an accumulating one.
MB_MODE = "frozen"


def _mb_capacity(warm_arrivals: int) -> int:
    """Entries the store may hold. Never below the warm-up's write volume.

    The warm-up offers 4 x warm_arrivals candidate writes (four training
    instances) and the store is frozen for eval, so 4 x warm_arrivals is the
    exact ceiling on what can ever be written. Unbounded is that number with
    headroom rather than an arbitrary large constant, so the saturation guard
    still has something real to compare against.
    """
    need = 4 * int(warm_arrivals)
    if MEMORY_CAPACITY_K is None:
        return max(4 * need, 2000)
    if MEMORY_CAPACITY_K < need:
        log.warning("MEMORY_CAPACITY_K=%d is below the warm-up write volume "
                    "(4 x %d = %d): the store will evict, and retrieval will "
                    "then be measuring eviction rather than the floor",
                    MEMORY_CAPACITY_K, warm_arrivals, need)
    return MEMORY_CAPACITY_K
# Plan cache. True for every run of record; --no-plan-cache sets it False and is
# refused unless the cells go to a scratch ORION_CELL_DIR. Module level rather
# than an argument because `_wire` runs per cell and would otherwise put the
# cache back on between cells.
USE_PLAN_CACHE = True
# Inline profiling, off unless --profile. Set to a live PowerSampler by main().
PROFILE_SAMPLER = None
PROFILE_DIR = Path(os.environ.get("ORION_PROFILE_DIR", "results/profile_cells"))
MEMORY_APPROACHES = {"Full": "selective"}  # approach -> M^B write_policy
# §AA (2026-08-17): the LLM plan path held to the same admissibility tests the
# partial-observability heuristic applies to its own proposal. See
# `partial_obs_prior.repair_plan` for the measurement that motivates them and for
# why this is a parity correction rather than a new advantage. Both are additive
# over `Memory-off`, so `Memory-off` itself is the control and the shipped cells do
# not move: the repair is OFF unless one of these names is requested.
#
#   Memory-off-rpg   REPAIR_MODE="guard"  per-VNF h^m repair, splits preserved
#   Memory-off-rp    REPAIR_MODE="full"   + collapse an elective split
#   Full-rp          REPAIR_MODE="full"   on the M^B approach, i.e. ORION as reported
REPAIR_APPROACHES = {"Memory-off-rpg": "guard", "Memory-off-rp": "full",
                     "Full-rp": "full"}
# The approach each repair variant is the twin of: same stack, same plan source,
# same mode, same M^B setting. Named once so the dispatch and the checkpoint
# lookup cannot disagree about what a repair cell IS.
REPAIR_BASE = {"Memory-off-rpg": "Memory-off", "Memory-off-rp": "Memory-off",
               "Full-rp": "Full"}
TRAINED_APPROACHES = {"RL-alone", "RL-advised", "RL-poprior", "Memory-off", "Full",
                      *REPAIR_APPROACHES}
# Which curriculum stack each trained approach evaluates. Same mapping run_cell
# dispatches on, named once so --eval-only can report the checkpoint per cell.
# The ladder, by how much of the planner each approach gets.
#
#   RL-alone    heuristic m̃, no advising, in training and at eval. The LLM-free
#               control: the MDO still receives a proposal, from a cheap planner.
#   Memory-off  llm_guided: Agent B builds m̃ during training AND at eval, advising on
#     / Full    in both. This is ORION. Full adds M^B on top of Memory-off.
#
# The planner is therefore in training, not only at inference, for the approaches whose
# claim requires it, and each approach evaluates the way it trained.
#
# The stack this replaces, `llm_prior`, put the LLM in training as a KL TARGET and is
# deleted. That channel was measured inert (agreement 0.007 at beta=25 on a fixed
# stream, trained argmax worse than untrained), the teacher it distilled never
# partitions so gating on it was circular, and swapping only those weights cost 6.35 pp
# at L2, 7.80 at L3 and 10.10 at L4. `llm_guided` keeps the LLM in training and drops
# only the mechanism that carried nothing.
STACK_FOR_APPROACH = {"RL-alone": "rl_alone", "RL-advised": "po_advised",
                      "RL-poprior": "po_prior",
                      "Memory-off": "llm_guided", "Full": "llm_guided"}
# Derived, never restated: a repair variant runs its base approach's stack, so the
# two cannot drift apart and a repair cell can never silently evaluate a different
# checkpoint than the row it is compared against.
STACK_FOR_APPROACH.update({a: STACK_FOR_APPROACH[b] for a, b in REPAIR_BASE.items()})
MEMORY_APPROACHES.update({a: MEMORY_APPROACHES[b] for a, b in REPAIR_BASE.items()
                          if b in MEMORY_APPROACHES})
# The planner advises the MDO at inference for these approaches (mode="advised"), and
# they are the ones that also train advised. RL-alone and RL-poprior decode
# deterministically, in training and at eval alike.
ADVISED_APPROACHES = {"Memory-off", "Full", "RL-advised",
                      *(a for a, b in REPAIR_BASE.items()
                        if b in {"Memory-off", "Full", "RL-advised"})}
# §Z.1 (2026-08-06): anneals to 0. The prior term is Kickstarting-style auxiliary
# shaping (arXiv:1803.03835: cross-entropy to the teacher on the STUDENT's own
# trajectories, weight decayed to zero), not a standing objective. Two reasons the
# old 0.2 floor was wrong. Shah et al. (arXiv:2512.21852) show a KL added to the
# loss alone over on-policy contexts keeps the pathwise term and drops the
# score-function one, so a permanent floor holds a structurally biased gradient
# alive forever; a floor of 0 lets it expire. And Kickstarting reports linear
# schedules reaching 0 outperform ones that do not. The old comment justified the
# floor with "the LLM always advises", which conflates two channels: inference-time
# advising is ADVISED_APPROACHES + prior_weight in AutoregMDOPolicy.forward, and it
# is untouched by beta. Beta is training-only.
BETA_FLOOR = 0.0

# ── §Y: which scenario classes this grid may run ─────────────────────────────
# `stress` (the pre-§Y rho-ramp class) is NOT a §Y axis and is excluded here
# rather than merely defaulted off. It was the default alongside `conventional`
# until 2026-08-03, so `grid_runner.py --part full` with no arguments produced 80
# cells of a scenario the registered design does not contain, at full cost and
# without complaint. The §Y axes are S1 `conventional` and S2 `complex`.
Y_SCENARIOS = ["conventional", "complex"]

# ── §Y.14: instance roles (pre-registered 4bf325d) ───────────────────────────
# Instance 100 is the EVALUATION instance and is reported. Instances 101-104 are
# spent on checkpoint selection. They are disjoint on purpose: selecting on 100
# would be selection on test.
EVAL_INSTANCE = HELDOUT_INSTANCES[0]
VALIDATION_INSTANCES = tuple(HELDOUT_INSTANCES[1:])
# Stack -> the approach whose eval path scores its segments on the validation
# instances. Both stacks are LLM-free here, so the probe IS the arm itself and a
# validation pass costs CPU only, no model calls.
#
# With `llm_prior` deleted, the LLM approaches share rl_alone's checkpoint sequence
# and therefore its selected segment. The §Y.14b proxy question, whether a segment
# chosen on an LLM-free probe is the right one for an advised arm, no longer arises
# as a choice between stacks: there is one policy, selected once, and the LLM
# approaches run it. Every row in the table is selected by the same rule.
SELECTABLE_STACKS = {"rl_alone": "RL-alone", "po_prior": "RL-poprior",
                     "llm_guided": "Memory-off"}


# ── per-cell wall-clock timeout (§Y, required before the scalability axis) ────
#
# A hung cell is not recoverable by the durable restart loop: the supervisor sees
# a live process and never restarts it, so one stalled cell parks a core
# indefinitely and the grid never completes. A cell must be able to FAIL.
#
# Honest limitation, stated because it decides where this can be relied on:
# SIGALRM is delivered between Python bytecodes, so it interrupts Python-level
# loops but NOT a single long-running C call. The known hang was exactly such a
# call (`list(itertools.product(...))`); that one is fixed at the source in
# `approach_runner.compute_ceiling` by streaming rather than materialising, and
# by refusing §Y-scale substrates outright. This timeout is the net for
# everything else, not a substitute for that fix. On Windows SIGALRM does not
# exist and the timeout is a no-op, logged once.
CELL_TIMEOUT_S = int(os.environ.get("ORION_CELL_TIMEOUT_S", 4 * 3600))


class CellTimeout(RuntimeError):
    """Raised when a cell exceeds CELL_TIMEOUT_S."""


@contextmanager
def cell_timeout(seconds: int, label: str):
    """Bound one cell's wall-clock time. No-op where SIGALRM is unavailable."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        if seconds > 0 and not _TIMEOUT_UNAVAILABLE_LOGGED:
            log.warning("per-cell timeout unavailable on this platform (no SIGALRM); "
                        "cells run unbounded")
            globals()["_TIMEOUT_UNAVAILABLE_LOGGED"] = True
        yield
        return

    def _fire(signum, frame):
        raise CellTimeout(f"cell {label} exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


_TIMEOUT_UNAVAILABLE_LOGGED = False


# ── resumable cell I/O (modeled on d_runner) ──────────────────────────────────
def _cell_path(scenario, approach, seed, level, instance):
    """Cell identity INCLUDES the evaluation instance.

    Without it, evaluating the same (scenario, approach, seed, level) on a second
    held-out instance is skipped as already banked -- so the instance-generalization
    axis would silently return the axis-1 cells instead of running.
    """
    return CELLS / f"{scenario}_{approach}_{seed}_{level}_i{instance}.json"


def _done(scenario, approach, seed, level, instance):
    p = _cell_path(scenario, approach, seed, level, instance)
    if p.exists():
        log.info("SKIP %s/%s seed=%d %s inst%s (banked)",
                 scenario, approach, seed, level, instance)
        return True
    return False


def _bank(scenario, approach, seed, level, instance, payload, prov):
    CELLS.mkdir(parents=True, exist_ok=True)
    # §AA — the two knobs that change what an approach IS ride into every cell,
    # unconditionally. A cell that does not state them cannot be compared with one
    # produced under the other setting, and both default to the shipped value, so an
    # absent field would be indistinguishable from a diagnostic value.
    import partial_obs_prior as _pop
    from orion.mdo.coordinator import ADVISE_WEIGHT as _aw
    payload = dict(payload, scenario=scenario, approach=approach, seed=seed,
                   level=level, instance=instance, provenance=prov,
                   knobs={"advise_weight": _aw,
                          "plan_repair": REPAIR_APPROACHES.get(approach, "off"),
                          "partial_guard": _pop.GUARD_MODE,
                          "partial_seq": _pop.SEQ_ACCOUNTING})
    path = _cell_path(scenario, approach, seed, level, instance)
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("BANKED %s", path)


# ── substrate / factory wiring ────────────────────────────────────────────────
# §Y.11 (2026-07-29): the size axis is REMOVED. There was an eval-only
# nodes-per-domain override addressed by a '@n8' family suffix that rode through
# the cell filename, the readout grouping and the substrate factory. The topology
# size is now fixed, so that machinery is gone rather than defaulted off: a
# disabled-by-default size knob is a path back to producing cells that cannot be
# compared with the rest of the grid.
def _substrate_fn(instance_seed):
    """§Y substrate: one generator, one fixed size, instance chosen by seed.

    The argument the callable receives is the RUN seed, which varies the arrival
    stream; the substrate instance is bound here and does NOT follow it. Keeping
    those separate is what makes the reported spread arrival-stream spread rather
    than a mixture of stream and instance variation.
    """
    return lambda _run_seed: generate_hierarchical_topology(instance_seed)


def _wire(scenario, level_name, instance_seed):
    """Point wp7's injection globals at this load level + instance + scenario."""
    OBS.USE_NODE_BASED_TIER_MASK = True
    W.MDO_POLICY_KIND = "autoreg"
    W.RC_SUBSTRATE_FN = _substrate_fn(instance_seed)
    W.RC_SLICE_FACTORY = make_scenario_slice_factory(scenario)
    W.RC_LOAD_LEVEL = get_level(level_name)   # calibrated lambda/mu (§Y.3)
    W.RC_NUM_ARRIVALS = NUM_ARRIVALS
    # Credit assignment: ONE-STEP bootstrap across arrivals, not the whole stream.
    #
    # wp7's module default is still ADV_MODE="stream_gae", which treats the entire
    # arrival stream as a single episode with a terminal only at the last arrival.
    # That is the formulation the §N.1 gate fault was traced to (the critic never
    # fits the returns, EV ~ 0, and the advantages degrade to position noise), and
    # §V.4/§W moved the mainline to td0 -- but only the probe scripts ever set it,
    # so the grid silently inherited the faulty mode. At N=2000 it would be worse
    # still: a 2000-arrival discounted return at gamma=0.99 has an effective
    # horizon of ~100, so most of the trace is discounted away while the critic is
    # still asked to fit it.
    #
    # td0:    A_a = r_a + gamma*V_{a+1} - V_a   one arrival of lookahead, so a
    #                                            placement that fragments the
    #                                            substrate is still charged for it
    # bandit: A_a = r_a - V_a                    fully myopic; this is the
    #                                            per-arrival-episode convention
    W.ADV_MODE = "td0"
    # Prior coupling: the SAME omission as ADV_MODE, found 2026-08-07, and this one
    # already cost a banked block.
    #
    # wp7's module default is PRIOR_LOSS="sampled_kl", whose own comment reads "a
    # contradictory objective ... measured: cannot align at any beta". For a
    # colocation m~, once step 0 samples a domain other than m~[0] the term pulls
    # step 1 toward m~[1] conditioned on the wrong prefix, against the decoder's
    # running-count conditioning. The grid pinned ADV_MODE and never pinned this,
    # so RL-poprior trained under it and lost to RL-alone in 20/20 paired cells,
    # -7.5 pp at L2 and -11.2 pp at L4.
    #
    # §Z.1 repaired distill's reduction (mean -> sum, so beta means the same thing
    # under both) and dropped BETA_FLOOR to 0, but neither changed WHICH term the
    # grid runs. This line does. distill is teacher-forced, so its known weakness
    # is exposure bias, measured at ~28 pp and only after ~100 rounds (KLS7), which
    # is a bias in the state distribution rather than an objective that contradicts
    # itself.
    W.PRIOR_LOSS = "distill"
    # Cache ON for every run of record: see wp7_runner.RC_USE_PLAN_CACHE. The
    # only way this reads False is --no-plan-cache, which is refused unless the
    # cells go to a scratch directory.
    W.RC_USE_PLAN_CACHE = USE_PLAN_CACHE
    W.RC_FIXED_TRAIN_STREAM = True   # content-memo => ~arrivals LLM calls / segment
    W.TIER_FILTER_LLM_PLANS = True
    W.CUSTOM_PLAN_BUILDER = None
    W.TRAIN_MDO_MODE = "sample"      # advised_sample is opt-in per curriculum config
    W.EVAL_LOAD_LEVEL = level_name   # rides into the M^B condition key (§Y.6)


# ── curriculum training ───────────────────────────────────────────────────────
def _curriculum_segments(seed, passes=1, n_instances=None):
    """The (level, instance) segment list for one seed's curriculum.

    Single definition, shared by `curriculum_train` (which walks it) and
    `load_stack` (which needs to know which segment index the FINAL checkpoint
    carries). Deriving the index instead of globbing for the highest `seg*.pt` is
    what makes a stale checkpoint set fail loudly: an 8-segment pre-§Y.13 stack
    leaves no `seg19`, so loading it raises rather than quietly evaluating a stack
    trained under a curriculum that no longer exists.
    """
    pool = list(TRAIN_INSTANCES)
    n_seg = len(pool) if n_instances is None else max(1, min(int(n_instances), len(pool)))
    rng = np.random.default_rng(seed)
    insts = list(rng.choice(pool, size=n_seg, replace=False)) * passes
    return [(TRAINING_LEVEL, int(inst)) for inst in insts]


def load_stack(scenario, seed, config, lr, passes=1, n_instances=None, seg=None):
    """Rebuild a trained coordinator from its banked segment checkpoint. NO training.

    §Y.13 (2026-08-03). `train_stack` had no skip-if-exists path, so any question
    that needed a fresh evaluation of an already-trained stack (re-deriving the
    rejection breakdown after the `unattributed` fix, scoring an earlier segment)
    cost a full ~8.5 h retrain of checkpoints that were already on disk. The
    weights are the experiment; the training loop is how they got there, and it
    does not need to run twice.

    Returns (coord, ckpt_path). `seg` selects an intermediate segment instead of
    the final one; the default is the last segment of the curriculum this seed
    would run, derived from `_curriculum_segments` rather than globbed.

    Fails loudly on a missing checkpoint or on an observation-layout mismatch.
    Both are cases where the alternative is an evaluation that looks normal and
    means nothing: an untrained policy still returns a plausible acceptance
    number, and so does one whose obs_dim no longer matches the substrate.
    """
    segments = _curriculum_segments(seed, passes, n_instances)
    idx = len(segments) - 1 if seg is None else int(seg)
    if not 0 <= idx < len(segments):
        raise SystemExit(f"segment {idx} is outside this curriculum (0..{len(segments) - 1})")
    level_i, inst_i = segments[idx]
    ck = CKPTS / f"{scenario}_{config}_{seed}_seg{idx}.pt"
    if not ck.exists():
        raise SystemExit(
            f"eval-only: no checkpoint {ck}. The curriculum for seed {seed} has "
            f"{len(segments)} segments, so the final one is seg{len(segments) - 1}. "
            "A stack trained under an earlier curriculum does not satisfy this "
            "request; retrain it rather than evaluating the segment that happens "
            "to exist.")

    # The segment INDEX does not identify the curriculum that wrote it. Asking for
    # an 8-segment curriculum against a 20-segment checkpoint set finds a live
    # `seg7` and loads a stack that is 8/20 of the way through a different
    # schedule, which returns a plausible acceptance number and is silently the
    # wrong experiment. The checkpoint set's own length is the identifying
    # evidence available on disk, so require it to match.
    on_disk = sorted(int(p.stem.rsplit("seg", 1)[1])
                     for p in CKPTS.glob(f"{scenario}_{config}_{seed}_seg*.pt"))
    if on_disk and on_disk[-1] != len(segments) - 1:
        raise SystemExit(
            f"eval-only: {CKPTS} holds seg0..seg{on_disk[-1]} for "
            f"({scenario}, {config}, seed {seed}), but the curriculum requested "
            f"here has {len(segments)} segments (final seg{len(segments) - 1}). "
            "These checkpoints were written by a different curriculum; seg"
            f"{idx} of that run is not seg{idx} of this one. Match "
            "--passes/--train-instances to the run that produced them, or retrain.")

    # Same wiring the checkpoint was written under, so obs_dim and the domain
    # count are read off the same substrate shape. `_wire` also pins
    # MDO_POLICY_KIND, which decides which policy class `build_stack` builds.
    _wire(scenario, level_i, inst_i)
    torch.manual_seed(seed)
    np.random.seed(seed)
    sub0 = W._make_sub(level_i, seed)
    policy, coord, critic, _opt, _optc, obs_dim, num_domains = W.build_stack(sub0, seed, lr)

    sd = torch.load(ck, map_location="cpu")
    for key, got in (("obs_dim", obs_dim), ("num_domains", num_domains)):
        want = sd.get(key)
        if want is not None and want != got:
            raise SystemExit(
                f"eval-only: {ck} was written with {key}={want}, this substrate "
                f"gives {got}. The checkpoint predates a change to the observation "
                "or the topology; it cannot be evaluated on this one.")
    policy.load_state_dict(sd["policy_state"])
    critic.load_state_dict(sd["critic_state"])
    policy.eval()
    fc = sd.get("final_curve") or {}
    log.info("### eval-only: loaded %s stack (%s seed=%d) from %s seg%d/%d "
             "[train-time eval_acceptance=%s]", config, scenario, seed, ck.name,
             idx, len(segments) - 1, fc.get("eval_acceptance"))
    return coord, str(ck)


def select_checkpoint(scenario, seed, config, args):
    """§Y.14 — choose this stack's segment by validation acceptance. NO training.

    Pre-registered at `4bf325d` before the run that established it: score every
    segment checkpoint on `VALIDATION_INSTANCES` at `TRAINING_LEVEL`, take the
    highest mean, and report that segment at `EVAL_INSTANCE`. Ties break to the
    earlier segment, which `max` gives for free by scanning in order.

    This exists because the stacks are NOT converged at the end of the
    curriculum: adjacent segment checkpoints ten rounds apart differ by up to
    11.9 pp on the evaluation instance, which exceeded the between-seed spread
    the §Y.13 criterion was computed on. Selecting cut RL-alone's L2 seed sd from
    0.0354 to 0.0070. It is the DEFAULT so that a run cannot quietly fall back to
    the superseded final-segment readout; `--final-segment` opts out explicitly.

    Returns (coord, ckpt_path, info) where `info` rides into every cell this
    stack produces, so a reader can see which segment was chosen and on what.
    """
    approach = SELECTABLE_STACKS.get(config)
    if approach is None:
        # Not a refusal to be worked around. Which probe a stack is selected on
        # changes what its arms mean, so it is registered before it is run.
        raise SystemExit(
            f"§Y.14 does not specify the checkpoint-selection probe for the "
            f"'{config}' stack. Selecting it on an unregistered probe would "
            "make the arm's meaning depend on an undocumented choice; running "
            "it on the final segment would use the superseded §Y.13 readout "
            "that missed its criterion. Amend "
            "docs/PREREG_AMENDMENT_2026-08-03_Y14.md first, or pass "
            "--final-segment to deliberately accept the old readout.")

    segments = _curriculum_segments(seed, args.passes, args.train_instances)
    log.info("### §Y.14 selecting %s checkpoint (%s seed=%d): %d segments x %d "
             "validation instances, probe=%s%s", config, scenario, seed,
             len(segments), len(VALIDATION_INSTANCES), approach)
    t0 = time.time()
    curve = []
    for g in range(len(segments)):
        coord, _ = load_stack(scenario, seed, config, args.lr, args.passes,
                              n_instances=args.train_instances, seg=g)
        scores = [eval_nonmemory(coord, approach, scenario, seed, TRAINING_LEVEL,
                                 inst, args.arrivals, None, None)["acceptance"]
                  for inst in VALIDATION_INSTANCES]
        curve.append(float(np.mean(scores)))
    best = int(np.argmax(curve))   # ties -> earliest, as registered
    coord, ck = load_stack(scenario, seed, config, args.lr, args.passes,
                           n_instances=args.train_instances, seg=best)
    log.info("### §Y.14 selected seg%d/%d val=%.4f (final seg val=%.4f, "
             "delta %+.1f pp) in %.1f min", best, len(segments) - 1, curve[best],
             curve[-1], 100 * (curve[best] - curve[-1]), (time.time() - t0) / 60)
    info = {"rule": "Y14_validation_selected", "prereg": "4bf325d",
            "stack": config, "selection_probe": approach,
            "selected_segment": best, "n_segments": len(segments),
            "validation_instances": list(VALIDATION_INSTANCES),
            "validation_score": round(curve[best], 4),
            "final_segment_score": round(curve[-1], 4),
            "validation_curve": [round(c, 4) for c in curve]}
    return coord, ck, info


def curriculum_train(scenario, seed, config, rounds, arrivals, lr, agent_b, kb, passes=1,
                     init_from=None, n_instances=None):
    """One stack, warm-started across the training INSTANCES `passes` times.

    config == "rl_alone"  -> beta=0, greedy m~, mock=True, no LLM.
    config == "po_prior"  -> §V.2: beta anneals 1->BETA_FLOOR, KL prior = the
                             partial-obs heuristic partition (LLM-free).
    config == "po_warmstart" -> §W.3: beta=0 (KL channel off, it cannot align
                             anyway), m~ = partial-obs heuristic (obs features +
                             eval builder), curriculum entered from `init_from`
                             (the §V.4 BC ckpt). LLM-free. The warm-start IS the
                             guidance channel here.
    `init_from`: optional ckpt path to warm-start SEGMENT 0 from (later segments
    chain as always). Returns (final_coord, final_ckpt_path).
    """
    CKPTS.mkdir(parents=True, exist_ok=True)
    # §Y.4: training is at ONE load level (TRAINING_LEVEL) over the TRAINING
    # INSTANCES. The pre-§Y curriculum rotated over 4 topology families, which no
    # longer exist; what rotates now is the network instance, so the stack sees
    # several draws of the same generator rather than one fixed graph. The load
    # level does NOT rotate -- L1/L3/L4 are held out and evaluated zero-shot,
    # which is the generalization claim §Y actually makes.
    #
    # §Y.13 (2026-08-02) — the curriculum covers the WHOLE training pool.
    # It previously drew 4 instances per seed and repeated them twice over 8
    # segments. That diverged from the registered protocol ("trained once, at R,
    # on the 20 training instances at L2") and it was the dominant term in the
    # RL-alone cross-seed spread: each seed learned a different 4-network sample
    # and was then scored on a held-out instance, so seed 42 reached 67-73%
    # acceptance on its own draw and 51.8% on instance 100. Covering the pool
    # costs nothing: `rps` divides the same `rounds` budget over more segments,
    # so 20 segments x 10 rounds replaces 8 x 25 at identical total updates.
    #
    # The draw is still seeded and still shuffled, so segment ORDER varies by
    # seed. What no longer varies is which networks the stack ever sees.
    segments = _curriculum_segments(seed, passes, n_instances)
    rps = max(4, rounds // len(segments))
    total = rps * len(segments)
    prev_ckpt = init_from
    coord = None
    for i, (level_i, inst_i) in enumerate(segments):
        _wire(scenario, level_i, inst_i)
        ck = CKPTS / f"{scenario}_{config}_{seed}_seg{i}.pt"
        last = (i == len(segments) - 1)
        if config == "rl_alone":
            # §X.3 — observation-legal m~: tier features and structural
            # feasibility come from the request's permitted nodes and domain
            # aggregates only (partial_obs_builder), never from a
            # full-substrate placement. beta=0: no prior coupling.
            W.CUSTOM_PLAN_BUILDER = partial_obs_builder
            W.RC_FIXED_TRAIN_STREAM = False
            bs = be = 0.0
            approach, mock, ab, k, ewt = "RL-alone", True, None, None, True
        elif config == "po_advised":
            # Control for the headline comparison. Identical to rl_alone in
            # every respect except that advising is ON in the rollouts, so
            # {RL-alone, RL-advised} isolates the decode bias of the advised
            # softmax and {RL-advised, Full} isolates who authored the plan.
            # LLM-free: the proposal is the partial-obs heuristic partition.
            W.CUSTOM_PLAN_BUILDER = partial_obs_builder
            W.RC_FIXED_TRAIN_STREAM = False
            W.TRAIN_MDO_MODE = "advised_sample"
            bs = be = 0.0
            approach, mock, ab, k, ewt = "RL-alone", True, None, None, True
        elif config == "po_warmstart":
            # §W.3 — heuristic-BC init + repaired update; guidance via init, not KL.
            W.CUSTOM_PLAN_BUILDER = partial_obs_builder
            W.RC_FIXED_TRAIN_STREAM = False
            bs = be = 0.0
            approach, mock, ab, k, ewt = "RL-alone", True, None, None, True
        elif config == "po_prior":
            # §V.2 — LLM-free good prior: partial-obs heuristic partition as the
            # KL target, injected via CUSTOM_PLAN_BUILDER (the §U.1h mechanism).
            # Builder is cheap and deterministic -> varied streams like rl_alone.
            W.CUSTOM_PLAN_BUILDER = partial_obs_builder
            W.RC_FIXED_TRAIN_STREAM = False
            bs = 1.0 - (i * rps / total) * (1.0 - BETA_FLOOR)
            be = 1.0 - ((i + 1) * rps / total) * (1.0 - BETA_FLOOR)
            approach, mock, ab, k, ewt = "RL-alone", True, None, None, True
        elif config == "llm_guided":
            # ORION's stack. The planner is present in training, not only at
            # inference, through the two channels that carry signal:
            #   1. it builds every m̃, so the plan features, the proposal one-hot in
            #      the observation and the substrate trajectory are all its doing;
            #   2. advising is ON in the rollouts (`advised_sample`), so the policy
            #      is optimised under the same biased distribution it will act with.
            # beta stays 0. The third channel, a KL term to the proposer, is not used:
            # measured at 0.007 agreement with beta=25 on a fixed stream, with the
            # trained argmax worse than an untrained one. Two working channels beat
            # three with one dead one, and its absence is stated rather than hidden.
            W.CUSTOM_PLAN_BUILDER = None      # -> make_llm_plan_builder
            W.RC_FIXED_TRAIN_STREAM = True    # content-memo: ~1 model call per distinct slice
            W.TRAIN_MDO_MODE = "advised_sample"
            bs = be = 0.0
            approach, mock, ab, k, ewt = "LLM+RL-memoff", False, agent_b, kb, True
        else:
            # This branch used to train `llm_prior`: Agent B in the loop as the KL
            # TARGET. Deleted 2026-08-12; `llm_guided` above replaces it with the
            # channels that work. A silent fallthrough would put the model back in
            # the hot loop for whatever config was misspelled.
            raise SystemExit(
                f"unknown curriculum config '{config}'. Trainable stacks are "
                "rl_alone, po_advised, po_warmstart, po_prior and llm_guided.")
        # NO second _wire() here. There was one until 2026-08-12, and because `_wire`
        # resets the per-cell knobs it silently undid everything the branch above had
        # just set. Measured consequences, all of which had been reported as design:
        #   * CUSTOM_PLAN_BUILDER back to None, so rl_alone / po_prior / po_warmstart
        #     fell through to `greedy_plan_builder` and trained on FULL-SUBSTRATE FFD
        #     partitions while being evaluated on `partial_obs_builder`. The §X.3
        #     "observation-legal m̃, never from a full-substrate placement" discipline
        #     was never in force, and po_prior's KL target was not the partial-obs
        #     heuristic it is defined as.
        #   * RC_FIXED_TRAIN_STREAM back to True, so the stacks that ask for varied
        #     streams got a fixed one.
        # The failure was invisible: every run completed and produced a plausible
        # acceptance number. `_wire` is called once, at the top of the loop, before the
        # branch; `test_the_curriculum_branch_is_not_overwritten` pins that.
        out = W.train_approach(approach, level_i, seed, rps, arrivals, lr, bs, be, ab, k,
                          mock=mock, actors=None, entropy_schedule=(0.03, 0.01),
                          eval_with_train_builder=ewt, ckpt_path=str(ck),
                          use_mb=False, init_from=prev_ckpt,
                          return_coord=last)
        if last:
            _, coord = out
        prev_ckpt = str(ck)
        log.info("  curriculum %s seg%d/%d fam=%s beta=%.2f->%.2f",
                 config, i + 1, len(segments), f"{level_i}/inst{inst_i}", bs, be)
    return coord, prev_ckpt


def _ap_for_level(sub, n, rng, slice_factory=None):
    """Arrival process at the CALIBRATED rate for the currently wired level.

    Single seam for every stream built in this module. `W.ARRIVAL_RATE` /
    `W.SERVICE_RATE` are the pre-§Y module constants and are a *fixed* load: using
    them makes an episode ignore the load level entirely, which is silent because
    the run still produces a plausible acceptance number for every cell.

    That is not hypothetical. `eval_plain` built its stream from the module
    constants, so Plain -- the reference the whole load axis is anchored on --
    returned the SAME acceptance (0.727) at L1, L2, L3 and L4, while every other
    approach varied correctly. The load axis would have been reported against a
    flat baseline.
    """
    lvl = W.RC_LOAD_LEVEL
    rate = lvl.arrival_rate if lvl is not None else W.ARRIVAL_RATE
    srate = lvl.service_rate if lvl is not None else W.SERVICE_RATE
    return ArrivalProcess(sub, n, rate, srate, rng,
                          slice_factory=slice_factory or W.RC_SLICE_FACTORY)


# ── Plain heuristic eval (stateful colocation_ffd over the stream) ─────────────
def eval_plain(scenario, seed, level, instance, arrivals):
    _wire(scenario, level, instance)
    sub = _substrate_fn(instance)(seed)
    rng = np.random.default_rng(seed + 777)
    ap = _ap_for_level(sub, arrivals, rng)
    ap.generate()
    active, admitted, total = {}, 0, 0
    cost_acc = CostAccumulator(sub)
    # §Y.13 — Plain gets a rejection taxonomy. It had none, so the reference
    # approach was the one row missing from every mix table. The bins are the
    # shared REJECT_BINS so the row sums and compares like the others; Plain has
    # no pre-commit / post-commit split because it checks before allocating.
    rejects = {b: 0 for b in REJECT_BINS}
    qos_split = {"C7": 0, "C9": 0}
    for ev in ap.events:
        if ev.event_type == EventType.DEPARTURE:
            p = active.pop(ev.request_id, None)
            if p is not None:
                sub.deallocate(p[0], p[1])
            continue
        if ev.event_type != EventType.ARRIVAL or ev.slice_request is None:
            continue
        total += 1
        sr = ev.slice_request
        # Same sampling point as every other approach: top of the arrival, before
        # the decision and before any allocation. Plain drives its own loop, so the
        # EpisodeRunner's `on_arrival` hook does not reach it and the call is
        # explicit here rather than implied.
        cost_acc.sample_utilization()
        # Plain runs its own loop rather than the EpisodeRunner, so it inherits
        # none of the `profiled()` wraps and had no cost row at all. It is the
        # reference approach, so without one there is nothing to state the
        # learned and LLM approaches' per-decision cost against.
        with profiled("plain.decision", {"k": len(sr.vnfs)}):
            res = colocation_ffd(sub, sr, GreedyConfig())
        # §X.2 — same QoS gate the coordinator's commits face (verifier model):
        # a placement that would violate C7/C9 is a reject, not an admission.
        if not res.feasible or res.plan is None:
            rejects["structural"] += 1
            continue
        reason = plan_qos_reason(sub, sr, res.plan)
        if reason is not None:
            rejects["qos_gate"] += 1
            qos_split[reason] = qos_split.get(reason, 0) + 1
            continue
        try:
            sub.allocate(res.plan, sr)
            active[sr.request_id] = (res.plan, sr)
            admitted += 1
            cost_acc.add_plan(sr, res.plan)
        except Exception:  # noqa: BLE001
            # The gate passed but the substrate refused the allocation. Rare and
            # previously invisible; binned rather than dropped so the row sums.
            rejects["unattributed"] += 1
    acceptance = admitted / total if total else 0.0
    _cost = cost_acc.summary()
    _cost["utilization"] = cost_acc.utilization_summary()
    out = {"acceptance": round(acceptance, 4), "admitted": admitted,
           "offered": total, "rejections": rejects, "cost": _cost,
           # Diagnostic, deliberately OUTSIDE the conserving bins: which
           # constraint the gate refused on. Plain's qos_gate column is the same
           # load-dependent model behind post_commit_c7_delay elsewhere.
           "qos_gate_split": qos_split}
    AcceptanceReport(admitted=admitted, offered=total,
                     rejections=rejects).check_conservation()
    return out


# ── heuristic partition through the coordinator pipeline (§V.1, §Y.8) ─────────
def eval_heuristic_pipeline(builder, scenario, seed, level, instance, arrivals):
    """`builder` partition -> coordinator follow_prior + greedy actors.

    No training, no LLM: follow_prior never consults the (untrained) policy, so
    this is a pure heuristic approach through the SAME pipeline as the learned
    approaches (identical stream, actors, routing, verifier).

    Parameterised on the builder so that the ONLY difference between
    `Plain-partial` and `Plain-fullobs` is how much of the substrate the partition
    builder may look at. See `eval_plain_fullobs` for why that matters.
    """
    _wire(scenario, level, instance)
    sub = _substrate_fn(instance)(seed)
    delays = W.build_delays(sub)
    _, coord, *_ = W.build_stack(_substrate_fn(instance)(seed), seed, 3e-3)
    cost = {}
    rejects = {}
    acc, adm, tot, _ = W.eval_acceptance(coord, level, seed, arrivals,
                                         delays, plan_builder=builder,
                                         mode="follow_prior", cost_out=cost,
                                         report_out=rejects)
    return {"acceptance": round(acc, 4), "admitted": adm, "offered": tot,
            "rejections": rejects.get("rejections"), "cost": cost}


def eval_plain_partial(scenario, seed, level, instance, arrivals):
    """§V.1: the heuristic idea at the MDO's OWN observability -- per-domain
    aggregates plus the node-based K x M feasibility mask, no node residuals."""
    return eval_heuristic_pipeline(partial_obs_builder, scenario, seed, level,
                                   instance, arrivals)


def _eval_partial_guard(mode, scenario, seed, level, instance, arrivals, seq=True):
    """MDO-partial with the h^m guard restricted to `mode`.

    Isolates how much of MDO-partial's margin over the learned approaches comes
    from the guard rather than from the heuristic. Everything else is held fixed:
    same builder, same colocation-first idea, same aggregate slack test, same
    pipeline, so any delta is the guard.

    The flag is set on the module rather than through the environment because
    `partial_obs_prior` is imported at startup; it is restored in `finally` so a
    later MDO-partial cell in the same process is unaffected.
    """
    import partial_obs_prior as _pop
    prev, prev_seq = _pop.GUARD_MODE, _pop.SEQ_ACCOUNTING
    _pop.GUARD_MODE, _pop.SEQ_ACCOUNTING = mode, seq
    try:
        return eval_heuristic_pipeline(partial_obs_builder, scenario, seed, level,
                                       instance, arrivals)
    finally:
        _pop.GUARD_MODE, _pop.SEQ_ACCOUNTING = prev, prev_seq


def eval_plain_partial_noh(scenario, seed, level, instance, arrivals):
    """Guard off entirely: commits on the per-domain aggregates alone."""
    return _eval_partial_guard("off", scenario, seed, level, instance, arrivals)


def eval_plain_partial_obsenc(scenario, seed, level, instance, arrivals):
    """Guard computed in the observation's own units, so the heuristic is held to
    exactly the evidence the policy receives. See partial_obs_prior.GUARD_MODE."""
    return _eval_partial_guard("obsenc", scenario, seed, level, instance, arrivals)


def eval_plain_partial_modal(scenario, seed, level, instance, arrivals):
    """Guard restricted to the VNF's modal required tier, the one tier index the
    MDO observation actually publishes per VNF. This is the observation-legal
    baseline: it keeps the guard but denies the heuristic the per-(k,m) tier SET
    that the observation has no field for."""
    return _eval_partial_guard("modal", scenario, seed, level, instance, arrivals)


def colocation_plan_builder(slice_req, substrate):
    """m~ from the FULL-substrate colocation-first heuristic (Plain's own placer).

    The full-observability twin of `partial_obs_builder`: same idea (try to fit the
    whole chain in one domain, split only if that fails), but reading node residuals
    instead of per-domain aggregates. Holding the IDEA fixed is what makes the pair
    an observability ablation.

    `W.greedy_plan_builder` is NOT that twin, despite also being full-obs: it is
    plain FFD, which best-fits each VNF over the whole substrate independently and
    therefore scatters chains across domains. Substituting it would confound
    observability with the heuristic itself, which is the error this function
    exists to avoid. Kept separately as the `Plain-ffd` approach because the size of
    that scattering penalty is worth having on the record.
    """
    result = colocation_ffd(substrate, slice_req, GreedyConfig())
    return F.plan_to_summary(result, slice_req, substrate)


def eval_plain_ffd(scenario, seed, level, instance, arrivals):
    """§Y.8 control: plain FFD partition (scatters chains) through the pipeline.

    Not part of the observability ablation. Its role is to show what the
    coordinator charges for a partition that ignores domain boundaries.
    """
    return eval_heuristic_pipeline(W.greedy_plan_builder, scenario, seed, level,
                                   instance, arrivals)


def eval_plain_fullobs(scenario, seed, level, instance, arrivals):
    """§Y.8: the FULL-substrate colocation partition, same pipeline as Plain-partial.

    Isolates a confound in the headline table. `Plain` and `Plain-partial` differ
    in two things at once, observability AND execution path: Plain calls
    colocation_ffd and allocates straight onto the substrate with a QoS check
    applied afterwards, while Plain-partial goes through the coordinator, the
    domain actors, k-shortest-path routing and the verifier. The gap between those
    two rows therefore does not measure the cost of partial observability.

    That this matters is not hypothetical: in the 2026-07-31 ladder the strictly
    LESS informed Plain-partial BEAT Plain at L1 (+2.6 pt) and L4 (+1.7 pt), which
    a pure observability ablation cannot produce.

    This approach holds BOTH the pipeline and the heuristic idea fixed and moves
    only what the builder may look at, so (Plain-fullobs - Plain-partial) is the
    cost of partial observability and (Plain - Plain-fullobs) is the pipeline
    difference. Untrained and LLM-free, so it costs one Plain row and nothing else.
    """
    return eval_heuristic_pipeline(colocation_plan_builder, scenario, seed, level,
                                   instance, arrivals)


# ── non-memory trained-approach eval (wp7 eval_foc) ─────────────────────────────────
def _with_plan_cache(pb):
    """Wrap an eval plan builder in the plan cache, and return its stats dict.

    RC_USE_PLAN_CACHE only ever wrapped `train_approach`'s builder, so every
    EVALUATION cell called the model once per arrival: 2000 calls at ~6s = 3.36 h
    against a 4 h cell timeout, and 538 h over the grid's 160 LLM cells. The
    cache belongs on both paths or the mechanism the design claims -- consult the
    model per distinct planning situation, not per arrival -- is only true during
    training.

    Note what a hit means for the memory approaches: it bypasses Agent B AND M^B.
    That is the intended shape rather than a leak, because the key now carries
    the network condition, so a miss is precisely a situation that is new in
    intent or in congestion, which is when the memory has something to say. It
    does mean M^B is consulted on the ~174 misses per 2000 arrivals, not 2000
    times, and the reported retrieval counts must be read against that.
    """
    if not USE_PLAN_CACHE:
        return pb, {"disabled": True}
    from orion.llm.plan_cache import PlanCache
    stats: dict = {}
    return W._cached_plan_builder(
        pb, PlanCache(capacity=W.PLAN_CACHE_CAPACITY), stats=stats), stats


def _cache_report(stats):
    if stats.get("disabled"):
        return {"disabled": True}
    lk = stats.get("lookups", 0)
    return {"lookups": lk, "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
            "hit_rate": round(stats.get("hits", 0) / lk, 4) if lk else None}


def eval_nonmemory(coord, approach, scenario, seed, level, instance, arrivals, agent_b, kb):
    _wire(scenario, level, instance)
    sub = _substrate_fn(instance)(seed)
    delays = W.build_delays(sub)
    cache_stats = None
    repair_stats = None
    if approach in ("Memory-off", "Memory-off-rpg", "Memory-off-rp"):
        pb = W.make_llm_plan_builder(agent_b, kb, lambda: None)
        if W.TIER_FILTER_LLM_PLANS:
            pb = W.tier_filtered(pb)
        pb, cache_stats = _with_plan_cache(pb)
        # §AA — repair OUTSIDE the cache, so EVERY served plan is checked against the
        # substrate as it is now. This ordering is the whole point.
        # `_cached_plan_builder` serves ~91% of arrivals from a stored plan whose
        # concrete `suggested_domains` were chosen for an earlier arrival, and its
        # only gate is `revalidate_plan`, which checks that each named domain still
        # CONTAINS a node of the required tier. Composition is fixed, so that test is
        # always true; it is topology revalidation, not capacity revalidation, and it
        # is why a cache hit can commit a domain that is now exhausted.
        # Repairing inside the cache would fix only the ~9% of arrivals that miss.
        if approach in REPAIR_APPROACHES:
            import partial_obs_prior as _pop
            repair_stats = _pop.reset_repair_stats()
            pb = _pop.plan_repaired(pb, mode=REPAIR_APPROACHES[approach])
    elif approach in ("RL-poprior", "RL-alone", "RL-advised"):
        # §V.2 / §X.3: eval on the SAME observation-legal m~ the approach trained on.
        pb = partial_obs_builder
    else:
        raise SystemExit(
            f"eval_nonmemory has no plan source for approach '{approach}'. This used to "
            "fall through to the full-substrate greedy builder for Prior-only, which "
            "gave a partial-observability policy a full-observability plan; the arm is "
            "deleted rather than repaired.")
    mode = "advised" if approach in ADVISED_APPROACHES else "deterministic"
    cost = {}
    rejects = {}
    acc, adm, tot, agree = W.eval_acceptance(coord, level, seed, arrivals,
                                             delays, plan_builder=pb, mode=mode,
                                             cost_out=cost, report_out=rejects)
    return {"acceptance": round(acc, 4), "admitted": adm, "offered": tot,
            "rejections": rejects.get("rejections"),
            "mtilde_agreement": agree, "cost": cost,
            "plan_cache": _cache_report(cache_stats) if cache_stats is not None else None,
            "plan_repair": dict(repair_stats) if repair_stats is not None else None}


# ── memory-approach eval (trained coord + live M^B write; warm on train families) ──
def grid_memory_instance(coord, sub, arrival_seed, agent_b, kb, mb, topo_sig,
                         slice_factory, delays, write, n_arrivals=None,
                         load_level=None):
    """One approach on one instance with the TRAINED coordinator (advised) and a
    live M^B: LLM+memory plan per arrival -> trained coordinator -> outcome ->
    write_to_mb. Mirrors approach_runner.run_instance but with the trained policy.
    n_arrivals caps the stream (warm-up uses a small cap; held-out eval the full).

    §Y.6 — STATEFUL. This loop previously resolved each arrival against a
    deepcopy of the substrate, so admissions were never charged and the stream
    ran forever on an empty network. Two consequences it fixes:
      * every warmed M^B entry recorded a free-network condition, so under the
        condition-keyed retrieval the store could never match a loaded network;
      * the warm-up saw none of the contention the eval path sees, which made it
        a different experiment from the one it was warming for.
    Arrivals now commit through the SAME mechanics as EpisodeRunner (build plan,
    allocate, post-commit verify, deallocate on hard penalty) and departures
    release their resources.
    """
    n = n_arrivals if n_arrivals is not None else F.ARRIVALS_PER_INSTANCE
    rng = np.random.default_rng(arrival_seed)
    # Calibrated rate, not the module constants: the warm-up exists to expose the
    # store to the contention the eval path sees, and a fixed rate would warm it
    # at a load level that no cell is ever evaluated at.
    ap = _ap_for_level(sub, n, rng, slice_factory=slice_factory)
    ap.generate()
    admitted = total = 0
    active: dict[str, tuple] = {}
    for ev in ap.events:
        if ev.event_type == EventType.DEPARTURE:
            held = active.pop(ev.request_id, None)
            if held is not None:
                sub.deallocate(held[0], held[1])
            continue
        if ev.event_type != EventType.ARRIVAL or ev.slice_request is None:
            continue
        total += 1
        sr = ev.slice_request
        # Condition AT DECISION TIME (pre-commit), so the stored state is the one
        # a future query at the same congestion can actually match.
        cond_sig = compute_condition_signature(sub, load_level)
        ok_b, builder_result, plan_dict, violations, plan_shape = F.run_llm_approach(
            "Full-M^B", sr, sub, agent_b, kb, mb, topo_sig, False)
        if not ok_b or builder_result is None:
            v_tags = [getattr(v, "constraint", v) for v in violations]
            if write:
                F.write_to_mb(mb, "Full-M^B", sr, False, plan_dict, v_tags,
                              topo_sig, plan_shape, condition_sig=cond_sig)
            continue
        plan_summary = F.plan_to_summary(builder_result, sr, sub)
        if plan_summary is None:
            continue
        res = coord.resolve_arrival(sub, sr, plan_summary, delays,
                                    mode="advised")  # LLM always advises the MDO
        ok = res.admitted
        if ok and res.partition is not None:
            placement = build_placement_plan(sr, res)
            if placement is None:
                ok = False
            else:
                sub.allocate(placement, sr)
                verdict = verify_committed_plan(sub, placement, sr,
                                                max_inter_domain_hops=3)
                if verdict.hard_penalty_fired:
                    sub.deallocate(placement, sr)
                    ok = False
                else:
                    active[sr.request_id] = (placement, sr)
        admitted += int(ok)
        if ok and plan_shape is None:
            plan_shape = F._extract_plan_shape(builder_result, sr, sub)
        if write:
            v_tag = None
            if not ok and res.decision is not None:
                last = res.decision
                if last.violation:
                    if last.violation.cross_domain_infeasible:
                        v_tag = "C5b"
                    elif last.violation.c7_violated:
                        v_tag = "C7"
                    elif last.violation.actor_infeasible:
                        v_tag = "actor_infeasible"
            # Second outcome loop: what the coordinator committed vs m~'s suggestion.
            committed = list(res.partition) if getattr(res, "partition", None) else None
            suggested = list(getattr(plan_summary, "suggested_domains", []) or [])
            diverged = committed is not None and committed != suggested
            F.write_to_mb(mb, "Full-M^B", sr, ok, plan_dict,
                          [v_tag] if v_tag else [], topo_sig, plan_shape,
                          committed_partition=committed, diverged=diverged,
                          condition_sig=cond_sig)
    return admitted, total


def _new_mb(write_policy, warm_arrivals):
    return EpisodicMemory(
        config=RetrievalConfig(mode=RetrievalMode.NO_RERANK, apply_recency=True, k_final=3),
        max_entries=_mb_capacity(warm_arrivals), write_policy=write_policy,
        evict_policy="importance")


def _plan_shape_from_partition(sr, partition):
    """Abstract plan shape from the COMMITTED partition.

    Abstract on purpose: shape and cut points, never concrete domain ids as an
    exemplar to copy. `_extract_plan_shape` needs node-level placements, which
    the eval path does not hand back, so this derives the same abstract fields
    from the partition the coordinator committed.
    """
    if not partition:
        return None
    dom = {v.vnf_id: d for v, d in zip(sr.vnfs, partition)}
    used = sorted(set(partition))
    cuts = [(fe.source_vnf, fe.target_vnf) for fe in sr.flow_edges
            if dom.get(fe.source_vnf) is not None
            and dom.get(fe.target_vnf) is not None
            and dom[fe.source_vnf] != dom[fe.target_vnf]]
    return {"strategy": "co-locate" if len(used) <= 1 else "split",
            "tier_assignment": [], "cut_points": cuts,
            "inter_domain_links": [], "domains_used": used}


def _accumulating(pb, mb, level):
    """(wrapped builder, on_decision) so M^B grows DURING the evaluation.

    The §P protocol froze the store for the held-out eval. That measures a memory
    with accumulation switched off, which is the one thing the mechanism claims
    to do, so `--mb-mode accumulate` drops the freeze.

    Two halves because the two facts arrive at different times. The condition has
    to be read PRE-decision or every entry records a post-commit state no later
    query can match, and the outcome is only final after the post-commit verify.
    They are paired by request id, so an arrival whose plan build fails (no MDO
    decision, no callback) cannot leak its condition onto the next arrival.
    """
    pending = {}

    def builder(slice_req, substrate):
        cond = compute_condition_signature(substrate, level)
        summary = pb(slice_req, substrate)
        if summary is None:
            pending.pop(slice_req.request_id, None)
            return None
        pending[slice_req.request_id] = (cond, summary)
        return summary

    def on_decision(slice_req, mdo_result, verdict, plan_summary):
        got = pending.pop(slice_req.request_id, None)
        if got is None:
            return
        cond, summary = got
        plan_dict = {"vnf_assignments": [
            {"vnf_id": v, "domain": d}
            for v, d in zip(summary.vnf_ids, summary.suggested_domains)]}
        committed = list(mdo_result.partition) if mdo_result.partition else None
        F.write_to_mb(
            mb, "Full-M^B", slice_req, bool(mdo_result.admitted), plan_dict,
            list(getattr(verdict, "violated", []) or []), None,
            _plan_shape_from_partition(slice_req, committed),
            committed_partition=committed,
            diverged=(committed is not None
                      and committed != list(summary.suggested_domains)),
            condition_sig=cond)

    return builder, on_decision


def eval_memory(coord, write_policy, scenario, seed, level, instance, arrivals, agent_b, kb,
                warm_arrivals, mb_mode, repair_mode=None):
    _wire(scenario, level, instance)
    sf = make_scenario_slice_factory(scenario)
    # Warm M^B ONCE per (scenario, write_policy, seed), capped at warm_arrivals per
    # training instance (the store is 50 entries; warm-up saturates it well before
    # 100). Snapshot to JSON and reused for every eval cell, so the warmed store is
    # identical across them and the comparison is controlled.
    #
    # This used to warm on the PRE-§Y topology families, which was a live defect
    # rather than dead code: the memory approaches would have filled their store
    # against the old 100-node four-tier substrate and then been evaluated on the
    # new one. Under the condition-keyed retrieval of §Y.6 those entries would
    # match rarely and misleadingly when they did. It warms on the §Y TRAINING
    # INSTANCES now, which is also what makes "warmed on train, evaluated on
    # held-out" true of the memory approaches in the same sense as the others.
    # The store is warmed at the TRAINING level, never at the eval level, and the
    # level is in the snapshot name.
    #
    # Two defects this closes, both found 2026-08-04:
    #  * `_wire(scenario, level, instance)` above points the warm-up's arrival
    #    process at the EVAL level, while the snapshot name carried no level. So
    #    the first cell to run created the store and every other level reused it
    #    -- with the default --levels L1 L2 L3 L4 that is an L1-warmed store
    #    (`free` for all 2000 arrivals) served to L2/L3/L4, which are `tight` for
    #    1200-1870 of 2000. Measured separation says such a store abstains on
    #    most arrivals at every level including L2, so Full would have come back
    #    equal to Memory-off with no visible cause.
    #  * load_level was never passed, so every warmed entry recorded
    #    load_level="" while every eval query carries the real level. The
    #    exact-match term in condition_similarity guards on `if la and lb`, so an
    #    empty string SKIPS the term rather than scoring it zero -- the one
    #    exact-match term in the state key has never contributed in any run.
    #
    # Warming at the training level is also what the protocol requires: L1/L3/L4
    # are evaluated zero-shot "without retraining or adaptation", and warming per
    # eval level would hand the memory approaches an adaptation the RL approaches
    # do not get. The consequence is intended and must be read as such: off the
    # training level the store is cross-condition, retrieval abstains on most
    # arrivals, and Full approaches Memory-off. That is the design refusing to
    # reuse a plan from a regime it was not taken in.
    # The warm-up budget and the capacity are IN THE NAME. Both decide what the
    # store contains, and the snapshot is reused whenever it exists, so leaving
    # them out means changing either one silently re-loads the store built under
    # the old one and the change reads as having no effect. The §Y.15 snapshots
    # are w35/k50; they stay on disk and stay loadable under their own name.
    cap = _mb_capacity(warm_arrivals)
    snap = (CKPTS / f"mb_{scenario}_{write_policy}_{seed}_{TRAINING_LEVEL}"
                    f"_w{warm_arrivals}k{cap}.json")
    # --warm-arrivals 0 starts the store EMPTY. Under accumulate that is the
    # honest form of the claim: the memory holds only what this run has lived
    # through, so nothing is carried in from a warm-up whose entries the eval
    # stream never produced.
    if warm_arrivals > 0 and not snap.exists():
        mbw = _new_mb(write_policy, warm_arrivals)
        _wire(scenario, TRAINING_LEVEL, instance)   # warm-up stream at the TRAINING load
        for wi, inst in enumerate(TRAIN_INSTANCES[:4]):
            wsub = _substrate_fn(inst)(seed)
            # Y.6: no topology key. The state term is the condition signature,
            # which grid_memory_instance computes per arrival; a topology key
            # is constant under the fixed Y substrate and scores nothing.
            wsig = None
            wdel = W.build_delays(wsub)
            grid_memory_instance(coord, wsub, seed + 100 + wi, agent_b, kb, mbw, wsig,
                                 sf, wdel, write=True, n_arrivals=warm_arrivals,
                                 load_level=TRAINING_LEVEL)
        CKPTS.mkdir(parents=True, exist_ok=True)
        mbw.save(snap)
        log.info("warmed M^B at %s (%d entries, cap=%d/instance) -> %s", TRAINING_LEVEL,
                 len(mbw._entries), warm_arrivals, snap)
        _wire(scenario, level, instance)         # back to the cell's eval level
    # Fresh copy of the warmed store, FROZEN for the held-out eval (§P transfer
    # protocol). Eval through the SAME stateful eval_foc + make_llm_plan_builder path
    # as Memory-off, so Full vs Memory-off differs ONLY in the memory (mb vs None) --
    # no stateless/stateful or plan-builder confound.
    mb = _new_mb(write_policy, warm_arrivals)
    if warm_arrivals > 0:
        mb.load(snap)
    # Saturation is the failure this guards. Every §Y.15 cell reported exactly
    # mb_entries=50 against a 50-entry cap, which is what eviction-bound looks
    # like from the outside, and nothing in the run said so. If it saturates
    # again the capacity is still the binding constraint and the retrieval
    # numbers are about eviction, not about the floor.
    if len(mb._entries) >= cap:
        log.warning("M^B is at capacity (%d/%d) after warm-up -- eviction is "
                    "binding, so retrieval is not measuring the floor",
                    len(mb._entries), cap)
    mb.reset_retrieval_log()   # count only the held-out eval, not the warm-up
    sub = _substrate_fn(instance)(seed)
    delays = W.build_delays(sub)
    W.EVAL_TOPO_SIG = None   # Y.6: retrieval is condition-keyed, not topology-keyed
    pb = W.make_llm_plan_builder(agent_b, kb, lambda: mb)
    if W.TIER_FILTER_LLM_PLANS:
        pb = W.tier_filtered(pb)
    # Accumulation wraps the builder BEFORE the cache, so a cache hit neither
    # records an episode nor re-reads the condition -- a hit did not consult the
    # model or the memory, and counting it as lived experience would inflate the
    # store with arrivals the mechanism never saw.
    if mb_mode == "accumulate":
        pb, W.EVAL_ON_DECISION = _accumulating(pb, mb, level)
    pb, cache_stats = _with_plan_cache(pb)
    # §AA — OUTSIDE the cache, and outside accumulation. Outside the cache because a
    # hit replays a plan chosen under an earlier substrate state and `revalidate_plan`
    # only re-checks topology; see eval_nonmemory. Outside accumulation because M^B
    # must record the plan the PLANNER authored, not the guard's correction, or the
    # store stops being evidence about the planner.
    repair_stats = None
    if repair_mode is not None:
        import partial_obs_prior as _pop
        repair_stats = _pop.reset_repair_stats()
        pb = _pop.plan_repaired(pb, mode=repair_mode)
    cost = {}
    rejects = {}
    entries_before = len(mb._entries)
    try:
        acc, adm, tot, _agree = W.eval_acceptance(coord, level, seed, arrivals,
                                                  delays, plan_builder=pb, mode="advised",
                                                  cost_out=cost, report_out=rejects)
    finally:
        W.EVAL_ON_DECISION = None
    W.EVAL_TOPO_SIG = None
    if mb_mode == "accumulate":
        log.info("M^B accumulated %d -> %d entries during eval",
                 entries_before, len(mb._entries))
    return {"acceptance": round(acc, 4), "admitted": adm, "offered": tot,
            "rejections": rejects.get("rejections"),
            "mb_entries": len(mb._entries), "mb_capacity": cap,
            "mb_capacity_setting": MEMORY_CAPACITY_K,
            "mb_mode": mb_mode, "mb_entries_before_eval": entries_before,
            "write_policy": write_policy,
            "warm_arrivals": warm_arrivals, "retrieval": mb.retrieval_stats(),
            "warm_level": TRAINING_LEVEL, "plan_cache": _cache_report(cache_stats),
            "plan_repair": dict(repair_stats) if repair_stats is not None else None,
            "cost": cost}


# ── orchestration ─────────────────────────────────────────────────────────────
def run_cell(scenario, approach, seed, level, instance, arrivals, stacks, agent_b, kb, prov,
             warm_arrivals):
    if _done(scenario, approach, seed, level, instance):
        return
    label = f"{scenario}/{approach}/seed{seed}/{level}/inst{instance}"
    t0 = time.time()
    # orion.profiling's `profiled()` wraps are already on the production path
    # (episode_runner plan_build/mdo.decision/verify, agent_b llm.generate/
    # struct.check, coordinator mdo.forward/actor.place) but they are no-ops
    # unless a collector is active, and nothing ever set one -- so no run has
    # recorded a single event. This is where it gets switched on.
    coll = None
    if PROFILE_SAMPLER is not None:
        from orion.profiling import ProfileCollector, set_collector
        coll = ProfileCollector(PROFILE_SAMPLER, label=label)
        set_collector(coll)
    try:
        with cell_timeout(CELL_TIMEOUT_S, label):
            if approach == "Plain":
                out = eval_plain(scenario, seed, level, instance, arrivals)
            elif approach == "MDO-fullobs":
                out = eval_plain_fullobs(scenario, seed, level, instance, arrivals)
            elif approach == "MDO-ffd":
                out = eval_plain_ffd(scenario, seed, level, instance, arrivals)
            elif approach == "MDO-partial":
                out = eval_plain_partial(scenario, seed, level, instance, arrivals)
            elif approach == "MDO-partial-noh":
                out = eval_plain_partial_noh(scenario, seed, level, instance, arrivals)
            elif approach == "MDO-partial-modal":
                out = eval_plain_partial_modal(scenario, seed, level, instance, arrivals)
            elif approach == "MDO-partial-obsenc":
                out = eval_plain_partial_obsenc(scenario, seed, level, instance, arrivals)
            elif approach == "MDO-partial-fact":
                out = _eval_partial_guard("full", scenario, seed, level, instance,
                                          arrivals, seq=False)
            elif approach == "MDO-partial-noh-fact":
                out = _eval_partial_guard("off", scenario, seed, level, instance,
                                          arrivals, seq=False)
            elif approach == "RL-alone":
                out = eval_nonmemory(stacks["rl_alone"], approach, scenario, seed, level, instance, arrivals, agent_b, kb)
            elif approach == "RL-advised":
                out = eval_nonmemory(stacks["po_advised"], approach, scenario, seed, level, instance, arrivals, agent_b, kb)
            elif approach == "RL-poprior":
                out = eval_nonmemory(stacks["po_prior"], approach, scenario, seed, level, instance, arrivals, agent_b, kb)
            elif approach in ("Memory-off", "Memory-off-rpg", "Memory-off-rp"):
                out = eval_nonmemory(stacks[STACK_FOR_APPROACH[approach]], approach, scenario, seed, level, instance, arrivals, agent_b, kb)
            elif approach not in ("Full", "Full-rp"):
                raise SystemExit(f"no eval dispatch for approach {approach!r}")
            else:  # Full / Full-rp
                out = eval_memory(stacks[STACK_FOR_APPROACH[approach]], MEMORY_APPROACHES[approach], scenario, seed,
                                  level, instance, arrivals, agent_b, kb, warm_arrivals,
                                  MB_MODE, repair_mode=REPAIR_APPROACHES.get(approach))
    except CellTimeout as exc:
        # Banked as a distinct outcome, not dropped: a missing cell is
        # indistinguishable from one never scheduled, and a silently absent cell
        # reads downstream as "this configuration was covered".
        out = {"status": "timeout", "timeout_s": CELL_TIMEOUT_S, "error": str(exc)}
        out["wall_s"] = round(time.time() - t0, 1)
        log.error("TIMEOUT %s after %.1f min -- banked as timeout, grid continues",
                  label, out["wall_s"] / 60)
        _bank(scenario, approach, seed, level, instance, out, prov)
        return
    finally:
        if coll is not None:
            from orion.profiling import set_collector
            set_collector(None)
    out["status"] = "ok"
    out["level"] = level
    out["instance"] = instance
    if approach in TRAINED_APPROACHES:
        # Which weights produced this number, and under which selection rule. A
        # cell that does not say cannot be compared with one produced under the
        # other rule, and the two differ by up to 4 pp at L2.
        _stack = STACK_FOR_APPROACH[approach]
        out["stack"] = _stack
        out["stack_ckpt"] = stacks.get("_source", {}).get(_stack)
        out["checkpoint_selection"] = stacks.get("_selection", {}).get(_stack)
    out["wall_s"] = round(time.time() - t0, 1)
    if coll is not None:
        raw = PROFILE_DIR / f"{scenario}_{approach}_{seed}_{level}_i{instance}.json"
        out["profile"] = {"raw": str(raw), "cell_totals": coll.cell_totals(),
                          "summary": coll.save(raw)}
        _pd = out["profile"]["summary"].get("llm.generate", {})
        log.info("  profile: %d events, llm.generate n=%s clean=%s",
                 len(coll.events), _pd.get("count"), _pd.get("n_clean"))
    metric = out.get("acceptance", out.get("foc"))
    log.info("%s -> %s=%s (%.1f min)", label,
             "acceptance" if "acceptance" in out else "FoC", metric,
             out["wall_s"] / 60)
    _bank(scenario, approach, seed, level, instance, out, prov)


def _train_arrivals(args):
    """Training episode length, defaulting to the eval one when unset."""
    return args.train_arrivals if args.train_arrivals else args.arrivals


def get_stacks(scenario, seed, args, agent_b, kb, need):
    """Train (or, under --eval-only, load) the stacks required by `need` approaches.

    `stacks["_source"]` maps each stack key to the checkpoint it was loaded from,
    or None when it was trained in this process. It rides into the banked cell so
    a re-derived cell is distinguishable from a trained-from-scratch one.
    """
    stacks = {"_source": {}, "_selection": {}}
    eval_only = getattr(args, "eval_only", False)
    # Derived from STACK_FOR_APPROACH rather than restated, so the two cannot
    # disagree with the dispatch below.
    by_stack: dict = {}
    for _ap, _st in STACK_FOR_APPROACH.items():
        by_stack.setdefault(_st, set()).add(_ap)
    # rl_alone, po_advised and po_prior train LLM-free, so their curricula get no agent.
    # `llm_guided` trains WITH the planner in the loop and needs both.
    wanted = [("rl_alone", by_stack.get("rl_alone", set()), "RL-alone", None, None),
              ("po_advised", by_stack.get("po_advised", set()), "RL-advised", None, None),
              ("po_prior", by_stack.get("po_prior", set()), "RL-poprior", None, None),
              ("llm_guided", by_stack.get("llm_guided", set()), "LLM-guided",
               agent_b, kb)]
    for key, approaches, label, ab, k in wanted:
        if not need & approaches:
            continue
        if not eval_only:
            log.info("### curriculum-train %s stack (%s seed=%d)", label, scenario, seed)
            curriculum_train(scenario, seed, key,
                             args.rounds, _train_arrivals(args), args.lr, ab, k,
                             args.passes, n_instances=args.train_instances)
        if args.eval_seg is not None:
            # Explicit segment: a diagnostic, already refused unless it writes to
            # a scratch cell directory.
            stacks[key], stacks["_source"][key] = load_stack(
                scenario, seed, key, args.lr, args.passes,
                n_instances=args.train_instances, seg=args.eval_seg)
        elif args.final_segment:
            stacks[key], stacks["_source"][key] = load_stack(
                scenario, seed, key, args.lr, args.passes,
                n_instances=args.train_instances)
            stacks["_selection"][key] = {"rule": "Y13_final_segment",
                                         "superseded_by": "Y14 (4bf325d)"}
        else:
            stacks[key], stacks["_source"][key], stacks["_selection"][key] = \
                select_checkpoint(scenario, seed, key, args)
    return stacks


def main():
    # The re-exec above only fires under __main__, so a run launched some other
    # way (module import, wrapper script) must still be pinned. Fail loudly
    # rather than produce a run whose hash order silently differs from the
    # banked cells (§U.2b: PYTHONHASHSEED drift moved greedy-replay state).
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "PYTHONHASHSEED=0 is required for BC/hash determinism (§U.2b). "
            "Run `PYTHONHASHSEED=0 python scripts/grid_runner.py ...`."
        )
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["smoke", "1", "full"], default="smoke")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--scenarios", nargs="+", default=list(Y_SCENARIOS),
                    choices=Y_SCENARIOS,
                    help="§Y scenario classes: S1 conventional, S2 complex. The "
                         "pre-§Y `stress` class is not a §Y axis and is not "
                         "accepted here.")
    ap.add_argument("--approaches", nargs="+", default=APPROACHES, choices=APPROACHES)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--train-arrivals", type=int, default=None,
                    help="episode length during CURRICULUM TRAINING only. Eval always\n"
                         "uses --arrivals, which must stay at the calibrated N or the\n"
                         "acceptance numbers no longer match the frozen ladder. Exists\n"
                         "because an LLM-in-the-loop stack blocks on one arrival at a\n"
                         "time: at 2000 training arrivals a single seed is >12 h.")
    ap.add_argument("--arrivals", type=int, default=NUM_ARRIVALS,
                    help="episode length; §Y.3 calibrated the levels at this value, "
                         "and a shorter episode cannot reach the offered concurrency")
    ap.add_argument("--levels", nargs="+", default=["L1", "L2", "L3", "L4"],
                    help="load levels to EVALUATE (training is always at TRAINING_LEVEL)")
    ap.add_argument("--eval-instances", type=int, nargs="+",
                    default=[EVAL_INSTANCE],
                    help=f"held-out topology instances to evaluate on. Default "
                         f"{EVAL_INSTANCE}. Instances "
                         f"{list(VALIDATION_INSTANCES)} are spent on §Y.14 "
                         "checkpoint selection and are refused here.")
    ap.add_argument("--warm-arrivals", type=int, default=200,
                    help="M^B warm-up arrivals PER training instance, over 4 "
                         "instances. Was 35, which offered 140 candidate writes "
                         "into a 50-entry store. Capacity is derived from this "
                         "now and never binds, so this argument alone decides "
                         "how much there is to retrieve. Costs ~7.5 s per warm "
                         "arrival, once per (scenario, write_policy, seed).")
    ap.add_argument("--passes", type=int, default=1,
                    help="curriculum passes over the training instances. Default 1 "
                         "since §Y.13: one pass now covers the whole pool, where it "
                         "previously covered a 4-instance sample.")
    ap.add_argument("--train-instances", type=int, default=None,
                    help="how many of the training instances the curriculum covers "
                         "(default: all of them). Lower it only for smoke runs.")
    ap.add_argument("--eval-only", action="store_true",
                    help="load the trained stacks from their banked segment "
                         "checkpoints instead of retraining them. Re-derives an "
                         "evaluation (a fixed rejection taxonomy, a different "
                         "eval instance) in minutes rather than a full retrain. "
                         "Raises if a checkpoint is missing: it never falls back "
                         "to training, and never evaluates an untrained policy.")
    ap.add_argument("--final-segment", action="store_true",
                    help="opt out of §Y.14 checkpoint selection and evaluate the "
                         "LAST curriculum segment, which is the superseded §Y.13 "
                         "readout. That readout missed its pre-registered "
                         "criterion (L2 seed sd 0.040 against 0.03) because the "
                         "stacks are not converged; selection brought it to "
                         "0.0070. Use only to reproduce a pre-§Y.14 cell.")
    ap.add_argument("--eval-seg", type=int, default=None,
                    help="with --eval-only, score an INTERMEDIATE curriculum "
                         "segment instead of the final one. Cells produced this "
                         "way are not the results of record; bank them under "
                         "ORION_CELL_DIR.")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--profile", action="store_true",
                    help="record per-decision wall / CPU-time / GPU energy from "
                         "the production `profiled()` wraps. Raw events per cell "
                         "to ORION_PROFILE_DIR. Serialise the run (one port, "
                         "other slot idle) or the two slots contaminate each "
                         "other's GPU windows.")
    ap.add_argument("--profile-hz", type=float, default=20.0,
                    help="power sampling rate")
    ap.add_argument("--mb-mode", choices=["frozen", "accumulate"], default="frozen",
                    help="frozen = the §P protocol (warm, freeze, evaluate). "
                         "accumulate = M^B writes during the evaluation, so the "
                         "store holds what the run has lived through. Pair with "
                         "--warm-arrivals 0 to start empty.")
    ap.add_argument("--retrieval-floor", type=float, default=None,
                    help="override episodic_memory.RETRIEVAL_FLOOR. The shipped "
                         "0.60 was calibrated on a synthetic probe with four "
                         "separated regimes; a live stream is a continuum, and "
                         "the calibration comment says to re-read the operating "
                         "point from retrieval_stats() on the first real run. "
                         "The floor in force is recorded in every cell.")
    ap.add_argument("--no-plan-cache", action="store_true",
                    help="build a plan per arrival instead of per cache key. "
                         "Diagnostic only: the cells are NOT results of record, "
                         "so ORION_CELL_DIR must point at a scratch path. "
                         "Measured 2.83 s per plan build, so an LLM cell is "
                         "~1.6 h at 2000 arrivals; raise ORION_CELL_TIMEOUT_S.")
    ap.add_argument("--tag", default="GRID")
    ap.add_argument("--no-prereg", action="store_true",
                    help="run without the pre-registration, which is not distributed with "
                         "this repository. Applies only when the document is absent; the "
                         "result JSON records prereg.status=\"skipped\".")
    args = ap.parse_args()

    # §Y.14 — 101 to 104 are spent on checkpoint selection. Reporting a cell on
    # one of them would be reporting the instances the checkpoint was chosen on.
    # `is None`, not falsiness: --eval-seg 0 is a legitimate segment.
    spent = sorted(set(args.eval_instances) & set(VALIDATION_INSTANCES))
    if spent and args.eval_seg is None:
        raise SystemExit(
            f"--eval-instances {spent} are §Y.14 validation instances, spent on "
            "checkpoint selection (pre-registered 4bf325d). Evaluating there "
            f"reports the instances the checkpoint was selected on. Instance "
            f"{EVAL_INSTANCE} is the evaluation instance of record.")

    # The plan cache is pipeline, not ablation, and a run of record never turns
    # it off. --no-plan-cache exists for one question -- whether the approach is
    # limited by how rarely the model is consulted -- and its cells answer that
    # question, not the grid's. So it is allowed, loudly, and only into scratch.
    #
    # Cost, re-measured 2026-08-11 off the 20 banked Memory-off cells: 2.83 s per
    # plan build, intercept ~0. The 6 s this refusal used to quote (hence 3.36 h
    # per cell, 538 h for the grid) is roughly 2x the measured figure.

    global USE_PLAN_CACHE
    if args.no_plan_cache:
        if CELLS == Path("data/grid_cells"):
            raise SystemExit(
                "--no-plan-cache writes cells that are not results of record. "
                "Set ORION_CELL_DIR to a scratch path.")
        USE_PLAN_CACHE = False
        log.warning("PLAN CACHE OFF: one plan build per arrival (~2.83 s each, "
                    "so ~%.1f h per LLM cell at %d arrivals). Cells go to %s and "
                    "are diagnostic, not results of record.",
                    args.arrivals * 2.83 / 3600, args.arrivals, CELLS)
    if not W.RC_USE_PLAN_CACHE and USE_PLAN_CACHE:
        raise SystemExit(
            "RC_USE_PLAN_CACHE is off but --no-plan-cache was not passed. The "
            "plan cache is part of the pipeline, not an option, and it must not "
            "drift off in a long run and be discovered afterwards.")

    # §AA — the advise weight is a property of the POLICY, not of the run: it enters
    # the softmax that PPO takes its ratio over, so a cell produced at a different
    # weight is a cell for a different policy under the same filename. Same refusal
    # as --no-plan-cache and --eval-seg, and for the same reason.
    from orion.mdo.coordinator import ADVISE_WEIGHT as _AW
    if _AW != 2.0:
        if CELLS == Path("data/grid_cells"):
            raise SystemExit(
                f"ORION_ADVISE_WEIGHT={_AW} writes cells that are not results of "
                "record. Set ORION_CELL_DIR to a scratch path.")
        log.warning("ADVISE WEIGHT %.3f (default 2.0): the advised approaches are "
                    "running a different policy. Cells go to %s and are "
                    "diagnostic.", _AW, CELLS)

    if args.eval_seg is not None:
        if not args.eval_only:
            raise SystemExit("--eval-seg requires --eval-only (there is no "
                             "intermediate segment to score during training).")
        # An intermediate segment answers a diagnostic question and produces a
        # cell that is NOT the result of record, but it is written under the same
        # filename as one. Refuse rather than overwrite; ORION_CKPT_DIR exists
        # because that exact mistake was made on 2026-08-02.
        if CELLS == Path("data/grid_cells"):
            raise SystemExit("--eval-seg writes cells that are not the results of "
                             "record. Set ORION_CELL_DIR to a scratch path.")

    # Part scoping.
    if args.part == "smoke":
        args.seeds, args.scenarios = [args.seeds[0]], [args.scenarios[0]]
        args.levels = args.levels[:1]
        args.rounds = min(args.rounds, 24)
        args.arrivals = min(args.arrivals, 200)
        args.passes = 1
        args.train_instances = 4  # a smoke checks the path, not the coverage
        args.warm_arrivals = min(args.warm_arrivals, 8)  # fast memory smoke
    elif args.part == "1":
        args.seeds, args.scenarios = [args.seeds[0]], [args.scenarios[0]]

    # The levels are only meaningful at the episode length they were calibrated
    # at: A = lambda/mu concurrent slices cannot be realised in fewer arrivals.
    # A smoke run is allowed to be short, a real run is not.
    if args.part != "smoke" and args.arrivals < NUM_ARRIVALS:
        raise SystemExit(
            f"--arrivals {args.arrivals} is below the calibrated episode length "
            f"{NUM_ARRIVALS} (§Y.2/§Y.3). The frozen levels offer up to "
            f"{max(get_level(l).erlangs for l in args.levels):.0f} concurrent "
            "slices, which a shorter episode cannot hold.")

    prov = git_provenance(serving=serving_provenance(args.port) if not args.mock else None,
                          tag=args.tag, prereg=PREREG,
                          allow_absent_prereg=args.no_prereg)
    log.info("provenance commit=%s dirty=%s part=%s", prov["git_commit"][:8],
             prov["git_dirty"], args.part)

    global MB_MODE
    MB_MODE = args.mb_mode
    if MB_MODE == "accumulate":
        log.warning("M^B ACCUMULATES during eval (not the frozen §P protocol); "
                    "warm_arrivals=%d", args.warm_arrivals)

    if args.retrieval_floor is not None:
        # Set on the module so the AUTO_FLOOR resolution inside retrieve() picks
        # it up; retrieval_stats() reports the floor actually applied, so the
        # cell records what ran rather than what the source says.
        import orion.llm.episodic_memory as _em
        log.warning("RETRIEVAL_FLOOR %.4f -> %.4f", _em.RETRIEVAL_FLOOR,
                    args.retrieval_floor)
        _em.RETRIEVAL_FLOOR = args.retrieval_floor

    if args.profile:
        global PROFILE_SAMPLER
        from orion.profiling import PowerSampler
        # Own PIDs: this process and the llama.cpp server it talks to, taken from
        # the serving provenance. Everything else on the A6000 is another user's
        # work and its power must not be charged to this experiment.
        own = {os.getpid()}
        _pid = ((prov.get("serving") or {}).get("server_pid"))
        if _pid:
            own.add(int(_pid))
        else:
            log.warning("--profile: no serving PID in provenance, so the LLM "
                        "server's own GPU work will read as foreign load")
        PROFILE_SAMPLER = PowerSampler(hz=args.profile_hz, own_pids=own)
        PROFILE_SAMPLER.start()
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        log.info("profiling ON: own_pids=%s hz=%.0f gpu=%s rapl=%s -> %s",
                 sorted(own), args.profile_hz, PROFILE_SAMPLER.gpu_available,
                 PROFILE_SAMPLER.rapl_available, PROFILE_DIR)
        if not PROFILE_SAMPLER.rapl_available:
            log.warning("RAPL is not readable (needs root): CPU energy is a "
                        "TDP ESTIMATE, not a measurement. Label it as such.")
    log.info("GRID part=%s seeds=%s scenarios=%s approaches=%s levels=%s "
             "eval_instances=%s train_level=%s arrivals=%d rounds=%d passes=%d "
             "train_instances=%s eval_only=%s eval_seg=%s cells=%s selection=%s",
             args.part, args.seeds, args.scenarios, args.approaches, args.levels,
             args.eval_instances, TRAINING_LEVEL, args.arrivals, args.rounds, args.passes,
             args.train_instances if args.train_instances is not None else len(TRAIN_INSTANCES),
             args.eval_only, args.eval_seg, CELLS,
             "Y13_final_segment" if args.final_segment else "Y14_validation_selected")

    need_llm = bool(set(args.approaches) & {"Memory-off", "Full"})
    agent_b = kb = None
    if need_llm and not args.mock:
        agent_b = R._build_local_agent(args.port)
        kb = R._load_kb()

    t_start = time.time()
    for scenario in args.scenarios:
        for seed in args.seeds:
            need = set(args.approaches) & TRAINED_APPROACHES
            # Skip training if every trained cell for this (scenario, seed) is banked.
            pending = {a for a in need
                       if any(not _done(scenario, a, seed, lv, inst)
                              for lv in args.levels for inst in args.eval_instances)}
            stacks = get_stacks(scenario, seed, args, agent_b, kb, pending) if pending else {}
            # Firing order is the order the approaches were REQUESTED in, not the
            # declaration order of APPROACHES. Cells are independent and banked as
            # they complete, so on a run that may be cut short the order decides
            # which rows exist, and that is the caller's judgement rather than an
            # artefact of how the list happens to be written. Duplicates collapse.
            for approach in _requested_order(args.approaches):
                for level in args.levels:
                    for inst in args.eval_instances:
                        run_cell(scenario, approach, seed, level, inst, args.arrivals,
                                 stacks, agent_b, kb, prov, args.warm_arrivals)

    log.info("GRID part=%s DONE in %.1f h", args.part, (time.time() - t_start) / 3600)
    _readout(args, args.levels)
    print("GRID_%s_DONE" % args.part.upper())


def _requested_order(approaches):
    """The requested approaches, de-duplicated, first-occurrence order preserved."""
    seen, out = set(), []
    for a in approaches:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _readout(args, levels):
    _readout_block(args, levels,
                   "§Y GRID — acceptance ratio by offered load "
                   f"(trained at {TRAINING_LEVEL}, L1/L3/L4 are zero-shot)")
    # The pre-§Y boundary note that stood here told the reader to compare Full
    # against Memory-off "per family under [stress]". There are no families and
    # no stress class under §Y, so it described an experiment this runner can no
    # longer perform.
    if not args.final_segment:
        print(f"\nTrained stacks: §Y.14 validation-selected checkpoints "
              f"(instances {list(VALIDATION_INSTANCES)}), reported at "
              f"instance {EVAL_INSTANCE}.")
    else:
        print("\nTrained stacks: FINAL SEGMENT (superseded §Y.13 readout, "
              "--final-segment).")


def _readout_block(args, levels, title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    for scenario in args.scenarios:
        print(f"\n[{scenario}]")
        hdr = f"  {'approach':11s} " + " ".join(f"{f:>12s}" for f in levels) + f" {'mean':>7s}"
        print(hdr)
        n_timeout = 0
        for approach in _requested_order(args.approaches):
            cells_by_fam = {}
            for fam in levels:
                vals = []
                for seed in args.seeds:
                  for inst in args.eval_instances:
                    p = _cell_path(scenario, approach, seed, fam, inst)
                    if p.exists():
                        cell = json.loads(p.read_text())
                        if cell.get("status") == "timeout":
                            n_timeout += 1
                            continue
                        # §Y.5: acceptance is the primary metric; `foc` only
                        # appears in pre-§Y banked cells.
                        vals.append(cell.get("acceptance", cell.get("foc")))
                vals = [v for v in vals if isinstance(v, (int, float))]
                cells_by_fam[fam] = (np.mean(vals), np.std(vals), len(vals)) if vals else None
            row = f"  {approach:11s} "
            allmeans = []
            for fam in levels:
                c = cells_by_fam[fam]
                if c:
                    row += f"{c[0]:6.1f}±{c[1]:4.1f} "
                    allmeans.append(c[0])
                else:
                    row += f"{'--':>12s} "
            row += f" {np.mean(allmeans):6.1f}" if allmeans else f" {'--':>6s}"
            print(row)
        if n_timeout:
            # Never let a truncated grid read as a complete one.
            print(f"  !! {n_timeout} cell(s) TIMED OUT and are excluded from these "
                  f"means (ORION_CELL_TIMEOUT_S={CELL_TIMEOUT_S})")


if __name__ == "__main__":
    main()
