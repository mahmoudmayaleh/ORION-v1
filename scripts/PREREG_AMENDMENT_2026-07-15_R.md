# Pre-registration Amendment §R — RC-family follow-up (2026-07-15)

Status: **COMMITTED (no git; provenance = `prereg_sha256` of this file in every result JSON).**
Family `C+_T-_B-_RC` (RC-v2, gen_seed 20260716) is **locked** — no family edits. The frontier
grid stays **STOPPED** (pilot gate, pre-committed) regardless of any §R outcome. Baseline refs
cited frozen from `results/ARCHIVE_2026-07-15_pre_Q/` and `results/rc_family_validity_RESULT.md`;
Plain-ColocFB and RA never re-run.

## Arms

| # | Arm | Plan source | Cache | Selector | Purpose |
|---|---|---|---|---|---|
| R.1 | ORION-local deployable | Agent B, LLaMA-3-8B | ON | follow_prior | **Primary deployable claim**: local beats Plain across seeds |
| R.2 | Local, diverse sampling | same | OFF | follow_prior | Signature-diversification; quantifies cache thinness |
| R.3 | Frontier, diverse sampling | Sonnet (pinned) | OFF | follow_prior | Does frontier ≥ local under fair sampling; hard $ cap, **100 calls / seed 42 only** |
| R.4 | Full ORION | Agent B local, cache-ON | ON | **trained MDO** (conformant §O) + BC actors | Does the RL selector add admission beyond plan-following |
| R.5 | RL-alone | greedy, no LLM | n/a | trained MDO, β=0 | Selector headroom on RC without LLM guidance |
| ref | Plain-ColocFB | — | — | — | Frozen from RC-v2 validity (mean 40.4%; per-seed 37.1/37.0/47.0) |

All arms: RC-v2, seeds 42/43/44 (bw sweep 70/90/110), byte-identical 100-arrival streams,
per-(seed, family) cold start (cache + M^B wiped, state hash asserted empty and logged). R.3 is
seed 42 only (100-call budget); its comparison is single-seed by construction.

## Δ1 — Training budget 60 → 250 rounds (calibration, dated 2026-07-15)

The ratified §O budget was **60 rounds** on a canary-calibrated critic transient (~20 rounds). §R
sets **250 rounds** for R.4/R.5. Justification, same discipline that stamped 15→60: the RC RL-alone
smoke (`data/rc_train_results_SMOKE100.json`, seed 42, greedy) shows FoC **plateaus by ~R30–100
with valid EV_tail5 = 0.686** and entropy converged by ~R50; 250 gives convergence margin without
open-ended drift. This is calibration on cited smoke evidence, not budget inflation.

## Δ2 — Entropy-collapse handling, pre-named (not a footnote)

The RC-v2 smoke collapsed MDO selection entropy **0.97 → 0.017 by ~R50** against the ratified
floor schedule (c0=0.03 → floor 0.01) that produced ~0.7 on the gate families — RC's admission
landscape is much sharper. Pre-named handling:

- **The run fires with the ratified entropy schedule untouched.** No mid-run tuning.
- **Per-round selection entropy is logged in the trace for BOTH arms** (already emitted as
  `mdo_entropy` in the round curve) — the collapse dynamics are data we want, not just a symptom.
- **Named finding (pre-registered):** if Full-ORION locks **below the R.1 follow_prior reference
  on all three seeds** with the §O.8 telemetry showing early entropy collapse, that is reported as
  *"RC's admission landscape induces premature commitment under the ratified exploration
  schedule."* Any floor/schedule change is a **§R Δ ratified before a re-run, never mid-run.**

## Δ3 — Comparison rewrite (blocker fix)

The comparator for "does the trained MDO add beyond plan-following" is **R.1** (three-seed local
follow_prior, same §R, same streams, same cold-start) — **not** the pilot's single-seed, cache-thin
(6 signatures) 44.4%. The pilot number is cited **only as the single-seed observation it is.**

Corrected pre-named comparisons (each reported with **sign per seed**; any inversion reported
as-is; no composite "system wins" unless all hold):
- **R-Primary (R.1):** R.1 > Plain (40.4 mean) in mean **AND** positive sign all three seeds →
  "the deployable ORION stack captures routing-critical headroom the best fixed heuristic misses"
  enters the paper. Fail → §P scoped paper stands; pilot noted as single-seed observation. No third
  reading.
- **R-Sampling:** `|R.2 − R.1|` reported as the cache-thinness measurement (characterization, no
  pass/fail).
- **R-Frontier (R.3 vs R.2, seed 42, same sampling regime):** R.3 ≥ R.2 → pilot gap was
  thin-sampling (evidence for a future separately-ratified frontier question; grid stays stopped);
  R.3 < R.2 → direction real, domain-tuned 8B beats frontier on this family, one paragraph, closed.
- **R-Selector (R.4, R.5 vs R.1 and Plain):**
  - R.5 > Plain → RL selector alone captures RC headroom.
  - **R.4 > R.1-mean on identical streams** → trained MDO adds admission beyond plan-following.
  - R.4 > R.5 → the LLM prior helps the RL selector on RC (the ORION thesis on the family where it
    can show).

## Fire sequence (amended — R.1/R.2 before R.4/R.5)

1. **§R committed** (this file, hashed).
2. **R.1 / R.2 fire** (follow_prior, box-only, minutes) → **readout to senior** — settles R-Primary,
   the deployable claim, independent of everything else.
3. **R.4 / R.5 fire** with **R.1 as the committed baseline**; BC-verify on seed 42 first, report
   timing before the batch.
4. **R.3** (frontier cache-off, 100 calls, ~$3.7, cap $6) slots wherever the API window is free —
   touches nothing on the box.
Full readout in the committed order at each group. DONE markers and silence between.

## Validity

- R.4/R.5 (training): EV_tail5 ≥ 0.5 per cell (§N Δ2); invalid-execution excluded + logged; §O.8
  telemetry on.
- R.1/R.2/R.3 (LLM): §P void triggers (malformed >10%, content >10%); schema-retry logging on the
  API path (R.3).
- All arms: §O.9 cost telemetry, model_id per call (R.3), `prereg_sha256` in every result JSON.

## Recording for the paper (capture everything on these long runs)

Every run writes, per seed and per arm:
- **Per-arrival trace:** request id, chain length, forced-vs-flexible, cache hit/miss, admitted,
  reject reason (structural / cross_domain_bw / C8 / C5 / c7_delay / actor_infeasible / api-fail).
- **Aggregates:** FoC per seed + plateau (last-10 mean, training), admitted/total, full reject
  taxonomy, cache hit rate, schema/api-fail rates.
- **M^B retrieval composition:** mean hits/query, pos/neg mix (where M^B live).
- **Training curves (R.4/R.5):** per-round eval FoC, EV, MDO selection entropy, KL mean, β,
  param-motion, m̃-agreement, value-norm stats, corr(adv,pos), negAdv@admitted (§O.8).
- **Plan/partition shape:** domains used, cut points, inter-domain links per admitted plan (where
  available).
- **Cost/§O.9:** wall-clock per round/cell, GPU energy measured, CPU energy TDP-estimate (labeled),
  API $ + tokens (R.3).
- **Provenance:** `prereg_sha256`, gen_seed, seeds, per-arm checkpoints (R.4/R.5), invalid flags.
Raw JSON retained; nothing averaged away that a per-seed/per-arrival table could show.

## Standing rules
Frozen refs cited from the archive + RC-v2 validity; no Plain/RA re-runs; RC-v2 locked; DONE
markers + silence between; first confirmed anomaly stops the affected arm group and reports, not
fixes. Frontier grid stays STOPPED.
