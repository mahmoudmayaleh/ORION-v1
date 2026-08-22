# Using ORION

Everything you need to run the system, reproduce a table, or add an approach.
`README.md` explains *what* ORION is; this explains *how to drive it*.

---

## 1. Setup

Python 3.11+.

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Check the install:

```bash
PYTHONHASHSEED=0 pytest -q
```

> **Always set `PYTHONHASHSEED=0`.** Set-iteration order reaches the greedy
> replay, so without it two runs of the same command can disagree. Every command
> below assumes it.

### The LLM server (only for LLM approaches)

`Plain`, `MDO-partial`, `MDO-fullobs` and `RL-alone` never construct an LLM
client, so they need no server. The `Full-*` and `Memory-off-*` approaches do.
Any OpenAI-compatible endpoint works:

```bash
./scripts/start_llm_gpu.sh 8000          # local llama.cpp on GPU
python scripts/llm_health_probe.py       # confirm it answers with non-zero tokens
```

Point the runner at it with `--port 8000`.

> Run **one** LLM job at a time. Two concurrent jobs wedge a local 8B server into
> returning empty completions. `/tmp/orion_local_llm.lock` enforces this; jobs
> with no LLM in them never take it, so they can run alongside freely.

---

## 2. Running an experiment

One entry point, `scripts/grid_runner.py`. A cell is one
(scenario, approach, seed, level, instance) and is written as one JSON file.

```bash
PYTHONHASHSEED=0 python scripts/grid_runner.py \
    --approaches Plain MDO-partial MDO-fullobs \
    --seeds 42 \
    --levels L1 L2 L3 L4 \
    --arrivals 2000 \
    --eval-instances 100
```

Common options:

| flag | meaning |
| --- | --- |
| `--approaches` | which approaches to run (see §3) |
| `--seeds` | one or more seeds; results are per seed |
| `--levels` | `L1`-`L4`, the calibrated load ladder |
| `--arrivals` | arrivals per episode (2000 is the committed value) |
| `--eval-instances` | which topology instance to evaluate on |
| `--train-instances` | curriculum length for the learned approaches |
| `--eval-only` | reuse banked checkpoints instead of retraining |
| `--final-segment` | skip §Y.14 selection, use the last segment |
| `--port` | LLM server port |
| `--mock` | no LLM calls, for wiring tests |
| `--prereg PATH` | verify and cite a pre-registration (optional) |

Useful environment variables:

| variable | effect |
| --- | --- |
| `ORION_CELL_DIR` | where cells are written (default `data/grid_cells`) |
| `ORION_POSTDECODE_GUARD=1` | re-check the committed partition against the h^m guard |
| `ORION_COLOCATION_CONTRACT=1` | make the LLM emit one host for the whole chain |
| `ORION_PLAN_REPAIR` | `off` / `guard` / `full` / `colo` |
| `ORION_CHAIN_ORDER=off` | disable the C10 contiguity constraint |
| `ORION_PREREG` | default pre-registration path |

### Long runs

Training runs take hours. Detach them so an SSH drop cannot kill them:

```bash
setsid nohup env PYTHONHASHSEED=0 ORION_CELL_DIR=data/my_run \
    python scripts/grid_runner.py --approaches RL-alone --seeds 42 \
    --levels L1 L2 L3 L4 --arrivals 2000 \
    > logs/my_run.log 2>&1 < /dev/null &
```

Run LLM-free and LLM jobs in parallel; they contend for nothing but CPU.

---

## 3. The approaches

| approach | what it is |
| --- | --- |
| `Plain` | colocation-first FFD over the **full** substrate, bypasses the coordinator. The info-privileged reference |
| `MDO-partial` | heuristic partition from the MDO's own observation surface: per-domain aggregates plus a per-tier largest-free-node statistic |
| `MDO-fullobs` | the same decision with **full** visibility: scores whole partitions on predicted delay and real inter-domain link state |
| `RL-alone` | the learned partition policy, no LLM anywhere |
| `Full` | ORION: Agent B plans, the policy disposes |
| `Memory-off` | ORION with episodic memory disabled |

Suffixes compose: `-rpg` (per-VNF h^m repair, splits preserved), `-rpc`
(colocation-preserving repair), `-rp` (repair plus collapse), `-fp`
(`follow_prior` executor — commit the plan verbatim instead of letting the policy
override it). So `Full-rpc-fp` is ORION with colocation repair and no policy
override.

`scripts/grid_runner.py --help` lists the full registry.

---

## 4. Reading the output

Each cell is JSON:

```jsonc
{
  "acceptance": 0.6430,
  "admitted": 1286, "offered": 2000,
  "rejections": { "structural": 15, "chain_order": 0, "actor_infeasible": 97, ... },
  "cost": {
    "structure":  { "split_rate": 0.84, "domains_per_chain_mean": 2.35, "domain_load_jain": 0.94 },
    "qos":        { "e2e_ms_p95": 16.2, "budget_ratio_p95": 0.76, "over_budget_frac": 0.0 },
    "timeseries": { "acceptance": [...], "steady_state_mean": 0.63 }
  }
}
```

The rejection taxonomy **conserves**: admitted plus all bins equals offered. If it
does not, something is being dropped silently and the run should not be trusted.

Bins worth knowing: `structural` (no plan could be built), `chain_order` (C10),
`actor_infeasible` (no node in the chosen domain), `cross_domain_infeasible` and
`c9_hops` (the price of splitting), `post_commit_c7_delay` (the verifier refused
an admitted plan on delay), `qos_gate` (refused before allocation).

---

## 5. The load ladder

Levels are **calibrated, not chosen**, and `get_level` refuses to hand out an
uncalibrated one rather than silently use a default arrival rate.

```bash
# measure the sweep (~40 min); pins levels on the partition-oracle reference
PYTHONHASHSEED=0 PYTHONPATH=src python scripts/calibrate_load_levels.py \
    --reference oracle --points 12 --seeds 42 43 44 45 46 --arrivals 2000

# change only the targets? reuse the cached sweep, no re-measurement
PYTHONHASHSEED=0 PYTHONPATH=src python scripts/calibrate_load_levels.py --reselect

# write the table into orion.sim.load_levels
PYTHONHASHSEED=0 PYTHONPATH=src python scripts/freeze_load_levels.py \
    --note "why this ladder changed"
```

The freeze script refuses a ladder that is not monotone in load or that gives two
levels the same arrival rate. Those refusals are the point; do not edit the table
by hand.

**Changing the substrate or the workload invalidates the ladder.** Recalibrate,
re-freeze, and retrain before comparing anything to older numbers.

---

## 6. Adding an approach

1. Register the name in `APPROACHES` in `scripts/grid_runner.py`. Variants that
   only change repair or executor go in `REPAIR_APPROACHES` / `FP_BASE` with a
   `REPAIR_BASE` entry naming the stack they train from.
2. Add a dispatch branch to the evaluation switch.
3. If it needs a new partition rule, write a builder with the signature
   `builder(slice_request, substrate) -> PlanSummary | None` and run it through
   `eval_heuristic_pipeline`, so it shares the coordinator, actors, routing and
   verifier with every other approach. Only the partition should differ.
4. Add a test that pins the behaviour, not the number.

---

## 7. Reproducibility rules

- **Untracked code refuses the run.** If `scripts/` or `src/` contains an
  untracked file, the runner stops. A number produced by untracked code can be
  neither reproduced nor refuted. Put scratch elsewhere; there is no bypass flag.
- **Pre-registration is optional.** Pass `--prereg PATH` to cite one; it must then
  be committed and byte-match its committed copy.
- **Experiment output is not versioned.** `data/`, `results/`, `runs/` and `logs/`
  are gitignored. Copy them off the machine that produced them.
- **Every cell records git provenance**, including whether the tree was dirty.

---

## 8. Troubleshooting

| symptom | cause |
| --- | --- |
| `REFUSING TO RUN: untracked code under scripts/ or src/` | working as designed; commit or move the file |
| `§Y.3 load calibration has not been run` | `CALIBRATED_LEVELS` is empty; run the calibration and freeze |
| LLM calls return empty completions | two LLM jobs are running; serialise them, then restart the server |
| `These checkpoints were written by a different curriculum` | `--train-instances` does not match the banked checkpoints |
| results differ between two identical runs | `PYTHONHASHSEED` was not set to `0` |
| a cell is missing `structure` / `qos` / `timeseries` | instrumentation raised and was swallowed; the cell is incomplete |
