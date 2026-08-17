# Scripts

Every script here is run from the repository root, not from this directory:

```bash
PYTHONHASHSEED=0 python scripts/grid_runner.py --part smoke --no-prereg
```

Scripts that write a result file pass through `orion.provenance` first. That guard refuses
to run if there is untracked code under `src/` or `scripts/`, and it cites a
pre-registration that is not distributed with this repository, so those scripts need
`--no-prereg` on a fresh clone. See the Provenance section of the top-level README.

## Experiment entry points

| Script | What it runs |
|---|---|
| `grid_runner.py` | The transfer grid. Approaches, two scenario classes, four calibrated load levels, several seeds, one fixed substrate. Resumable: each cell is written and skipped if present. |
| `approach_runner.py` | A single approach on a single instance. The unit `grid_runner` calls per cell. |
| `milp_approach_runner.py` | Per-arrival optimal embedding, as an achievability reference. |
| `track_b_runner.py` | The local model against a frozen benchmark, optionally alongside a frontier model. Requires `data/benchmark_S/`. |
| `r_local_runner.py` | The local planner comparison on the frozen RC draw. |
| `wp7_runner.py` | The training and evaluation harness the trained approaches are built on. |

## Planner and heuristics

| Script | What it does |
|---|---|
| `partial_obs_prior.py` | The partial-observability heuristic planner, and `repair_plan`, which holds a language model's plan to the same admissibility tests the heuristic applies to its own proposal. |
| `demo_solver.py` | Standalone MILP placement demonstration. |

## Calibration

Run in this order. The second reads what the first writes.

| Script | What it does |
|---|---|
| `calibrate_load_levels.py` | Sweeps offered load and derives the level ladder. Writes `results/y3_load_calibration.json`. |
| `freeze_load_levels.py` | Turns that sweep into the frozen level table. |
| `build_benchmark_S.py` | Builds the frozen benchmark `track_b_runner` reads. |

## Diagnostics and probes

These answer one question each and are kept because published claims rest on them.

| Script | Question |
|---|---|
| `llm_health_probe.py` | Is the language-model server answering, before a long run commits to it. |
| `probe_retrieval_floor.py` | Where the M^B retrieval abstain floor should sit. |
| `probe_kl_sanity.py` | Whether the prior-coupling loss moves the policy at all. |
| `probe_llm_tokens.py` | Token accounting for a planner call. |
| `probe_mtilde_support.py` | Whether the proposed partition is in the policy's support. |
| `diag_delay_budget.py` | Where the end-to-end delay budget is spent on rejected arrivals. |
| `kill_sanity_check.py` | Replays arrivals classified as search failures through the real verifier, to check that classification is not inflated. |
| `check_summary_updates.py` | Whether the aggregate surface the planner reads actually moves as the substrate fills. |
| `eval/retrieval_quality.py` | Recall@k for each retrieval mode against the ground-truth fixture. |

## Analysis and figures

| Script | Output |
|---|---|
| `y15_figures.py` | Acceptance figures and tables from banked grid cells. |
| `y15_cost_figures.py` | Cost figures from the profiling sidecars. |
| `analyze_results.py` | Summary tables from a result file. |
| `analyze_m2.py` | The M.2 readout. |
| `cost_metrics.py`, `cost_secondary_table.py` | Placement-cost aggregates. |
| `derive_a2.py` | Re-derives the spread-and-fail dose-response from a result file. |

## Language model serving

| Script | What it does |
|---|---|
| `start_llm_gpu.sh` | Serves the quantized planner model through `llama-cpp-python` with full GPU offload, on an OpenAI-compatible endpoint. |
| `llm_eval/` | Standalone planner evaluation against the fixtures in `data/placement_eval/`, including `setup.sh` for building `llama.cpp`. |

## Not distributed

Pre-§Y scripts are not included. They reference names the redesign removed, such as the
`MEC` and `RAN_EDGE` tiers merged into `EDGE`, the multi-attempt coordinator, and the
fraction-of-ceiling metric, so they cannot run against this code.
