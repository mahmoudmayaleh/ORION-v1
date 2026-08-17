# ORION

[![CI](https://github.com/mahmoudmayaleh/ORION-v1/actions/workflows/ci.yml/badge.svg)](https://github.com/mahmoudmayaleh/ORION-v1/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ORION admits and places 6G network slices across federated administrative domains, using a
language model to propose a placement and reinforcement learning to dispose of it.

A slice request arrives as an operator intent. A language model turns it into a structured
slice specification and an abstract placement plan over domains. A PPO-trained Multi-Domain
Orchestrator decides whether to admit the slice and how to partition the plan across
domains, seeing only per-domain aggregates rather than the full substrate. Behaviour-cloned
per-domain actors then place the assigned VNFs on concrete nodes inside their own domain,
and a verifier checks the result against the slice's QoS and capacity constraints before
anything is committed.

![ORION system design](Orion-arch.jpg)

## Why the orchestrator sees less than the baselines

The central constraint is federation. A domain operator does not publish its internal
topology, so the orchestrator partitions a plan from per-domain summaries and a
feasibility mask, not from node-level state. Several baselines in this repository read the
full substrate. They are information-privileged references that bound what is achievable,
not peers competing under the same rules, and the comparison is meaningful only when that
asymmetry is stated. `MDO-partial` exists to isolate it: it is the same heuristic idea as
the full-visibility baseline, restricted to the orchestrator's observation.

## Repository layout

```
src/orion/
  llm/         Agent A (intent to spec), Agent B (abstract plan), K^B semantic memory,
               M^B episodic memory, retrieval pipeline, plan cache, structural checker
  mdo/         Multi-Domain Orchestrator: policy, coordinator, pre-commit checks, KL prior
  actors/      Per-domain actors, action masking, GATv2 backbone, intra-domain routing
  sim/         Substrate and slice generators, episode runner, delay model, verifier, reward
  training/    PPO and MAPPO update, GAE, replay buffer, behaviour cloning, KL schedule
  baselines/   Colocation-first and first-fit-decreasing heuristic placers
  monitor/     Strategy monitor and Page-Hinkley drift detection
  provenance.py  Run provenance guard stamped into every result file
scripts/       Experiment entry points, calibration, diagnostics and figure generation
tests/         Test suite
data/          Input fixtures: knowledge bases, placement examples, frozen benchmark
```

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/mahmoudmayaleh/ORION-v1.git
cd ORION-v1
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,actors,retrieval]"
```

The extras are separated because they carry real weight:

| Extra       | Brings                                        | Needed for                                        |
|-------------|-----------------------------------------------|---------------------------------------------------|
| `dev`       | pytest, pytest-cov, ruff, mypy                | Running the test suite and the linters            |
| `actors`    | torch-geometric                               | The GATv2 domain actors                           |
| `retrieval` | rank-bm25, faiss-cpu, sentence-transformers   | M^B retrieval. `rank-bm25` is the one that binds  |

`rank-bm25` is not optional in practice. Without it `build_index` fails loudly rather than
degrading to recency-only retrieval, because a silent degradation there would change
results without changing any setting.

Verify the install:

```bash
PYTHONHASHSEED=0 pytest
```

`PYTHONHASHSEED=0` is required, not advisory. Behaviour cloning replays a greedy placement
whose tie-breaking reads set iteration order, so an unpinned hash seed makes the
demonstrations non-deterministic. The entry points enforce it.

## Serving the language model

Agent A and Agent B talk to an OpenAI-compatible endpoint. Any server that exposes one
works. The runs in this repository used a 4-bit Llama-3-8B telecom-tuned model served
locally by `llama-cpp-python`:

```bash
bash scripts/start_llm_gpu.sh 8000        # full GPU offload, llama-3 chat format
curl http://localhost:8000/v1/models      # confirm it answers
```

The default endpoint is `http://localhost:8000/v1`, set in `LLMConfig`. For the frontier
comparison in `track_b_runner.py`, export an Anthropic API key as
`ORION_FRONTIER_API_KEY`; the backend enforces a spend cap and refuses to start without a
key rather than failing part-way through a run.

Runs that do not need a language model at all (`Plain`, `MDO-*`, `RL-alone`, `RL-poprior`)
need no server.

## Running an experiment

The main entry point is the transfer grid: approaches, across two scenario classes, four
calibrated load levels, and several seeds, on one fixed hierarchical substrate.

```bash
# smoke: one family, one seed, all approaches
PYTHONHASHSEED=0 python scripts/grid_runner.py --part smoke --no-prereg

# LLM-free subset, no server required
PYTHONHASHSEED=0 python scripts/grid_runner.py \
    --part 1 --approaches Plain MDO-partial RL-alone --no-prereg
```

Every cell writes `data/grid_cells/<scenario>_<approach>_<seed>_<family>.json` the moment
it finishes and is skipped if that file already exists, so the grid is resumable and
killing it is always safe. Trained stacks cache their checkpoints under
`results/wp7/ckpt_grid/`.

### The approach ladder

Each approach differs from its neighbour in one thing, so a gap between two rows names a
mechanism rather than a bundle of changes.

| Approach       | Planner        | Observability   | Trained | Memory |
|----------------|----------------|-----------------|---------|--------|
| `Plain`        | Heuristic      | Full substrate  | No      | No     |
| `MDO-fullobs`  | Heuristic      | Full substrate  | No      | No     |
| `MDO-partial`  | Heuristic      | Domain summaries| No      | No     |
| `MDO-ffd`      | First-fit      | Domain summaries| No      | No     |
| `RL-alone`     | Heuristic      | Domain summaries| Yes     | No     |
| `RL-advised`   | Heuristic      | Domain summaries| Yes     | No     |
| `RL-poprior`   | Heuristic prior| Domain summaries| Yes     | No     |
| `Memory-off`   | Agent B        | Domain summaries| Yes     | No     |
| `Full`         | Agent B        | Domain summaries| Yes     | M^B    |

`Plain` allocates directly; `MDO-fullobs` runs the same placer through the coordinator and
the domain actors, so the two differ only in pipeline while `MDO-fullobs` and
`MDO-partial` differ only in observability. Variants suffixed `-rp` and `-rpg` apply plan
repair, holding the language model's plan to the same admissibility tests the heuristic
applies to its own proposal. They are additive over their base approach, which stays the
control.

### Other entry points

| Script                     | Purpose                                                        |
|----------------------------|----------------------------------------------------------------|
| `approach_runner.py`       | Single approach on a single instance                           |
| `milp_approach_runner.py`  | Per-arrival optimal embedding reference                        |
| `track_b_runner.py`        | Local model against a frozen benchmark and a frontier model    |
| `partial_obs_prior.py`     | The partial-observability heuristic planner and its plan repair|
| `calibrate_load_levels.py` | Re-derive the load ladder, then `freeze_load_levels.py`        |
| `y15_figures.py`           | Acceptance figures and tables from banked cells                |
| `llm_health_probe.py`      | Check the language-model server is answering before a long run |

`scripts/README.md` describes all of them.

## Provenance

Every result-writing runner passes through `orion.provenance`, which stamps the commit,
the dirty-file list and the serving model into the result JSON, and refuses to run at all
if there is untracked code under `src/` or `scripts/`. A number produced by code that
exists in no revision cannot later be reproduced or refuted, which is the failure this
guard was built after. There is no bypass for it.

The pre-registrations that govern the reported experiments are not distributed here. The
entry points cite them by path and hash, so on a fresh clone they refuse to start. Pass
`--no-prereg` to run anyway. It applies only when the document is genuinely absent, it
relaxes nothing else, and the result JSON records `prereg.status = "skipped"` so a number
produced without a pinned protocol says so on its face.

## Testing

```bash
PYTHONHASHSEED=0 pytest                              # full suite with coverage
PYTHONHASHSEED=0 pytest --no-cov -q                  # faster
ruff check . --select E9,F63,F7,F82,F811             # the blocking lint gate
ruff check .                                         # full style set, advisory
```

Two tests exercise a timeout path through `SIGALRM` and skip on Windows. CI runs the suite
on Python 3.11 and 3.12.

The blocking lint gate is the defect subset: syntax errors, undefined names, redefinitions
and invalid comparisons. The wider style set in `pyproject.toml` is reported without
blocking, because reformatting the experiment code in bulk would make every result file's
commit stamp point at a tree that differs from the one the run used.

## Citation

If you use this software, cite it via `CITATION.cff`, which GitHub renders as a citation
on the repository page.

## License

MIT. See [LICENSE](LICENSE).
