# ORION

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ORION is a hierarchical framework for placing network slices across multiple administrative
domains, where no operator has node-level visibility of the others. It separates proposing a
slice decomposition from committing it: Large Language Model (LLM) agents propose, and a
reinforcement learning orchestrator disposes.

The setting is what makes this hard. Domains expose only aggregate summaries and hold
different compute tiers, so placement restrictions can force a slice to span domains even
when aggregate resources suffice, and utilization-dependent delays can break a latency
budget before capacity is exhausted.

LLM agents translate the service intent into a structured request and propose an abstract
partition across domains and infrastructure tiers. A multi-domain orchestrator trained by
reinforcement learning commits the partition, treating the proposal as a soft prior rather
than an instruction, and per-domain actors place the functions on physical nodes. The
proposal guides the orchestrator in training as well as at deployment, and verified
placements are stored with the conditions under which they held, so the agents reuse
structures that succeeded under comparable conditions.

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
| `RL-alone`     | Policy         | Domain summaries| Yes     | No     |
| `Memory-off`   | LLM            | Domain summaries| Yes     | No     |
| `Full`         | LLM            | Domain summaries| Yes     | Yes    |

`Plain` allocates directly; `MDO-fullobs` runs the same placer through the coordinator and
the domain actors, so the two differ only in pipeline while `MDO-fullobs` and
`MDO-partial` differ only in observability. Variants suffixed `-rp` and `-rpg` apply plan
repair, holding the language model's plan to the same admissibility tests the heuristic
applies to its own proposal. They are additive over their base approach, which stays the
control.


## Citation

If you use this software, cite it via `CITATION.cff`, which GitHub renders as a citation
on the repository page.

## License

MIT. See [LICENSE](LICENSE).
