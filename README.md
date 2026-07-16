# ORION

**Slice Orchestrator Assistant for Multi-Domain 6G Network Slicing** — an LLM-guided
reinforcement-learning system for admitting and placing network slices across federated
administrative domains.

![ORION system design](V6-ORION.png)

ORION composes four pieces:

| component | role |
|---|---|
| **Agent A** | natural-language intent → structured slice specification |
| **Agent B** | slice specification → abstract placement plan (`m̃`), grounded by retrieval over a knowledge base (K^B) and an episodic memory (M^B) |
| **MDO** | a PPO-trained Multi-Domain Orchestrator that decides admission and partitions the plan across domains, KL-regularized toward Agent B's plan |
| **Domain actors** | per-domain placement within a domain's own substrate, behaviour-cloned then frozen |

The research question is whether the LLM's plan **helps the RL selector** — so the
central experiment is an ablation of the KL prior that ties the selector to the plan,
not a demonstration that the pipeline runs.

---

## Status

**This is an active research repository, published alongside a paper in preparation.**
The code is the code that produced the reported experiments; it is not a product, and
the experiment record is still moving. Read `docs/` for what was registered and when.

Results are **not** in this repository — they are reported in the paper. What ships here
is the system, the tests, the experiment entry points, and the pre-registrations they
cite.

## Install

```bash
git clone https://github.com/mahmoudmayaleh/ORION-v1.git
cd ORION-v1
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,retrieval,actors]"
```

Requires Python ≥ 3.11. GPU experiments were run on an RTX A6000 (48 GB) with
`torch==2.5.1+cu121`; `pyproject.toml` floors torch at 2.2, but the reported runs used
that pin, and PPO results are sensitive to the torch/CUDA build.

```bash
pytest -q      # 419 tests
```

## Layout

```
src/orion/
  llm/          Agent A, Agent B, LLM backend, K^B semantic memory, M^B episodic memory,
                plan cache, constrained JSON decoding
  mdo/          the Multi-Domain Orchestrator: observation, policy, coordinator
  actors/       per-domain placement + cross-domain routing
  training/     MAPPO trainer, PPO update, KL prior schedule, value normalisation
  sim/          substrate, arrival process, episode runner, delay model, verifier
  substrate/    topology families, incl. the routing-critical (RC) family
  baselines/    colocation-FFD and routability-aware references
  monitor/      drift + telemetry
  provenance.py run-provenance recorder (see below)
scripts/        experiment entry points (one per registered experiment)
docs/           pre-registrations + protocol + technical reference
tests/          419 tests
data/           fixtures: knowledge base, memory seeds, sample intents
```

## Running an experiment

Entry points live in `scripts/`. Each stamps its own provenance and refuses to run on a
dirty tree (see below). The LLM-backed arms need an OpenAI-compatible server on
`localhost:8000` — the reported runs used `llama.cpp` serving a Q4_K_M GGUF of an
8B LLaMA-3 telecom-tuned model. Arms that do not need the LLM run with `--mock`.

```bash
python scripts/d_runner.py --help          # the 300-arrival adaptation run
python scripts/rc_train_runner.py --mock   # training loop, no server needed
```

Model weights are not distributed here.

## Provenance and pre-registration

Two conventions in this repository are load-bearing, and are the reason it is worth
reading rather than just running:

**Experiments are pre-registered.** `docs/PREREG_*.md` state hypotheses, arms, and
firing conditions *before* the evidence exists; amendments (`Δ1`, `Δ2`, …) append rather
than rewrite, so a claim that failed stays visible next to the claim that replaced it.
`§R Δ3-R`, for instance, withdraws a cache-pathology claim that an earlier amendment had
relied on. Code comments cite these sections by name.

**Runs refuse to be unprovenanced.** `src/orion/provenance.py` stamps every result with
its commit, records the serving model and its process incarnation, and **raises** if
there is untracked code under `scripts/` or `src/`, or if a cited pre-registration's
hash does not match its committed copy. There is deliberately no bypass flag. This
exists because a result once survived review on code that was never committed and a
server whose identity was never recorded; it did not reproduce.

The experiment protocol, including the serving-layer nondeterminism measurement that
forces repeated LLM-path cells, is in `docs/EXPERIMENT_PROTOCOL.md`.

## Citing

See `CITATION.cff`. The paper reference will be added on publication.

## License

MIT — see `LICENSE`.
