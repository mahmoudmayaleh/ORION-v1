# ORION-v1

![ORION_System_design](Orion-arch.jpg)

ORION is an LLM-guided reinforcement-learning system for admission and placement of
network slices across federated administrative domains. A language model turns an
operator intent into a structured slice specification and an abstract placement plan;
a PPO-trained Multi-Domain Orchestrator, KL-regularised toward that plan, decides
admission and partitions the plan across domains; behaviour-cloned per-domain actors
place within each domain.

## Layout

| Path | Contents |
| --- | --- |
| `src/orion/` | The system: substrate model, simulator, LLM agents, memory/retrieval, MDO, per-domain actors, training. |
| `tests/` | Test suite. |
| `scripts/` | Experiment entry points and analysis tools. |
| `data/` | Fixtures used by the test suite. |

## Install

```
pip install -e ".[dev]"
```

Optional extras: `retrieval` (BM25/FAISS/sentence-transformers), `actors` (torch-geometric).

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.
