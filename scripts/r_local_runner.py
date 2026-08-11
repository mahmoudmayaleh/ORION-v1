"""§R R.1/R.2 — local ORION follow_prior baselines on RC-v2 (2026-07-15).

R.1  ORION-local deployable : Agent B (LLaMA-3-8B) plans, plan-cache ON, follow_prior.
R.2  Local, diverse sampling : same, plan-cache OFF (every arrival re-plans).

Three seeds 42/43/44 (bw sweep 70/90/110), byte-identical 100-arrival RC streams,
per-(seed, approach) cold start (cache + M^B wiped, state hash asserted empty). Reuses
q_pilot_runner.run_q_cell (now emitting a per-arrival trace) so R.1/R.2 are the
SAME deployable stack the pilot measured, just cache ON vs OFF.

Settles (PREREG_AMENDMENT_2026-07-15_R.md):
  R-Primary  : R.1 > Plain (per-seed) in mean AND positive sign all three seeds.
  R-Sampling : |R.2 - R.1| = the cache-thinness measurement (characterization).

Needs the local llama.cpp server on :8000. --mock swaps Agent B for FFD (wiring smoke,
no server). Box-only, minutes-to-hours. Records everything per approach/seed for the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np



def prereg_sha256() -> str:
    """§R provenance: hash of the §R amendment (this run's pre-registration)."""
    p = Path(__file__).resolve().parent.parent / "docs" / "PREREG_AMENDMENT_2026-07-15_R.md"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
from orion.llm.plan_cache import PlanCache
from orion.provenance import git_provenance, serving_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r_local")

# ── FROZEN RC-v2 validity draw (results/rc_family_validity_RESULT.md) ──────────
# §R rule: Plain-ColocFB and the ceiling are NEVER re-run; cite these verbatim.
# R.1/R.2 build the SAME (substrate, arrival stream) as the validity draw
# (a pre-§Y routing-critical instance + arrival_seed=seed +
# rc_slice_factory), so admits are directly comparable against these ceilings.
# Re-running _evaluate drifts across machines (36.4 vs 37.9 vs frozen 37.1) — the
# reason the prereg freezes it.
FROZEN_RC = {
    42: {"bw": 70.0,  "total": 100, "ceiling": 97,  "plain_admits": 36, "plain_foc": 37.1},
    43: {"bw": 90.0,  "total": 100, "ceiling": 100, "plain_admits": 37, "plain_foc": 37.0},
    44: {"bw": 110.0, "total": 100, "ceiling": 100, "plain_admits": 47, "plain_foc": 47.0},
}
BW_FOR_SEED = {s: FROZEN_RC[s]["bw"] for s in FROZEN_RC}
PLAIN_FOC_RC_MEAN = 40.4        # RC-v2 validity draw (frozen reference, context)
MEMORY_CAPACITY_K = 50


def _build_local_agent(port: int):
    from orion.llm.llm_backend import LLMBackend, LLMConfig
    from orion.llm.agent_b import AgentB
    cfg = LLMConfig(base_url=f"http://localhost:{port}/v1", api_key="EMPTY",
                    model="default", temperature=0.0, max_tokens=2048)
    return AgentB(LLMBackend(cfg))


def _load_kb():
    from orion.llm.semantic_memory import SemanticMemory
    kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
    if kb_path.exists():
        kb = SemanticMemory.from_json(kb_path)
        logger.info("K^B loaded: %d entries", len(kb.entries))
        return kb
    return None


def _fresh_mb():
    from orion.llm.episodic_memory import EpisodicMemory
    from orion.retrieval import RetrievalConfig, RetrievalMode
    return EpisodicMemory(
        config=RetrievalConfig(mode=RetrievalMode.NO_RERANK, apply_recency=True, k_final=3),
        max_entries=MEMORY_CAPACITY_K, write_policy="selective", evict_policy="importance")


def _mb_composition(mb):
    """Composition of M^B at cell end.

    The success/failure label lives in `entry.tags["label"]` -- a LIST, e.g.
    {"label": ["success"]} (see EpisodicMemory.write). MemoryEntry has no
    `admitted` and no `success` attribute, so the previous
    `getattr(e, "admitted", getattr(e, "success", False))` fell through to the
    False default for EVERY entry and reported mb_pos=0 on every run ever
    recorded -- including R.2|42's "mb=50(+0/-50)", which read as "M^B learned
    only from failures" when in fact 84 admissions had written positives and 50
    was merely the capacity cap. A telemetry field that cannot express one of its
    two states is not a measurement, so pin it with a test rather than an idiom:
    tests/test_mb_composition.py fails if this silently returns to all-zero.
    """
    entries = list(getattr(mb, "_entries", []))
    pos = sum(1 for e in entries
              if ((getattr(e, "tags", None) or {}).get("label") or [None])[0] == "success")
    return {"mb_entries": len(entries), "mb_pos": pos, "mb_neg": len(entries) - pos}








