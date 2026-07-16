#!/usr/bin/env python
"""§R R.3 — frontier (Sonnet) cache-off, seed 42, 100 arrivals, $6 subcap (within the shared
$10 ledger). Closes the §R paragraph with a measured frontier number vs local R.2 = 86.6%."""
import sys, json, logging
from pathlib import Path
sys.path.insert(0, "scripts")
import q_pilot_runner as Q

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("R3")
ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "data" / "api_cost_ledger.json"
SUBCAP = 6.0


def main():
    from orion.llm.frontier_backend import (FrontierBackend, CostMeter,
                                            make_frontier_config, PILOT_COST_CAP_USD)
    from orion.llm.agent_b import AgentB
    from orion.llm.semantic_memory import SemanticMemory
    prior = json.loads(LEDGER.read_text())["spent_usd"] if LEDGER.exists() else 0.0
    rem = max(0.0, min(SUBCAP, PILOT_COST_CAP_USD - prior))
    if rem <= 0:
        log.error("no budget for R.3 (prior spent=$%.2f, cap=$%.2f)", prior, PILOT_COST_CAP_USD)
        return
    log.info("R.3 budget = $%.2f (min of $6 subcap, $%.2f remaining)", rem, PILOT_COST_CAP_USD - prior)
    meter = CostMeter(cap_usd=rem)
    agent_b = AgentB(FrontierBackend(make_frontier_config("claude-sonnet-5", temperature=0.0), meter))
    kbp = ROOT / "data" / "kb_entries.json"
    kb = SemanticMemory.from_json(kbp) if kbp.exists() else None
    sub = Q.generate_rc_instance(seed=Q.RC_GEN_SEED, inter_domain_bw_override=70)
    m = Q.run_q_cell("R.3-frontier", agent_b, kb, None, None, sub, 42, False)
    spent = meter.spent_usd
    LEDGER.write_text(json.dumps({"spent_usd": round(prior + spent, 6), "updated": "r3", "cap": 10.0}, indent=2))
    foc = 100.0 * m["admitted"] / 97
    out = {"track": "R.3", "seed": 42, "admitted": m["admitted"], "total": m["total"],
           "foc97": round(foc, 1), "reasons": m.get("reasons"), "sonnet_usd": round(spent, 4),
           "refs": {"R2_local_foc": 86.6, "ceiling": 97}}
    (ROOT / "data" / "r3_results.json").write_text(json.dumps(out, indent=2))
    log.info("R.3: admit=%d/%d FoC97=%.1f%% spent=$%.4f -> data/r3_results.json",
             m["admitted"], m["total"], foc, spent)
    print("\n=== R.3 FRONTIER READOUT ===")
    print(f"  admit={m['admitted']}/{m['total']}  FoC97={foc:.1f}%  reasons={m.get('reasons')}  spent=${spent:.4f}")
    print("  vs local R.2 (Tele-8B) = 86.6%")


if __name__ == "__main__":
    main()
