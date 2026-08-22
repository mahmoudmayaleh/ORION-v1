"""Build the cost-per-decision row for each approach from the profile cells.

The decision stage is `plain.decision` for Plain, which never enters the
coordinator, and `mdo.decision` for everything else. Verification is timed
separately and is reported in the prose, not the table.

Energy: GPU draw is charged by the sampler to this experiment's own PIDs, which
include the resident llama.cpp server. That server sits loaded on the A6000 for
the whole session, so its idle draw lands on every cell whether or not the cell
consulted it. Only ORION actually calls the model, so GPU energy is reported for
ORION alone and the LLM-free approaches carry the CPU estimate only.
"""
import json
import os

CELLS = "data/profile_run_cells"
APPROACHES = ["Plain", "MDO-fullobs", "MDO-partial", "RL-alone", "Full"]
LLM = {"Full"}

print("%-12s %8s %8s %8s %8s %8s %8s" % (
    "approach", "n", "med_ms", "p90_ms", "p99_ms", "mJ/dec", "cache"))
for a in APPROACHES:
    p = os.path.join(CELLS, "conventional_%s_42_L2_i100.json" % a)
    if not os.path.exists(p):
        print("%-12s  (pending)" % a)
        continue
    d = json.load(open(p))
    prof = d["profile"]
    s = prof["summary"]
    ct = prof["cell_totals"]
    stage = "plain.decision" if "plain.decision" in s else "mdo.decision"
    w = s[stage]["wall_s"]
    n = w["n"]
    energy_j = ct["cpu_energy_j_est"] + (ct["gpu_energy_j"] if a in LLM else 0.0)
    cache = (d.get("plan_cache") or {}).get("hit_rate")
    print("%-12s %8d %8.3f %8.3f %8.3f %8.1f %8s" % (
        a, n, w["p50"] * 1e3, w["p90"] * 1e3, w["p99"] * 1e3,
        energy_j / n * 1e3, "--" if cache is None else "%.3f" % cache))

    for k in ("llm.generate", "struct.check", "plan_build", "verify", "mdo.forward",
              "actor.place"):
        if k in s:
            kw = s[k]["wall_s"]
            print("      %-14s n=%-6d mean=%8.3f ms  p50=%8.3f  p99=%8.3f" % (
                k, kw["n"], kw["mean"] * 1e3, kw["p50"] * 1e3, kw["p99"] * 1e3))
    print("      cell wall=%.1fs cpu=%.1fs gpu_j=%.1f cpu_j=%.1f" % (
        ct["wall_s"], ct["cpu_s"], ct["gpu_energy_j"], ct["cpu_energy_j_est"]))
