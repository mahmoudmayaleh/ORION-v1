#!/usr/bin/env python
"""§S Track B — Agent-A intent->spec evaluation, 2x2 {Tele-8B, Sonnet} x {K^A on, off}.

M^A OFF for the main table (independent calls). Scores each returned spec against the frozen
K^A-consistent gold. Sonnet calls run through FrontierBackend's CostMeter; a persistent ledger
(data/api_cost_ledger.json) enforces the SHARED $10 cap across Track B + C + R.3.

Usage:
  python scripts/track_b_runner.py --approaches tele                 # local only, $0
  python scripts/track_b_runner.py --approaches sonnet --smoke 2     # 2-call infra+guard smoke
  python scripts/track_b_runner.py --approaches sonnet               # full Sonnet approaches under cap
Frozen inputs: data/benchmark_S/benchmark_S.json (+ MANIFEST hash), docs/PREREG_S_2026-07-15.md
"""
import sys, json, argparse, hashlib, time, logging, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trackB")

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "benchmark_S" / "benchmark_S.json"
LEDGER = ROOT / "data" / "api_cost_ledger.json"
PREREG = ROOT / "docs" / "PREREG_S_2026-07-15.md"
QOS_BAND = {"eMBB": {"delay": (20, 100), "tput": (50, 500)},
            "URLLC": {"delay": (1, 10), "tput": (10, 50)},
            "mMTC": {"delay": (50, 500), "tput": (1, 10)},
            "V2X": {"delay": (5, 20), "tput": (20, 80)},
            "XR": {"delay": (5, 30), "tput": (100, 500)}}
TOL = 0.15


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serving_provenance(port=8000):
    """Record the SERVING LAYER per run: model_id + chat_format + server incarnation.
    The 2026-07-15 incident showed weights+prompts were pinned but the chat template
    (the layer between them) was not — so pin it in provenance from here on."""
    prov = {"port": port, "model_id": None, "chat_format": None,
            "server_pid": None, "server_start": None}
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5) as r:
            prov["model_id"] = json.loads(r.read())["data"][0]["id"]
    except Exception:  # noqa: BLE001
        pass
    try:
        import subprocess
        out = subprocess.run(["ps", "-eo", "pid,lstart,cmd"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "llama_cpp.server" in line and f"--port {port}" in line:
                parts = line.split()
                prov["server_pid"] = parts[0]
                prov["server_start"] = " ".join(parts[1:6])
                m = re.search(r"--chat_format\s+(\S+)", line)
                prov["chat_format"] = m.group(1) if m else "(auto-detected)"
                break
    except Exception:  # noqa: BLE001
        pass
    return prov


def _within(x, gold, band):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    lo, hi = gold * (1 - TOL), gold * (1 + TOL)
    return (lo <= x <= hi) and (band[0] <= x <= band[1])


def score_item(spec, gold, valid):
    """Field/SFC/VCR/QoS scoring of one returned spec vs its gold. Robust to malformed spec."""
    s = {"schema_valid": bool(valid)}
    gclass = gold["slice_type"]
    band = QOS_BAND[gclass]
    # class (tier-1 strict). tier-2 handled by caller for ambiguous items.
    s["slice_type_pred"] = (spec or {}).get("slice_type")
    s["slice_type_correct"] = s["slice_type_pred"] == gclass
    # SFC
    gold_sfc = gold["sfc"]
    pred_vnfs = (spec or {}).get("vnfs") or []
    pred_sfc = [ (v or {}).get("vnf_type") for v in pred_vnfs ]
    s["sfc_len_ok"] = len(pred_sfc) == len(gold_sfc)
    s["sfc_order_ok"] = pred_sfc == gold_sfc
    s["sfc_family_ok"] = set(pred_sfc) == set(gold_sfc)
    s["sfc_all_ok"] = s["sfc_len_ok"] and s["sfc_order_ok"] and s["sfc_family_ok"]
    # VCR (per position present in BOTH; exact +-0.05 vs K^A constant)
    vcr_ok = vcr_n = 0
    for k, gv in enumerate(gold["vnfs"]):
        if k < len(pred_vnfs):
            vcr_n += 1
            try:
                if abs(float(pred_vnfs[k].get("vcr")) - gv["vcr"]) <= 0.05:
                    vcr_ok += 1
            except (TypeError, ValueError):
                pass
    s["vcr_acc"] = (vcr_ok / vcr_n) if vcr_n else 0.0
    # QoS
    q = (spec or {}).get("qos") or {}
    s["delay_ok"] = _within(q.get("max_e2e_delay"), gold["qos"]["max_e2e_delay"], band["delay"])
    s["tput_ok"] = _within(q.get("min_throughput"), gold["qos"]["min_throughput"], band["tput"])
    # bw edges (+-15% of gold edge; no class-band conjunct)
    ge = gold["flow_edges"]; pe = (spec or {}).get("flow_edges") or []
    bw_ok = bw_n = 0
    for k, g in enumerate(ge):
        bw_n += 1
        if k < len(pe):
            try:
                x = float(pe[k].get("bandwidth_demand"))
                lo, hi = g["bandwidth_demand"] * (1 - TOL), g["bandwidth_demand"] * (1 + TOL)
                if lo <= x <= hi:
                    bw_ok += 1
            except (TypeError, ValueError):
                pass
    s["bw_acc"] = (bw_ok / bw_n) if bw_n else 1.0
    return s


def run_approach(agent, items, kb, approach_label):
    """One approach over all 100 items. Returns per-item records. kb=None => K^A off."""
    recs = []
    t0 = time.time()
    for it in items:
        gold = it["gold"]
        try:
            spec, res = agent.translate_with_memory(it["intent"], kb=kb, mb=None,
                                                    request_id=it["id"], max_retries=1)
            valid = bool(res.is_valid)
        except Exception as e:  # api-fail / cost-cap propagate up; parse errors -> invalid
            from orion.llm.frontier_backend import CostCapExceeded
            if isinstance(e, CostCapExceeded):
                log.error("COST CAP hit in %s at item %s — stopping approach.", approach_label, it["id"])
                raise
            spec, valid = {}, False
            log.warning("%s %s: %s", approach_label, it["id"], str(e)[:100])
        sc = score_item(spec, gold, valid)
        # two-tier for ambiguous
        if it["stratum"] == "ambiguous":
            defensible = gold.get("defensible_classes", [gold["slice_type"]])
            sc["tier1_correct"] = sc["slice_type_pred"] == gold["slice_type"]
            sc["tier2_correct"] = sc["slice_type_pred"] in defensible
        recs.append({"id": it["id"], "stratum": it["stratum"], **sc})
    log.info("%s done: %d items in %.1fs", approach_label, len(recs), time.time() - t0)
    return recs


def aggregate(recs):
    n = len(recs)
    def rate(key): return round(100.0 * sum(1 for r in recs if r.get(key)) / n, 1)
    def mean(key): return round(100.0 * sum(r.get(key, 0.0) for r in recs) / n, 1)
    single = [r for r in recs if r["stratum"] != "ambiguous"]
    amb = [r for r in recs if r["stratum"] == "ambiguous"]
    def srate(rs, key): return round(100.0 * sum(1 for r in rs if r.get(key)) / len(rs), 1) if rs else None
    return {
        "n": n,
        "schema_valid_pct": rate("schema_valid"),
        "slice_type_pct": srate(single, "slice_type_correct"),
        "sfc_all_pct": rate("sfc_all_ok"),
        "sfc_order_pct": rate("sfc_order_ok"),
        "sfc_family_pct": rate("sfc_family_ok"),
        "vcr_acc_pct": mean("vcr_acc"),
        "delay_pct": rate("delay_ok"),
        "tput_pct": rate("tput_ok"),
        "bw_acc_pct": mean("bw_acc"),
        "ambiguous_tier1_pct": srate(amb, "tier1_correct"),
        "ambiguous_tier2_pct": srate(amb, "tier2_correct"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approaches", default="tele", help="comma list: tele,sonnet")
    ap.add_argument("--snapshot", default="claude-sonnet-5", help="pinned Sonnet snapshot")
    ap.add_argument("--port", type=int, default=8000, help="local Tele-8B port")
    ap.add_argument("--smoke", type=int, default=0, help="if >0, run only N items (infra check)")
    ap.add_argument("--rep", type=int, default=0,
                    help="repeat index (EXPERIMENT_PROTOCOL Amendment 9: LLM-path cells run n=3, "
                         "reported median+range). Suffixes the output file; 0 = unrepeated.")
    args = ap.parse_args()

    # Provenance guard (hard): refuses on untracked code under scripts/ or src/,
    # and on a pre-registration that has drifted from its committed copy.
    from orion.provenance import git_provenance, serving_provenance
    _prov = git_provenance(serving=serving_provenance(args.port),
                           tag=f"B-rep{args.rep}",
                           prereg="docs/PREREG_S_2026-07-15.md")
    log.info("provenance: commit=%s dirty=%s serving=%s",
             _prov["git_commit"][:8], _prov["git_dirty"], _prov["serving"])

    assert BENCH.exists(), f"missing frozen benchmark {BENCH}"
    bench_hash = _sha(BENCH)[:16]
    prereg_hash = _sha(PREREG)[:12]
    items = json.loads(BENCH.read_text(encoding="utf-8"))
    if args.smoke:
        items = items[:args.smoke]
    log.info("Track B | benchmark_S_sha256=%s prereg_S_sha256=%s | %d items | approaches=%s",
             bench_hash, prereg_hash, len(items), args.approaches)

    from orion.llm.llm_backend import LLMBackend, LLMConfig
    from orion.llm.agent_a import AgentA
    from orion.llm.semantic_memory import SemanticMemory
    kb_path = ROOT / "data" / "kb_entries.json"
    kb = SemanticMemory.from_json(kb_path) if kb_path.exists() else None
    if kb is None:
        log.warning("K^A (kb_entries.json) NOT found — K^A-on approach will equal K^A-off!")

    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    # Serving-layer provenance (chat_format alongside model_id) — pinned per run.
    serving = serving_provenance(args.port) if "tele" in approaches else None
    if serving:
        log.info("serving provenance (local): %s", serving)
    results = {"track": "B", "benchmark_S_sha256": _sha(BENCH), "prereg_S_sha256": prereg_hash,
               "snapshot": args.snapshot, "smoke": args.smoke,
               "serving_provenance_local": serving, "approaches": {}}

    for approach in approaches:
        if approach == "tele":
            backend = LLMBackend(LLMConfig(base_url=f"http://localhost:{args.port}/v1",
                                           api_key="EMPTY", model="default",
                                           temperature=0.0, max_tokens=2048))
            meter = None
        elif approach == "sonnet":
            from orion.llm.frontier_backend import (FrontierBackend, CostMeter,
                                                    make_frontier_config, PILOT_COST_CAP_USD)
            prior = json.loads(LEDGER.read_text())["spent_usd"] if LEDGER.exists() else 0.0
            remaining = max(0.0, PILOT_COST_CAP_USD - prior)
            log.info("SHARED cost ledger: prior_spent=$%.4f  cap=$%.2f  remaining=$%.4f",
                     prior, PILOT_COST_CAP_USD, remaining)
            if remaining <= 0:
                log.error("cap exhausted; refusing Sonnet approach."); continue
            meter = CostMeter(cap_usd=remaining)
            cfg = make_frontier_config(args.snapshot, temperature=0.0)
            backend = FrontierBackend(cfg, meter)
        else:
            log.warning("unknown approach %s", approach); continue

        agent = AgentA(backend)
        for kb_on in (True, False):
            label = f"{approach}|K^A={'on' if kb_on else 'off'}"
            try:
                recs = run_approach(agent, items, kb if kb_on else None, label)
            except Exception as e:
                from orion.llm.frontier_backend import CostCapExceeded
                if isinstance(e, CostCapExceeded):
                    log.error("stopping: cost cap. Partial results saved.")
                    break
                raise
            results["approaches"][label] = {"aggregate": aggregate(recs), "items": recs}
            log.info("%s -> %s", label, results["approaches"][label]["aggregate"])

        if meter is not None:
            spent = meter.spent_usd
            prior = json.loads(LEDGER.read_text())["spent_usd"] if LEDGER.exists() else 0.0
            LEDGER.write_text(json.dumps({"spent_usd": round(prior + spent, 6),
                                          "updated": "track_b", "cap": 10.0}, indent=2))
            log.info("Sonnet spent this run=$%.4f  ledger total=$%.4f", spent, prior + spent)
            results["approaches"].setdefault("_cost", {})["sonnet_run_usd"] = round(spent, 4)

    tag = "smoke" if args.smoke else "full"
    rep = f"_rep{args.rep}" if args.rep else ""
    results["provenance"] = _prov
    results["rep"] = args.rep
    outp = ROOT / "data" / f"track_b_results_{'-'.join(approaches)}_{tag}{rep}.json"
    outp.write_text(json.dumps(results, indent=2))
    log.info("results -> %s", outp)
    # readout
    print("\n" + "=" * 84)
    print(f"TRACK B READOUT ({tag}) — benchmark {bench_hash} | prereg {prereg_hash}")
    print("=" * 84)
    hdr = ["approach", "schema", "slice", "sfc_all", "vcr", "delay", "tput", "bw", "amb_t1", "amb_t2"]
    print("{:<16}{:>7}{:>7}{:>8}{:>7}{:>7}{:>7}{:>7}{:>8}{:>8}".format(*hdr))
    for label, d in results["approaches"].items():
        if label.startswith("_"):
            continue
        a = d["aggregate"]
        print("{:<16}{:>6}%{:>6}%{:>7}%{:>6}%{:>6}%{:>6}%{:>6}%{:>7}%{:>7}%".format(
            label, a["schema_valid_pct"], a["slice_type_pct"] or 0, a["sfc_all_pct"],
            a["vcr_acc_pct"], a["delay_pct"], a["tput_pct"], a["bw_acc_pct"],
            a["ambiguous_tier1_pct"] or 0, a["ambiguous_tier2_pct"] or 0))


if __name__ == "__main__":
    main()
