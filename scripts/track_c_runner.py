#!/usr/bin/env python
"""§S Track C — Agent B plan-quality probe. 30 RC arrivals (10/seed 42/43/44), both
{Tele-8B, Sonnet} generate one plan each cache-off; scored offline (admit + reject taxonomy +
forced/flex spread). Tests the A.2 dated prediction: Sonnet collapses the same way as Tele-8B
(observation-adequacy failure at the plan layer, not model capacity). Sonnet metered via the
shared $10 ledger."""
import sys, json, argparse, copy, logging
from collections import Counter
from pathlib import Path
sys.path.insert(0, "scripts")
import numpy as np
import q_pilot_runner as Q
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.actors.greedy_domain_actor import GreedyDomainActor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trackC")
ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "data" / "api_cost_ledger.json"
BW = {42: 70.0, 43: 90.0, 44: 110.0}
N_PER_SEED = 10


def _delays(sub):
    g = sub.graph; d = {}
    for u, v in g.edges():
        ud, vd = g.nodes[u].get("domain_id", -1), g.nodes[v].get("domain_id", -1)
        if ud != vd and ud >= 0 and vd >= 0:
            k = (min(ud, vd), max(ud, vd))
            d[k] = min(d.get(k, 999.0), float(g[u][v]["propagation_delay"]))
    return d


def score_model(agent_b, kb, label):
    from orion.llm.frontier_backend import CostCapExceeded
    recs = []
    for seed in (42, 43, 44):
        sub = Q.generate_rc_instance(seed=Q.RC_GEN_SEED + (seed - 42), inter_domain_bw_override=BW[seed])
        rng = np.random.default_rng(seed)
        ap = ArrivalProcess(sub, 100, Q.ARRIVAL_RATE, Q.SERVICE_RATE, rng, slice_factory=Q.rc_slice_factory)
        ap.generate()
        actors = {dd: GreedyDomainActor(dd) for dd in range(sub.num_domains)}
        coord = MDOCoordinator(None, actors, MDOConfig(n_part=1))
        delays = _delays(sub)
        n = 0
        for ev in ap.events:
            if ev.event_type != EventType.ARRIVAL or ev.slice_request is None:
                continue
            if n >= N_PER_SEED:
                break
            n += 1
            sr = ev.slice_request
            forced = Q._rc_forced(sr, sub)
            try:
                sr_dict = Q._slice_request_to_dict(sr, sub)
                topo = Q.build_abstract_topology(sub)
                plan_dict, check = agent_b.generate_with_memory(sr_dict, topo, kb=kb, mb=None, max_retries=0)
            except CostCapExceeded:
                raise
            except Exception:  # noqa: BLE001
                recs.append({"seed": seed, "forced": forced, "admit": False, "reason": "gen_fail"}); continue
            if not getattr(check, "is_valid", False):
                recs.append({"seed": seed, "forced": forced, "admit": False, "reason": "structural"}); continue
            greedy = Q._llm_plan_to_greedy_result(plan_dict, sr, sub)
            if greedy is None:
                recs.append({"seed": seed, "forced": forced, "admit": False, "reason": "structural"}); continue
            ps = Q.plan_to_summary(greedy, sr, sub)
            if ps is None:
                recs.append({"seed": seed, "forced": forced, "admit": False, "reason": "plan_conv_fail"}); continue
            mdo = coord.resolve_arrival(copy.deepcopy(sub), sr, ps, delays, mode="follow_prior")
            if mdo.admitted:
                recs.append({"seed": seed, "forced": forced, "admit": True, "reason": None})
            else:
                v = mdo.retry_history.attempts[-1].violation if mdo.retry_history.attempts else None
                tag = (("actor_infeasible" if v.actor_infeasible else
                        "cross_domain_bw" if v.cross_domain_infeasible else
                        "c7_delay" if v.c7_violated else "other") if v else "coordinator")
                recs.append({"seed": seed, "forced": forced, "admit": False, "reason": tag})
        log.info("%s seed %d: %d arrivals scored", label, seed, n)
    return recs


def agg(recs):
    n = len(recs); adm = sum(1 for r in recs if r["admit"])
    tax = Counter(r["reason"] for r in recs if not r["admit"])
    flex = [r for r in recs if not r["forced"]]
    flex_spread = sum(1 for r in flex if not r["admit"] and r["reason"] == "cross_domain_bw")
    return {"n": n, "admit": adm, "admit_pct": round(100.0 * adm / max(1, n), 1),
            "taxonomy": dict(tax), "flex_n": len(flex), "flex_spread_and_fail": flex_spread}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="tele,sonnet")
    ap.add_argument("--snapshot", default="claude-sonnet-5")
    ap.add_argument("--rep", type=int, default=0,
                    help="repeat index (Amendment 9: LLM-path cells run n=3, median+range)")
    a = ap.parse_args()

    from orion.provenance import git_provenance, serving_provenance
    _prov = git_provenance(serving=serving_provenance(8000), tag=f"C-rep{a.rep}",
                           prereg="docs/PREREG_S_2026-07-15.md")
    print("provenance: commit=%s dirty=%s serving=%s"
          % (_prov["git_commit"][:8], _prov["git_dirty"], _prov["serving"]))
    from orion.llm.semantic_memory import SemanticMemory
    kbp = ROOT / "data" / "kb_entries.json"
    kb = SemanticMemory.from_json(kbp) if kbp.exists() else None
    res = {"track": "C", "snapshot": a.snapshot, "rep": a.rep,
           "provenance": _prov, "arms": {}}
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        meter = None
        if arm == "tele":
            from orion.llm.llm_backend import LLMBackend, LLMConfig
            from orion.llm.agent_b import AgentB
            agent = AgentB(LLMBackend(LLMConfig(base_url="http://localhost:8000/v1", api_key="EMPTY",
                                                model="default", temperature=0.0, max_tokens=2048)))
        else:
            from orion.llm.frontier_backend import (FrontierBackend, CostMeter,
                                                    make_frontier_config, PILOT_COST_CAP_USD)
            from orion.llm.agent_b import AgentB
            prior = json.loads(LEDGER.read_text())["spent_usd"] if LEDGER.exists() else 0.0
            rem = max(0.0, PILOT_COST_CAP_USD - prior)
            if rem <= 0:
                log.error("cap exhausted; skipping sonnet"); continue
            meter = CostMeter(cap_usd=rem)
            agent = AgentB(FrontierBackend(make_frontier_config(a.snapshot, temperature=0.0), meter))
        try:
            recs = score_model(agent, kb, arm)
        except Exception as e:  # noqa: BLE001
            log.error("%s stopped: %s", arm, str(e)[:120]); recs = []
        res["arms"][arm] = {"aggregate": agg(recs) if recs else None, "items": recs}
        log.info("%s -> %s", arm, res["arms"][arm]["aggregate"])
        if meter is not None:
            spent = meter.spent_usd
            prior = json.loads(LEDGER.read_text())["spent_usd"] if LEDGER.exists() else 0.0
            LEDGER.write_text(json.dumps({"spent_usd": round(prior + spent, 6),
                                          "updated": "track_c", "cap": 10.0}, indent=2))
            log.info("Sonnet spent=$%.4f ledger=$%.4f", spent, prior + spent)
    _rep = f"_rep{a.rep}" if a.rep else ""
    (ROOT / "data" / f"track_c_results{_rep}.json").write_text(json.dumps(res, indent=2, default=str))
    log.info("results -> data/track_c_results.json")
    print("\n=== TRACK C READOUT — Agent B plan quality (30 RC arrivals) ===")
    for arm, d in res["arms"].items():
        a2 = d.get("aggregate")
        if a2:
            print(f"  {arm:8} admit={a2['admit']}/{a2['n']} ({a2['admit_pct']}%)  "
                  f"flex_spread_fail={a2['flex_spread_and_fail']}/{a2['flex_n']}  tax={a2['taxonomy']}")
    print("  A.2 prediction: Sonnet ~= Tele (both collapse) => observation-adequacy confirmed")


if __name__ == "__main__":
    main()
