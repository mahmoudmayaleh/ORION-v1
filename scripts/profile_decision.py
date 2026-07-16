#!/usr/bin/env python3
"""Per-decision profiler for the ORION paper.

Measures, per ARM x per STAGE, the cost of a single admission decision:
  - wall-clock latency (ms)
  - CPU time (ms, process_time) and derived CPU utilisation
  - GPU utilisation / power / ENERGY (J) for the LLM stage (NVML, ~20 Hz)
  - memory: peak VRAM (from GPU samples) and process RSS

Decisions are run ONE AT A TIME against a single LLM server on an otherwise idle
A6000, i.e. the deployment-representative cost of a single decision (NOT the
throughput-optimised 2-server path used for the batch run).

Stages
  RA-ColocFB (static, no GPU):  plan_build -> plan_summary -> mdo_routing
  Memory-off / Full-M^B (LLM):  prep -> retrieval -> llm_inference -> struct_check
                                -> plan_convert -> plan_summary -> mdo_routing
                                [-> mb_write   (Full-M^B only)]

The profiler calls the SAME production functions used by five_arm_runner.py;
only the stage boundaries are added, so the measured work is identical to a real
run. Outputs (in --out dir):
  profile_raw.csv       one row per (arm, family, decision, stage)
  profile_summary.csv   aggregated per (arm, stage)
  profile_summary.md    paper-ready table
  profile_meta.json     run metadata (idle power, GPU name, config)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import resource
import statistics
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parent / "src"))

import five_arm_runner as R  # noqa: E402  reuse the exact production helpers
from orion.actors.greedy_domain_actor import GreedyDomainActor  # noqa: E402
from orion.mdo.coordinator import MDOConfig, MDOCoordinator  # noqa: E402
from orion.llm.llm_backend import LLMBackend, LLMConfig  # noqa: E402
from orion.llm.agent_b import AgentB  # noqa: E402
from orion.llm.semantic_memory import SemanticMemory, build_query_from_slice  # noqa: E402
from orion.llm.episodic_memory import EpisodicMemory  # noqa: E402
from orion.llm.structural_checker import check_plan  # noqa: E402
from orion.retrieval import RetrievalConfig, RetrievalMode  # noqa: E402


# ── GPU sampler (NVML preferred, nvidia-smi fallback) ───────────────────────

class GPUSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []            # (t_perf, util%, power_W, mem_MB, sm_clk)
        self._stop = threading.Event()
        self.idle_power_w = 0.0
        self._read, self.backend, self.gpu_name = _make_gpu_reader()

    def run(self):
        while not self._stop.is_set():
            try:
                u, p, m, c = self._read()
                self.samples.append((time.perf_counter(), u, p, m, c))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()

    def _window(self, t0, t1):
        return [s for s in self.samples if t0 <= s[0] <= t1]

    def energy_j(self, t0, t1, subtract_idle=False):
        pts = self._window(t0, t1)
        if len(pts) < 2:
            return 0.0
        e = 0.0
        for i in range(1, len(pts)):
            dt = pts[i][0] - pts[i - 1][0]
            p0, p1 = pts[i - 1][2], pts[i][2]
            if subtract_idle:
                p0 = max(0.0, p0 - self.idle_power_w)
                p1 = max(0.0, p1 - self.idle_power_w)
            e += 0.5 * (p0 + p1) * dt
        return e

    def stats(self, t0, t1):
        pts = self._window(t0, t1)
        if not pts:
            return 0.0, 0.0, 0.0
        util = statistics.mean(s[1] for s in pts)
        power = statistics.mean(s[2] for s in pts)
        vram = max(s[3] for s in pts)
        return util, power, vram

    def measure_idle(self, seconds=5.0):
        t0 = time.perf_counter()
        time.sleep(seconds)
        pts = self._window(t0, time.perf_counter())
        if pts:
            self.idle_power_w = statistics.mean(s[2] for s in pts)
        return self.idle_power_w


def _make_gpu_reader():
    """Return (read_fn, backend_name, gpu_name). read_fn -> (util%,power_W,mem_MB,sm_clk)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = None
        name = "unknown"
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            nm = pynvml.nvmlDeviceGetName(h)
            nm = nm.decode() if isinstance(nm, bytes) else nm
            if "A6000" in nm:
                handle, name = h, nm
                break
        if handle is None:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            nm = pynvml.nvmlDeviceGetName(handle)
            name = nm.decode() if isinstance(nm, bytes) else nm

        def read():
            u = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            m = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
            try:
                c = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            except Exception:
                c = 0
            return u, p, m, c

        return read, "pynvml", name
    except Exception:
        import subprocess
        # Find the A6000 index.
        idx = "0"
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"]).decode()
            for line in out.splitlines():
                if "A6000" in line and line.startswith("GPU "):
                    idx = line.split(":")[0].split()[1]
                    break
        except Exception:
            pass

        def read():
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,power.draw,memory.used,clocks.sm",
                "--format=csv,noheader,nounits", "-i", idx,
            ]).decode().strip()
            u, p, m, c = [x.strip() for x in out.split(",")]
            return float(u), float(p), float(m), float(c)

        return read, "nvidia-smi", f"A6000(idx {idx})"


# ── Stage timing ────────────────────────────────────────────────────────────

@contextmanager
def stage(rows, arm, fam, idx, name, gpu=None):
    w0, c0 = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        w1, c1 = time.perf_counter(), time.process_time()
        row = {"arm": arm, "family": fam, "decision": idx, "stage": name,
               "wall_ms": (w1 - w0) * 1000.0, "cpu_ms": (c1 - c0) * 1000.0,
               "gpu_energy_J": "", "gpu_energy_above_idle_J": "",
               "gpu_util_mean": "", "gpu_power_mean_W": "", "vram_peak_MB": ""}
        if gpu is not None:
            row["gpu_energy_J"] = gpu.energy_j(w0, w1)
            row["gpu_energy_above_idle_J"] = gpu.energy_j(w0, w1, subtract_idle=True)
            u, p, m = gpu.stats(w0, w1)
            row["gpu_util_mean"], row["gpu_power_mean_W"], row["vram_peak_MB"] = u, p, m
        rows.append(row)


# ── Pipeline (mirrors run_instance / run_llm_arm, timed per stage) ───────────

def build_coord(substrate):
    actors = {d: GreedyDomainActor(d) for d in range(substrate.num_domains)}
    coord = MDOCoordinator(None, actors, MDOConfig(n_part=1))
    delays = {}
    g = substrate.graph
    for u, v in g.edges():
        ud, vd = g.nodes[u].get("domain_id", -1), g.nodes[v].get("domain_id", -1)
        if ud != vd and ud >= 0 and vd >= 0:
            key = (min(ud, vd), max(ud, vd))
            delays[key] = min(delays.get(key, 999.0), float(g[u][v]["propagation_delay"]))
    return coord, delays


def profile_static_decision(rows, arm, fam, idx, sr, substrate, coord, delays):
    with stage(rows, arm, fam, idx, "plan_build"):
        ok, builder = R.run_static_arm(arm, substrate, sr)
    if not ok:
        return
    with stage(rows, arm, fam, idx, "plan_summary"):
        summary = R.plan_to_summary(builder, sr, substrate)
    if summary is None:
        return
    with stage(rows, arm, fam, idx, "mdo_routing"):
        coord.resolve_arrival(copy.deepcopy(substrate), sr, summary, delays, mode="follow_prior")


def profile_llm_decision(rows, arm, fam, idx, sr, substrate, agent_b, kb, mb,
                         coord, delays, topo_sig, gpu, record=True):
    r = rows if record else []
    with stage(r, arm, fam, idx, "prep"):
        abstract_topo = R.build_abstract_topology(substrate)
        sr_dict = R._slice_request_to_dict(sr, substrate)

    with stage(r, arm, fam, idx, "retrieval"):
        query = build_query_from_slice(sr_dict)
        ref_knowledge = None
        if kb is not None:
            formatted = kb.format_for_prompt(kb.retrieve(query, top_k=5))
            ref_knowledge = formatted or None
        few_shot = None
        if mb is not None:
            converted = mb.to_few_shot(mb.retrieve(query, top_k=3))
            few_shot = converted or None

    plan_dict, valid = {}, False
    with stage(r, arm, fam, idx, "llm_inference", gpu=gpu if record else None):
        try:
            plan_dict = agent_b.generate_plan(sr_dict, abstract_topo, few_shot, None, ref_knowledge)
        except Exception:
            plan_dict = {}

    with stage(r, arm, fam, idx, "struct_check"):
        check = check_plan(plan_dict, sr_dict, abstract_topo) if plan_dict else None
        valid = bool(check and check.is_valid)

    admitted, plan_shape = False, None
    if valid:
        with stage(r, arm, fam, idx, "plan_convert"):
            plan_result = R._llm_plan_to_greedy_result(plan_dict, sr, substrate)
            feasible = plan_result is not None and plan_result.feasible
            plan_shape = R._extract_plan_shape(plan_result, sr, substrate) if feasible else None
        if feasible:
            with stage(r, arm, fam, idx, "plan_summary"):
                summary = R.plan_to_summary(plan_result, sr, substrate)
            if summary is not None:
                with stage(r, arm, fam, idx, "mdo_routing"):
                    res = coord.resolve_arrival(copy.deepcopy(substrate), sr, summary,
                                                delays, mode="follow_prior")
                    admitted = res.admitted

    if arm == "Full-M^B" and mb is not None:
        with stage(r, arm, fam, idx, "mb_write"):
            R.write_to_mb(mb, arm, sr, admitted, plan_dict, [], topo_sig, plan_shape)


# ── Aggregation & output ────────────────────────────────────────────────────

def pctile(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


STAGE_ORDER = ["prep", "retrieval", "llm_inference", "struct_check", "plan_convert",
               "plan_build", "plan_summary", "mdo_routing", "mb_write"]


def aggregate(rows):
    by = defaultdict(list)
    for row in rows:
        by[(row["arm"], row["stage"])].append(row)
    summary = []
    for (arm, st), rs in by.items():
        walls = [x["wall_ms"] for x in rs]
        cpus = [x["cpu_ms"] for x in rs]
        rec = {
            "arm": arm, "stage": st, "n": len(rs),
            "wall_ms_mean": statistics.mean(walls),
            "wall_ms_std": statistics.pstdev(walls) if len(walls) > 1 else 0.0,
            "wall_ms_p50": pctile(walls, 0.50),
            "wall_ms_p95": pctile(walls, 0.95),
            "cpu_ms_mean": statistics.mean(cpus),
        }
        if st == "llm_inference":
            en = [x["gpu_energy_J"] for x in rs if x["gpu_energy_J"] != ""]
            eni = [x["gpu_energy_above_idle_J"] for x in rs if x["gpu_energy_above_idle_J"] != ""]
            ut = [x["gpu_util_mean"] for x in rs if x["gpu_util_mean"] != ""]
            pw = [x["gpu_power_mean_W"] for x in rs if x["gpu_power_mean_W"] != ""]
            vr = [x["vram_peak_MB"] for x in rs if x["vram_peak_MB"] != ""]
            rec["gpu_energy_J_mean"] = statistics.mean(en) if en else 0.0
            rec["gpu_energy_above_idle_J_mean"] = statistics.mean(eni) if eni else 0.0
            rec["gpu_util_mean"] = statistics.mean(ut) if ut else 0.0
            rec["gpu_power_mean_W"] = statistics.mean(pw) if pw else 0.0
            rec["vram_peak_MB"] = max(vr) if vr else 0.0
        summary.append(rec)
    summary.sort(key=lambda r: (r["arm"], STAGE_ORDER.index(r["stage"])
                                if r["stage"] in STAGE_ORDER else 99))
    return summary


def end_to_end(rows):
    per = defaultdict(lambda: {"wall": 0.0, "cpu": 0.0, "energy": 0.0})
    for row in rows:
        k = (row["arm"], row["family"], row["decision"])
        per[k]["wall"] += row["wall_ms"]
        per[k]["cpu"] += row["cpu_ms"]
        if row["gpu_energy_J"] != "":
            per[k]["energy"] += row["gpu_energy_J"]
    by_arm = defaultdict(list)
    for (arm, _f, _d), v in per.items():
        by_arm[arm].append(v)
    out = {}
    for arm, vs in by_arm.items():
        walls = [v["wall"] for v in vs]
        out[arm] = {
            "n": len(vs),
            "wall_ms_mean": statistics.mean(walls),
            "wall_ms_std": statistics.pstdev(walls) if len(walls) > 1 else 0.0,
            "wall_ms_p95": pctile(walls, 0.95),
            "cpu_ms_mean": statistics.mean(v["cpu"] for v in vs),
            "gpu_energy_J_mean": statistics.mean(v["energy"] for v in vs),
            "decisions_per_s": 1000.0 / statistics.mean(walls) if walls else 0.0,
        }
    return out


def write_outputs(outdir, rows, summary, e2e, meta):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_cols = ["arm", "family", "decision", "stage", "wall_ms", "cpu_ms",
                "gpu_energy_J", "gpu_energy_above_idle_J", "gpu_util_mean",
                "gpu_power_mean_W", "vram_peak_MB"]
    with open(outdir / "profile_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_cols)
        w.writeheader()
        w.writerows(rows)

    sum_cols = ["arm", "stage", "n", "wall_ms_mean", "wall_ms_std", "wall_ms_p50",
                "wall_ms_p95", "cpu_ms_mean", "gpu_energy_J_mean",
                "gpu_energy_above_idle_J_mean", "gpu_util_mean", "gpu_power_mean_W",
                "vram_peak_MB"]
    with open(outdir / "profile_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_cols, extrasaction="ignore")
        w.writeheader()
        for rec in summary:
            w.writerow(rec)

    with open(outdir / "profile_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Paper-ready markdown.
    lines = ["# ORION per-decision profile", ""]
    lines.append(f"GPU: {meta['gpu_name']} · idle power {meta['idle_power_W']:.1f} W · "
                 f"sampler {meta['gpu_backend']} @ {1/meta['sample_interval_s']:.0f} Hz · "
                 f"{meta['n_decisions']} decisions/arm/family, warmup {meta['warmup']}")
    lines += ["", "## Per-arm end-to-end (one decision)", "",
              "| Arm | n | Latency ms (mean±std) | p95 ms | CPU ms | GPU energy J | dec/s |",
              "|---|---|---|---|---|---|---|"]
    for arm in ["RA-ColocFB", "Memory-off", "Full-M^B"]:
        if arm not in e2e:
            continue
        e = e2e[arm]
        lines.append(f"| {arm} | {e['n']} | {e['wall_ms_mean']:.1f}±{e['wall_ms_std']:.1f} | "
                     f"{e['wall_ms_p95']:.1f} | {e['cpu_ms_mean']:.1f} | "
                     f"{e['gpu_energy_J_mean']:.2f} | {e['decisions_per_s']:.2f} |")

    lines += ["", "## Per-arm × per-stage", "",
              "| Arm | Stage | n | Wall ms (mean±std) | p95 ms | CPU ms | GPU J | GPU W | Util % | VRAM MB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for rec in summary:
        gj = f"{rec.get('gpu_energy_J_mean', 0):.2f}" if rec["stage"] == "llm_inference" else ""
        gw = f"{rec.get('gpu_power_mean_W', 0):.0f}" if rec["stage"] == "llm_inference" else ""
        gu = f"{rec.get('gpu_util_mean', 0):.0f}" if rec["stage"] == "llm_inference" else ""
        vr = f"{rec.get('vram_peak_MB', 0):.0f}" if rec["stage"] == "llm_inference" else ""
        lines.append(f"| {rec['arm']} | {rec['stage']} | {rec['n']} | "
                     f"{rec['wall_ms_mean']:.2f}±{rec['wall_ms_std']:.2f} | "
                     f"{rec['wall_ms_p95']:.2f} | {rec['cpu_ms_mean']:.2f} | {gj} | {gw} | {gu} | {vr} |")
    (outdir / "profile_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--families", nargs="+", default=["C+_T+_B+", "C+_T-_B-"])
    ap.add_argument("--n-decisions", type=int, default=50, help="measured decisions per arm per family")
    ap.add_argument("--warmup", type=int, default=3, help="discarded warm-up decisions per arm per family")
    ap.add_argument("--inst-seed", type=int, default=0)
    ap.add_argument("--arrival-seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    fam_lookup = {f.short_name: f for f in R.ALL_FAMILIES}
    gpu = GPUSampler(interval=0.05)
    gpu.start()
    print(f"GPU sampler: {gpu.backend} ({gpu.gpu_name}); measuring idle power (5s)...")
    idle = gpu.measure_idle(5.0)
    print(f"idle power ~ {idle:.1f} W")

    rows = []
    for fam in args.families:
        family = fam_lookup[fam]
        substrate = R.generate_family_instance(family, seed=args.inst_seed)
        topo_sig = R.compute_signature(substrate, fam).to_dict()
        coord, delays = build_coord(substrate)

        rng = np.random.default_rng(args.arrival_seed)
        ap_ = R.ArrivalProcess(substrate, R.ARRIVALS_PER_INSTANCE, R.ARRIVAL_RATE,
                               R.SERVICE_RATE, rng)
        ap_.generate()
        arrivals = [e.slice_request for e in ap_.events
                    if e.event_type == R.EventType.ARRIVAL and e.slice_request is not None]

        need = args.warmup + args.n_decisions
        arrivals = arrivals[:need]
        print(f"\n[{fam}] {len(arrivals)} arrivals (warmup {args.warmup} + measure {args.n_decisions})")

        # Fresh Agent B (own server) and K^B/M^B per family, mirroring the runner.
        cfg = LLMConfig(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY",
                        model="default", temperature=0.05, max_tokens=2048)
        agent_b = AgentB(LLMBackend(cfg))
        kb_path = _SCRIPTS.parent / "data" / "kb_entries.json"
        kb = SemanticMemory.from_json(kb_path) if kb_path.exists() else None
        mb_full = EpisodicMemory(
            config=RetrievalConfig(mode=RetrievalMode.NO_RERANK, apply_recency=True, k_final=3),
            max_entries=R.MEMORY_CAPACITY_K)

        # Pre-populate M^B so Full-M^B is measured with a realistic (full) memory,
        # matching the eval phase. Use the warm-up arrivals for writes (unmeasured).
        for sr in arrivals[:args.warmup]:
            profile_llm_decision([], "Full-M^B", fam, -1, sr, substrate, agent_b,
                                 kb, mb_full, coord, delays, topo_sig, gpu, record=False)

        for arm in R.ARM_NAMES:
            # LLM-server / code warm-up (discarded).
            for sr in arrivals[:args.warmup]:
                if arm in R.STATIC_ARMS:
                    profile_static_decision([], arm, fam, -1, sr, substrate, coord, delays)
                else:
                    mb = mb_full if arm == "Full-M^B" else None
                    profile_llm_decision([], arm, fam, -1, sr, substrate, agent_b,
                                         kb, mb, coord, delays, topo_sig, gpu, record=False)
            # Measured decisions.
            for i, sr in enumerate(arrivals[args.warmup:]):
                if arm in R.STATIC_ARMS:
                    profile_static_decision(rows, arm, fam, i, sr, substrate, coord, delays)
                else:
                    mb = mb_full if arm == "Full-M^B" else None
                    profile_llm_decision(rows, arm, fam, i, sr, substrate, agent_b,
                                         kb, mb, coord, delays, topo_sig, gpu)
            print(f"  {arm}: measured {args.n_decisions} decisions")

    gpu.stop()
    summary = aggregate(rows)
    e2e = end_to_end(rows)
    meta = {
        "gpu_name": gpu.gpu_name, "gpu_backend": gpu.backend,
        "idle_power_W": gpu.idle_power_w, "sample_interval_s": gpu.interval,
        "n_decisions": args.n_decisions, "warmup": args.warmup,
        "families": args.families, "port": args.port,
        "peak_rss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    write_outputs(args.out, rows, summary, e2e, meta)
    print(f"\nPeak runner RSS: {meta['peak_rss_MB']:.0f} MB")
    print(f"Outputs in {args.out}/: profile_raw.csv, profile_summary.csv, profile_summary.md")


if __name__ == "__main__":
    main()
