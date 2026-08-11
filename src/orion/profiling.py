"""Inline cost profiling — per-decision wall / CPU-time / GPU energy / CPU energy.

Captured DURING experiments (PREREG 2026-07-11), not in a separate profiling run, and stored
RAW per decision (not just averages) so the paper can report full distributions/percentiles.

Design:
  - A background PowerSampler thread polls NVML (RTX A6000) power (W) and AMD-RAPL package
    energy (uJ, if readable) at a fixed rate. This decouples measurement from the hot path.
  - `profiled(name, meta)` is a context manager recording {name, t0, t1, wall_s, cpu_s, meta}
    into the ACTIVE collector. It is a near-zero-cost no-op when no collector is active, so it
    is always safe to leave in the code.
  - At save time each event is joined with the sampler's power/energy trace over its [t0,t1]
    window: gpu_energy_j = ∫P dt (trapezoid), cpu_energy_j = ΔRAPL (wrap-corrected). CPU-time
    is the portable per-decision proxy; GPU energy is exact per LLM call (A6000 is the only
    load on that card).

Honest limits, for the methods section:
  - CPU package energy (RAPL) is socket-wide, apportioned to a window by its wall interval; on a
    near-idle experiment box this ≈ the experiment's CPU energy but is NOT per-process isolated.
  - GPU energy is whole-A6000 (all CUDA contexts); during a gate the LLM server is its only user.
  - If RAPL is not readable (root-gated), cpu_energy_j is None and CPU cost is reported as cpu_s.
"""
from __future__ import annotations

import glob
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

# ---- NVML (GPU power) ---------------------------------------------------------------------
_NVML_OK = False
_GPU_HANDLE = None
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
    # Pick the RTX A6000 by name (nvidia-smi index 1 here; the T400 reports N/A power).
    for _i in range(pynvml.nvmlDeviceGetCount()):
        _h = pynvml.nvmlDeviceGetHandleByIndex(_i)
        _name = pynvml.nvmlDeviceGetName(_h)
        if isinstance(_name, bytes):
            _name = _name.decode()
        if "A6000" in _name:
            _GPU_HANDLE = _h
            break
    if _GPU_HANDLE is None and pynvml.nvmlDeviceGetCount() > 0:
        _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # noqa: BLE001
    _NVML_OK = False


def _gpu_power_w() -> float | None:
    if not (_NVML_OK and _GPU_HANDLE is not None):
        return None
    try:
        return pynvml.nvmlDeviceGetPowerUsage(_GPU_HANDLE) / 1000.0  # mW -> W
    except Exception:  # noqa: BLE001
        return None


def _gpu_mem_used_bytes() -> int | None:
    """Current used memory on the profiled GPU (whole-card, all contexts)."""
    if not (_NVML_OK and _GPU_HANDLE is not None):
        return None
    try:
        return int(pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE).used)
    except Exception:  # noqa: BLE001
        return None


def _gpu_proc_util(since_us: int) -> dict[int, int] | None:
    """{pid: sm_util%} for processes with samples since `since_us` (epoch us).

    The A6000 is shared: another user runs a vLLM server on the same card, so
    whole-card power is not this experiment's power. NVML reports SM utilisation
    per PID, which is what makes a foreign-load window detectable instead of
    silently added to the energy figure. Returns None if the query is
    unavailable; an empty dict means nothing used the GPU in the window.
    """
    if not (_NVML_OK and _GPU_HANDLE is not None):
        return None
    try:
        return {s.pid: int(s.smUtil)
                for s in pynvml.nvmlDeviceGetProcessUtilization(_GPU_HANDLE, since_us)}
    except Exception:  # noqa: BLE001
        # NVML raises NOT_FOUND when no samples fall in the window, which is not
        # the same as "no foreign load" -- report unknown rather than clean.
        return None


# ---- RAPL (AMD CPU package energy via intel-rapl driver) ----------------------------------
def _discover_rapl() -> list[tuple[str, str, int]]:
    """Return [(name, energy_uj_path, max_range_uj)] for readable RAPL package zones."""
    zones = []
    for path in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        ep = f"{path}/energy_uj"
        try:
            with open(ep) as f:
                f.read()  # readability probe
            name = "pkg"
            try:
                name = open(f"{path}/name").read().strip()
            except Exception:  # noqa: BLE001
                pass
            mx = 0
            try:
                mx = int(open(f"{path}/max_energy_range_uj").read().strip())
            except Exception:  # noqa: BLE001
                pass
            zones.append((name, ep, mx))
        except Exception:  # noqa: BLE001
            continue  # not readable (root-gated) -> skip
    return zones


def _rapl_read(zones) -> float | None:
    """Sum of readable RAPL zones in uJ (monotonic counters, may wrap)."""
    if not zones:
        return None
    tot = 0.0
    for _n, ep, _mx in zones:
        try:
            tot += float(open(ep).read().strip())
        except Exception:  # noqa: BLE001
            return None
    return tot


class PowerSampler(threading.Thread):
    """Background power/energy trace. Samples (t_epoch, gpu_w, rapl_uj) at `hz`."""

    def __init__(self, hz: float = 20.0, own_pids: set[int] | None = None):
        super().__init__(daemon=True)
        self.dt = 1.0 / hz
        self._stop_evt = threading.Event()
        self.rapl_zones = _discover_rapl()
        self.rapl_max = sum(mx for _n, _e, mx in self.rapl_zones) if self.rapl_zones else 0
        self.ts: list[float] = []
        self.gpu_w: list[float] = []
        self.rapl_uj: list[float] = []
        # PIDs whose GPU work IS this experiment: this process plus the llama.cpp
        # server it talks to. Anything else on the card is foreign load.
        self.own_pids = set(own_pids or ())
        self.foreign: list[bool | None] = []   # per tick: True/False, None = unknown
        self.foreign_pids_seen: dict[int, int] = {}   # pid -> max sm_util observed

    @property
    def rapl_available(self) -> bool:
        return bool(self.rapl_zones)

    @property
    def gpu_available(self) -> bool:
        return _gpu_power_w() is not None

    def run(self):
        while not self._stop_evt.is_set():
            now = time.time()
            self.ts.append(now)
            self.gpu_w.append(_gpu_power_w() or 0.0)
            self.rapl_uj.append(_rapl_read(self.rapl_zones) or 0.0)
            util = _gpu_proc_util(int((now - self.dt * 2) * 1e6))
            if util is None:
                self.foreign.append(None)
            else:
                hot = {p: u for p, u in util.items()
                       if u > 0 and p not in self.own_pids}
                for p, u in hot.items():
                    self.foreign_pids_seen[p] = max(self.foreign_pids_seen.get(p, 0), u)
                self.foreign.append(bool(hot))
            time.sleep(self.dt)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)

    def gpu_energy_j(self, t0: float, t1: float) -> float | None:
        """Trapezoidal ∫ P dt over [t0,t1] from samples."""
        if not self.gpu_available or len(self.ts) < 2:
            return None
        e = 0.0
        for i in range(1, len(self.ts)):
            a, b = self.ts[i - 1], self.ts[i]
            if b <= t0 or a >= t1:
                continue
            lo, hi = max(a, t0), min(b, t1)
            if hi <= lo:
                continue
            # linear interp of power at window edges within [a,b]
            span = b - a if b > a else 1e-9
            pa = self.gpu_w[i - 1] + (self.gpu_w[i] - self.gpu_w[i - 1]) * ((lo - a) / span)
            pb = self.gpu_w[i - 1] + (self.gpu_w[i] - self.gpu_w[i - 1]) * ((hi - a) / span)
            e += 0.5 * (pa + pb) * (hi - lo)
        return e

    def gpu_clean_fraction(self, t0: float, t1: float) -> float | None:
        """Share of [t0,t1] during which no foreign PID used the GPU.

        1.0 means the card was ours for the whole window and its energy is
        attributable. Anything less means another user's process was running and
        the whole-card integral is not this experiment's cost. None when NVML
        could not tell us, which must not be read as clean.
        """
        if len(self.ts) < 2:
            return None
        span = max(t1 - t0, 1e-9)
        clean = 0.0
        for i in range(1, len(self.ts)):
            a, b = self.ts[i - 1], self.ts[i]
            lo, hi = max(a, t0), min(b, t1)
            if hi <= lo:
                continue
            f = self.foreign[i] if i < len(self.foreign) else None
            if f is None:
                return None
            if not f:
                clean += hi - lo
        return clean / span

    def cpu_energy_j(self, t0: float, t1: float) -> float | None:
        """ΔRAPL over [t0,t1] (uJ->J), wrap-corrected, interpolated at edges."""
        if not self.rapl_available or len(self.ts) < 2:
            return None

        def interp(tq):
            if tq <= self.ts[0]:
                return self.rapl_uj[0]
            if tq >= self.ts[-1]:
                return self.rapl_uj[-1]
            for i in range(1, len(self.ts)):
                if self.ts[i] >= tq:
                    a, b = self.ts[i - 1], self.ts[i]
                    ua, ub = self.rapl_uj[i - 1], self.rapl_uj[i]
                    if ub < ua and self.rapl_max:  # counter wrapped
                        ub += self.rapl_max
                    span = b - a if b > a else 1e-9
                    return ua + (ub - ua) * ((tq - a) / span)
            return self.rapl_uj[-1]

        d_uj = interp(t1) - interp(t0)
        if d_uj < 0 and self.rapl_max:
            d_uj += self.rapl_max
        return d_uj / 1e6


# ---- Collector + profiled() context -------------------------------------------------------
class ProfileCollector:
    """Accumulates raw per-decision events; joins with a PowerSampler at save time."""

    def __init__(self, sampler: PowerSampler | None = None, label: str = ""):
        self.sampler = sampler
        self.label = label
        self.events: list[dict] = []
        self._proc = psutil.Process()
        self._lock = threading.Lock()
        # §O.9 — cell-level accounting baseline + peak-memory tracking.
        self._t0_epoch = time.time()
        _ct = self._proc.cpu_times()
        self._cpu0 = _ct.user + _ct.system
        self.peak_rss_bytes: int = 0
        self.peak_gpu_mem_bytes: int | None = None
        self.sample_memory()

    def add(self, ev: dict):
        with self._lock:
            self.events.append(ev)

    def sample_memory(self):
        """§O.9 (lead addition): sample peak RSS / GPU memory; call at round
        boundaries. GPU number is whole-card (all contexts), same caveat as
        GPU energy."""
        try:
            rss = self._proc.memory_info().rss
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
        except Exception:  # noqa: BLE001
            pass
        gm = _gpu_mem_used_bytes()
        if gm is not None and (self.peak_gpu_mem_bytes is None or gm > self.peak_gpu_mem_bytes):
            self.peak_gpu_mem_bytes = gm

    def cell_totals(self) -> dict:
        """§O.9 — per-cell totals: wall clock, CPU time, GPU/CPU energy
        (measured or explicitly-labeled estimate), peak memory."""
        from orion.config import MDO_CPU_WATT_PER_CORE

        self.sample_memory()
        now = time.time()
        _ct = self._proc.cpu_times()
        cpu_total = (_ct.user + _ct.system) - self._cpu0
        s = self.sampler
        rapl_ok = bool(s and s.rapl_available)
        out = {
            "wall_s": now - self._t0_epoch,
            "cpu_s": cpu_total,
            "gpu_energy_j": s.gpu_energy_j(self._t0_epoch, now) if s else None,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_gpu_mem_bytes": self.peak_gpu_mem_bytes,
        }
        if rapl_ok:
            out["cpu_energy_j"] = s.cpu_energy_j(self._t0_epoch, now)
            out["cpu_energy_method"] = "rapl-measured"
        else:
            out["cpu_energy_j_est"] = cpu_total * MDO_CPU_WATT_PER_CORE
            out["cpu_energy_method"] = "tdp-estimate"
        return out

    def summary(self) -> dict:
        """Per-decision-type distributions (count, mean, p50/p90/p99, sum) for wall/cpu/energy."""
        import numpy as np

        by = {}
        for e in self.events:
            by.setdefault(e["name"], []).append(e)
        out = {}
        def dist(vals):
            a = np.asarray(vals, dtype=float)
            return {"mean": float(a.mean()), "sum": float(a.sum()),
                    "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
                    "p99": float(np.percentile(a, 99)), "max": float(a.max()),
                    "n": int(a.size)}

        for name, evs in by.items():
            row = {"count": len(evs)}
            for k in ("wall_s", "cpu_s", "gpu_energy_j", "cpu_energy_j",
                      "cpu_energy_j_est", "gpu_clean_frac"):
                vals = [e[k] for e in evs if e.get(k) is not None]
                if vals:
                    row[k] = dist(vals)
            # Energy over the windows the card was ours for the WHOLE event. This
            # is the attributable number; `gpu_energy_j` above is whole-card and
            # includes any foreign load. n_clean vs count says how much survived.
            clean = [e["gpu_energy_j"] for e in evs
                     if e.get("gpu_clean_frac") == 1.0 and e.get("gpu_energy_j") is not None]
            row["n_clean"] = len(clean)
            if clean:
                row["gpu_energy_j_clean"] = dist(clean)
            out[name] = row
        return out

    def save(self, raw_path: str | Path):
        """Fill energy windows from the sampler, dump raw events + summary sidecar.
        GPU energy is MEASURED (NVML). CPU energy is MEASURED (RAPL) if readable, else a
        clearly-flagged TDP ESTIMATE = cpu_s * MDO_CPU_WATT_PER_CORE (cpu_energy_j_est)."""
        from orion.config import MDO_CPU_WATT_PER_CORE

        s = self.sampler
        rapl_ok = bool(s and s.rapl_available)
        for e in self.events:
            if s is not None and "t0" in e and "t1" in e:
                if e.get("gpu_energy_j") is None:
                    e["gpu_energy_j"] = s.gpu_energy_j(e["t0"], e["t1"])
                # Whole-card energy is only this experiment's energy when nothing
                # else was on the card. Carry the share so a contaminated window
                # can be dropped downstream instead of averaged in.
                if e.get("gpu_clean_frac") is None:
                    e["gpu_clean_frac"] = s.gpu_clean_fraction(e["t0"], e["t1"])
                if rapl_ok and e.get("cpu_energy_j") is None:
                    e["cpu_energy_j"] = s.cpu_energy_j(e["t0"], e["t1"])
            if not rapl_ok and e.get("cpu_s") is not None:  # measured energy unavailable
                e["cpu_energy_j_est"] = e["cpu_s"] * MDO_CPU_WATT_PER_CORE
        raw_path = Path(raw_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"label": self.label,
                   "cpu_energy_method": "rapl-measured" if rapl_ok else "tdp-estimate",
                   "cpu_watt_per_core": None if rapl_ok else MDO_CPU_WATT_PER_CORE,
                   "rapl_available": rapl_ok,
                   "gpu_available": bool(s and s.gpu_available),
                   # Who else was on the card, and how hard. A non-empty map means
                   # some windows' whole-card energy is not attributable here.
                   "gpu_own_pids": sorted(s.own_pids) if s else [],
                   "gpu_foreign_pids": (dict(sorted(s.foreign_pids_seen.items()))
                                        if s else {}),
                   "cell_totals": self.cell_totals(),  # §O.9
                   "events": self.events, "summary": self.summary()},
                  open(raw_path, "w"))
        return self.summary()


_ACTIVE: ProfileCollector | None = None


def set_collector(c: ProfileCollector | None):
    global _ACTIVE
    _ACTIVE = c


def get_collector() -> ProfileCollector | None:
    return _ACTIVE


@contextmanager
def profiled(name: str, meta: dict | None = None):
    """Record one decision's wall + cpu-time (+ energy window). No-op if no active collector."""
    c = _ACTIVE
    if c is None:
        yield
        return
    t0 = time.time()
    w0 = time.perf_counter()
    c0 = time.process_time()  # CLOCK_PROCESS_CPUTIME_ID: user+sys CPU-seconds, ~µs resolution
    try:
        yield
    finally:
        w1 = time.perf_counter()
        c1 = time.process_time()
        ev = {"name": name, "t0": t0, "t1": time.time(),
              "wall_s": w1 - w0, "cpu_s": c1 - c0,
              "gpu_energy_j": None, "cpu_energy_j": None}
        if meta:
            ev["meta"] = meta
        c.add(ev)
