"""Per-decision cost tables and figures from the inline profiler.

Reads the sidecars written by `grid_runner --profile` (ORION_PROFILE_DIR, default
results/profile_cells) and the matching cells, and writes to results/y15_cost/.

Three things this refuses to do, because each would produce a confident wrong
number:

  * GPU energy is attributed ONLY to `llm.generate`. The A6000 draws ~30 W idle
    and more when another user's server is on it, so integrating whole-card power
    over a heuristic approach's windows would charge Plain hundreds of joules for
    work it never sent to the GPU. Approaches that make no model call report `--`,
    not a number.
  * Only windows with gpu_clean_frac == 1.0 count. The card is shared; a window
    with foreign load is dropped and the drop is reported, never averaged in.
  * CPU energy is a TDP estimate (cpu_s x MDO_CPU_WATT_PER_CORE), because RAPL is
    root-gated on this box. It is labelled as an estimate in every output.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROFILES = Path("results/profile_cells")
CELLS = Path("data/grid_cells")
OUT = Path("results/y15_cost")

# Decision stages in pipeline order. `plain.decision` is Plain's whole decision;
# it has no partition or routing stage to break out.
STAGES = ["plain.decision", "plan_build", "llm.generate", "struct.check",
          "mdo.decision", "mdo.forward", "actor.place", "verify"]
GPU_STAGE = "llm.generate"          # the only stage that runs on the GPU
NESTED = ("mdo.forward", "actor.place")   # inside mdo.decision, never added to it

# Categorical slots in fixed order, assigned to stages in pipeline order and not
# by rank, so a stage keeps its colour when another approach drops out. Validated
# (light, 6 slots): worst adjacent CVD dE 9.1, normal-vision floor 19.6. Three
# slots warn on contrast against the surface, which the direct labels and the
# table twin in cost_tables.md discharge.
COLOR = {"plain.decision": "#2a78d6", "plan_build": "#eb6834",
         "llm.generate": "#1baf7a", "struct.check": "#eda100",
         "mdo.decision": "#e87ba4", "verify": "#008300"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(profiles=PROFILES, cells=CELLS):
    """[{approach, level, seed, scenario, summary, totals, admitted, offered, ...}]"""
    recs = []
    for f in sorted(Path(profiles).glob("*.json")):
        p = json.loads(f.read_text())
        scenario, approach, seed, level, inst = f.stem.split("_")
        cell = Path(cells) / f.name
        c = json.loads(cell.read_text()) if cell.exists() else {}
        recs.append({
            "scenario": scenario, "approach": approach, "seed": int(seed),
            "level": level, "summary": p["summary"], "totals": p["cell_totals"],
            "cpu_energy_method": p["cpu_energy_method"],
            "foreign": p.get("gpu_foreign_pids", {}),
            "admitted": c.get("admitted"), "offered": c.get("offered"),
            "acceptance": c.get("acceptance"),
        })
    return recs


def per_decision(rec, stage, field="wall_s"):
    """Mean per-event value for one stage, or None if the stage never ran."""
    row = rec["summary"].get(stage)
    if not row or field not in row:
        return None
    return row[field]["mean"]


def decision_latency_s(rec):
    """End-to-end per-decision latency: the sum of the stages, per arrival.

    Summed from stage totals over the arrival count rather than averaged over
    stages, since not every stage fires on every arrival (a rejected plan never
    reaches routing).
    """
    n = rec["offered"]
    if not n:
        return None
    tot = 0.0
    for s in STAGES:
        row = rec["summary"].get(s)
        if row and "wall_s" in row:
            # mdo.forward / actor.place are nested inside mdo.decision; counting
            # both would double-charge the same wall time.
            if s in NESTED:
                continue
            tot += row["wall_s"]["sum"]
    return tot / n


def gpu_energy_per_decision(rec):
    """(J per model call, n_clean, n_total) over uncontaminated windows only."""
    row = rec["summary"].get(GPU_STAGE)
    if not row:
        return None, 0, 0
    n_total, n_clean = row["count"], row.get("n_clean", 0)
    clean = row.get("gpu_energy_j_clean")
    return (clean["mean"] if clean else None), n_clean, n_total


def cpu_energy_j(rec):
    """TDP-estimated CPU energy for the whole cell."""
    t = rec["totals"]
    return t.get("cpu_energy_j", t.get("cpu_energy_j_est"))


# ── tables ───────────────────────────────────────────────────────────────────
def table_cost(recs):
    out = ["| approach | n | decision ms | GPU J/call | calls clean/total | "
           "cell CPU s | cell CPU J (est) | J per admitted |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    by = defaultdict(list)
    for r in recs:
        by[r["approach"]].append(r)
    for a in sorted(by, key=lambda x: (GPU_STAGE not in
                                       (by[x][0]["summary"] or {}), x)):
        rs = by[a]
        lat = [x for x in (decision_latency_s(r) for r in rs) if x is not None]
        gj, nc, nt = zip(*(gpu_energy_per_decision(r) for r in rs))
        gj = [x for x in gj if x is not None]
        cpu_s = [r["totals"]["cpu_s"] for r in rs]
        cpu_j = [x for x in (cpu_energy_j(r) for r in rs) if x is not None]
        adm = sum(r["admitted"] or 0 for r in rs)
        # Energy per admitted slice: GPU where the model ran, plus estimated CPU.
        tot_gpu = sum(g * n for g, n in zip(gj, nc)) if gj else 0.0
        per_adm = (tot_gpu + sum(cpu_j)) / adm if adm else None
        out.append(
            f"| {a} | {len(rs)} | {np.mean(lat) * 1000:.2f} |"
            f" {f'{np.mean(gj):.1f}' if gj else '--'} |"
            f" {sum(nc)}/{sum(nt) if sum(nt) else 0} |"
            f" {np.mean(cpu_s):.1f} |"
            f" {f'{np.mean(cpu_j):.0f}' if cpu_j else '--'} |"
            f" {f'{per_adm:.2f}' if per_adm else '--'} |")
    return "\n".join(out)


def table_stages(recs):
    by = defaultdict(list)
    for r in recs:
        by[r["approach"]].append(r)
    out = ["| approach | stage | calls/arrival | ms p50 | ms p90 | ms p99 | "
           "share of decision |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for a in sorted(by):
        rs = by[a]
        total = np.mean([x for x in (decision_latency_s(r) for r in rs)
                         if x is not None])
        for s in STAGES:
            rows = [r["summary"][s] for r in rs if s in r["summary"]]
            if not rows:
                continue
            n_per = np.mean([w["count"] / r["offered"] for w, r in
                             zip(rows, rs) if r["offered"]])
            p50 = np.mean([w["wall_s"]["p50"] for w in rows]) * 1000
            p90 = np.mean([w["wall_s"]["p90"] for w in rows]) * 1000
            p99 = np.mean([w["wall_s"]["p99"] for w in rows]) * 1000
            share = np.mean([w["wall_s"]["sum"] / r["offered"] for w, r in
                             zip(rows, rs) if r["offered"]]) / total
            nested = " (nested)" if s in NESTED else ""
            out.append(f"| {a} | {s}{nested} | {n_per:.2f} | {p50:.3f} | "
                       f"{p90:.3f} | {p99:.3f} | "
                       f"{'--' if nested else f'{share * 100:.1f}%'} |")
    return "\n".join(out)


# ── figures ──────────────────────────────────────────────────────────────────
def fig_stage_stack(recs):
    """Where a decision's time goes, per approach.

    A dot plot, not a stacked bar. The stages span three orders of magnitude, so
    the axis has to be logarithmic, and on a log axis a stacked bar is simply
    wrong: segment lengths no longer add, and a bar encodes length from a zero
    the axis does not have. Position encodes the value instead, and the total is
    direct-labelled since it is the one number per row that has to be readable.
    """
    by = defaultdict(list)
    for r in recs:
        by[r["approach"]].append(r)
    approaches = sorted(by, key=lambda a: np.mean(
        [x for x in (decision_latency_s(r) for r in by[a]) if x is not None]))

    fig, ax = plt.subplots(figsize=(9.5, 0.66 * len(approaches) + 2.4))
    for i, a in enumerate(approaches):
        total = 0.0
        for s in STAGES:
            if s in NESTED:
                continue
            rows = [(r["summary"][s], r) for r in by[a] if s in r["summary"]]
            if not rows:
                continue
            ms = np.mean([w["wall_s"]["sum"] / r["offered"] for w, r in rows
                          if r["offered"]]) * 1000
            total += ms
            ax.plot([ms], [i], "o", ms=9, color=COLOR.get(s, MUTED),
                    mec="white", mew=1.6, label=s, zorder=3)
        # One label per row, at the row's total. A value on every dot would be
        # unreadable at this density and the axis already carries them.
        ax.text(1.02, i, f"{total:.1f} ms total", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.5, color=MUTED)

    ax.set_yticks(range(len(approaches)))
    ax.set_yticklabels(approaches, fontsize=9.5, color=INK)
    ax.set_ylim(-0.6, len(approaches) - 0.4)
    ax.set_xscale("log")
    ax.set_xlabel("mean wall time per arrival (ms, log scale)", fontsize=9.5,
                  color=MUTED)
    ax.set_title("Where a decision's time goes, by stage", fontsize=11,
                 color=INK, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    seen, h2, l2 = set(), [], []
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    ax.legend(h2, l2, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    return fig


def fig_energy(recs):
    """GPU energy per model call, clean windows only, and the drop count."""
    by = defaultdict(list)
    for r in recs:
        g, nc, nt = gpu_energy_per_decision(r)
        if g is not None:
            by[r["approach"]].append((g, nc, nt))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    if not by:
        ax.text(0.5, 0.5, "no GPU-attributable events\n(no approach made a "
                          "model call, or every window was contaminated)",
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return fig
    names = sorted(by)
    means = [np.mean([g for g, _, _ in by[a]]) for a in names]
    ax.bar(names, means, color="#b5341f", alpha=0.85, width=0.55)
    for i, a in enumerate(names):
        nc = sum(n for _, n, _ in by[a])
        nt = sum(n for _, _, n in by[a])
        ax.text(i, means[i], f"\n{nc}/{nt} clean", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylabel("GPU energy per model call (J)")
    ax.set_title("GPU energy per Agent B call, whole-card over uncontaminated "
                 "windows only", fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    return fig


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=PROFILES)
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    recs = load(args.profiles, args.cells)
    if not recs:
        raise SystemExit(f"no profile sidecars in {args.profiles}. Run "
                         "grid_runner with --profile first.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, fig in (("stage_breakdown", fig_stage_stack(recs)),
                      ("gpu_energy_per_call", fig_energy(recs))):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    method = {r["cpu_energy_method"] for r in recs}
    foreign = {p for r in recs for p in r["foreign"]}
    md = [
        "# §Y.15 per-decision cost",
        "",
        f"Cells profiled: {len(recs)}. CPU energy method: {', '.join(sorted(method))}.",
        "",
        "CPU energy is a **TDP estimate** (`cpu_s` x `MDO_CPU_WATT_PER_CORE`), not a",
        "measurement: RAPL is root-gated on this box. GPU energy is measured whole-card",
        "by NVML and attributed only to `llm.generate`, over windows where no foreign",
        "process used the card.",
        "",
    ]
    if foreign:
        md += [f"Foreign GPU processes seen during profiling: "
               f"`{sorted(foreign)}`. Their windows are excluded from the energy "
               "figures; the clean/total column says how many survived.", ""]
    md += ["## Per-decision cost", "", table_cost(recs), "",
           "## Stage breakdown", "",
           "`mdo.forward` and `actor.place` are nested inside `mdo.decision` and are",
           "shown for detail only; they are not added to the decision total.", "",
           table_stages(recs), ""]
    (out / "cost_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote 2 figures (png+pdf) and cost_tables.md to {out}")


if __name__ == "__main__":
    main()
