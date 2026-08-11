"""§Y.15 acceptance figures and tables from the banked grid cells.

Reads data/grid_cells/*.json and writes figures + tables to results/y15_figures/.

Acceptance only. The cells carry no energy data and their `cost` block is a
placement-cost summary (demand, hops, bandwidth), not power. `wall_s` is per
cell, not per decision, and is dominated by the plan-cache miss count times the
LLM serving latency, plus the one-off M^B warm-up and the Y.14 selection pass,
so it is not an inference-cost metric and is deliberately not plotted as one.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELLS = Path("data/grid_cells")
OUT = Path("results/y15_figures")

# Code's own order and names (grid_runner.APPROACHES), heuristic then trained.
APPROACHES = ["Plain", "MDO-fullobs", "MDO-partial", "RL-alone", "RL-poprior",
              "Memory-off", "Full"]
LEVELS = ["L1", "L2", "L3", "L4"]
SEEDS = [42, 43, 44, 45, 46]

# MDO-fullobs is information privileged: it reads the full substrate while the
# partial-obs approaches see per-domain abstract state. Drawn dashed so it does
# not read as a peer.
CEILING = {"MDO-fullobs"}
COLOR = {"Plain": "#6b6b6b", "MDO-fullobs": "#b07d2b", "MDO-partial": "#2f7d32",
         "RL-alone": "#1f6fb2", "RL-poprior": "#7a4fa3", "Memory-off": "#c1554e",
         "Full": "#8c2f28"}


def load():
    """acc[(scenario, approach, level)][seed] -> acceptance.

    Returned as a plain dict: a defaultdict would materialise an empty entry on
    every miss, so the absent cells (RL-poprior ran conventional only, complex
    ran L2 only) would come back as present-but-empty and reach the mean.
    """
    acc: dict = defaultdict(dict)
    for f in CELLS.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("status") != "ok":
            continue
        acc[(d["scenario"], d["approach"], d["level"])][d["seed"]] = d["acceptance"]
    return dict(acc)


def series(acc, scenario, approach, level):
    cell = acc.get((scenario, approach, level), {})
    return [cell[s] for s in SEEDS if s in cell]


def present(acc, scenario, approach):
    return any(acc.get((scenario, approach, l)) for l in LEVELS)


# ── tables ───────────────────────────────────────────────────────────────────
def table_markdown(acc, scenario, levels):
    """Mean +- sd over seeds, one row per approach."""
    rows = [a for a in APPROACHES if present(acc, scenario, a)]
    head = "| approach | " + " | ".join(levels) + " |"
    rule = "| " + " | ".join(["---"] + ["---:"] * len(levels)) + " |"
    out = [head, rule]
    for a in rows:
        cells = []
        for l in levels:
            xs = series(acc, scenario, a, l)
            cells.append("--" if not xs
                         else f"{np.mean(xs):.3f} ± {np.std(xs, ddof=1):.3f}"
                         if len(xs) > 1 else f"{np.mean(xs):.3f}")
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_csv(acc, path):
    lines = ["scenario,approach,level,n_seeds,mean,sd,min,max"]
    for (sc, ap, lv), d in sorted(acc.items()):
        xs = list(d.values())
        sd = f"{np.std(xs, ddof=1):.4f}" if len(xs) > 1 else ""
        lines.append(f"{sc},{ap},{lv},{len(xs)},{np.mean(xs):.4f},{sd},"
                     f"{min(xs):.4f},{max(xs):.4f}")
    path.write_text("\n".join(lines) + "\n")


def table_paired(acc, scenario, ref, levels):
    """Paired within-seed difference against a reference approach.

    Paired because one llm_prior checkpoint sequence per (scenario, seed) serves
    Prior-only, Memory-off and Full, so the seed is the unit of independence and
    an unpaired sd across cells would overstate the spread.
    """
    rows = [a for a in APPROACHES if a != ref and present(acc, scenario, a)]
    out = [f"| approach vs {ref} | " + " | ".join(levels) + " |",
           "| " + " | ".join(["---"] + ["---:"] * len(levels)) + " |"]
    for a in rows:
        cells = []
        for l in levels:
            arm = acc.get((scenario, a, l), {})
            base = acc.get((scenario, ref, l), {})
            d = [arm[s] - base[s] for s in SEEDS if s in arm and s in base]
            cells.append("--" if not d else
                         f"{np.mean(d) * 100:+.2f} ± {np.std(d, ddof=1) * 100:.2f}"
                         if len(d) > 1 else f"{np.mean(d) * 100:+.2f}")
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# ── figures ──────────────────────────────────────────────────────────────────
def fig_lines(acc):
    """Acceptance against offered load, one panel per scenario."""
    scen = [("conventional", LEVELS), ("complex", ["L2"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": [4, 1.15]})
    for ax, (sc, levels) in zip(axes, scen):
        for a in APPROACHES:
            if not present(acc, sc, a):
                continue
            xs, ys, es = [], [], []
            for i, l in enumerate(levels):
                v = series(acc, sc, a, l)
                if v:
                    xs.append(i)
                    ys.append(np.mean(v))
                    es.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=3, lw=1.6,
                        color=COLOR[a], label=a,
                        ls="--" if a in CEILING else "-")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_title(sc)
        ax.set_xlabel("load level")
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("acceptance ratio")
    # Legend off the conventional panel: complex ran L2 only and never ran
    # RL-poprior, so a legend built from that axis silently drops a series.
    axes[1].legend(*axes[0].get_legend_handles_labels(), fontsize=7.5,
                   loc="upper right", framealpha=0.95)
    fig.suptitle("Acceptance against offered load (mean ± sd over 5 seeds, "
                 "held-out instance 100)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_box(acc):
    """Seed spread per approach, one panel per level."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
    for ax, l in zip(axes, LEVELS):
        rows = [a for a in APPROACHES if series(acc, "conventional", a, l)]
        data = [series(acc, "conventional", a, l) for a in rows]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops={"color": "black", "lw": 1.3},
                        flierprops={"marker": ".", "ms": 4})
        for patch, a in zip(bp["boxes"], rows):
            patch.set_facecolor(COLOR[a])
            patch.set_alpha(0.45 if a in CEILING else 0.8)
            patch.set_edgecolor(COLOR[a])
        for i, (a, xs) in enumerate(zip(rows, data), start=1):
            ax.scatter(np.full(len(xs), i), xs, s=9, color="black", zorder=3,
                       alpha=0.65)
        ax.set_xticks(range(1, len(rows) + 1))
        ax.set_xticklabels(rows, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(l)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
    axes[0].set_ylabel("acceptance ratio")
    fig.suptitle("Seed spread by approach, conventional scenario "
                 "(5 seeds, points overlaid)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_paired(acc, ref="MDO-partial"):
    """Paired within-seed difference against the deployable reference."""
    rows = [a for a in APPROACHES
            if a != ref and a not in CEILING and present(acc, "conventional", a)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(rows)
    for j, a in enumerate(rows):
        means, sds = [], []
        for l in LEVELS:
            arm = acc.get(("conventional", a, l), {})
            base = acc.get(("conventional", ref, l), {})
            d = [arm[s] - base[s] for s in SEEDS if s in arm and s in base]
            means.append(np.mean(d) * 100 if d else np.nan)
            sds.append(np.std(d, ddof=1) * 100 if len(d) > 1 else 0.0)
        pos = np.arange(len(LEVELS)) + (j - (len(rows) - 1) / 2) * width
        ax.bar(pos, means, width=width * 0.92, yerr=sds, capsize=2.5,
               color=COLOR[a], label=a, alpha=0.88,
               error_kw={"lw": 0.9})
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(LEVELS)
    ax.set_xlabel("load level")
    ax.set_ylabel(f"acceptance minus {ref} (percentage points)")
    ax.set_title(f"Paired within-seed difference against {ref}, conventional "
                 "(error bars are paired sd)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    return fig


def fig_scenario(acc):
    """Conventional against complex at the one level both were run at."""
    rows = [a for a in APPROACHES if series(acc, "complex", a, "L2")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(rows))
    for off, sc in ((-0.19, "conventional"), (0.19, "complex")):
        m = [np.mean(series(acc, sc, a, "L2")) for a in rows]
        e = [np.std(series(acc, sc, a, "L2"), ddof=1) for a in rows]
        ax.bar(x + off, m, 0.36, yerr=e, capsize=3, label=sc,
               alpha=0.85, error_kw={"lw": 0.9})
    ax.set_xticks(x)
    ax.set_xticklabels(rows, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("acceptance ratio")
    ax.set_title("L2 by scenario. The complex mix asks ~40% more CPU per slice, "
                 "so its L2 is 0.94 of capacity, not 0.67", fontsize=9.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    return fig


def main():
    acc = load()
    OUT.mkdir(parents=True, exist_ok=True)

    for name, fig in (("acceptance_vs_load", fig_lines(acc)),
                      ("seed_spread_box", fig_box(acc)),
                      ("paired_vs_mdo_partial", fig_paired(acc)),
                      ("scenario_l2", fig_scenario(acc))):
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    table_csv(acc, OUT / "acceptance_all_cells.csv")
    md = [
        "# §Y.15 acceptance tables",
        "",
        "Mean ± sd over 5 seeds (42 to 46), held-out instance 100.",
        "`MDO-fullobs` is information privileged and reads as a ceiling, not a peer.",
        "",
        "## conventional",
        "",
        table_markdown(acc, "conventional", LEVELS),
        "",
        "## complex",
        "",
        table_markdown(acc, "complex", ["L2"]),
        "",
        "## Paired within-seed difference, conventional (percentage points)",
        "",
        "Paired because one `llm_prior` checkpoint sequence per (scenario, seed)",
        "serves Prior-only, Memory-off and Full, so the seed is the unit of",
        "independence and an unpaired spread would overstate it.",
        "",
        table_paired(acc, "conventional", "MDO-partial", LEVELS),
        "",
        table_paired(acc, "conventional", "Plain", LEVELS),
        "",
    ]
    (OUT / "acceptance_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote 4 figures (png+pdf), acceptance_all_cells.csv and "
          f"acceptance_tables.md to {OUT}")


if __name__ == "__main__":
    main()
