#!/usr/bin/env python3
"""P0-A: historical 2x2 ablation interaction analysis (zero training).

Recovers per-seed Energy/Consensus x Random/SVD cells (C/D/E/F) from
glue_main_ablation_3seeds_gpu12_20261016_v3, computes per-seed interaction
I_s = (F-D)_s - (E-C)_s, verifies the MRPC +1.634+-0.283 replication value,
and writes INTERACTION_2x2.csv, a paired bar chart, and module-level
rank-map difference heatmaps (all from the frozen seed42 allocations).
"""
import csv
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

EXP_ROOT = "/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1"
ABL_ROOT = "/workspace/SpecCon-LoRA/experiments/glue_main_ablation_3seeds_gpu12_20261016_v3"
RANK_CSV = "/workspace/SpecCon-LoRA/experiments/chapter4_core_ablation_v1/rank_allocation_summary.csv"
OUT_DIR = os.path.join(EXP_ROOT, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

TASKS = ["cola", "mrpc", "rte"]
SEEDS = ["42", "777", "2026"]
VARIANTS = ["C", "D", "E", "F"]
VARIANT_LABEL = {"C": "Energy+Random", "D": "Energy+SVD", "E": "Consensus+Random", "F": "Consensus+SVD"}
TASK_LABEL = {"cola": "CoLA (MCC)", "mrpc": "MRPC (Acc)", "rte": "RTE (Acc)"}
MODULES = ["Q", "K", "V", "Attn Out", "FFN In", "FFN Out"]

# Reference palette (dataviz skill, light mode)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIV_CMAP = LinearSegmentedColormap.from_list("div_blue_red", ["#2a78d6", "#f0efec", "#e34948"])

# Values claimed by the chapter-4 report, for replication check (tolerance 0.002).
REPORTED = {
    "F_minus_D": {"cola": 1.632, "mrpc": 1.716, "rte": 1.805},
    "E_minus_C": {"cola": -0.799, "mrpc": 0.081, "rte": 1.083},
    "I": {"cola": 2.431, "mrpc": 1.635, "rte": 0.722},
    "mrpc_I_sd": 0.283,
}


def load_scores():
    """Return {(task, seed): {variant: unrounded score}} from metrics.json.

    MRPC uses final_accuracy_percent (accuracy is the reported metric; the
    runner's combined score averages accuracy and F1). CoLA (MCC) and RTE
    (accuracy) use final_score_percent.
    """
    cells = {}
    with open(os.path.join(ABL_ROOT, "results.csv")) as f:
        for row in csv.DictReader(f):
            if row["variant"] not in VARIANTS:
                continue
            with open(row["metrics"]) as mf:
                m = json.load(mf)
            field = "final_accuracy_percent" if row["task"] == "mrpc" else "final_score_percent"
            cells.setdefault((row["task"], row["seed"]), {})[row["variant"]] = m[field]
    return cells


def build_table(cells):
    """Per-seed cells, paired deltas, means and sample SDs."""
    out_rows = []
    summary = {}
    for task in TASKS:
        per_seed = {}
        for seed in SEEDS:
            c, d, e, f = (cells[(task, seed)][v] for v in VARIANTS)
            ec = e - c  # consensus gain under random init
            fd = f - d  # consensus gain under SVD init
            per_seed[seed] = dict(C=c, D=d, E=e, F=f, E_minus_C=ec, F_minus_D=fd, I=fd - ec,
                                  D_minus_C=d - c, F_minus_E=f - e)
            out_rows.append(dict(task=task, seed=seed, **per_seed[seed]))
        summary[task] = {
            key: (statistics.mean([per_seed[s][key] for s in SEEDS]),
                  statistics.stdev([per_seed[s][key] for s in SEEDS]) if len(SEEDS) > 1 else float("nan"))
            for key in ("E_minus_C", "F_minus_D", "I", "D_minus_C", "F_minus_E")
        }
    return out_rows, summary


def check_replication(summary):
    results = []
    for task in TASKS:
        for key, rep in (("F_minus_D", "F_minus_D"), ("E_minus_C", "E_minus_C"), ("I", "I")):
            mean, _ = summary[task][key]
            ok = abs(mean - REPORTED[rep][task]) <= 0.002
            results.append((task, key, mean, REPORTED[rep][task], ok))
    mi_mean, mi_sd = summary["mrpc"]["I"]
    mrpc_ok = abs(mi_mean - 1.634) <= 0.002 and abs(mi_sd - REPORTED["mrpc_I_sd"]) <= 0.002
    results.append(("mrpc", "I mean/SD vs 1.634+-0.283", mi_mean, 1.634, mrpc_ok))
    return results, mrpc_ok


def write_csv(rows, summary):
    path = os.path.join(OUT_DIR, "INTERACTION_2x2.csv")
    header = ["task", "seed", "C_energy_random", "D_energy_svd", "E_consensus_random", "F_consensus_svd",
              "E_minus_C", "F_minus_D", "I_interaction", "D_minus_C", "F_minus_E"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r["task"], r["seed"], r["C"], r["D"], r["E"], r["F"],
                        r["E_minus_C"], r["F_minus_D"], r["I"], r["D_minus_C"], r["F_minus_E"]])
        w.writerow([])
        w.writerow(["task", "stat", "E_minus_C_mean", "E_minus_C_sd", "F_minus_D_mean", "F_minus_D_sd",
                    "I_mean", "I_sd"])
        for task in TASKS:
            s = summary[task]
            w.writerow([task, "mean+-sampleSD", s["E_minus_C"][0], s["E_minus_C"][1],
                        s["F_minus_D"][0], s["F_minus_D"][1], s["I"][0], s["I"][1]])
    print("wrote", path)


def paired_bar_chart(rows, summary):
    """Three panels, one per task: per-seed E-C (blue) vs F-D (orange) bars + mean+-SD note."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharey=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, task in zip(axes, TASKS):
        ax.set_facecolor(SURFACE)
        per = {r["seed"]: r for r in rows if r["task"] == task}
        x = [0, 1, 2]
        w = 0.38
        ec = [per[s]["E_minus_C"] for s in SEEDS]
        fd = [per[s]["F_minus_D"] for s in SEEDS]
        b1 = ax.bar([i - w / 2 - 0.01 for i in x], ec, width=w, color=BLUE, label="Consensus gain, Random init (E−C)")
        b2 = ax.bar([i + w / 2 + 0.01 for i in x], fd, width=w, color=ORANGE, label="Consensus gain, SVD init (F−D)")
        for bars in (b1, b2):
            for b in bars:
                val = b.get_height()
                if val == 0:
                    continue
                ax.annotate(f"{val:+.2f}", (b.get_x() + b.get_width() / 2, val),
                            xytext=(0, 3 if val > 0 else -3), textcoords="offset points",
                            ha="center", va="bottom" if val > 0 else "top",
                            fontsize=7.5, color=INK)
        i_mean, i_sd = summary[task]["I"]
        f_mean, f_sd = summary[task]["F_minus_D"]
        ax.set_title(f"{TASK_LABEL[task]}\nF−D = {f_mean:+.2f} ± {f_sd:.2f};  I = {i_mean:+.2f} ± {i_sd:.2f}",
                     fontsize=9, color=INK, pad=6)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=8, color=INK2)
        ax.tick_params(axis="y", labelsize=8, colors=INK2, length=2)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS)
        ymax = max(0.6, max(abs(v) for v in ec + fd) * 1.35)
        ax.set_ylim(-ymax, ymax)
        ax.set_ylabel("paired difference (points)", fontsize=8, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["Consensus gain, Random init (E−C)", "Consensus gain, SVD init (F−D)"],
               loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"interaction_2x2_paired_bars.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote paired bar chart")


def rank_map_diff():
    """Module-level consensus-minus-energy rank differences from the frozen seed42 maps."""
    rows = list(csv.DictReader(open(RANK_CSV)))
    rank = {(r["task"], r["method"], int(r["layer"]), r["module"]): int(r["rank"]) for r in rows}
    out = []
    for task in TASKS:
        for layer in range(12):
            for mod in MODULES:
                e = rank[(task, "energy", layer, mod)]
                c = rank[(task, "consensus", layer, mod)]
                out.append(dict(task=task, layer=layer, module=mod, energy=e, consensus=c, diff=c - e))
    path = os.path.join(OUT_DIR, "rank_map_difference.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "layer", "module", "energy", "consensus", "diff"])
        w.writeheader()
        w.writerows(out)
    print("wrote", path)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, task in zip(axes, TASKS):
        mat = [[next(r["diff"] for r in out if r["task"] == task and r["layer"] == l and r["module"] == m)
                for m in MODULES] for l in range(12)]
        vmax = max(abs(v) for row in mat for v in row) or 1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(mat, cmap=DIV_CMAP, norm=norm, aspect="auto")
        ax.set_xticks(range(len(MODULES)))
        ax.set_xticklabels(MODULES, fontsize=8, color=INK2, rotation=30, ha="right")
        ax.set_yticks(range(12))
        ax.set_yticklabels([f"L{l}" for l in range(12)], fontsize=7.5, color=INK2)
        ax.set_title(f"{TASK_LABEL[task]}  Δrank (Consensus − Energy)", fontsize=9, color=INK, pad=8)
        for i in range(12):
            for j in range(len(MODULES)):
                v = mat[i][j]
                if v != 0:
                    rgba = DIV_CMAP(norm(v))
                    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    ax.text(j, i, f"{v:+d}", ha="center", va="center", fontsize=7.5,
                            color=INK if lum > 0.55 else "#ffffff")
        ax.tick_params(length=0)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_color(AXIS)
    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    cbar.ax.tick_params(labelsize=7.5, colors=INK2, length=0)
    cbar.set_label("rank difference (consensus − energy)", fontsize=8, color=INK2)
    fig.suptitle("Module-level rank-map differences, frozen seed42 allocations, 755,712 parameters",
                 fontsize=10, color=INK)
    fig.subplots_adjust(left=0.06, right=0.93, top=0.88, bottom=0.10, wspace=0.32)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"rank_map_difference_heatmaps.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote rank-map difference heatmaps")


def main():
    cells = load_scores()
    assert all(len(cells[(t, s)]) == 4 for t in TASKS for s in SEEDS), "missing 2x2 cells"
    rows, summary = build_table(cells)
    checks, mrpc_ok = check_replication(summary)
    print("replication checks (task, contrast, computed, reported, ok):")
    for c in checks:
        print(" ", c)
    print("MRPC I = 1.634 +- 0.283 replication:", "PASS" if mrpc_ok else "FAIL")
    write_csv(rows, summary)
    paired_bar_chart(rows, summary)
    rank_map_diff()


if __name__ == "__main__":
    main()
