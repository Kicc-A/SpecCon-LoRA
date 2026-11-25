#!/usr/bin/env python3
"""P0-C: grouped mechanism figures (zero training).

Builds rank_maps.pdf from the frozen seed42 allocations in
chapter4_core_ablation_v1/rank_allocation_summary.csv (3 tasks x 2 methods x
72 modules, exactly 755,712 parameters). Three pages:
  1. Energy-only / Consensus rank maps + difference per task (3x3 grid).
  2. Attention vs FFN budget distribution per task (grouped bars).
  3. Layer-level capacity migration (signed bars, who pays for the gain).
Figures describe the allocation mechanism; they are not per-module causal evidence.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

EXP_ROOT = "/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1"
RANK_CSV = "/workspace/SpecCon-LoRA/experiments/chapter4_core_ablation_v1/rank_allocation_summary.csv"
OUT_DIR = os.path.join(EXP_ROOT, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

TASKS = ["cola", "mrpc", "rte"]
TASK_LABEL = {"cola": "CoLA", "mrpc": "MRPC", "rte": "RTE"}
MODULES = ["Q", "K", "V", "Attn Out", "FFN In", "FFN Out"]
ATTN_MODS = {"Q", "K", "V", "Attn Out"}
COST = {"Q": 1536, "K": 1536, "V": 1536, "Attn Out": 1536, "FFN In": 3840, "FFN Out": 3840}

# Reference palette (dataviz skill, light mode)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RED = "#e34948"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div_blue_red", ["#2a78d6", "#f0efec", "#e34948"])


def load():
    ranks = {}
    for r in csv.DictReader(open(RANK_CSV)):
        ranks[(r["task"], r["method"], int(r["layer"]), r["module"])] = int(r["rank"])
    return ranks


def page1_rank_maps(pdf, ranks):
    """3x3: rows = tasks, cols = Energy map / Consensus map / difference."""
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 7.6))
    fig.patch.set_facecolor(SURFACE)
    for ti, task in enumerate(TASKS):
        e = [[ranks[(task, "energy", l, m)] for m in MODULES] for l in range(12)]
        c = [[ranks[(task, "consensus", l, m)] for m in MODULES] for l in range(12)]
        d = [[c[l][j] - e[l][j] for j in range(len(MODULES))] for l in range(12)]
        vmin, vmax = min(min(x) for x in e + c), max(max(x) for x in e + c)
        specs = [(e, "Energy-only ranks", SEQ_CMAP, (vmin, vmax), None),
                 (c, "Consensus ranks", SEQ_CMAP, (vmin, vmax), None),
                 (d, "Δ rank (Consensus − Energy)", DIV_CMAP, None,
                  TwoSlopeNorm(vmin=-max(abs(v) for row in d for v in row),
                               vcenter=0, vmax=max(abs(v) for row in d for v in row)))]
        for j, (mat, title, cmap, vr, norm) in enumerate(specs):
            ax = axes[ti, j]
            ax.set_facecolor(SURFACE)
            im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vr[0] if vr else None,
                           vmax=vr[1] if vr else None, norm=norm)
            if j == 0:
                ax.set_ylabel(TASK_LABEL[task], fontsize=9, color=INK)
            if ti == 0:
                ax.set_title(title, fontsize=9, color=INK, pad=6)
            ax.set_xticks(range(len(MODULES)))
            ax.set_xticklabels(MODULES if ti == 2 else [], fontsize=7.5, color=INK2,
                               rotation=30, ha="right")
            ax.set_yticks(range(12))
            ax.set_yticklabels([f"L{l}" for l in range(12)], fontsize=7, color=INK2)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_color(AXIS)
            if title.startswith("Δ"):
                for i in range(12):
                    for k in range(len(MODULES)):
                        v = mat[i][k]
                        if v != 0:
                            rgba = DIV_CMAP(norm(v))
                            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                            ax.text(k, i, f"{v:+d}", ha="center", va="center", fontsize=7,
                                    color=INK if lum > 0.55 else "#ffffff")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("Seed42 rank allocations: 72 target matrices, 755,712 parameters each",
                 fontsize=11, color=INK)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.93, bottom=0.07, wspace=0.45, hspace=0.30)
    pdf.savefig(fig)
    fig.savefig(os.path.join(OUT_DIR, "rank_maps_p1.png"), dpi=150)
    plt.close(fig)
    print("page 1 done")


def page2_budget(pdf, ranks):
    """Grouped bars: attention vs FFN parameter share, energy vs consensus, per task."""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, task in zip(axes, TASKS):
        ax.set_facecolor(SURFACE)
        groups = ["Attention", "FFN"]
        vals = {"energy": [], "consensus": []}
        for method in ("energy", "consensus"):
            for grp in groups:
                mods = ATTN_MODS if grp == "Attention" else {"FFN In", "FFN Out"}
                total = sum(ranks[(task, method, l, m)] * COST[m]
                            for l in range(12) for m in mods)
                vals[method].append(total / 1000)
        x = [0, 1]
        w = 0.38
        ax.bar([i - w / 2 - 0.01 for i in x], vals["energy"], width=w, color=BLUE,
               label="Energy-only")
        ax.bar([i + w / 2 + 0.01 for i in x], vals["consensus"], width=w, color=ORANGE,
               label="Consensus")
        for i, grp in enumerate(groups):
            e, c = vals["energy"][i], vals["consensus"][i]
            for xx, v, col in ((i - w / 2 - 0.01, e, INK2), (i + w / 2 + 0.01, c, INK)):
                ax.annotate(f"{v:.0f}k", (xx, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8, color=col)
        d = (vals["consensus"][1] - vals["energy"][1]) - (vals["consensus"][0] - vals["energy"][0])
        ax.set_title(f"{TASK_LABEL[task]}\nnet transfer: Attention → FFN = {d:+.1f}k" if d >= 0
                     else f"{TASK_LABEL[task]}\nnet transfer: FFN → Attention = {-d:+.1f}k",
                     fontsize=9, color=INK, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9, color=INK2)
        ax.tick_params(axis="y", labelsize=8, colors=INK2, length=2)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS)
        ax.set_ylim(0, max(vals["energy"] + vals["consensus"]) * 1.22)
        ax.set_ylabel("adapter parameters (x1000)", fontsize=8, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["Energy-only", "Consensus"], loc="lower center", ncol=2,
               fontsize=8, frameon=False)
    fig.suptitle("Attention vs FFN budget share (755,712 parameters total)",
                 fontsize=10, color=INK)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.16, wspace=0.30)
    pdf.savefig(fig)
    fig.savefig(os.path.join(OUT_DIR, "rank_maps_p2.png"), dpi=150)
    plt.close(fig)
    print("page 2 done")


def page3_migration(pdf, ranks):
    """Signed bars: per-layer net parameter change (consensus minus energy)."""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, task in zip(axes, TASKS):
        ax.set_facecolor(SURFACE)
        net = []
        for l in range(12):
            delta = sum((ranks[(task, "consensus", l, m)] - ranks[(task, "energy", l, m)]) * COST[m]
                        for m in MODULES)
            net.append(delta)
        colors = [BLUE if v < 0 else RED for v in net]
        bars = ax.bar(range(12), [v / 1000 for v in net], width=0.62, color=colors)
        for i, v in enumerate(net):
            if v != 0:
                ax.annotate(f"{v / 1000:+.1f}k", (i, v / 1000),
                            xytext=(0, 3 if v > 0 else -3), textcoords="offset points",
                            ha="center", va="bottom" if v > 0 else "top", fontsize=6.5,
                            color=INK2)
        ax.axhline(0, color=AXIS, lw=1)
        pos = sum(max(v, 0) for v in net)
        neg = sum(min(v, 0) for v in net)
        ax.set_title(f"{TASK_LABEL[task]}\nlayers gain {pos / 1000:+.1f}k / lose {neg / 1000:.1f}k",
                     fontsize=9, color=INK, pad=6)
        ax.set_xticks(range(12))
        ax.set_xticklabels([f"L{l}" for l in range(12)], fontsize=7, color=INK2)
        ax.tick_params(axis="y", labelsize=8, colors=INK2, length=2)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS)
        ax.set_ylabel("net parameter change (x1000)", fontsize=8, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=RED)]
    fig.legend(handles, ["capacity released", "capacity added"], loc="lower center", ncol=2,
               fontsize=8, frameon=False)
    fig.suptitle("Layer capacity migration, Consensus minus Energy (who pays for the gain)",
                 fontsize=10, color=INK)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.16, wspace=0.30)
    pdf.savefig(fig)
    fig.savefig(os.path.join(OUT_DIR, "rank_maps_p3.png"), dpi=150)
    plt.close(fig)
    print("page 3 done")


def main():
    ranks = load()
    for task in TASKS:
        for method in ("energy", "consensus"):
            total = sum(ranks[(task, method, l, m)] * COST[m]
                        for l in range(12) for m in MODULES)
            assert total == 755712, f"{task}/{method} budget {total}"
    with PdfPages(os.path.join(OUT_DIR, "rank_maps.pdf")) as pdf:
        page1_rank_maps(pdf, ranks)
        page2_budget(pdf, ranks)
        page3_migration(pdf, ranks)
    print("wrote rank_maps.pdf")


if __name__ == "__main__":
    main()
