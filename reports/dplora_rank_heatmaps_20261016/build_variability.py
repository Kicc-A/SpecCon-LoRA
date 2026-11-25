"""Recreate the reference rank-SD style using completed DPLoRA runs."""
import csv
import json

from build_figures import OUT, TASKS, MODULES, load_runs
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED = "#172D45", "#63748A"
STEM = "04_dplora_rank_seed_variability"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "axes.titleweight": "bold", "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": "#C5CFDB", "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
})


def main():
    groups, _, sources = load_runs()
    values = {
        task: np.stack([run["rank"] for run in runs]).std(axis=0, ddof=1)
        for task, runs in groups.items()
    }
    # The supplied SpecCon figure peaks below 7. DPLoRA exceeds 7, so expand
    # the shared color scale to preserve every value instead of clipping.
    upper = float(np.ceil(max(matrix.max() for matrix in values.values())))
    fig, axes = plt.subplots(1, len(groups), figsize=(11.5, 8.4), squeeze=False)
    fig.subplots_adjust(left=.075, right=.865, bottom=.22, top=.76, wspace=.38)
    rows = []
    for ax, (task, runs) in zip(axes.flat, groups.items()):
        matrix = values[task]
        im = ax.imshow(matrix, vmin=0, vmax=upper, cmap="magma_r", aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(6), [label for _, label in MODULES], rotation=30, ha="right")
        ax.set_yticks(range(12), range(12))
        ax.set_ylabel("Encoder layer (0-based)")
        ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.45, alpha=.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        seeds = ", ".join(str(r["seed"]) for r in runs)
        ax.set_title(f"{TASKS[task]} | p=0.4 | n={len(runs)}", loc="left", pad=25)
        ax.text(0, 1.018, f"Seeds: {seeds}", transform=ax.transAxes,
                fontsize=9, color=MUTED, va="bottom")
        for layer in range(12):
            for col, (_, module) in enumerate(MODULES):
                value = float(matrix[layer, col])
                ax.text(col, layer, f"{value:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if value / upper > .6 else INK)
                rows.append({"task": task, "seeds": seeds, "n_seeds": len(runs),
                             "layer": layer, "module": module, "sample_sd": value})
    fig.text(.075, .95, "Rank allocation variability across seeds", fontsize=22, weight="bold")
    fig.text(.075, .902, "DPLoRA | RoBERTa-base | final ranks after pruning", fontsize=12, color=MUTED)
    fig.text(.075, .864, "12 encoder layers x 6 adapted projections | same color scale for both tasks", fontsize=10, color=MUTED)
    cb = fig.colorbar(im, cax=fig.add_axes([.905, .31, .018, .39]), ticks=np.arange(0, upper + 1, 2))
    cb.set_label("Rank sample SD")
    for y, text in [
        (.115, "Cell values are sample SD across available seeds (ddof=1); higher values indicate greater seed variability."),
        (.087, "Ranks come from pruning/final_r_config.json. SD=0 means identical ranks, including ranks that are always zero."),
        (.059, "Only CoLA (3 seeds) and RTE (2 seeds) have completed local runs; the other six GLUE tasks are unavailable."),
        (.031, "Q/K/V/Attn-O are attention projections. FFN-in/out are feed-forward projections. Shared color scale: 0–12."),
    ]:
        fig.text(.075, y, text, fontsize=8.5, color=MUTED)
    for fmt in ["png", "pdf", "svg"]:
        (OUT / fmt).mkdir(exist_ok=True)
        fig.savefig(OUT / fmt / f"{STEM}.{fmt}", dpi=300)
    plt.close(fig)
    (OUT / "data").mkdir(exist_ok=True)
    with (OUT / "data/rank_seed_variability.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "statistic": "sample standard deviation", "ddof": 1,
        "stage": "final_after_pruning", "colormap": "magma_r", "color_scale": [0, upper],
        "reference": "/workspace/SpecCon-LoRA/reports/glue_best_42_777_2026_20261013/figures_20261014/png/04_rank_seed_variability.png",
        "tasks": {task: {"seeds": [r["seed"] for r in groups[task]],
                         "maximum_sd": float(matrix.max()), "zero_sd_modules": int((matrix == 0).sum())}
                  for task, matrix in values.items()},
        "sources": sources,
        "outputs": [f"{fmt}/{STEM}.{fmt}" for fmt in ["png", "pdf", "svg"]],
    }
    (OUT / "data/variability_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"outputs": manifest["outputs"], "color_scale": [0, upper], "tasks": manifest["tasks"]}, indent=2))


if __name__ == "__main__":
    main()
