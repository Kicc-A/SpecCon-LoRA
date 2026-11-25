"""Plot final DPLoRA GLUE ranks from completed local runs; no training needed."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dplora-rank-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

OUT = Path(__file__).resolve().parent
COMPLETED = Path("/workspace/DPLoRA/experiments/glue_dp04_full_20260922/completed_runs.json")
TASKS = {"cola": "CoLA", "rte": "RTE"}
ALL_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]
MODULES = [
    ("attention.self.query", "Q"), ("attention.self.key", "K"),
    ("attention.self.value", "V"), ("attention.output.dense", "Attn-O"),
    ("intermediate.dense", "FFN-in"), ("output.dense", "FFN-out"),
]
COLORS = LinearSegmentedColormap.from_list(
    "rank_blues", ["#f5f7fa", "#d5e5f3", "#8cbddc", "#397bb1", "#153b68"]
)
NORM = Normalize(0, 16)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": "#192f49", "axes.labelcolor": "#192f49",
    "xtick.color": "#425b73", "ytick.color": "#425b73",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(path):
    data = read_json(path)
    keys = [f"roberta.encoder.layer.{layer}.{suffix}"
            for layer in range(12) for suffix, _ in MODULES]
    assert set(data) == set(keys), f"Incomplete/unexpected module map: {path}"
    assert all(type(data[k]) is int and 0 <= data[k] <= 16 for k in keys), path
    return np.array([data[k] for k in keys], dtype=int).reshape(12, 6)


def load_runs():
    groups = {task: [] for task in TASKS}
    rows, sources = [], []
    for name, record in read_json(COMPLETED).items():
        task = name.split("/")[0]
        if task not in groups:
            continue
        root = Path(record["output_dir"])
        final_path = root / "pruning/final_r_config.json"
        initial_path = root / "lora_r_config.json"
        pruning_path = root / "pruning/pruning_metrics.json"
        final, initial = matrix(final_path), matrix(initial_path)
        pruning = read_json(pruning_path)
        assert (root / "eval_metrics.json").is_file(), root
        assert (root / "train_metrics.json").is_file(), root
        assert np.all(final <= initial), root
        costs = np.array([1536, 1536, 1536, 1536, 3840, 3840])
        assert int((final * costs).sum()) == pruning["final_parameters"], root
        assert int((initial * costs).sum()) == pruning["initial_parameters"], root
        assert pruning["pruning_steps"] == 10, root
        assert abs(pruning["target_reduction_rate"] - 0.4) < 1e-9, root
        seed = record["seed"]
        groups[task].append({"seed": seed, "rank": final})
        source = {
            "task": task, "seed": seed, "run": str(root),
            "final_parameters": pruning["final_parameters"],
            "zero_rank_modules": int((final == 0).sum()),
            "final_rank_file": str(final_path), "final_rank_sha256": sha256(final_path),
            "initial_rank_file": str(initial_path), "initial_rank_sha256": sha256(initial_path),
            "pruning_metrics_file": str(pruning_path), "pruning_metrics_sha256": sha256(pruning_path),
        }
        sources.append(source)
        for layer in range(12):
            for col, (suffix, label) in enumerate(MODULES):
                rows.append({"task": task, "seed": seed, "layer": layer,
                             "module": label, "module_name": f"roberta.encoder.layer.{layer}.{suffix}",
                             "initial_rank": int(initial[layer, col]),
                             "final_rank": int(final[layer, col]), "source": str(final_path)})
    for task, runs in groups.items():
        runs.sort(key=lambda r: r["seed"])
        assert runs, task
        assert len({r["seed"] for r in runs}) == len(runs), task
    return groups, rows, sources


def heatmap(ax, values, title, is_mean=False):
    im = ax.imshow(values, cmap=COLORS, norm=NORM, aspect="auto")
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=12)
    ax.set_xticks(range(6), [label for _, label in MODULES], rotation=35, ha="right")
    ax.set_yticks(range(12))
    ax.set_ylabel("Encoder layer (0-based)")
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for layer in range(12):
        for col in range(6):
            value = values[layer, col]
            label = f"{value:.1f}" if is_mean else str(int(value))
            color = "white" if value >= 11 else "#192f49"
            ax.text(col, layer, label, ha="center", va="center", fontsize=9, color=color)
    return im


def compose(panels, title, subtitle):
    count = len(panels)
    width = 4.0 * count + 0.9
    fig = plt.figure(figsize=(width, 7.8))
    gs = fig.add_gridspec(1, count + 1, width_ratios=[1] * count + [0.055],
                          left=0.075, right=1 - 0.9 / width, bottom=0.19, top=0.78, wspace=0.33)
    for i, (values, label, is_mean) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        im = heatmap(ax, values, label, is_mean)
    cax = fig.add_subplot(gs[0, -1])
    fig.colorbar(im, cax=cax, ticks=[0, 4, 8, 12, 16], label="Final LoRA rank")
    fig.text(0.075, 0.945, title, fontsize=20, weight="bold")
    fig.text(0.075, 0.895, subtitle, fontsize=10.5, color="#536d85")
    fig.text(0.075, 0.078, "Final ranks after progressive pruning; shared color scale: 0–16. Zero means no LoRA rank remains.", fontsize=9)
    fig.text(0.075, 0.043, "Q/K/V: attention projections; Attn-O: attention output; FFN-in/out: feed-forward projections.", fontsize=9, color="#536d85")
    return fig


def main():
    groups, rows, sources = load_runs()
    for directory in ["png", "pdf", "svg", "data"]:
        (OUT / directory).mkdir(exist_ok=True)
    with (OUT / "data/ranks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for task, runs in groups.items():
        values = np.stack([r["rank"] for r in runs])
        for layer in range(12):
            for col, (_, label) in enumerate(MODULES):
                summaries.append({"task": task, "layer": layer, "module": label,
                                  "n_seeds": len(runs), "mean_final_rank": float(values[:, layer, col].mean()),
                                  "sample_sd_final_rank": float(values[:, layer, col].std(ddof=1))})
    with (OUT / "data/summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    artifacts = []
    with PdfPages(OUT / "DPLoRA_GLUE_final_rank_heatmaps.pdf") as bundle:
        def save(fig, stem):
            for fmt in ["png", "pdf", "svg"]:
                fig.savefig(OUT / fmt / f"{stem}.{fmt}", dpi=240)
            bundle.savefig(fig)
            plt.close(fig)
            artifacts.append(stem)

        overview = []
        for task, runs in groups.items():
            mean = np.stack([r["rank"] for r in runs]).mean(axis=0)
            seeds = ", ".join(str(r["seed"]) for r in runs)
            overview.append((mean, f"{TASKS[task]} · mean of {len(runs)} seeds\n{seeds}", True))
        save(compose(overview, "DPLoRA | Final rank allocation",
                     "RoBERTa-base · p=0.4 · available completed local GLUE runs · cell values are seed means"),
             "01_final_rank_mean_overview")
        for index, (task, runs) in enumerate(groups.items(), 2):
            panels = [(r["rank"], f"Seed {r['seed']}\n{int((r['rank'] == 0).sum())}/72 modules at rank 0", False) for r in runs]
            panels.append((np.stack([r["rank"] for r in runs]).mean(axis=0), f"Mean across {len(runs)} seeds", True))
            save(compose(panels, f"DPLoRA | {TASKS[task]}",
                         "RoBERTa-base · p=0.4 · individual panels show integer ranks; the last panel shows the seed mean"),
                 f"{index:02d}_{task}_final_rank_by_seed")

    manifest = {
        "method": "DPLoRA", "model": "RoBERTa-base", "pruning_target": 0.4,
        "rank_stage": "final_after_pruning", "color_range": [0, 16],
        "completed_registry": str(COMPLETED), "completed_registry_sha256": sha256(COMPLETED),
        "coverage": {task: [r["seed"] for r in runs] for task, runs in groups.items()},
        "unavailable_tasks": [task for task in ALL_TASKS if task not in groups],
        "excluded": "Preflight runs and unfinished/failed runs; no imputation of missing tasks or seeds.",
        "checks": ["Exact 12 x 6 module coverage", "Integer ranks in [0,16]", "Final ranks <= initial ranks",
                   "Rank-derived initial/final parameter counts match pruning_metrics.json", "10 pruning steps; target p=0.4"],
        "sources": sources, "figures": artifacts,
    }
    (OUT / "data/manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "runs": len(sources), "coverage": manifest["coverage"],
                      "figures": artifacts, "parameter_counts_verified": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
