#!/usr/bin/env python3
"""Rebuild the selected GLUE experiment figures from recorded artifacts."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
REPORT = OUT.parent
ROOT = REPORT.parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/speccon_glue_figure_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np

TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]
LABELS = dict(zip(TASKS, ["CoLA", "MNLI", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "STS-B"]))
SEEDS = [42, 777, 2026]
COLORS = {42: "#1677B8", 777: "#E88923", 2026: "#269578"}
INK, MUTED, GRID, PAPER = "#172D45", "#63748A", "#E5EAF0", "#B65465"
MODULES = ["attention.self.query", "attention.self.key", "attention.self.value",
           "attention.output.dense", "intermediate.dense", "output.dense"]
MODULE_LABELS = ["Q", "K", "V", "Attn-O", "FFN-in", "FFN-out"]
# RoBERTa-base: attention weights 768 x 768; FFN weights 768 x 3072.
COST_PER_RANK = np.array([1536, 1536, 1536, 1536, 3840, 3840])
METRICS = {t: "Accuracy (%)" for t in TASKS}
METRICS.update(cola="MCC x 100", stsb="Pearson x 100", mnli="Mean m/mm accuracy (%)")
STYLES = dict(zip(SEEDS, ["-", "--", "-."]))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "axes.titleweight": "bold", "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": "#C5CFDB", "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": .8, "grid.linewidth": .6,
    "axes.axisbelow": True, "lines.linewidth": 1.7, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
for directory in ["png", "pdf", "svg", "thumbnails", "data"]:
    (OUT / directory).mkdir(exist_ok=True)

def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def write_csv(name, rows):
    with (OUT / "data" / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def score(task, h):
    splits = h["splits"]
    if task == "mnli":
        return 50 * (splits["matched"]["accuracy"] + splits["mismatched"]["accuracy"])
    metric = {"cola": "matthews_correlation", "stsb": "pearson"}.get(task, "accuracy")
    return 100 * splits["validation"][metric]

def valid_loss(h):
    # MNLI validation loss is example-weighted; its score is the unweighted m/mm mean.
    values = list(h["splits"].values())
    return sum(x["loss"] * x["examples"] for x in values) / sum(x["examples"] for x in values)

def extract():
    report = json.loads((REPORT / "results.json").read_text())
    runs, curves, ranks, diagnostics, provenance, summary, split_rows = {}, [], [], [], [], [], []
    max_score_error, max_objective_error = 0., 0.
    for task in TASKS:
        t = report["tasks"][task]
        runs[task] = {}
        assert sorted(x["seed"] for x in t["runs"]) == sorted(SEEDS)
        for selected in sorted(t["runs"], key=lambda x: SEEDS.index(x["seed"])):
            seed = selected["seed"]
            source = ROOT / selected["source"]
            history = [json.loads(x) for x in (source / "history.jsonl").read_text().splitlines() if x.strip()]
            allocation = json.loads((source / "allocation.json").read_text())
            metrics = json.loads((source / "metrics.json").read_text())
            assert metrics["state"] == "complete", (task, seed, metrics["state"])
            assert len(history) == t["hyperparameters"]["epochs"] == metrics["epochs"]
            assert [h["epoch"] for h in history] == list(range(1, len(history) + 1))
            assert len(allocation["rank_pattern"]) == 72
            matrix = np.full((12, 6), np.nan)
            for name, rank in allocation["rank_pattern"].items():
                layer, module = re.fullmatch(r"roberta\.encoder\.layer\.(\d+)\.(.+)", name).groups()
                row, col = int(layer), MODULES.index(module)
                assert 2 <= rank <= 16 and int(rank) == rank
                assert np.isnan(matrix[row, col])
                matrix[row, col] = rank
                ranks.append(dict(task=task, seed=seed, layer=row, module=MODULE_LABELS[col],
                                  module_path=name, rank=rank, cost_per_rank=int(COST_PER_RANK[col]),
                                  adapter_parameters=int(rank * COST_PER_RANK[col]), source=str(source)))
            assert np.isfinite(matrix).all()
            parameter_matrix = matrix * COST_PER_RANK
            assert int(parameter_matrix.sum()) == selected["adapter_parameters"] == 755712
            assert int(parameter_matrix.sum()) == allocation["rank_report"]["budget_used"]
            rows = []
            for h in history:
                supervised = h.get("train_supervised_loss", h.get("train_ce_loss"))
                consistency = h.get("train_consistency_loss", h.get("train_kl_loss"))
                assert supervised is not None and consistency is not None
                objective_error = abs(h["train_loss"] - supervised - h["rdrop_alpha"] * consistency)
                max_objective_error = max(max_objective_error, objective_error)
                assert objective_error < 2e-6, (task, seed, h["epoch"], objective_error)
                row = dict(task=task, seed=seed, epoch=h["epoch"], optimizer_step=h["step"],
                           train_total_loss=h["train_loss"], train_supervised_loss=supervised,
                           train_consistency_loss=consistency, rdrop_alpha=h["rdrop_alpha"],
                           validation_loss=valid_loss(h), score=score(task, h), metric=METRICS[task])
                assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
                rows.append(row)
                curves.append(row)
                for split, values in h["splits"].items():
                    for metric, value in values.items():
                        split_rows.append(dict(task=task, seed=seed, epoch=h["epoch"], split=split,
                                               metric=metric, value=value))
            score_error = abs(rows[-1]["score"] - selected["score"])
            max_score_error = max(max_score_error, score_error)
            assert score_error < 1e-9, (task, seed, score_error)
            arrays = {k: np.array([h[k] for h in rows]) for k in [
                "epoch", "train_total_loss", "train_supervised_loss", "train_consistency_loss", "validation_loss", "score"]}
            best_i, min_i = int(arrays["score"].argmax()), int(arrays["validation_loss"].argmin())
            diag = dict(task=task, seed=seed, epochs=len(history), metric=METRICS[task],
                        final_score=rows[-1]["score"], peak_score=float(arrays["score"][best_i]),
                        peak_epoch=best_i + 1, peak_minus_final=float(arrays["score"][best_i] - rows[-1]["score"]),
                        min_validation_loss=float(arrays["validation_loss"][min_i]), min_loss_epoch=min_i + 1,
                        final_validation_loss=rows[-1]["validation_loss"],
                        final_over_min_loss=float(arrays["validation_loss"][-1] / arrays["validation_loss"][min_i]),
                        adapter_parameters=int(parameter_matrix.sum()), mean_rank=float(matrix.mean()),
                        rank_min=int(matrix.min()), rank_max=int(matrix.max()),
                        modules_at_rank2=int((matrix == 2).sum()), modules_at_rank16=int((matrix == 16).sum()),
                        attention_parameter_share=float(parameter_matrix[:, :4].sum() / parameter_matrix.sum()),
                        upper_half_parameter_share=float(parameter_matrix[6:].sum() / parameter_matrix.sum()),
                        elapsed_hours=metrics["elapsed_seconds"] / 3600, source=str(source))
            diagnostics.append(diag)
            runs[task][seed] = dict(history=history, rank=matrix, params=parameter_matrix,
                                    arrays=arrays, diagnostic=diag, source=source)
            provenance.append(dict(task=task, seed=seed, source=str(source),
                                   files={name: sha256(source / name) for name in ["history.jsonl", "allocation.json", "metrics.json", "run_config.json"]}))
        scores = [runs[task][s]["arrays"]["score"][-1] for s in SEEDS]
        assert abs(np.mean(scores) - t["mean"]) < 1e-9
        assert abs(np.std(scores, ddof=1) - t["sample_sd"]) < 1e-9
        summary.append(dict(task=task, metric=METRICS[task], seed42=scores[0], seed777=scores[1], seed2026=scores[2],
                            speccon_mean=t["mean"], speccon_sample_sd=t["sample_sd"],
                            dplora_paper_mean=t["paper_mean"], dplora_paper_sd=t["paper_sd"],
                            delta=t["delta"], **t["hyperparameters"]))
    assert abs(np.mean([report["tasks"][t]["mean"] for t in TASKS]) - report["our_avg"]) < 1e-9
    write_csv("learning_curves.csv", curves)
    write_csv("rank_allocation.csv", ranks)
    write_csv("run_diagnostics.csv", diagnostics)
    # Different experiments have different optional hyperparameter keys.
    keys = list(dict.fromkeys(k for row in summary for k in row))
    write_csv("final_results_and_hyperparameters.csv", [{k: row.get(k, "") for k in keys} for row in summary])
    write_csv("validation_split_metrics.csv", split_rows)
    shutil.copy2(REPORT / "results.json", OUT / "data" / "selected_report_results.json")
    dump_json(OUT / "data" / "provenance.json", {"report_path": str(REPORT / "results.json"),
              "report_sha256": sha256(REPORT / "results.json"), "report_snapshot": report["snapshot"], "runs": provenance})
    dump_json(OUT / "data" / "validation.json", dict(status="passed", runs=len(diagnostics),
              epoch_records=len(curves), module_rank_records=len(ranks),
              max_report_score_error=max_score_error, max_loss_decomposition_error=max_objective_error,
              all_runs_complete=True, all_parameter_budgets=755712,
              all_epochs_complete=True, all_rank_matrices_shape=[12, 6]))
    return report, runs

def decorate(fig, title, subtitle, footer, rect=(.055, .10, .95, .82)):
    fig.suptitle(title, x=.055, y=.968, ha="left", fontsize=20, fontweight="bold")
    fig.text(.055, .919, subtitle, ha="left", va="top", fontsize=10.3, color=MUTED)
    fig.text(.055, .032, footer, ha="left", va="bottom", fontsize=8.8, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=rect[0], bottom=rect[1], right=rect[2], top=rect[3], hspace=.52, wspace=.32)

def seed_handles():
    return [Line2D([0], [0], color=COLORS[s], linestyle=STYLES[s], label=f"seed {s}") for s in SEEDS]

def epoch_axis(ax, epochs):
    ax.set_xlabel("Epoch")
    ax.set_xlim(1, epochs)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))

def log_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(which="minor", alpha=.23)

def task_subtitle(task):
    hp = REPORT_DATA["tasks"][task]["hyperparameters"]
    return f"{hp['epochs']} epochs | lambda={hp['lambda']:g} | lr={hp['lr']:g} | dropout={hp['dropout']:g} | R-Drop alpha={hp['rdrop_alpha']:g}"

CATALOG = []

def save(fig, stem, title_cn, section, caption, atlas):
    fig.savefig(OUT / "png" / f"{stem}.png", dpi=300)
    fig.savefig(OUT / "pdf" / f"{stem}.pdf")
    fig.savefig(OUT / "svg" / f"{stem}.svg")
    fig.savefig(OUT / "thumbnails" / f"{stem}.jpg", dpi=85, pil_kwargs={"quality": 88})
    atlas.savefig(fig)
    CATALOG.append(dict(number=len(CATALOG) + 1, stem=stem, title=title_cn, section=section, caption=caption))
    plt.close(fig)
    print(f"[{len(CATALOG):02d}] {stem}", flush=True)

def comparison(atlas):
    fig, axes = plt.subplots(2, 4, figsize=(17, 9.5))
    for task, ax in zip(TASKS, axes.flat):
        t = REPORT_DATA["tasks"][task]
        ax.errorbar(t["paper_mean"], 0, xerr=t["paper_sd"], fmt="D", color=PAPER, capsize=5, ms=6, lw=2)
        ax.errorbar(t["mean"], 1, xerr=t["sample_sd"], fmt="D", color=INK, capsize=5, ms=6, lw=2)
        for i, seed in enumerate(SEEDS):
            ax.scatter(RUNS[task][seed]["arrays"]["score"][-1], 1.22 + i * .13,
                       color=COLORS[seed], s=32, zorder=4, edgecolors="white", linewidths=.5)
        ax.set_title(f"{LABELS[task]}  |  delta {t['delta']:+.2f}", loc="left")
        ax.set_yticks([0, 1], ["DPLoRA", "SpecCon"])
        ax.set_ylim(-.6, 2.3)
        ax.set_xlabel(METRICS[task])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="y", visible=False)
        ax.text(.02, .94, f"SpecCon {t['mean']:.2f} +/- {t['sample_sd']:.2f}\nDPLoRA {t['paper_mean']:.2f} +/- {t['paper_sd']:.2f}",
                transform=ax.transAxes, va="top", fontsize=9, color=MUTED)
    decorate(fig, "Final validation performance | SpecCon-LoRA vs DPLoRA",
             f"8 tasks x 3 seeds | task-specific axes | macro mean: {REPORT_DATA['our_avg']:.2f} vs {REPORT_DATA['paper_avg']:.2f} ({REPORT_DATA['delta_avg']:+.2f} points)",
             "SpecCon: final epoch, mean +/- sample SD (n=3). DPLoRA: paper-reported mean +/- SD, p=0.4.\nDifferent evaluation protocols may affect comparison; these error bars are not confidence intervals or significance tests.",
             rect=(.065, .15, .975, .80))
    fig.legend(handles=seed_handles(), loc="upper right", bbox_to_anchor=(.97, .90), ncol=3, frameon=False)
    save(fig, "01_final_performance_comparison", "最终成绩：SpecCon-LoRA 与 DPLoRA", "overview",
         "采用报告入选配置的最终 epoch 成绩。SpecCon 误差棒为 3 个 seed 的样本标准差；DPLoRA 为原报告引用的论文数据。各任务横轴范围不同。", atlas)

def score_heatmap(atlas):
    values = np.array([[RUNS[t][s]["arrays"]["score"][-1] for s in SEEDS] + [REPORT_DATA["tasks"][t]["mean"]] for t in TASKS])
    delta = values - np.array([REPORT_DATA["tasks"][t]["paper_mean"] for t in TASKS])[:, None]
    limit = max(abs(delta.min()), abs(delta.max()))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [1.8, 1]})
    im = ax.imshow(delta, cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(4), ["seed 42", "seed 777", "seed 2026", "Mean"])
    ax.set_yticks(range(8), [LABELS[t] for t in TASKS])
    ax.set_title("Cell value: final metric | color: delta vs paper", pad=14)
    for i in range(8):
        for j in range(4):
            ax.text(j, i, f"{values[i,j]:.2f}\n({delta[i,j]:+.2f})", ha="center", va="center", fontsize=12,
                    color="white" if abs(delta[i, j]) / limit > .65 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.035)
    cb.set_label("SpecCon - DPLoRA (metric points)")
    means = np.array([REPORT_DATA["tasks"][t]["delta"] for t in TASKS])
    bx.barh(np.arange(8), means, color="#1677B8", height=.58)
    bx.axvline(0, color=MUTED, lw=1)
    bx.set_yticks(range(8), [LABELS[t] for t in TASKS])
    bx.invert_yaxis()
    bx.set_xlim(min(0, means.min()) - .04, means.max() + .22)
    for i, v in enumerate(means):
        bx.text(v + .014, i, f"{v:+.3f}", va="center", fontsize=11)
    bx.set_title("Mean-score difference", pad=14)
    bx.set_xlabel("Metric points")
    decorate(fig, "Final-score heatmap and task-wise improvements",
             "Metric scale x100: CoLA MCC | STS-B Pearson | MNLI mean m/mm accuracy | other tasks accuracy",
             "Color is centered at each task's DPLoRA paper mean, not at zero raw accuracy. Mean-column values are three-seed averages.\nThe macro mean is the arithmetic average of eight task metrics; no pooled uncertainty estimate is inferred.",
             rect=(.08, .15, .95, .82))
    save(fig, "02_score_heatmap_and_deltas", "成绩热力图与逐任务提升", "overview",
         "格内为实际成绩及相对 DPLoRA 论文均值的差值，颜色表示差值，不是原始准确率。", atlas)

def heat(ax, matrix, vmax=16, vmin=2, decimals=0, cmap="YlGnBu", fontsize=8):
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(6), MODULE_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(12), range(12))
    ax.set_ylabel("Encoder layer (0-based)")
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.45, alpha=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(12):
        for j in range(6):
            ax.text(j, i, f"{matrix[i,j]:.{decimals}f}", ha="center", va="center", fontsize=fontsize,
                    color="white" if (matrix[i,j] - vmin) / (vmax - vmin) > .6 else INK)
    return im

def rank_overview(atlas, sd=False):
    mats = {t: np.stack([RUNS[t][s]["rank"] for s in SEEDS]) for t in TASKS}
    data = {t: m.std(axis=0, ddof=1) if sd else m.mean(axis=0) for t, m in mats.items()}
    upper = np.ceil(max(x.max() for x in data.values())) if sd else 16
    fig, axes = plt.subplots(2, 4, figsize=(18, 12))
    for task, ax in zip(TASKS, axes.flat):
        im = heat(ax, data[task], vmax=upper, vmin=0 if sd else 2, decimals=1,
                  cmap="magma_r" if sd else "YlGnBu", fontsize=7.8)
        ax.set_title(f"{LABELS[task]} | lambda={REPORT_DATA['tasks'][task]['hyperparameters']['lambda']:g}", loc="left")
    decorate(fig, "Rank allocation variability across seeds" if sd else "Rank allocation across tasks | three-seed mean",
             "12 encoder layers x 6 adapted projections | same color scale for all tasks | RoBERTa-base",
             ("Cell values are sample SD across seeds 42, 777 and 2026; higher values indicate seed-sensitive allocation.\n" if sd else "Cell values are mean ranks across seeds 42, 777 and 2026, so fractions are expected. Individual allocations use integer ranks.\n") +
             "Ranks come from allocation.json at initialization; they are not attention weights or a time series. Q/K/V/Attn-O are attention projections.",
             rect=(.055, .14, .91, .84))
    fig.subplots_adjust(hspace=.42, wspace=.35)
    cb = fig.colorbar(im, cax=fig.add_axes([.937, .25, .012, .48]))
    cb.set_label("Rank sample SD" if sd else "Mean rank")
    save(fig, "04_rank_seed_variability" if sd else "03_rank_mean_heatmaps",
         "Rank 分配的 seed 间波动" if sd else "8 个任务的平均 rank 热力图", "overview",
         "每格为三个 seed 的 rank 样本标准差，用于观察初始化分配的稳定性。" if sd else "每格为三个 seed 的平均 rank，所有任务统一采用 2–16 色标。层编号为 0–11。", atlas)

def parameter_budget(atlas):
    matrices = np.stack([np.mean([RUNS[t][s]["params"] for s in SEEDS], axis=0) for t in TASKS])
    module_shares = matrices.sum(axis=1) / 755712 * 100
    layer_shares = matrices.sum(axis=2) / 755712 * 100
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1, 1.8]})
    palette = ["#335C81", "#568BB1", "#79B2C7", "#AAD3D2", "#E4B777", "#CC8152"]
    left = np.zeros(8)
    for j, (name, color) in enumerate(zip(MODULE_LABELS, palette)):
        ax.barh(range(8), module_shares[:, j], left=left, color=color, label=name, height=.65)
        for i in range(8):
            ax.text(left[i] + module_shares[i, j] / 2, i, f"{module_shares[i,j]:.0f}%", va="center", ha="center",
                    fontsize=8.5, color="white" if j < 2 else INK)
        left += module_shares[:, j]
    ax.set_yticks(range(8), [LABELS[t] for t in TASKS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of adapter parameters (%)")
    ax.set_title("By projection type", loc="left", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, frameon=False)
    im = bx.imshow(layer_shares, aspect="auto", cmap="YlGnBu", vmin=0, vmax=np.ceil(layer_shares.max()))
    bx.grid(False)
    bx.set_xticks(range(12), range(12))
    bx.set_yticks(range(8), [LABELS[t] for t in TASKS])
    bx.set_xlabel("Encoder layer (0-based)")
    bx.set_title("By encoder layer (%)", loc="left", pad=14)
    for i in range(8):
        for j in range(12):
            bx.text(j, i, f"{layer_shares[i,j]:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if layer_shares[i,j] > .6 * layer_shares.max() else INK)
    fig.colorbar(im, ax=bx, fraction=.027, pad=.02, label="Parameter share (%)")
    decorate(fig, "Where the adapter parameter budget goes",
             "Exactly 755,712 adapter parameters per run | each panel averages seeds 42, 777 and 2026",
             "Parameter cost = rank x (input dimension + output dimension). Attention: 1,536 parameters/rank; FFN: 3,840 parameters/rank.\nThese plots measure parameter allocation, not computational cost, attention importance or causal contribution to performance.",
             rect=(.06, .22, .95, .82))
    save(fig, "05_parameter_budget_distribution", "参数预算：模块与层的分配", "overview",
         "按 rank × (输入维度 + 输出维度) 计算真实参数量，避免把 FFN 和注意力模块的相同 rank 当成相同预算。每次训练均为 755,712 个适配器参数。", atlas)

def loss_overview(atlas, key):
    training = key == "train_total_loss"
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    for task, ax in zip(TASKS, axes.flat):
        stack = np.stack([RUNS[task][s]["arrays"][key] for s in SEEDS])
        epochs = RUNS[task][42]["arrays"]["epoch"]
        mean, sd = stack.mean(axis=0), stack.std(axis=0, ddof=1)
        for s, y in zip(SEEDS, stack):
            ax.plot(epochs, y, color=COLORS[s], lw=.8, alpha=.45, linestyle=STYLES[s])
        ax.fill_between(epochs, np.maximum(mean - sd, np.min(stack) * .1), mean + sd, color="#1677B8", alpha=.14)
        ax.plot(epochs, mean, color=INK, lw=2)
        ax.set_title(LABELS[task], loc="left")
        ax.set_ylabel(("Training objective" if training else ("Validation MSE" if task == "stsb" else "Validation CE")) + " (log)")
        log_axis(ax)
        epoch_axis(ax, int(epochs[-1]))
    decorate(fig, "Training loss | all selected experiments" if training else "Validation loss | all selected experiments",
             "Bold line: three-seed mean | shaded band: +/- sample SD | thin lines: individual seeds | log y-axis; task-specific ranges",
             ("Training objective = supervised loss + alpha x consistency loss, alpha=1. MRPC supervised loss includes label smoothing (0.05).\n" if training else "Validation uses a single deterministic forward pass: cross-entropy for classification and MSE for STS-B.\n") +
             "No smoothing or interpolation. MNLI validation loss is weighted by the matched/mismatched validation-set sizes.",
             rect=(.065, .15, .975, .82))
    save(fig, "06_training_loss_overview" if training else "07_validation_loss_overview",
         "训练 loss 总览" if training else "验证 loss 总览", "overview",
         "粗线为三个 seed 均值，阴影为 ±1 样本标准差，细线为各 seed 原始记录。采用对数纵轴以兼顾训练初期和后期。", atlas)

def metric_overview(atlas, late=False):
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    for task, ax in zip(TASKS, axes.flat):
        for seed in SEEDS:
            a = RUNS[task][seed]["arrays"]
            start_epoch = int(np.floor(a["epoch"][-1] / 2)) + 1 if late else 1
            mask = a["epoch"] >= start_epoch
            ax.plot(a["epoch"][mask], a["score"][mask], color=COLORS[seed], linestyle=STYLES[seed])
            ax.scatter(a["epoch"][-1], a["score"][-1], color=COLORS[seed], s=25, zorder=4, clip_on=False)
        ax.axhline(REPORT_DATA["tasks"][task]["paper_mean"], color=PAPER, linestyle=":", lw=1.5)
        ax.set_title(LABELS[task] + (f" | epochs {start_epoch}-{int(a['epoch'][-1])}" if late else ""), loc="left")
        ax.set_ylabel(METRICS[task])
        epoch_axis(ax, int(a["epoch"][-1]))
        if late:
            ax.set_xlim(start_epoch, int(a["epoch"][-1]))
    decorate(fig, "Validation metrics | late-epoch detail" if late else "Validation metric curves | all three seeds",
             "Last half of each training schedule | unsmoothed scores | task-specific axes" if late else "Raw per-epoch scores | end markers show the final epoch used in the report | axes vary by task",
             "Dotted red line is a DPLoRA paper final-score reference, not its learning curve. Only SpecCon training trajectories are available here.\nMRPC uses accuracy (not the logged accuracy/F1 composite); CoLA and STS-B are not accuracy tasks.",
             rect=(.065, .15, .975, .79))
    fig.legend(handles=seed_handles() + [Line2D([0], [0], color=PAPER, ls=":", label="DPLoRA paper mean")],
               loc="upper right", bbox_to_anchor=(.97, .89), ncol=4, frameon=False)
    save(fig, "27_validation_metric_late_epochs" if late else "08_validation_metric_overview",
         "后半程验证指标曲线放大图" if late else "验证精度 / MCC / Pearson 曲线总览", "overview",
         "仅放大各任务后半程 epoch，便于观察收敛后的细微差异；不平滑、不修改最终分数。" if late else "所有曲线均来自逐 epoch 日志，不平滑。末端圆点为报告采用的最终 epoch；DPLoRA 虚线仅为论文成绩参考。", atlas)

def diagnostics(atlas):
    gap = np.array([[RUNS[t][s]["diagnostic"]["peak_minus_final"] for s in SEEDS] for t in TASKS])
    ratio = np.array([[RUNS[t][s]["diagnostic"]["final_over_min_loss"] for s in SEEDS] for t in TASKS])
    fig, axes = plt.subplots(1, 2, figsize=(14, 9.5))
    for ax, data, cmap, minimum, title, clabel in zip(axes, [gap, ratio], ["YlOrRd", "YlOrRd"], [0, 1],
            ["Peak metric minus final metric", "Final validation loss / minimum loss"], ["Metric points", "Loss ratio"]):
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=minimum, vmax=data.max())
        ax.grid(False)
        ax.set_yticks(range(8), [LABELS[t] for t in TASKS])
        ax.set_xticks(range(3), [f"seed {s}" for s in SEEDS])
        ax.set_title(title, pad=14)
        for i in range(8):
            for j, s in enumerate(SEEDS):
                d = RUNS[TASKS[i]][s]["diagnostic"]
                ep = d["peak_epoch"] if minimum == 0 else d["min_loss_epoch"]
                suffix = "" if minimum == 0 else "x"
                ax.text(j, i, f"{data[i,j]:.2f}{suffix}\n@ epoch {ep}", ha="center", va="center", fontsize=11,
                        color="white" if (data[i,j] - minimum) / (data.max() - minimum) > .58 else INK)
        fig.colorbar(im, ax=ax, fraction=.04, pad=.04, label=clabel)
    decorate(fig, "Late-training diagnostics",
             "Peak scores and minimum validation losses are diagnostic only; report scores remain fixed-final-epoch values",
             "Left: maximum of the report metric over recorded epochs minus final metric. Right: final / minimum validation loss.\nThe annotated epoch is the first peak or minimum. This view does not establish a causal link between rank allocation and generalization.",
             rect=(.08, .15, .94, .82))
    save(fig, "09_late_training_diagnostics", "后期训练诊断：峰值差距与验证 loss 回升", "overview",
         "左侧显示历次验证最高指标到最终指标的差距，右侧为最终验证 loss / 最小验证 loss；附峰值或最小 loss 的首次 epoch。仅供诊断，不改用最佳 checkpoint。", atlas)

def auxiliary(atlas):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for task, ax, key, label in zip(["mnli", "mrpc", "qqp", "stsb"], axes.flat,
                                  ["accuracy", "f1", "f1", "spearmanr"],
                                  ["Split accuracy (%)", "F1 x 100", "F1 x 100", "Spearman x 100"]):
        for seed in SEEDS:
            history = RUNS[task][seed]["history"]
            epochs = [x["epoch"] for x in history]
            if task == "mnli":
                for split, style in [("matched", "-"), ("mismatched", "--")]:
                    ax.plot(epochs, [100 * h["splits"][split][key] for h in history], color=COLORS[seed], ls=style, lw=1.3)
            else:
                ax.plot(epochs, [100 * h["splits"]["validation"][key] for h in history], color=COLORS[seed], ls=STYLES[seed])
            epoch_axis(ax, epochs[-1])
        ax.set_title(LABELS[task] + (" | solid: matched; dashed: mismatched" if task == "mnli" else ""), loc="left")
        ax.set_ylabel(label)
    decorate(fig, "Additional validation metrics",
             "Logged auxiliary metrics complement the primary report metric; they do not change the selected experiment scores",
             "MNLI report metric averages matched and mismatched accuracy. MRPC/QQP report accuracy; F1 is shown here for completeness.\nSTS-B report metric is Pearson; Spearman is an additional rank-correlation metric.",
             rect=(.065, .15, .975, .79))
    fig.legend(handles=seed_handles(), loc="upper right", bbox_to_anchor=(.97, .89), ncol=3, frameon=False)
    save(fig, "10_auxiliary_validation_metrics", "补充指标：MNLI 双验证集、F1、Spearman", "overview",
         "展示日志中实际记录的补充指标：MNLI matched/mismatched accuracy、MRPC/QQP F1、STS-B Spearman。", atlas)

def task_curves(task, atlas, number):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    train_ax, val_ax, score_ax, decomp_ax = axes.flat
    for seed in SEEDS:
        a = RUNS[task][seed]["arrays"]
        for ax, key in [(train_ax, "train_total_loss"), (val_ax, "validation_loss"), (score_ax, "score")]:
            ax.plot(a["epoch"], a[key], color=COLORS[seed], ls=STYLES[seed], lw=1.8)
            if key == "score":
                ax.scatter(a["epoch"][-1], a[key][-1], color=COLORS[seed], s=36, clip_on=False, zorder=5)
        epoch_axis(train_ax, int(a["epoch"][-1]))
    val_name = "MSE" if task == "stsb" else "Cross-entropy"
    for ax, title, ylabel in [(train_ax, "Training total loss", "R-Drop objective (log)"),
                             (val_ax, "Validation loss", f"{val_name} (log)"),
                             (score_ax, "Validation metric | full recorded trajectory", METRICS[task])]:
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
    for ax in [train_ax, val_ax]:
        log_axis(ax)
    score_ax.axhline(REPORT_DATA["tasks"][task]["paper_mean"], color=PAPER, ls=":", label="DPLoRA paper mean", lw=1.5)
    score_ax.legend(frameon=False, fontsize=8, loc="lower right")
    for key, color, style, label in [("train_total_loss", INK, "-", "Total"),
                                   ("train_supervised_loss", "#1677B8", "--", "Supervised"),
                                   ("train_consistency_loss", "#E88923", "-.", "Consistency x alpha")]:
        stack = np.stack([RUNS[task][s]["arrays"][key] for s in SEEDS])
        if key == "train_consistency_loss":
            stack = stack * REPORT_DATA["tasks"][task]["hyperparameters"]["rdrop_alpha"]
        mean, sd = stack.mean(0), stack.std(0, ddof=1)
        decomp_ax.plot(a["epoch"], mean, color=color, ls=style, label=label, lw=2)
        decomp_ax.fill_between(a["epoch"], np.maximum(mean - sd, np.min(stack) * .1), mean + sd, color=color, alpha=.12)
    decomp_ax.set_title("Training objective components | mean +/- SD", loc="left")
    decomp_ax.set_ylabel("Loss (log)")
    decomp_ax.legend(frameon=False, fontsize=8, loc="best")
    log_axis(decomp_ax)
    for ax in axes.flat:
        epoch_axis(ax, int(a["epoch"][-1]))
    scores = " | ".join(f"seed {s}: {RUNS[task][s]['arrays']['score'][-1]:.2f}" for s in SEEDS)
    decorate(fig, f"{LABELS[task]} | training and validation curves", task_subtitle(task),
             f"Final {METRICS[task]}: {scores}. All trajectories are unsmoothed epoch-level records.\nTraining objective includes R-Drop consistency; validation loss is a separate deterministic supervised evaluation.",
             rect=(.07, .15, .97, .79))
    fig.legend(handles=seed_handles(), loc="upper right", bbox_to_anchor=(.97, .89), ncol=3, frameon=False)
    save(fig, f"{number:02d}_{task}_learning_curves", f"{LABELS[task]}：训练、验证、指标与 loss 分解", task,
         f"{task_subtitle(task)}。三个 seed 的原始曲线及训练目标分解；纵轴注明 log 的面板采用对数尺度。", atlas)

def task_ranks(task, atlas, number):
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    for seed, ax in zip(SEEDS, axes):
        run = RUNS[task][seed]
        im = heat(ax, run["rank"], fontsize=11)
        d = run["diagnostic"]
        ax.set_title(f"seed {seed} | final {d['final_score']:.2f}\nmean rank {d['mean_rank']:.2f}; at rank=2: {d['modules_at_rank2']}/72", fontsize=11, pad=15)
    decorate(fig, f"{LABELS[task]} | individual rank-allocation heatmaps", task_subtitle(task),
             "All runs: 755,712 adapter parameters; 72 projections; allowed rank range 2-16. One shared color scale; layer indices are 0-based.\nQ/K/V = query/key/value; Attn-O = attention output; FFN-in/out = intermediate/output dense. Allocation is recorded at initialization.",
             rect=(.055, .17, .91, .81))
    cb = fig.colorbar(im, cax=fig.add_axes([.94, .25, .015, .48]), ticks=[2, 4, 6, 8, 10, 12, 14, 16])
    cb.set_label("Rank")
    save(fig, f"{number:02d}_{task}_rank_heatmaps", f"{LABELS[task]}：三个 seed 的 rank 热力图", task,
         "实际整数 rank；每个面板为一个 seed，统一 2–16 色标。标题中的 final 使用对应任务的报告指标。", atlas)

def write_readme_and_gallery():
    findings = []
    for task in TASKS:
        ds = [RUNS[task][s]["diagnostic"] for s in SEEDS]
        findings.append(f"| {LABELS[task]} | {np.mean([d['upper_half_parameter_share'] for d in ds])*100:.1f}% | "
                        f"{np.mean([d['modules_at_rank2'] for d in ds]):.1f}/72 | "
                        f"{np.mean([d['peak_minus_final'] for d in ds]):.2f} | "
                        f"{np.mean([d['final_over_min_loss'] for d in ds]):.2f}x |")
    text = f"""# SpecCon-LoRA GLUE 实验图册

生成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}  
报告快照：{REPORT_DATA['snapshot']}  
范围：刚更新的 `SpecCon-LoRA_GLUE_best_42_777_2026_vs_DPLoRA.pdf` 中入选的 8 个任务 × 3 个 seed（42、777、2026），共 24 次已完成训练。

## 文件入口

- `index.html`：本地图片浏览页，支持按任务筛选，链接到高清 PNG、PDF、SVG。
- `SpecCon_GLUE_experiment_figure_atlas.pdf`：{len(CATALOG)} 页合并图册。
- `png/`：300 dpi 高清图片；`pdf/`、`svg/`：单张矢量图片。
- `data/`：逐 epoch 曲线 CSV、rank CSV、最终成绩及超参数 CSV、诊断 CSV、验证集补充指标 CSV、来源哈希与核验结果。
- `build_figures.py`：可复现绘图脚本（依赖 Python、NumPy、Matplotlib）。在本目录运行 `python build_figures.py` 可重新读取选中的实验日志并生成图片。

## 统计口径

1. 报告分数固定取最后一个 epoch，不使用最佳 checkpoint。每个任务三个 seed 使用同一组超参数。原报告按完成的三 seed 配置的最终验证均值选择配置；SST-2 按用户要求更新为 lambda=2、38 epochs，与此前配置的最终均值持平。
2. CoLA 使用 MCC × 100；STS-B 使用 Pearson × 100；MNLI 使用 matched/mismatched accuracy 的算术平均 × 100；其余任务使用 accuracy × 100。MRPC 的日志 `combined_score` 是 accuracy/F1 综合指标，因此本图册直接取 validation.accuracy，与报告一致。
3. 阴影与 SpecCon 误差棒是三个 seed 的样本标准差（ddof=1），不是置信区间。所有曲线不平滑、不插值，各任务轴范围可能不同；标为 log 的 loss 面板采用对数纵轴。对数图中的均值减标准差若不为正，仅在绘制阴影下沿时截到最小观测值的 0.1 倍，不修改 CSV 原值。
4. 训练总 loss = supervised loss + alpha × consistency loss，alpha=1。分类任务监督项是交叉熵（MRPC 含 0.05 label smoothing），STS-B 是 MSE。验证 loss 是确定性单次前向计算的监督损失，与训练总目标定义不同，不能将二者之差直接当成泛化误差。
5. MNLI 验证 loss 按两个验证集的样本数加权；MNLI score 按两个验证集 accuracy 等权平均。图册提供两套原始 accuracy 曲线。
6. Rank 热力图来自各 run 的 `allocation.json`，为初始化时固定分配，不是 attention 权重、梯度强度或 rank 随 epoch 的演化。模块为 Q、K、V、Attn-O、FFN-in、FFN-out，层编号 0–11。
7. 参数预算按 rank × (in_features + out_features) 计算：注意力每 rank 1,536 参数，FFN 每 rank 3,840 参数。24 次运行均核验为 755,712 个适配器参数，实际整数 rank 范围均在 [2,16]。
8. DPLoRA 数据沿用原报告记录的论文 Table 8、p=0.4 的均值和标准差，出处：{REPORT_DATA['paper_source']}。本次选中的运行记录只有 SpecCon 训练日志，因此 DPLoRA 仅显示最终分数参考线或误差棒，不绘制其训练曲线或 rank 分配。不同实验协议可能影响比较；图中不作显著性结论。
9. 日志没有训练集 accuracy、逐 step loss、预测级混淆矩阵或 attention 激活值，因此未生成这些图；曲线分辨率为每 epoch。验证峰值与 loss 最小值只用于诊断，不参与最终分数替换。
10. SST-2 使用新完成的 lambda=2、38 epochs 配置；QNLI 使用报告入选的 lambda=0、49 epochs 配置，不混入仍在补跑的 lambda=2 配置。

## 数据显示的现象

以下指标均在每个 seed 内计算，再取三个 seed 的均值。高层定义为 encoder layer 6–11；峰值差距使用该任务的报告指标。

| 任务 | 高层参数占比 | rank=2 模块数 | 峰值减最终分数 | 最终 / 最小验证 loss |
|---|---:|---:|---:|---:|
{chr(10).join(findings)}

这些图可用于观察 rank 集中程度、seed 波动和后期训练走势，但不能仅凭 rank 热力图认定性能变化的因果来源。验证 loss 上升、指标基本持平可以同时出现，应结合两种曲线判断。

## 图表目录

"""
    text += "\n".join(f"- {x['number']:02d}. [{x['title']}](png/{x['stem']}.png)：{x['caption']}" for x in CATALOG) + "\n"
    (OUT / "README.md").write_text(text, encoding="utf-8")
    cards = []
    for x in CATALOG:
        stem = x['stem']
        cards.append(f'''<article data-section="{x['section']}"><a href="png/{stem}.png" target="_blank"><img loading="lazy" src="thumbnails/{stem}.jpg" alt="{html.escape(x['title'])}"></a>
<div class="body"><h2>{x['number']:02d} · {html.escape(x['title'])}</h2><p>{html.escape(x['caption'])}</p><nav><a href="png/{stem}.png">高清 PNG</a><a href="pdf/{stem}.pdf">矢量 PDF</a><a href="svg/{stem}.svg">SVG</a></nav></div></article>''')
    filters = '<button class="active" data-filter="all">全部</button><button data-filter="overview">总览</button>' + ''.join(f'<button data-filter="{t}">{LABELS[t]}</button>' for t in TASKS)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SpecCon-LoRA · GLUE 实验图册</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#172d45;font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}main{max-width:1440px;margin:auto;padding:44px 32px}h1{font-size:32px;margin:8px 0 12px}header p{line-height:1.9;color:#63748a;max-width:1050px}.meta{letter-spacing:2px;font-size:13px;color:#1677b8;font-weight:700}.links,nav{display:flex;gap:14px;flex-wrap:wrap}a{color:#1677b8;text-decoration:none}a:hover{text-decoration:underline}.links a{background:white;padding:10px 16px;border-radius:8px;border:1px solid #dde5ed}.filters{position:sticky;top:0;background:#f3f6faf2;padding:24px 0 18px;display:flex;gap:8px;flex-wrap:wrap;z-index:2}button{border:1px solid #d4deea;background:white;color:#63748a;padding:8px 15px;border-radius:30px;cursor:pointer}button.active{color:white;background:#172d45;border-color:#172d45}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article{border:1px solid #dee6ef;border-radius:12px;background:white;overflow:hidden}article[hidden]{display:none}img{width:100%;display:block}.body{padding:18px 22px 24px;border-top:1px solid #edf0f5}h2{font-size:17px;margin:0 0 10px}.body p{font-size:14px;color:#63748a;line-height:1.8;min-height:50px}nav{font-size:13px;font-weight:600}footer{padding:32px 0;color:#63748a;font-size:13px}@media(max-width:850px){main{padding:24px 16px}.grid{grid-template-columns:1fr}h1{font-size:26px}}</style>
<main><header><div class="meta">SpecCon-LoRA / RoBERTa-base / GLUE</div><h1>实验图册：Rank、Loss 与验证指标</h1>
<p>8 个任务 × 3 个 seed（42 / 777 / 2026），与已更新报告的入选实验一致。SST-2 使用 λ=2、38 epochs；QNLI 使用 λ=0、49 epochs。分数采用最终 epoch，曲线均来自原始训练日志。CoLA 显示 MCC，STS-B 显示 Pearson，其余显示 accuracy。</p>
<div class="links"><a href="SpecCon_GLUE_experiment_figure_atlas.pdf">打开合并图册 PDF</a><a href="README.md">口径与图表说明</a><a href="data/final_results_and_hyperparameters.csv">成绩与超参数 CSV</a><a href="data/learning_curves.csv">曲线原始数据 CSV</a><a href="data/rank_allocation.csv">Rank 原始数据 CSV</a></div></header>
<div class="filters">FILTERS</div><section class="grid">CARDS</section><footer>报告快照：SNAPSHOT。DPLoRA 仅提供论文最终分数参考；误差棒不是置信区间。</footer></main>
<script>document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('article').forEach(c=>c.hidden=b.dataset.filter!=='all'&&c.dataset.section!==b.dataset.filter)}));</script></html>'''
    (OUT / "index.html").write_text(page.replace("FILTERS", filters).replace("CARDS", '\n'.join(cards)).replace("SNAPSHOT", REPORT_DATA['snapshot']), encoding="utf-8")
    dump_json(OUT / "figure_manifest.json", dict(generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              report_snapshot=REPORT_DATA["snapshot"], count=len(CATALOG), formats=["png", "pdf", "svg"],
              png_dpi=300, atlas="SpecCon_GLUE_experiment_figure_atlas.pdf", figures=CATALOG))

if __name__ == "__main__":
    REPORT_DATA, RUNS = extract()
    print("Validated 24 complete runs, 1,074 epoch records and 1,728 module ranks.", flush=True)
    with PdfPages(OUT / "SpecCon_GLUE_experiment_figure_atlas.pdf") as atlas:
        atlas.infodict().update(Title="SpecCon-LoRA GLUE experiment figures", Author="Experiment report generator",
                               Subject="Final-epoch validation, rank allocation and learning curves")
        comparison(atlas)
        score_heatmap(atlas)
        rank_overview(atlas)
        rank_overview(atlas, sd=True)
        parameter_budget(atlas)
        loss_overview(atlas, "train_total_loss")
        loss_overview(atlas, "validation_loss")
        metric_overview(atlas)
        diagnostics(atlas)
        auxiliary(atlas)
        for i, task in enumerate(TASKS):
            task_curves(task, atlas, 11 + 2*i)
            task_ranks(task, atlas, 12 + 2*i)
        metric_overview(atlas, late=True)
    write_readme_and_gallery()
    print(f"Complete: {len(CATALOG)} figures in {OUT}", flush=True)
