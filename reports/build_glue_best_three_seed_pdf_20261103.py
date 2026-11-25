#!/usr/bin/env python3
"""Build a paper-ready PDF of the best completed three-seed GLUE results."""
import json
import statistics
from pathlib import Path

import fitz

ROOT = Path("/workspace/SpecCon-LoRA")
OUT = ROOT / "reports/GLUE_三随机种子最佳结果_20261103.pdf"
REGULAR_FILE = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD_FILE = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

base = json.loads((ROOT / "reports/glue_best_42_777_2026_20261013/results.json").read_text())["tasks"]
sweep = json.loads((ROOT / "experiments/qnli_qqp_hparam_sweep_20261022_v1/results.json").read_text())["runs"]

specs = [
    ("CoLA", "MCC", "λ=1.5 · 42轮", "cola"),
    ("MNLI", "Accuracy", "λ=1.5 · 32轮", "mnli"),
    ("MRPC", "Accuracy", "λ=1.5 · 57轮", "mrpc"),
    ("QNLI", "Accuracy", "λ=0.25 · 49轮", "qnli"),
    ("QQP", "Accuracy", "λ=1.5 · 15轮", "qqp"),
    ("RTE", "Accuracy", "λ=0.5 · 63轮", "rte"),
    ("SST-2", "Accuracy", "λ=2.0 · 38轮", "sst2"),
    ("STS-B", "Pearson", "λ=1.5 · 67轮", "stsb"),
]

def scores(task):
    if task in ("qnli", "qqp"):
        campaign = "qnli_lr0.000480_e49_lambda0.25" if task == "qnli" else "qqp_lr0.000440_e15_lambda1.5"
        runs = [r for r in sweep if r["config"]["id"] == campaign]
    else:
        runs = base[task]["runs"]
    rows = {int(r["seed"]): float(r.get("score", 0)) for r in runs}
    assert sorted(rows) == [42, 777, 2026], (task, rows)
    return [rows[s] for s in (42, 777, 2026)]

rows = []
for task, metric, config, key in specs:
    values = scores(key)
    rows.append([task, metric, config, *[f"{v:.4f}" for v in values],
                 f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"])

doc = fitz.open()
page = doc.new_page(width=841.68, height=595.44)  # A4 landscape in points
page.insert_font(fontname="noto", fontfile=REGULAR_FILE)
page.insert_font(fontname="notob", fontfile=BOLD_FILE)
navy = (0.09, 0.20, 0.30)
muted = (0.36, 0.40, 0.45)
pale = (0.93, 0.95, 0.97)
rule = (0.83, 0.87, 0.91)

page.insert_text((45, 57), "SpecCon-LoRA 在 GLUE 验证集上的三随机种子结果",
                 fontname="notob", fontsize=18, color=navy)
page.insert_text((45, 82), "统一 γ=1.2；各任务选取已完成配置中固定末轮验证指标的最高三种子均值 · seed = 42 / 777 / 2026",
                 fontname="noto", fontsize=9.5, color=muted)
page.draw_line((45, 98), (797, 98), color=rule, width=0.7)

headers = ["任务", "指标", "配置", "seed 42", "seed 777", "seed 2026", "均值 ± 样本 SD"]
widths = [75, 88, 160, 90, 90, 90, 159]
xs = [45]
for width in widths[:-1]:
    xs.append(xs[-1] + width)
page.draw_rect(fitz.Rect(45, 109, 797, 141), color=navy, fill=navy)
for j, header in enumerate(headers):
    page.insert_text((xs[j]+7, 130), header, fontname="notob", fontsize=9.5, color=(1, 1, 1))

for i, row in enumerate(rows):
    top = 141 + i * 39
    if i % 2 == 0:
        page.draw_rect(fitz.Rect(45, top, 797, top+39), color=pale, fill=pale)
    for j, value in enumerate(row):
        page.insert_text((xs[j]+7, top+25), value,
                         fontname="notob" if j == 6 else "noto",
                         fontsize=9.2 if j != 2 else 8.9,
                         color=navy if j == 6 else (0.14, 0.19, 0.25))
    page.draw_line((45, top+39), (797, top+39), color=rule, width=0.45)

page.insert_text((45, 477), "表注", fontname="notob", fontsize=10, color=navy)
notes = [
    "1. 数值均为 ×100 后的验证集指标；CoLA 为 Matthews 相关系数，STS-B 为 Pearson，其余为准确率。",
    "2. 所有配置均使用 RoBERTa-base 和 755,712 个可训练适配器参数；均值和样本 SD 基于同一配置的三个种子。",
    "3. CoLA 使用 γ=1.2 的主配置；QNLI 和 QQP 使用后续三种子复核。QQP λ=0 对照尚未完成，未纳入。",
    "4. 配置在验证集上筛选；本表是探索性验证成绩，不是 GLUE 官方测试集结果。",
]
for i, note in enumerate(notes):
    page.insert_text((45, 495+i*16), note, fontname="noto", fontsize=8.2,
                     color=(0.24, 0.29, 0.35))
page.draw_line((45, 561), (797, 561), color=rule, width=0.6)
page.insert_text((45, 577),
                 "来源：glue_best_42_777_2026_20261013；qnli_qqp_hparam_sweep_20261022_v1",
                 fontname="noto", fontsize=7.4, color=muted)
page.insert_text((747, 577), "2026-11-03", fontname="noto", fontsize=7.4, color=muted)
doc.set_metadata({"title": "SpecCon-LoRA GLUE 三随机种子最佳结果",
                  "author": "SpecCon-LoRA experiment records",
                  "subject": "Final-epoch validation scores for seeds 42, 777, and 2026"})
doc.save(OUT, garbage=4, deflate=True)
doc.close()
print(OUT)
