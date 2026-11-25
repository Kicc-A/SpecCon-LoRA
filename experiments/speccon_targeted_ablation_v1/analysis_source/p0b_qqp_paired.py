#!/usr/bin/env python3
"""Summarize frozen, final-epoch QQP lambda 0 versus lambda 0.25 pairs."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path("/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1")
SWEEP = Path("/workspace/SpecCon-LoRA/experiments/qnli_qqp_hparam_sweep_20261022_v1")
RUNNER = SWEEP / "source_snapshot/reproduction/run_roberta_other_glue_rdrop.py"
EXPECTED_SHA = "db24fdd1c745b1c56c6d7aa7f8545b4b2cc0b4b27ad09c234222ed40e21ad7af"
assert hashlib.sha256(RUNNER.read_bytes()).hexdigest() == EXPECTED_SHA

sweep = json.loads((SWEEP / "results.json").read_text())["runs"]
rows = []
for seed in (42, 777, 2026):
    run = ROOT / f"runs/qqp_lambda0/restart_20261103/seed{seed}"
    result = json.loads((run / "metrics.json").read_text())
    config = json.loads((run / "run_config.json").read_text())
    assert result["state"] == "complete" and result["epochs"] == 10
    assert result["adapter_parameters"] == 755712
    assert config["source_sha256"] == EXPECTED_SHA
    assert config["subspace_consistency_lambda"] == 0.0
    reference = next(r for r in sweep if r["config"]["id"] == "qqp_lr0.000440_e10_lambda0.25" and r["seed"] == seed)
    assert reference["final_epoch"] == 10
    acc0 = result["final_score_percent"]
    rows.append({"seed": seed, "lambda0_accuracy": acc0,
                 "lambda025_accuracy": reference["score"],
                 "paired_gain_lambda025_minus_lambda0": reference["score"] - acc0,
                 "lambda0_f1": 100 * result["final"]["splits"]["validation"]["f1"],
                 "lambda025_f1": reference["f1"],
                 "lambda0_output": str(run), "lambda025_output": reference["output"]})

numeric = ["lambda0_accuracy", "lambda025_accuracy",
           "paired_gain_lambda025_minus_lambda0", "lambda0_f1", "lambda025_f1"]
summary = []
for label, fun in (("mean", statistics.mean), ("sample_sd", statistics.stdev)):
    summary.append({"seed": label, **{key: fun(row[key] for row in rows) for key in numeric}})
output = ROOT / "analysis/QQP_PAIRED.csv"
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows + summary)
print(output)
