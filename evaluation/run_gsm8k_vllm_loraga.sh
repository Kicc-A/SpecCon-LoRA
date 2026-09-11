#!/usr/bin/env bash
set -euo pipefail

PYTHON=/workspace/.venvs/gora-vllm/bin/python
RUN_VLLM=/workspace/MyTransformers/scripts/run_vllm.py
MODEL=${MODEL:-/workspace/MyTransformers/experiments/gora_gsm8k_vllm_loraga_tp4/merged_hf}
DATASET=/workspace/MyTransformers/experiments/gora_gsm8k_vllm_loraga_tp4/data/gsm8k_official_1319.jsonl
OUTPUT=${OUTPUT:-/workspace/MyTransformers/experiments/gora_gsm8k_vllm_loraga_tp4/result}

export DATA_PARALLEL_WORLD_SIZE="${DATA_PARALLEL_WORLD_SIZE:-1}"
export TENSOR_PARALLEL_WORLD_SIZE="${TENSOR_PARALLEL_WORLD_SIZE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH=/workspace/MyTransformers

mkdir -p "$OUTPUT"
cd /workspace/MyTransformers

"$PYTHON" -u "$RUN_VLLM" \
  --model_name "$MODEL" \
  --datasets "$DATASET" \
  --save_path "$OUTPUT" \
  --save_name generations.jsonl \
  --seed 42 \
  --temperature 0 \
  --top_p 0.1 \
  --top_k 1 \
  --max_tokens 512 \
  --max_model_len 1024 \
  --repetition_penalty 1.1

"$PYTHON" - "$OUTPUT/generations.jsonl" <<'PY'
import json
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in source.open(encoding="utf-8")]
pattern = re.compile(r"####\s*(\d+)")

def extract(text):
    match = pattern.search(text)
    return int(match.group(1).replace(",", "")) if match else 0

correct = 0
missing = 0
scored = []
for row in rows:
    match = pattern.search(row["test_result"])
    prediction = extract(row["test_result"])
    gold = extract(row["answer"])
    is_correct = prediction == gold
    correct += int(is_correct)
    missing += int(match is None)
    scored.append({**row, "pred_answer_loraga": prediction,
                   "gold_answer_loraga": gold,
                   "correct_loraga": is_correct,
                   "missing_pred_marker": match is None})

metrics = {"accuracy": correct / len(rows), "correct": correct,
           "total": len(rows), "missing_pred_marker": missing,
           "extractor": "LoRA-GA examples/eval_gsm8k.py: re.search(r\"####\\s*(\\d+)\") then int; missing -> 0"}
(source.parent / "metrics_loraga.json").write_text(json.dumps(metrics, indent=2) + "\n")
(source.parent / "scored_predictions_loraga.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored))
print(json.dumps(metrics, indent=2))
PY
