#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/SpecCon-LoRA
HERE=$ROOT/reproduction/gora_protocol_20261112
SEED=${SEED:-42}
OUT=$HERE/seed${SEED}
test -f "$OUT/output/last_adapter/adapter_config.json"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}
export TENSOR_PARALLEL_WORLD_SIZE=2
export DATA_PARALLEL_WORLD_SIZE=1
export PYTHONPATH=/workspace/MyTransformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PY=/workspace/.venvs/gora-vllm/bin/python
"$PY" "$HERE/merge_adapter.py" \
  --adapter "$OUT/output/last_adapter" \
  --output "$OUT/merged_hf" \
  --runtime-template /workspace/SpecCon-LoRA/experiments/llama31_metamathqa_specconlora_fair_rmin6_rmax24_gamma0.7_lambda0.5_seed77_gpu0123/eval/merged_hf \
  --scaling-mode rank >"$OUT/merge.log" 2>&1

"$PY" /workspace/MyTransformers/scripts/run_vllm.py \
  --model_name "$OUT/merged_hf" \
  --datasets "$HERE/gsm8k_official_1319.jsonl" \
  --save_path "$OUT/eval" --save_name generations.jsonl \
  --seed 42 --temperature 0 --top_p 0.1 --top_k 1 \
  --max_tokens 512 --max_model_len 1024 --repetition_penalty 1.1 \
  >"$OUT/eval.log" 2>&1

"$PY" "$HERE/score.py" "$OUT/eval/generations.jsonl" "$OUT/eval/metrics.json"
