#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/SpecCon-LoRA/reproduction/gora_paper_loraga_20261117
RUN="$ROOT/paper-seed42"
SOURCE="$ROOT/source"

bash "$ROOT/launch.sh" full > "$ROOT/full.log" 2>&1
test -s "$RUN/final.ckpt"
test -s "$RUN/rank.json"
test -s "$RUN/config.json"
if rg -q 'Traceback \(most recent call last\)|RuntimeError:' "$ROOT/full.log"; then
  echo 'Trainer printed an exception; aborting export' >&2
  exit 1
fi

export PYTHONPATH="$SOURCE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
/root/anaconda3/envs/gora/bin/python "$ROOT/export_merge.py" --run "$RUN" > "$ROOT/export.log" 2>&1
CUDA_VISIBLE_DEVICES=2 /workspace/.venvs/gora-vllm/bin/python "$ROOT/eval_loraga.py" \
  --model "$RUN/merged_hf" --output "$RUN/eval_loraga_gsm8k" \
  > "$ROOT/eval.log" 2>&1
