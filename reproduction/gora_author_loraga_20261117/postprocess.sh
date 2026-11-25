#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/SpecCon-LoRA/reproduction/gora_author_loraga_20261117
RUN="$ROOT/run-seed42"
SOURCE="$ROOT/source"

while pgrep -f "$SOURCE/train/u_train.py" > /dev/null; do
  sleep 60
done

test -s "$RUN/final.ckpt" || { echo 'GoRA training stopped without final.ckpt'; exit 1; }
test -s "$RUN/rank.json" || { echo 'GoRA training stopped without rank.json'; exit 1; }
test -s "$RUN/config.json" || { echo 'GoRA training stopped without config.json'; exit 1; }
if rg -q 'Traceback \(most recent call last\)|RuntimeError:' "$ROOT/launcher.log"; then
  echo 'Training log contains an exception; inspect before export'
  exit 1
fi

export PYTHONPATH="$SOURCE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
/root/anaconda3/envs/gora/bin/python "$ROOT/export_author_merge.py" --run "$RUN"
CUDA_VISIBLE_DEVICES=2 /workspace/.venvs/gora-vllm/bin/python "$ROOT/eval_loraga.py" \
  --model "$RUN/merged_hf" --output "$RUN/eval_loraga_gsm8k"
