#!/usr/bin/env bash
set -euo pipefail
HERE=/workspace/SpecCon-LoRA/reproduction/gora_paper_20261115
SEED=${SEED:-42}
OUT=$HERE/seed${SEED}
LORA=$OUT/gora-paper-gsm8k-seed${SEED}
test -f "$LORA/final.ckpt"
test -f "$LORA/rank.json"
mkdir -p "$OUT/eval"
export CUDA_VISIBLE_DEVICES=0,1
export TENSOR_PARALLEL_WORLD_SIZE=2 DATA_PARALLEL_WORLD_SIZE=1
export PYTHONPATH=/workspace/MyTransformers
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/workspace/.venvs/gora-vllm/bin/python
"$PY" /workspace/MyTransformers/scripts/run_vllm.py \
  --model_name /workspace/Models/Llama-3.1-8B-Base \
  --lora_path "$LORA" --load_lora_from_mt \
  --datasets /workspace/SpecCon-LoRA/reproduction/gora_protocol_20261112/gsm8k_official_1319.jsonl \
  --save_path "$OUT/eval" --save_name generations.jsonl \
  --seed 42 --temperature 0 --top_p 0.1 --top_k 1 \
  --max_tokens 512 --max_model_len 1024 --repetition_penalty 1.1 \
  > "$OUT/eval.log" 2>&1
"$PY" /workspace/SpecCon-LoRA/reproduction/gora_protocol_20261112/score.py \
  "$OUT/eval/generations.jsonl" "$OUT/eval/metrics.json"
