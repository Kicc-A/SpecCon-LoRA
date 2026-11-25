#!/usr/bin/env bash
set -euo pipefail

HERE=/workspace/SpecCon-LoRA/reproduction/gora_official_20261114
ROOT=/workspace/MyTransformers
SEED=${SEED:-42}
OUT=$HERE/seed${SEED}
mkdir -p "$OUT"
export PYTHONPATH=$ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

exec /root/anaconda3/envs/gora/bin/deepspeed --include localhost:0,2 \
  --master_port 16667 train/u_train.py \
  --huggingface --model-name-or-path /workspace/Models/Llama-3.1-8B-Base \
  --train-dataset-name metamathqa --eval-dataset-name gsm8k \
  --train-dataset-path /workspace/MyTransformers/data/metamathqa_lorapro/metamathqa_lorapro_gsm_tok512_train100k.input_output.jsonl \
  --eval-dataset-path /workspace/MyTransformers/data/metamathqa_lorapro/metamathqa_lorapro_gsm_tok512_eval10k.input_output.jsonl \
  --dataset-class-name iterable --mode sft \
  --output-path "$OUT" --experiment-name "gora-gsm8k-seed${SEED}" \
  --use-lora --use-lora-plus --lora-plus-scaler 16 --use-gora \
  --gradient-est-n-steps 32 --gora-init-method compress \
  --gora-max-rank 32 --gora-min-rank 4 --gora-scale-by-lr \
  --gora-lr 5e-2 --gora-rank-stablize --gora-dynamic-scaling \
  --gora-importance-type union_mean --lora-rank 8 --lora-scaler 16 \
  --replace-modules q_proj k_proj v_proj o_proj \
  --run-lora-in-fp32 --weight-a-init-method kaiming \
  --epochs 1 --batch-size-per-gpu 1 --eval-batch-size-per-gpu 1 \
  --gradient-accumulation-steps 16 --bf16 --activation-checkpoint \
  --max-len 1024 --max-src-len 1024 --eval-max-len 1024 --eval-max-src-len 1024 \
  --seed "$SEED" --zero-stage 2 --lr 5e-5 --warmup 0.03 \
  --weight-decay 5e-4 --lr-decay-style cosine --lr-decay-ratio 0.1 \
  --save-trainable --skip-eval --diy-optimizer \
  --ds-config-path /workspace/MyTransformers/ds_config/zero2_config.json \
  "$@"
