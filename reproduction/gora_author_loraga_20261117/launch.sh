#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/SpecCon-LoRA/reproduction/gora_author_loraga_20261117
SOURCE="$ROOT/source"
RUN="$ROOT/run-seed42"
PYTHON=/root/anaconda3/envs/gora/bin/python

cd "$SOURCE"
export PYTHONPATH="$SOURCE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PYTHON" -m deepspeed.launcher.runner \
  --include localhost:2,3 --master_port 16823 \
  "$SOURCE/train/u_train.py" \
  --huggingface --model-name-or-path /workspace/Models/Llama-3.1-8B-Base \
  --train-dataset-name metamathqa --eval-dataset-name gsm8k \
  --train-dataset-path "$ROOT/data/train100k.jsonl" \
  --eval-dataset-path "$ROOT/data/eval10k.jsonl" \
  --dataset-class-name iterable --mode sft \
  --output-path "$ROOT" --experiment-name run-seed42 \
  --use-lora --use-lora-plus --lora-plus-scaler 16 --use-gora \
  --gradient-est-n-steps 32 --gora-init-method compress \
  --gora-max-rank 32 --gora-min-rank 4 --gora-scale-by-lr \
  --gora-lr 0.05 --gora-rank-stablize --gora-dynamic-scaling \
  --gora-importance-type union_mean --lora-rank 8 --lora-scaler 16 \
  --replace-modules q_proj k_proj v_proj o_proj \
  --run-lora-in-fp32 --weight-a-init-method kaiming \
  --batch-size-per-gpu 1 --eval-batch-size-per-gpu 1 \
  --gradient-accumulation-steps 32 --bf16 --activation-checkpoint \
  --max-len 1024 --max-src-len 1024 --eval-max-len 1024 --eval-max-src-len 1024 \
  --seed 42 --dataset-seed 42 --zero-stage 2 --lr 5e-5 \
  --warmup .03 --auto-warmup-steps 10 --auto-warmup-rate .05 \
  --betas .9 .999 --weight-decay 5e-4 --lr-decay-style cosine \
  --lr-decay-ratio .1 --atten-type flash --fuse-linear-loss \
  --all-reduce-loss --save-trainable --skip-eval --diy-optimizer \
  --ds-config-path "$SOURCE/ds_config/zero2_config.json" --epochs 1
