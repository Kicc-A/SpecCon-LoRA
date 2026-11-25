#!/usr/bin/env bash
set -euo pipefail

HERE=/workspace/SpecCon-LoRA/reproduction/gora_paper_20261115
ROOT=/workspace/MyTransformers-gora-author
SEED=${SEED:-42}
OUT=$HERE/seed${SEED}
mkdir -p "$OUT"
export PYTHONPATH=$ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

# Paper protocol: global batch 64 = 2 * 1 * 32; initialization N=64,
# reference rank 8, gamma 0.08. Main-text warmup is 0.03 (Table 11 says 0.3).
# The released shell script uses --gora-n-steps, but this checkout parses
# --gradient-est-n-steps. Use physical GPUs 0 and 1, currently idle.
/root/anaconda3/envs/gora/bin/deepspeed --include localhost:0,1 \
  --master_port 16677 train/u_train.py \
  --huggingface --model-name-or-path /workspace/Models/Llama-3.1-8B-Base \
  --train-dataset-name metamathqa --eval-dataset-name gsm8k \
  --train-dataset-path /workspace/MyTransformers/data/metamathqa_lorapro/metamathqa_lorapro_gsm_tok512_train100k.input_output.jsonl \
  --eval-dataset-path /workspace/MyTransformers/data/metamathqa_lorapro/metamathqa_lorapro_gsm_tok512_eval10k.input_output.jsonl \
  --dataset-class-name iterable --mode sft \
  --output-path "$OUT" --experiment-name "gora-paper-gsm8k-seed${SEED}" \
  --use-lora --use-lora-plus --lora-plus-scaler 16 --use-gora \
  --gradient-est-n-steps 64 --gora-init-method compress \
  --gora-max-rank 32 --gora-min-rank 4 --gora-scale-by-lr \
  --gora-lr 8e-2 --gora-rank-stablize --gora-dynamic-scaling \
  --gora-importance-type union_mean --lora-rank 8 --lora-scaler 16 \
  --replace-modules q_proj k_proj v_proj o_proj \
  --run-lora-in-fp32 --weight-a-init-method kaiming \
  --epochs 1 --batch-size-per-gpu 1 --eval-batch-size-per-gpu 1 \
  --gradient-accumulation-steps 32 --bf16 --activation-checkpoint \
  --max-len 1024 --max-src-len 1024 --eval-max-len 1024 --eval-max-src-len 1024 \
  --seed "$SEED" --zero-stage 2 --lr 5e-5 --warmup 0.03 \
  --auto-warmup-steps 10 --auto-warmup-rate 0.05 \
  --weight-decay 5e-4 --lr-decay-style cosine --lr-decay-ratio 0.1 \
  --atten-type flash --fuse-linear-loss --all-reduce-loss \
  --save-trainable --skip-eval --diy-optimizer \
  --ds-config-path "$ROOT/ds_config/zero2_config.json" \
  > "$OUT/train.log" 2>&1

bash "$HERE/eval.sh" > "$OUT/finish.log" 2>&1
