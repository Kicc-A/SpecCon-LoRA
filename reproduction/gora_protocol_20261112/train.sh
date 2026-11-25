#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/SpecCon-LoRA
HERE=$ROOT/reproduction/gora_protocol_20261112
SEED=${SEED:-42}
OUT=$HERE/seed${SEED}
test ! -e "$OUT/output/last_adapter" || { echo "Existing adapter: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}
export PYTHONPATH=$ROOT
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

exec /root/anaconda3/envs/gslora/bin/torchrun --standalone --nproc_per_node=2 \
  "$HERE/run_math_specconlora.py" \
  --model_name_or_path /workspace/Models/Llama-3.1-8B-Base \
  --output_dir "$OUT/output" \
  --train_dataset_name /workspace/MyTransformers/data/metamathqa_lorapro/metamathqa_lorapro_gsm_tok512_train100k.input_output.jsonl \
  --train_question_column input --train_answer_column output \
  --eval_dataset_name /workspace/MyTransformers/data/metamathqa_lorapro/metamathqa_lorapro_gsm_tok512_eval10k.input_output.jsonl \
  --eval_question_column input --eval_answer_column output \
  --dataset_prefix '' --dataset_postfix '' \
  --target_modules q_proj k_proj v_proj o_proj \
  --base_rank 8 --r_min 4 --r_max 32 --rank_budget_mode param \
  --rank_score_gamma 1.2 --calibration_steps 64 --calibration_blocks 4 \
  --subspace_consistency_lambda 1.5 --subspace_consistency_floor 0.05 \
  --spectral_oversample 8 --spectral_power_iterations 1 --spectral_seed 3407 \
  --lora_alpha 16 --lora_dropout 0 --use_lora_plus --lora_plus_scaler 16 \
  --init_method svd_a_zero_b --scaling_mode rank \
  --max_length 1024 --max_src_len 1024 --max_new_tokens 1024 --no_pad_to_max_length \
  --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
  --learning_rate 5e-5 --weight_decay 5e-4 --warmup_ratio 0.03 \
  --lr_scheduler cosine --lr_decay_ratio 0.1 \
  --seed "$SEED" --bf16 --gradient_checkpointing --trust_remote_code \
  >"$OUT/train.log" 2>&1
