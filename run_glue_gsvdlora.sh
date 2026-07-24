#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/GSVD-LoRA
TASKS=${TASKS:-"mnli sst2 cola qnli mrpc"}
SEEDS=${SEEDS:-42}
cd "$ROOT"
for seed in $SEEDS; do
  for task in $TASKS; do
    python run_gsvdlora_glue_t5.py \
      --task_name "$task" --seed "$seed" \
      --model_name_or_path /workspace/Models/T5-Base \
      --glue_root /workspace/datasets/glue \
      --output_dir "$ROOT/outputs/${task}_gsvdlora_consensus_seed${seed}" \
      --target_modules q k v o wi wo \
      --base_rank 8 --r_min 4 --r_max 32 --rank_budget_mode param \
      --calibration_steps 64 --calibration_blocks 4 \
      --subspace_consistency_lambda 1.0 --subspace_consistency_floor 0.05 \
      --spectral_oversample 8 --spectral_power_iterations 1 \
      --init_method svd_a_energy_small_b --scaling_mode sqrt_rank \
      --use_loraplus --loraplus_lr_ratio 16
  done
done
