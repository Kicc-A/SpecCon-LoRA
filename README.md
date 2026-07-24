# GSVD-LoRA

GSVD-LoRA is a clean research implementation of **budget-constrained LoRA rank allocation by cross-block gradient subspace consensus and cost-aware marginal spectral utility**.

It is intentionally separate from `/workspace/GS-LoRA`. The existing GS-LoRA code and results are not modified.

## Core idea

For target matrix `W_i`, split `C` calibration batches into `H` disjoint blocks and compute the block-mean gradient `G_i^(h)`. A deterministic randomized SVD retains only the leading `r_max` left singular subspace and energy:

```
G_i^(h) ~= U_i^(h) diag(sigma_i^(h)) V_i^(h)^T
e_i,k^(h) = sigma_i,k^(h)^2
mu_i,k = mean_h e_i,k^(h)
```

The agreement of the top-k gradient subspaces is the average normalized projector overlap:

```
C_i,k = 2/[H(H-1)] * sum_(a<b) || U_i,1:k^(a)^T U_i,1:k^(b) ||_F^2 / k
```

`C_i,k` is sign-invariant, basis-invariant inside a repeated-singular-value subspace, and lies in `[0,1]` up to numerical error. The consensus-adjusted energy is

```
e_tilde_i,k = mu_i,k * max(C_i,k, C_floor)^lambda
```

Each additional LoRA rank is a separate candidate with cost `c_i = d_in_i + d_out_i` and marginal utility

```
q_i,k = e_tilde_i,k * (e_tilde_i,k / sum_j e_tilde_i,j)^(gamma-1) / c_i
```

All modules start at `r_min`. Candidates are admitted in descending utility while respecting rank-prefix feasibility and the exact global parameter budget. If `param_budget` is omitted, the budget equals uniform `base_rank=8` LoRA:

```
sum_i r_i * (d_in_i + d_out_i) <= sum_i 8 * (d_in_i + d_out_i)
```

## Difference from nearby methods

- GoRA uses a matrix-level gradient/weight importance to map budget to layer ranks. GSVD-LoRA scores every incremental singular direction.
- IGU-LoRA uses integrated-gradient sensitivity with EMA/deviation uncertainty. GSVD-LoRA uses a one-shot, pre-training, cross-block projector-overlap statistic.
- AdaLoRA/DoRA prune rank components during training. GSVD-LoRA allocates once before training and keeps ordinary PEFT compatibility.
- LoRA-GA focuses on gradient-aligned initialization. GSVD-LoRA uses spectral initialization as a secondary component; the allocator is the main method.

## Layout

- `gsvd_lora/consensus.py`: deterministic randomized SVD and cross-block projector agreement.
- `gsvd_lora/calibration.py`: blockwise calibration without retaining full gradients for every block.
- `gsvd_lora/rank_allocator.py`: marginal spectral utility and exact parameter budget.
- `gsvd_lora/init.py`: optional SVD initialization and rank-aware scaling.
- `examples/NLG/run_math_gsvdlora.py`: MetaMathQA training.
- `run_gsvdlora_glue_t5.py`: GLUE training.
- `evaluation/`: the retained four-GPU vLLM + LoRA-GA-style GSM8K extraction protocol.

## MetaMathQA

```bash
cd /workspace/GSVD-LoRA
SEED=42 CONSISTENCY_LAMBDA=1.0 bash run_metamathqa_gsvdlora_4gpu.sh
```

The script matches the current GS-LoRA/GoRA comparison settings: Llama-3.1-8B-Base, MetaMathQA 100k, `q/k/v/o`, reference rank 8, rank range 4-32, alpha 16, sqrt-rank scaling, LoRA+, one epoch, and four GPUs. The intended first ablation is `CONSISTENCY_LAMBDA=0`, which retains block-mean spectral energy but removes subspace agreement.

## GLUE

```bash
TASKS="mnli sst2 cola qnli mrpc" SEEDS=42 bash run_glue_gsvdlora.sh
```

## Evaluation

Merge with the training scaling before vLLM evaluation:

```bash
python evaluation/merge_adapter.py \
  --adapter EXP/output/last_adapter \
  --output EXP/eval/merged_hf \
  --runtime-template /workspace/MyTransformers/experiments/gora_gsm8k_vllm_loraga_tp4/merged_hf \
  --scaling-mode sqrt_rank
MODEL=EXP/eval/merged_hf OUTPUT=EXP/eval/result \
  bash evaluation/run_gsm8k_vllm_loraga.sh
```

## Fair-comparison protocol

Keep base model, data, target modules, reference parameter budget, optimizer, schedule, training tokens, decoding, and answer extractor fixed. Compare at least:

1. uniform rank-8 LoRA;
2. GoRA;
3. GSVD-LoRA with `lambda=0`;
4. GSVD-LoRA with `lambda=1`;
5. optionally energy-magnitude stability as a separate ablation.

Choose `lambda`, `gamma`, and block count on a development split, not GSM8K test accuracy.
