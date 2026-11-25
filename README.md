# SpecCon-LoRA

SpecCon-LoRA is a clean research implementation of **budget-constrained LoRA rank allocation by cross-block gradient subspace consensus and cost-aware marginal spectral utility**.

Current experiments and launchers are documented in [CURRENT_GLUE.md](CURRENT_GLUE.md). The two retained historical math/code results are indexed in [BEST_NLG_RESULTS.md](BEST_NLG_RESULTS.md).

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

- GoRA uses a matrix-level gradient/weight importance to map budget to layer ranks. SpecCon-LoRA scores every incremental singular direction.
- IGU-LoRA uses integrated-gradient sensitivity with EMA/deviation uncertainty. SpecCon-LoRA uses a one-shot, pre-training, cross-block projector-overlap statistic.
- AdaLoRA/DoRA prune rank components during training. SpecCon-LoRA allocates once before training and keeps ordinary PEFT compatibility.
- LoRA-GA focuses on gradient-aligned initialization. SpecCon-LoRA uses spectral initialization as a secondary component; the allocator is the main method.

## Layout

- `speccon_lora/consensus.py`: deterministic randomized SVD and cross-block projector agreement.
- `speccon_lora/calibration.py`: blockwise calibration without retaining full gradients for every block.
- `speccon_lora/rank_allocator.py`: marginal spectral utility and exact parameter budget.
- `speccon_lora/init.py`: optional SVD initialization and rank-aware scaling.
- `reproduction/run_current_glue.py`: retained RoBERTa GLUE runs; see [CURRENT_GLUE.md](CURRENT_GLUE.md).
- `reproduction/launch_rte_qnli_lambda15_20261015.py`: current RTE/QNLI lambda=1.5 experiment.
- `experiments/`: current GLUE runs and retained historical results.
- `BEST_NLG_RESULTS.md`: the best recorded MetaMathQA and CodeFeedback runs, with adapters, merged models and evaluation outputs.

## GLUE

```bash
/root/anaconda3/bin/python reproduction/run_current_glue.py \
  --task cola --gpu 0 --output-root /workspace/glue_rerun --dry-run
```

Historical NLG launch/evaluation scripts and discarded GoRA-protocol reproduction outputs were removed on 2026-10-15. Retained NLG predictions and models remain in their original directories. Shared method code, datasets, base models, DPLoRA comparisons and the independent MT-Bench experiment snapshot are preserved.

## Fair-comparison protocol

Keep base model, data, target modules, reference parameter budget, optimizer, schedule, training tokens, decoding, and answer extractor fixed. Compare at least:

1. uniform rank-8 LoRA;
2. GoRA;
3. SpecCon-LoRA with `lambda=0`;
4. SpecCon-LoRA with `lambda=1`;
5. optionally energy-magnitude stability as a separate ablation.

Choose `lambda`, `gamma`, and block count on a development split, not GSM8K test accuracy.
