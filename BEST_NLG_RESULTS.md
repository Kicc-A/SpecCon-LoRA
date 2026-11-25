# 保留的数学与代码最佳结果

2026-10-15 按用户要求，仅保留以下两次 SpecCon-LoRA 历史实验的完整结果。选择依据是现有完整评测记录的最高单次成绩，不是多种子均值；CodeFeedback 按 HumanEval pass@1 选择。两次实验的基座均为 Llama-3.1-8B-Base。

| 训练任务 | 评测任务 | 种子 | 记录成绩 |
|---|---|---:|---|
| MetaMathQA | GSM8K，1319 题 | 77 | accuracy 73.3131%，967/1319 |
| CodeFeedback | HumanEval，164 题、每题 5 个样本 | 42 | pass@1 39.5122%，pass@5 65.8537% |

## MetaMathQA

[实验目录](experiments/llama31_metamathqa_specconlora_fair_rmin6_rmax24_gamma0.7_lambda0.5_seed77_gpu0123/)

- [评测指标](experiments/llama31_metamathqa_specconlora_fair_rmin6_rmax24_gamma0.7_lambda0.5_seed77_gpu0123/eval/gsm8k_vllm_loraga/result/metrics_loraga.json)
- [训练配置](experiments/llama31_metamathqa_specconlora_fair_rmin6_rmax24_gamma0.7_lambda0.5_seed77_gpu0123/output/run_config.json)
- [最终 adapter](experiments/llama31_metamathqa_specconlora_fair_rmin6_rmax24_gamma0.7_lambda0.5_seed77_gpu0123/output/last_adapter/)
- [合并模型](experiments/llama31_metamathqa_specconlora_fair_rmin6_rmax24_gamma0.7_lambda0.5_seed77_gpu0123/eval/merged_hf/)

## CodeFeedback

[实验目录](experiments/llama31_codefeedback_specconlora_metaconfig_rmin6_rmax24_gamma0.7_lambda0.5_seed42_gpu0123/)

- [评测指标](experiments/llama31_codefeedback_specconlora_metaconfig_rmin6_rmax24_gamma0.7_lambda0.5_seed42_gpu0123/eval/humaneval_official.jsonl.metrics.json)
- [训练配置](experiments/llama31_codefeedback_specconlora_metaconfig_rmin6_rmax24_gamma0.7_lambda0.5_seed42_gpu0123/output/run_config.json)
- [最终 adapter](experiments/llama31_codefeedback_specconlora_metaconfig_rmin6_rmax24_gamma0.7_lambda0.5_seed42_gpu0123/output/last_adapter/)
- [合并模型](experiments/llama31_codefeedback_specconlora_metaconfig_rmin6_rmax24_gamma0.7_lambda0.5_seed42_gpu0123/eval/merged_hf/)

训练日志、rank 分配、原始预测、逐题判定和两次运行对应的本地 W&B 记录均保留。数学及代码的旧工作入口已删除；这些结果作为历史成果保存。旧日志中的命令和路径是当时的运行记录，部分旧评测工具路径已随本次清理移除。

当前训练工作入口见 [CURRENT_GLUE.md](CURRENT_GLUE.md)。MT-Bench 结果和独立源码快照仍在原实验目录；主工作区的重复 MT-Bench 启动脚本已清理。

[清理范围](../.cleanup_reviews/20261015_nlg_gora/CLEANUP_PLAN.md) · [选择记录](../.cleanup_reviews/20261015_nlg_gora/selection_audit.json)
