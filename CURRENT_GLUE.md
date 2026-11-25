# 当前保留的 GLUE 实验

2026-10-15 按用户选定范围清理：保留当前报告的 8 个任务 × seed 42、777、2026，共 24 次训练。原始目录与文件内容保持不变，保留最终及诊断 adapter、配置、逐轮日志、rank 分配、随机状态和实际训练的源码快照。

| 任务 | 轮数 | λ | 额外设置 | 三种子均值 |
|---|---:|---:|---|---:|
| CoLA | 42 | 1.5 | R-Drop α=1 | 63.9783 |
| MNLI | 32 | 1.5 | R-Drop α=1 | 87.8946 |
| MRPC | 57 | 1.5 | label smoothing=0.05 | 89.7876 |
| QNLI | 49 | 0 | R-Drop α=1 | 93.0319 |
| QQP | 10 | 1.5 | R-Drop α=1 | 91.4618 |
| RTE | 63 | 0.5 | R-Drop α=1 | 80.5054 |
| SST-2 | 38 | 2 | R-Drop α=1 | 94.8012 |
| STS-B | 67 | 1.5 | 回归一致性正则 | 90.8103 |

分数来自固定末轮验证集；指标与统计口径沿用原报告。完整路径、参数和启动命令见 [current_glue_manifest.json](reproduction/current_glue_manifest.json)。

- [当前汇总 PDF](reports/glue_best_42_777_2026_20261013/SpecCon-LoRA_GLUE_best_42_777_2026_vs_DPLoRA.pdf)
- [原始汇总 JSON](reports/glue_best_42_777_2026_20261013/results.json)
- [图册入口](reports/glue_best_42_777_2026_20261013/figures_20261014/index.html)
- [逐文件清理记录](../.cleanup_reviews/20261015_glue/CLEANUP_PLAN.md)

## 复现入口

入口直接调用各批次 `source_snapshot` 中与 `run_config.json` 哈希一致的训练脚本，并使用该次实验的实际超参数。历史调参队列已删除；源码快照中的旧队列仅作为训练来源记录，不再作为启动入口。

```bash
cd /workspace/SpecCon-LoRA
# 只显示命令，不加载模型、不启动训练。
/root/anaconda3/bin/python reproduction/run_current_glue.py \
  --task sst2 --gpu 0 --output-root /workspace/glue_rerun --dry-run

# 去掉 --dry-run 复跑；默认顺序运行 42、777、2026，可用 --seeds 42 指定种子。
```

输出必须使用新目录，脚本拒绝覆盖现有结果。依赖原训练环境 `/root/anaconda3/bin/python`、`/workspace/Models/RoBERTa-base` 和 `/workspace/datasets/glue`。

QNLI、RTE、SST-2 复现依赖的 9 个旧 `training_rng.pt` 及说明文件仍保留在原路径；这些旧目录只提供随机状态，不再保留旧模型和训练成绩。

独立的 GLUE 发布仓库的实现与配置保留，其历史调参输出已清理。该仓库旧默认配置与本表并不完全一致；复现本表请使用上面的入口。

后续清理已将数学、代码实验精简为 [两次最高分结果](BEST_NLG_RESULTS.md)，旧工作入口和 GoRA 复现产物已删除。DPLoRA 对照、MT-Bench 结果及其独立快照保留。当前新增的 RTE/QNLI λ=1.5 实验使用 `reproduction/launch_rte_qnli_lambda15_20261015.py`，其目录和随机状态依赖受保护。
