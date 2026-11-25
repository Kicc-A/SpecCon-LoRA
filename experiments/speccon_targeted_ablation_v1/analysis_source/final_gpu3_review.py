#!/usr/bin/env python3
"""Audit and summarize the 27 GPU-3 runs with fixed-final paired metrics."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev

EXP = Path('/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1')
MAIN = Path('/workspace/SpecCon-LoRA/experiments/glue_main_ablation_3seeds_gpu12_20261016_v3')
OUT = EXP / 'analysis'


def read(path):
    return json.loads(path.read_text())


def metric(path, task):
    data = read(path)
    assert data['state'] == 'complete' and data['adapter_parameters'] == 755712, path
    return float(data['final_accuracy_percent' if task == 'mrpc' else 'final_score_percent'])


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summary(values):
    return f'{mean(values):+.3f}±{stdev(values):.3f}', f'{sum(v>1e-8 for v in values)}/{sum(abs(v)<=1e-8 for v in values)}/{sum(v<-1e-8 for v in values)}'


def audit(run, ref, task, seed):
    allocation = read(run / 'allocation.json')
    assert allocation['rank_report']['budget_used'] == 755712
    new_rng = read(run / 'training_rng.json')
    old_rng = read(ref / 'reused_F/training_rng.json')
    assert new_rng['cpu_sha256'] == old_rng['cpu_sha256']
    assert new_rng['cuda_sha256'] == old_rng['cuda_sha256']
    initial = read(run / 'initial_state.json')
    old_initial = read(MAIN / 'runs' / task / 'E' / f'seed{seed}' / 'initial_state.json')
    assert initial['base_classifier_sha256'] == old_initial['base_classifier_sha256']
    full_rank = read(ref / 'reused_F/allocation.json')['rank_pattern']
    changed = sum(allocation['rank_pattern'][name] != full_rank[name] for name in full_rank)
    return changed


def main():
    OUT.mkdir(exist_ok=True)
    for status_name, count in [('status_p1a_gpu3_extension.json', 9), ('status_p1b_gpu3_score_side.json', 12),
                               ('status_p1b_rte_gpu3.json', 6)]:
        status = read(EXP / status_name)
        assert len(status['entries']) == count and all(e['state'] == 'complete' for e in status['entries'])
    shuffle_rows = []
    side_rows = []
    for task, shuf in [('rte', 20261106), ('mrpc', 20261107), ('cola', 20261107)]:
        for seed in (42, 777, 2026):
            run = EXP / 'runs/p1a_gpu3_extension_20261106' / task / f'shuffle{shuf}' / f'seed{seed}'
            ref = MAIN / 'references' / task / f'seed{seed}'
            full = metric(ref / 'reused_F/metrics.json', task)
            shuffled = metric(run / 'metrics.json', task)
            manifest = read(run / 'shuffle_manifest.json')
            assert manifest['seed'] == shuf and manifest['unchanged_modules'] == 0 and manifest['total_modules'] == 72
            changed = audit(run, ref, task, seed)
            shuffle_rows.append({'task': task, 'shuffle_seed': shuf, 'train_seed': seed,
                                 'full': full, 'shuffled': shuffled, 'full_minus_shuffled': full-shuffled,
                                 'rank_modules_changed_from_full': changed,
                                 'metrics_path': str(run / 'metrics.json'),
                                 'manifest_path': str(run / 'shuffle_manifest.json')})
    for task in ('mrpc', 'cola', 'rte'):
        for seed in (42, 777, 2026):
            ref = MAIN / 'references' / task / f'seed{seed}'
            full = metric(ref / 'reused_F/metrics.json', task)
            vals = {}
            paths = {}
            for side in ('right', 'joint'):
                root = 'runs/p1b_rte_gpu3_score_side_20261107' if task == 'rte' else 'runs/p1b_gpu3_score_side_20261106'
                run = EXP / root / task / side / f'seed{seed}' if task != 'rte' else EXP / root / side / f'seed{seed}'
                vals[side] = metric(run / 'metrics.json', task)
                assert read(run / 'initial_state.json')['score_side'] == side
                audit(run, ref, task, seed)
                paths[side] = str(run / 'metrics.json')
            side_rows.append({'task': task, 'train_seed': seed, 'left': full,
                              'right': vals['right'], 'joint': vals['joint'],
                              'left_minus_right': full-vals['right'],
                              'left_minus_joint': full-vals['joint'],
                              'joint_minus_right': vals['joint']-vals['right'],
                              'right_metrics_path': paths['right'], 'joint_metrics_path': paths['joint']})
    write_csv(OUT / 'GPU3_SHUFFLE_EXTENSION.csv', shuffle_rows)
    write_csv(OUT / 'GPU3_SCORE_SIDE.csv', side_rows)

    lines = ['# SpecCon-LoRA GPU 3 后续消融结果', '',
             '运行日期：2026-11-06 至 2026-11-07。三阶段 9/9、12/12、6/6 次新训练全部正常完成。CoLA 为固定末轮 MCC×100；MRPC/RTE 为固定末轮 Accuracy×100。下列 ± 为三个训练 seed 的配对样本标准差，属于描述性统计。', '',
             '## 1. 第二次打乱共识及 RTE 扩展', '',
             '| 任务 | shuffle seed | 原始共识 − 打乱共识 | 胜/平/负 |', '|---|---:|---:|---:|']
    for task, shuf in [('rte', 20261106), ('mrpc', 20261107), ('cola', 20261107)]:
        rows = [r for r in shuffle_rows if r['task'] == task]
        value, wtl = summary([r['full_minus_shuffled'] for r in rows])
        lines.append(f'| {task.upper()} | {shuf} | {value} | {wtl} |')
    lines += ['', '先前第一组 shuffle 的 MRPC/CoLA 差值分别为 +2.124±0.861、+2.543±1.287；本次第二组仍无负配对，但 MRPC 效应较小。不同 shuffle 结果必须同时保留，不能只报最大收益。', '',
              '## 2. 共识方向对照', '',
              '| 任务 | 左侧 − 右侧 | 胜/平/负 | 左侧 − 联合 | 胜/平/负 | 联合 − 右侧 | 胜/平/负 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for task in ('mrpc', 'cola', 'rte'):
        rows = [r for r in side_rows if r['task'] == task]
        parts = []
        for key in ('left_minus_right', 'left_minus_joint', 'joint_minus_right'):
            value, wtl = summary([r[key] for r in rows])
            parts.extend([value, wtl])
        lines.append('| ' + task.upper() + ' | ' + ' | '.join(parts) + ' |')
    lines += ['', '右侧和联合评分均使用与左侧相同随机化 SVD 的奇异值/能量；联合评分为每个 rank 前缀 sqrt(C_left × C_right)，再执行同一 C_floor、lambda、成本归一化与预算分配。左侧原始 Full 结果复用，三组按任务和训练 seed 配对。', '',
              '## 3. 结论与边界', '',
              '- 这些结果支持当前 RoBERTa-base、默认预算与参考校准下“共识曲线的模块对应关系”和“左侧评分选择”有实际作用。左侧对右侧在 MRPC、CoLA、RTE 各三种子全胜；左侧对联合的优势较小且并非每个种子都胜。',
              '- 第二个 shuffle 只覆盖 MRPC/CoLA。每任务仅三个训练种子，不能把描述性均值当统计显著或跨校准普遍收益。',
              '- 先前 MRPC 跨校准 15 对的平均差 -0.147 点，以及真实局部 rank 交换预测的负结果，仍需在论文中呈现。',
              '- 27 次新训练均核验：755,712 适配器参数、同 seed 分类头哈希、CPU/CUDA 训练 RNG 哈希与原 Full 一致；shuffle 的 72 个模块均来自另一同类型同形状模块。',
              '', '逐次结果：`GPU3_SHUFFLE_EXTENSION.csv` 与 `GPU3_SCORE_SIDE.csv`。预注册文件及原始 metrics 位于实验根目录。']
    (OUT / 'GPU3_消融结果_20261107.md').write_text('\n'.join(lines) + '\n')
    print(OUT / 'GPU3_消融结果_20261107.md')


if __name__ == '__main__':
    main()
