#!/usr/bin/env python3
"""Summarize every preregistered P1-A run using fixed-final task metrics."""
import csv
import json
from pathlib import Path
import statistics

EXP = Path('/workspace/SpecCon-LoRA/experiments/speccon_targeted_ablation_v1')
MAIN = Path('/workspace/SpecCon-LoRA/experiments/glue_main_ablation_3seeds_gpu12_20261016_v3')
RUNS = EXP / 'runs/p1a_shuffle_seed20261106'
CSV = EXP / 'analysis/SHUFFLE_CONSENSUS.csv'
SUMMARY = EXP / 'analysis/SUMMARY_zh.md'


def read(path):
    return json.loads(path.read_text())


def score(path, task):
    data = read(path)
    assert data['state'] == 'complete' and data['adapter_parameters'] == 755712, path
    return float(data['final_accuracy_percent' if task == 'mrpc' else 'final_score_percent'])


def main():
    prereg = read(EXP / 'prereg_p1a_shuffle_20261106.json')
    assert prereg['shuffle_seed'] == 20261106
    rows = []
    for task in ('mrpc', 'cola'):
        for seed in (42, 777, 2026):
            run = RUNS / task / f'seed{seed}'
            ref = MAIN / 'references' / task / f'seed{seed}'
            full = score(ref / 'reused_F/metrics.json', task)
            energy_path = ref / 'reused_D/metrics.json'
            if not energy_path.exists():
                energy_path = MAIN / 'runs' / task / 'D' / f'seed{seed}' / 'metrics.json'
            energy = score(energy_path, task)
            shuffle = score(run / 'metrics.json', task)
            manifest = read(run / 'shuffle_manifest.json')
            allocation = read(run / 'allocation.json')
            full_allocation = read(ref / 'reused_F/allocation.json')
            current_rng = read(run / 'training_rng.json')
            ref_rng = read(ref / 'reused_F/training_rng.json')
            initial = read(run / 'initial_state.json')
            original_initial = read(MAIN / 'runs' / task / 'E' / f'seed{seed}' / 'initial_state.json')
            assert manifest['seed'] == 20261106 and manifest['unchanged_modules'] == 0
            assert allocation['rank_report']['budget_used'] == 755712
            assert current_rng['cpu_sha256'] == ref_rng['cpu_sha256']
            assert current_rng['cuda_sha256'] == ref_rng['cuda_sha256']
            assert initial['base_classifier_sha256'] == original_initial['base_classifier_sha256']
            changed = sum(allocation['rank_pattern'][k] != full_allocation['rank_pattern'][k]
                          for k in allocation['rank_pattern'])
            rows.append({'task': task, 'seed': seed, 'full': full, 'shuffle': shuffle,
                         'energy_only': energy, 'full_minus_shuffle': full - shuffle,
                         'full_minus_energy': full - energy,
                         'shuffle_minus_energy': shuffle - energy,
                         'rank_modules_changed_from_full': changed,
                         'shuffle_manifest': str(run / 'shuffle_manifest.json'),
                         'shuffle_metrics': str(run / 'metrics.json'),
                         'full_metrics': str(ref / 'reused_F/metrics.json'),
                         'energy_metrics': str(energy_path)})
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    lines = ['# P1-A 正确对应的共识 vs 打乱共识', '',
             '六次预注册正式训练全部完成。CoLA 使用固定末轮 MCC×100；MRPC 使用固定末轮 Accuracy×100（不使用 Accuracy/F1 混合指标）。三种子结果仅为描述性统计。', '',
             '| 任务 | Full − Shuffle，均值±配对样本SD | 胜/平/负 | Full − Energy-only | Shuffle − Energy-only |',
             '|---|---:|---:|---:|---:|']
    for task in ('mrpc', 'cola'):
        subset = [row for row in rows if row['task'] == task]
        a = [row['full_minus_shuffle'] for row in subset]
        b = [row['full_minus_energy'] for row in subset]
        c = [row['shuffle_minus_energy'] for row in subset]
        wtl = f'{sum(x > 1e-8 for x in a)}/{sum(abs(x) <= 1e-8 for x in a)}/{sum(x < -1e-8 for x in a)}'
        lines.append(f'| {task.upper()} | +{statistics.mean(a):.3f}±{statistics.stdev(a):.3f} | {wtl} | {statistics.mean(b):+.3f} | {statistics.mean(c):+.3f} |')
    lines += ['', '## 逐种子', '',
              '| 任务 | seed | Full | Shuffle | Energy-only | Full−Shuffle | 变化的 rank 模块 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['task'].upper()} | {row['seed']} | {row['full']:.3f} | {row['shuffle']:.3f} | {row['energy_only']:.3f} | {row['full_minus_shuffle']:+.3f} | {row['rank_modules_changed_from_full']}/72 |")
    lines += ['', '## 解释边界', '',
              '- 固定任务、训练 seed、分类头、训练 RNG、参数预算及 SVD 初始化后，原始共识的模块对应关系在这一个预注册 shuffle 下带来正向收益。两任务六个配对全部为正。',
              '- 仅有一个 shuffle seed、两个任务和每任务三个训练种子。该结果支持当前条件下的共识特异性，不能推断对任意打乱、所有任务、所有校准子集均成立。',
              '- MRPC 跨校准 15 对平均 -0.147 点、CoLA/RTE 固定 seed42 的跨校准负结果仍须在论文中保留；本实验不能抵消这些边界。',
              '- 核对通过：六次预算均为 755,712；每次 72/72 模块的共识曲线来自另一同类型同形状模块；分类头哈希、CPU/CUDA 训练 RNG 哈希与同 seed 参考一致。',
              '', f'逐次数据与源文件路径：`{CSV}`。']
    SUMMARY.write_text('\n'.join(lines) + '\n')
    print(SUMMARY)


if __name__ == '__main__':
    main()
