#!/usr/bin/env python3
"""Independent statistics on frozen A/B calibration candidate records."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT = Path('/workspace/SpecCon-LoRA')
SOURCE = PROJECT / 'reports/independent_calibration_mechanism_20261019'
MECH = PROJECT / 'experiments/independent_calibration_mechanism_20261019_v1'
OUT = PROJECT / 'experiments/geometric_predictivity_validation_v1'
TASKS = ('cola', 'mrpc', 'rte')
SEED = 20261027
PERMUTATIONS = 999
BOOTSTRAPS = 999


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(rows, filename):
    pd.DataFrame(rows).to_csv(OUT / filename, index=False)


def as_cube(frame):
    """Return [draw, type, module, nested-rank] arrays in a fixed module order."""
    fields = ('consistency', 'log_cumulative_energy', 'recurrence')
    cubes = {field: [] for field in fields}
    types = sorted(frame.module_type.unique())
    for rep in range(5):
        by_type = {field: [] for field in fields}
        selected = frame[frame.replicate == rep]
        for typ in types:
            group = selected[selected.module_type == typ]
            modules = sorted(group.module.unique())
            assert len(modules) == 12
            for field in fields:
                matrix = np.vstack([group[group.module == module].sort_values('k')[field].to_numpy()
                                    for module in modules])
                assert matrix.shape == (12, 14)
                by_type[field].append(matrix)
        for field in fields:
            cubes[field].append(np.stack(by_type[field]))
    return {field: np.stack(cubes[field]) for field in fields}


def standard_rank(array):
    ranked = rankdata(array, axis=2)
    return (ranked - ranked.mean(axis=2, keepdims=True)) / np.maximum(
        ranked.std(axis=2, keepdims=True), 1e-12)


def corr(x, y, adjust=None):
    if adjust is not None:
        x = x - adjust * np.mean(x * adjust, axis=2, keepdims=True)
        y = y - adjust * np.mean(y * adjust, axis=2, keepdims=True)
    numerator = np.sum(x * y, axis=(1, 2, 3))
    denominator = np.sqrt(np.sum(x*x, axis=(1, 2, 3)) * np.sum(y*y, axis=(1, 2, 3)))
    return numerator / denominator


def evaluate(cubes):
    c = standard_rank(cubes['consistency'])
    e = standard_rank(cubes['log_cumulative_energy'])
    y = standard_rank(cubes['recurrence'])
    return {'raw_energy': corr(e, y), 'consensus': corr(c, y),
            'consensus_given_energy': corr(c, y, e),
            'energy_given_consensus': corr(e, y, c)}


def energy_bins(energy):
    """Four modules per tertile, per draw and module type."""
    means = energy.mean(axis=3)
    bins = np.empty(means.shape, dtype=int)
    for draw in range(means.shape[0]):
        for typ in range(means.shape[1]):
            bins[draw, typ, np.argsort(means[draw, typ])] = np.repeat(range(3), 4)
    return bins


def permuted_consensus_rank(ranked, bins, rng):
    result = ranked.copy()
    for draw in range(ranked.shape[0]):
        for typ in range(ranked.shape[1]):
            for group in range(3):
                indices = np.flatnonzero(bins[draw, typ] == group)
                result[draw, typ, indices] = ranked[draw, typ, rng.permutation(indices)]
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = SOURCE / 'candidate_metrics.csv'
    frame = pd.read_csv(source)
    assert len(frame) == 3 * 5 * 2 * 72 * 16
    assert set(frame.task) == set(TASKS)
    assert set(frame['k']) == set(range(1, 17))
    assert frame.groupby(['task', 'replicate', 'order', 'module']).size().eq(16).all()
    assert frame.groupby(['task', 'replicate', 'order']).module.nunique().eq(72).all()
    frame = frame[(frame['order'] == 0) & (frame['k'] >= 3)].copy()
    assert frame.groupby(['task', 'replicate']).size().eq(72 * 14).all()
    manifests = json.loads((MECH / 'manifest.json').read_text())
    for task in TASKS:
        assert len(manifests['tasks'][task]['replicates']) == 5
        done = json.loads((MECH / 'runs' / task / 'DONE.json').read_text())
        assert done['complete'] and done['model_unchanged']
        for artifact, digest in done['artifacts'].items():
            assert sha(MECH / 'runs' / task / artifact) == digest
    rng = np.random.default_rng(SEED)
    adjusted, permutations, bootstraps = [], [], []
    for task in TASKS:
        cubes = as_cube(frame[frame.task == task])
        observed = evaluate(cubes)
        for rep in range(5):
            adjusted.append({'task': task, 'replicate': rep, 'unit': 'A calibration draw',
                             **{key: float(values[rep]) for key, values in observed.items()}})
        obs = float(observed['consensus_given_energy'].mean())
        c = standard_rank(cubes['consistency'])
        e = standard_rank(cubes['log_cumulative_energy'])
        y = standard_rank(cubes['recurrence'])
        bins = energy_bins(cubes['log_cumulative_energy'])
        null = []
        for iteration in range(PERMUTATIONS):
            shuffled = permuted_consensus_rank(c, bins, rng)
            null.append(float(corr(shuffled, y, e).mean()))
        null = np.asarray(null)
        permutations.append({'task': task, 'statistic': 'mean partial rank correlation across five A draws',
            'observed': obs, 'null_mean': float(null.mean()), 'null_sd': float(null.std(ddof=1)),
            'p_two_sided': float((1 + np.count_nonzero(abs(null) >= abs(obs))) / (PERMUTATIONS + 1)),
            'iterations': PERMUTATIONS,
            'permutation_unit': 'entire nested-rank module curve within module type and module-energy tertile',
            'seed': SEED})
        values = []
        for iteration in range(BOOTSTRAPS):
            selected = rng.integers(0, 12, size=(1, c.shape[1], 12, 1))
            sampled = {key: np.take_along_axis(array, selected, axis=2)
                       for key, array in cubes.items()}
            vals = evaluate(sampled)
            values.append({key: float(val.mean()) for key, val in vals.items()})
        for key in observed:
            series = np.array([v[key] for v in values])
            bootstraps.append({'task': task, 'statistic': key,
                'observed_mean': float(observed[key].mean()),
                'bootstrap_mean': float(series.mean()),
                'ci_lower_2_5': float(np.quantile(series, .025)),
                'ci_upper_97_5': float(np.quantile(series, .975)),
                'iterations': BOOTSTRAPS, 'cluster_unit': 'module, all nested ranks and A draws',
                'seed': SEED})
        print(task, 'observed partial', round(obs, 4), 'perm p', permutations[-1]['p_two_sided'], flush=True)
    save(adjusted, 'adjusted_correlations.csv')
    save(permutations, 'permutation_tests.csv')
    save(bootstraps, 'group_bootstrap.csv')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    keys = [('raw_energy', 'Raw energy'), ('consensus', 'Consensus'),
            ('consensus_given_energy', 'Consensus | energy'),
            ('energy_given_consensus', 'Energy | consensus')]
    for ax, task in zip(axes, TASKS):
        row = {r['statistic']: r for r in bootstraps if r['task'] == task}
        for index, (key, label) in enumerate(keys):
            item = row[key]
            mean = item['observed_mean']
            ax.errorbar(index, mean, yerr=[[mean-item['ci_lower_2_5']],
                [item['ci_upper_97_5']-mean]], fmt='o', capsize=3)
        ax.axhline(0, color='.6', lw=.8)
        ax.set_xticks(range(4), [v[1] for v in keys], rotation=35, ha='right')
        ax.set_title(task.upper())
        ax.set_ylim(-1, 1)
    axes[0].set_ylabel('Within-rank/type rank correlation')
    fig.suptitle('Module-cluster bootstrap; 5 overlapping A draws, shared B')
    fig.tight_layout()
    fig.savefig(OUT / 'correlation_controls.png', dpi=180)
    plt.close(fig)
    lines = ['# EXP-03 几何相关性的独立统计复核', '',
        f'输入：`{source}`，SHA256 `{sha(source)}`。随机种子 {SEED}。',
        '仅使用预设 order0 的五组 A 校准、可分配 k=3…16、三个任务。每组72模块×14个 nested rank。',
        '原始机制实验的 DONE 哈希均通过复核；A/B 样本互斥，五组 A 可重叠且共用 B。', '',
        '## 观察事实', '',
        '|任务|Raw energy|Consensus|Consensus 控制 energy|Energy 控制 Consensus|模块置换 p|',
        '|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        rows = {r['statistic']: r for r in bootstraps if r['task'] == task}
        p = next(r['p_two_sided'] for r in permutations if r['task'] == task)
        lines.append('|'+task.upper()+'|'+ '|'.join(f"{rows[k]['observed_mean']:.3f} [{rows[k]['ci_lower_2_5']:.3f}, {rows[k]['ci_upper_97_5']:.3f}]" for k,_ in keys)+f'|{p:.4f}|')
    lines += ['',
        '每个候选先在同 rank、模块类型内做秩标准化；条件相关再对 log 累积能量或 Consensus 做线性残差化。',
        '置换把整个模块的 k=3…16 Consensus 曲线分配给同类型、模块平均谱能量三分位内的另一模块；'
        '每次 A 校准独立置换，保留 nested-rank 曲线、模块形状、rank 与大致能量结构。双侧 p 使用加一校正。',
        'Bootstrap 在模块层分层重采样，并将选中的模块带入全部五组 A 校准及其所有 k；'
        '区间只描述模块抽样敏感性，不覆盖共同 B、A 重叠、模型种子与训练任务不确定性。', '',
        '## 可能机制解释', '',
        'Consensus 与 held-out recurrence 都包含子空间投影结构，因此较高相关可能部分来自共同的几何定义。'
        '加入模块类型和谱能量控制后若仍正相关，说明它不只是形状、rank 或能量的简单代理。', '',
        '## 目前不能确定', '',
        '这里的 held-out recurrence 仍是固定初始化点的几何量，无法代替真实下游 Accuracy 或证明因果关系。'
        '同一 B 被重复使用，置换与 Bootstrap 的 p/区间不应解释为跨独立数据集的显著性。'
        '原始 B 投影基底和梯度虽有保存，但本次未计算高维随机重合的精确零基线；该问题留作补充诊断。', '',
        '所有原始逐次统计见 `adjusted_correlations.csv`，置换和 Bootstrap 见对应 CSV。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT / 'config.json').write_text(json.dumps({'source':str(source),'source_sha256':sha(source),
        'mechanism_manifest_sha256':sha(MECH/'manifest.json'),'seed':SEED,
        'permutations':PERMUTATIONS,'bootstraps':BOOTSTRAPS},indent=2)+'\n')


if __name__ == '__main__':
    main()
