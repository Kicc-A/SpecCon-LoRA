#!/usr/bin/env python3
"""Refresh EXP-01 tables and figures from validated final-checkpoint metrics."""
import csv
import itertools
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/workspace/SpecCon-LoRA/experiments/mrpc_calibration_robustness_v1')
OLD = Path('/workspace/SpecCon-LoRA/experiments/extended_ablation_20261021_v2')
SEEDS = (42, 777, 2026)
LAMBDAS = (0.0, 1.5)
P = 755712


def read(path):
    return json.loads(Path(path).read_text())


def save_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = read(ROOT / 'manifest.json')
    old = read(OLD / 'manifest.json')
    prior = {(j['spec']['replicate'], j['spec']['lambda']): j for j in old['jobs']
             if j['spec']['group'] == 'calibration_subset' and j['spec']['task'] == 'mrpc'
             and j['spec']['seed'] == 42}
    assert len(prior) == 10
    new = {(j['spec']['replicate'], j['spec']['seed'], j['spec']['lambda']): j
           for j in manifest['jobs']}
    assert len(new) == 20
    runs = []
    for rep in range(5):
        for seed in SEEDS:
            for lam in LAMBDAS:
                j = prior[rep, lam] if seed == 42 else new[rep, seed, lam]
                folder = Path(j['output'])
                if not (folder / 'metrics.json').exists():
                    continue
                metric = read(folder / 'metrics.json')
                initial = read(folder / 'initial_state.json')
                alloc = read(folder / 'allocation.json')
                calib = read(folder / 'calibration_rows.json')
                rng = read(folder / 'training_rng.json')
                assert metric['state'] == 'complete' and metric['final'] == metric['selected']
                assert metric['adapter_parameters'] == P and metric['epochs'] == 57
                assert initial['classifier_sha256'] == '8525992b7c0f62ea5e7d4d9ba33a20aa71acf73016426ef8315c03aa6386e112'
                assert calib['ids'] == j['spec']['independent_ids']
                assert alloc['rank_report']['budget_used'] == P
                if seed != 42:
                    old_alloc = read(Path(prior[rep, lam]['output']) / 'allocation.json')
                    assert alloc['rank_pattern'] == old_alloc['rank_pattern']
                    assert initial['adapter_sha256'] == read(Path(prior[rep, lam]['output']) / 'initial_state.json')['adapter_sha256']
                runs.append({'replicate': rep, 'training_seed': seed, 'lambda': lam,
                    'method': 'Energy' if lam == 0 else 'Consensus',
                    'final_accuracy_percent': 100 * metric['final']['splits']['validation']['accuracy'],
                    'head_sha256': initial['classifier_sha256'],
                    'adapter_initial_sha256': initial['adapter_sha256'],
                    'training_cpu_rng_sha256': rng['cpu_sha256'],
                    'training_cuda_rng_sha256': rng['cuda_sha256'],
                    'output': str(folder), 'reused': seed == 42})
    save_csv(ROOT / 'runs.csv', runs, list(runs[0]))
    lookup = {(r['replicate'], r['training_seed'], r['lambda']): r for r in runs}
    pairs = []
    for rep in range(5):
        for seed in SEEDS:
            e, c = lookup.get((rep, seed, 0.0)), lookup.get((rep, seed, 1.5))
            if e is None or c is None:
                continue
            assert e['training_cpu_rng_sha256'] == c['training_cpu_rng_sha256']
            assert e['training_cuda_rng_sha256'] == c['training_cuda_rng_sha256']
            pairs.append({'replicate': rep, 'training_seed': seed,
                'energy_accuracy': e['final_accuracy_percent'],
                'consensus_accuracy': c['final_accuracy_percent'],
                'delta_consensus_minus_energy': c['final_accuracy_percent']-e['final_accuracy_percent']})
    save_csv(ROOT / 'paired_results.csv', pairs,
        ['replicate', 'training_seed', 'energy_accuracy', 'consensus_accuracy', 'delta_consensus_minus_energy'])
    ranks = {}
    for lam in LAMBDAS:
        for rep in range(5):
            ranks[rep, lam] = read(Path(prior[rep,lam]['output']) / 'allocation.json')['rank_pattern']
    (ROOT/'rank_allocations.json').write_text(json.dumps({
        'model_initialization_seed':42,
        'rank_maps_shared_across_training_seeds':True,
        'source':'frozen MRPC A0..A4 seed42 allocation.json; new runs must match',
        'maps':{f'A{rep}_{"Energy" if lam==0 else "Consensus"}':ranks[rep,lam]
                for rep in range(5) for lam in LAMBDAS}},indent=2)+'\n')
    names = sorted(ranks[0,0.0])
    costs = {}
    for name in names:
        # Frozen reference reports precise shape-dependent rank cost.
        details = read(Path(prior[0,0.0]['output']) / 'allocation_details.json')
        costs[name] = int(details[name]['rank_cost'])
    distances = []
    for seed in SEEDS:
        for lam in LAMBDAS:
            for a,b in itertools.combinations(range(5),2):
                distance = sum(costs[name]*abs(ranks[a,lam][name]-ranks[b,lam][name])
                               for name in names)/(2*P)
                distances.append({'training_seed': seed, 'method': 'Energy' if lam == 0 else 'Consensus',
                    'calibration_a': a, 'calibration_b': b, 'cost_weighted_rank_distance': distance,
                    'rank_map_source': 'frozen seed42; identical model/gradient cache across seeds'})
    save_csv(ROOT / 'rank_distance.csv', distances,
        ['training_seed','method','calibration_a','calibration_b','cost_weighted_rank_distance','rank_map_source'])
    heat = np.full((5,3), np.nan)
    for pair in pairs:
        heat[pair['replicate'], SEEDS.index(pair['training_seed'])] = pair['delta_consensus_minus_energy']
    fig, ax = plt.subplots(figsize=(6,4))
    im = ax.imshow(heat, cmap='coolwarm', vmin=-2.5, vmax=2.5, aspect='auto')
    ax.set_xticks(range(3), SEEDS); ax.set_yticks(range(5), [f'A{i}' for i in range(5)])
    ax.set_xlabel('Training seed'); ax.set_ylabel('Calibration set')
    ax.set_title('Consensus minus Energy: MRPC final Accuracy points')
    for i in range(5):
        for j in range(3):
            if np.isfinite(heat[i,j]): ax.text(j,i,f'{heat[i,j]:+.2f}',ha='center',va='center')
    fig.colorbar(im, ax=ax, label='Paired difference (pp)')
    fig.tight_layout(); fig.savefig(ROOT/'calibration_seed_heatmap.png',dpi=180); plt.close(fig)
    matrix = np.array([[ranks[rep,lam][name] for rep in range(5)] for lam in LAMBDAS for name in names])
    fig, ax = plt.subplots(figsize=(7,14))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=2, vmax=16)
    ax.set_xticks(range(5), [f'A{i}' for i in range(5)])
    ax.set_yticks([0,72], ['Energy', 'Consensus'])
    ax.axhline(71.5,color='white',lw=1)
    ax.set_xlabel('Calibration set');ax.set_title('Module ranks: 72 modules × two methods')
    fig.colorbar(im, ax=ax, label='Rank')
    fig.tight_layout();fig.savefig(ROOT/'rank_variability_heatmap.png',dpi=180);plt.close(fig)
    deltas = [p['delta_consensus_minus_energy'] for p in pairs]
    by_seed = {seed:[p['delta_consensus_minus_energy'] for p in pairs if p['training_seed']==seed] for seed in SEEDS}
    by_rep = {rep:[p['delta_consensus_minus_energy'] for p in pairs if p['replicate']==rep] for rep in range(5)}
    sensitive = sorted(((name, float(np.std([ranks[rep,1.5][name] for rep in range(5)],ddof=1)))
        for name in names), key=lambda x:-x[1])[:12]
    sensitivity=[{'module':name,'rank_cost':costs[name],
        'energy_rank_sd':float(np.std([ranks[rep,0.0][name] for rep in range(5)],ddof=1)),
        'consensus_rank_sd':float(np.std([ranks[rep,1.5][name] for rep in range(5)],ddof=1)),
        'energy_ranks':','.join(str(ranks[rep,0.0][name]) for rep in range(5)),
        'consensus_ranks':','.join(str(ranks[rep,1.5][name]) for rep in range(5))}
        for name in names]
    save_csv(ROOT/'rank_sensitivity.csv',sensitivity,list(sensitivity[0]))
    lines = ['# EXP-01 MRPC 跨校准多种子可靠性', '',
        f'已完成 {len(runs)}/30 次，其中原有10次；已形成 {len(pairs)}/15 对。所有分数均为最终 checkpoint Accuracy (%)。',
        '新训练固定与缓存梯度一致的 seed42 模型和分类头，训练顺序及 dropout RNG 按 777/2026 变化。',
        '因此结果检验共同初始模型下的训练随机性，不检验不同分类头初始化。', '',
        '原默认配置用训练集前512条即时采集校准梯度；A0–A4 用独立机制实验预采集的随机512条梯度，'
        '校准 dropout seed 固定7001。默认配置与A组的绝对分数对比同时改变了样本及校准阶段随机流，'
        '不能称只改变样本的纯因果对照。A组内部同一(子集,训练seed)的两种lambda共享梯度缓存、'
        '样本ID、分块、分类头和训练随机状态，因此其配对差值可检验该A组协议下的Consensus贡献。', '',
        '## 观察事实', '',
        '|校准集|seed42 Δ|seed777 Δ|seed2026 Δ|', '|---|---:|---:|---:|']
    for rep in range(5):
        cells = [next((p['delta_consensus_minus_energy'] for p in pairs if p['replicate']==rep
                       and p['training_seed']==seed),None) for seed in SEEDS]
        lines.append('|A'+str(rep)+'|'+'|'.join('—' if v is None else f'{v:+.3f}' for v in cells)+'|')
    lines += ['', '|训练 seed|已完成配对|平均 Δ|', '|---:|---:|---:|']
    for seed, values in by_seed.items():
        lines.append(f'|{seed}|{len(values)}/5|'+('—' if not values else f'{statistics.mean(values):+.3f}')+'|')
    if deltas:
        win = sum(v>0 for v in deltas);tie=sum(v==0 for v in deltas)
        lines += ['',f'当前 {len(deltas)} 对描述性均值 {statistics.mean(deltas):+.3f} pp，样本 SD '
            + (f'{statistics.stdev(deltas):.3f}' if len(deltas)>1 else '—')
            + f'；胜/平/负 {win}/{tie}/{len(deltas)-win-tie}。']
    lines += ['', '不同 A 子集可能重叠，五组共用训练任务；15 对不能视作独立随机样本。',
        'rank map 与训练 seed 无关：三个种子使用相同 seed42 模型、A 梯度缓存与样本 ID；'
        '新训练的 rank map 与对应 seed42 参考逐项比较。完整72模块×五子集×两方法配置见 '
        '`rank_allocations.json`，逐模块敏感性见 `rank_sensitivity.csv`。', '',
        '|方法|10个 A 子集两两距离均值|', '|---|---:|']
    for method in ('Energy','Consensus'):
        values = [r['cost_weighted_rank_distance'] for r in distances if r['training_seed']==42 and r['method']==method]
        lines.append(f'|{method}|{statistics.mean(values):.4f}|')
    lines += ['', '跨子集 rank 变化最大的模块（共识配置，五个 A 的 rank 样本 SD）：', '',
        '|模块|rank SD|', '|---|---:|']
    lines += [f'|{name}|{sd:.3f}|' for name,sd in sensitive]
    lines += ['', '## 可能机制解释', '',
        'Consensus 改变同一预算下的模块 rank；若新种子仍以负配对收益为主，几何评分信息可能未转化为下游收益。'
        '这只是待检验解释，不能把相关性直接解释为原因。', '',
        '## 目前不能确定', '',
        '在30次训练全部完成之前，不能把部分结果当作完整三种子矩阵。'
        '校准子集重叠、固定分类头和任务单一限制外推。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
